"""Rede de segmentação pré-treinada para anatomia do tórax (torchxrayvision).

A PSPNet do `torchxrayvision` (Cohen et al., licença Apache-2.0) segmenta 14
estruturas em raio-X frontal; aqui só interessam **coração** e os **dois
pulmões**. Foi treinada no ChestX-Det, um recorte de 3.578 imagens do mesmo
ChestX-ray14 do NIH de onde vem a nossa base, com contornos de radiologistas.
Por isso alguns raio-X do nosso teste podem ter estado no treino dela: o Dice
contra o gabarito pode sair otimista (o mesmo aviso do exemplo de RM cerebral
com U-Net pré-treinada).

A rede espera 512 × 512 com intensidades em [−1024, 1024]; as nossas imagens
são 128 × 128 em `uint8`. A conversão vai e volta aqui dentro, e a saída volta
para 128 × 128 por média de área. O resultado fica em cache em
`data/processed/pretreinada.npz`, porque rodar a base inteira leva minutos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src import config

ESTRUTURAS = ("Heart", "Left Lung", "Right Lung")
CACHE = config.DATA_PROCESSED / "pretreinada.npz"


def _dispositivo() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def segmentar(imagens: np.ndarray, lote: int = 8, dispositivo: str | None = None) -> np.ndarray:
    """Probabilidades (n, 3, 128, 128) em float16: coração, pulmão esquerdo, pulmão direito."""
    import torch
    import torch.nn.functional as F
    import torchxrayvision as xrv

    dispositivo = dispositivo or _dispositivo()
    modelo = xrv.baseline_models.chestx_det.PSPNet().eval().to(dispositivo)
    canais = [modelo.targets.index(e) for e in ESTRUTURAS]
    saida = np.zeros((len(imagens), len(ESTRUTURAS), config.TAMANHO, config.TAMANHO), np.float16)
    with torch.no_grad():
        for inicio in range(0, len(imagens), lote):
            bloco = imagens[inicio : inicio + lote].astype(np.float32)
            x = torch.from_numpy(xrv.datasets.normalize(bloco, 255)[:, None]).to(dispositivo)
            x = F.interpolate(x, size=512, mode="bilinear", align_corners=False)
            # Já em 512, o redimensionamento interno do modelo não faz nada; a saída são logits.
            p = torch.sigmoid(modelo(x))[:, canais]
            p = F.interpolate(p, size=config.TAMANHO, mode="area")
            saida[inicio : inicio + lote] = p.cpu().numpy()
    return saida


def segmentar_com_cache(
    ids: np.ndarray, imagens: np.ndarray, cache: Path = CACHE, refazer: bool = False
) -> np.ndarray:
    """Como `segmentar`, mas lê e grava o cache; refaz se os ids não baterem."""
    if cache.exists() and not refazer:
        guardado = np.load(cache, allow_pickle=False)
        if np.array_equal(guardado["ids"], np.asarray(ids, str)):
            return guardado["probabilidades"]
    probabilidades = segmentar(imagens)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, ids=np.asarray(ids, str), probabilidades=probabilidades)
    return probabilidades
