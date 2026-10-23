"""Transfer learning com poucos dados: bases congeladas, features e a cabeça.

Exercício "poucas imagens por classe": uma rede treinada em outro problema
vira um extrator de features congelado, e só a última camada aprende com os
nossos rótulos. Com a base congelada, a imagem passa por ela **uma vez só**: as
features ficam em cache e cada treino da cabeça leva segundos, o que permite
repetir o sorteio muitas vezes por tamanho de amostra.

Três bases, todas congeladas:

* **MobileNetV2 (ImageNet)**: a do exercício e do notebook 05;
* **EfficientNetB0 (ImageNet)**: mais moderna, do mesmo porte;
* **DenseNet121 do CheXpert** (`torchxrayvision`, Apache-2.0): treinada em
  raio-X de tórax de Stanford. É transferência **do mesmo domínio**. Não há
  imagem do NIH no CheXpert; as versões do `torchxrayvision` treinadas com o
  NIH ficam de fora, porque viram os nossos exames com rótulo de cardiomegalia.

A cabeça é uma camada densa com sigmoide, o mesmo que uma **regressão
logística** sobre as features: o `scikit-learn` resolve esse problema exato e
sem sorteio de inicialização. A força da regularização é escolhida por
validação cruzada **dentro da própria amostra de treino**: com 10 exames por
classe não há outra validação honesta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config

BASES = ("MobileNetV2 · ImageNet", "EfficientNetB0 · ImageNet", "DenseNet121 · CheXpert")
LADO_IMAGENET = 224


def _keras(nome: str, imagens: np.ndarray, lote: int) -> np.ndarray:
    import keras
    import tensorflow as tf

    if nome.startswith("MobileNetV2"):
        base = keras.applications.MobileNetV2(include_top=False, pooling="avg", weights="imagenet")
        preparar = keras.applications.mobilenet_v2.preprocess_input
    else:
        base = keras.applications.EfficientNetB0(
            include_top=False, pooling="avg", weights="imagenet"
        )
        preparar = keras.applications.efficientnet.preprocess_input
    saida = []
    for inicio in range(0, len(imagens), lote):
        x = imagens[inicio : inicio + lote].astype(np.float32)[..., None]
        x = tf.image.resize(np.repeat(x, 3, axis=-1), (LADO_IMAGENET, LADO_IMAGENET))
        saida.append(base.predict(preparar(x), verbose=0))
    return np.concatenate(saida)


def _chexpert(imagens: np.ndarray, lote: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    import torchxrayvision as xrv

    dispositivo = "mps" if torch.backends.mps.is_available() else "cpu"
    modelo = xrv.models.DenseNet(weights="densenet121-res224-chex").eval().to(dispositivo)
    saida = []
    with torch.no_grad():
        for inicio in range(0, len(imagens), lote):
            bloco = xrv.datasets.normalize(imagens[inicio : inicio + lote].astype(np.float32), 255)
            x = torch.from_numpy(bloco[:, None]).to(dispositivo)
            x = F.interpolate(x, size=LADO_IMAGENET, mode="bilinear", align_corners=False)
            saida.append(modelo.features2(x).cpu().numpy())
    return np.concatenate(saida)


def extrair(nome: str, imagens: np.ndarray, lote: int = 64) -> np.ndarray:
    """Features (n, d) da base `nome` (uma de `BASES`), imagens `uint8` 128 × 128."""
    if nome not in BASES:
        raise ValueError(f"base desconhecida: {nome}")
    if nome.startswith("DenseNet121"):
        return _chexpert(imagens, lote)
    return _keras(nome, imagens, lote)


def extrair_com_cache(
    nome: str, ids: np.ndarray, imagens: np.ndarray, espelhar: bool = False, refazer: bool = False
):
    """Como `extrair`, com cache em `data/processed/features/`; refaz se os ids mudarem.

    Com `espelhar=True`, extrai do raio-X espelhado na horizontal (o coração
    passa para o lado direito), em cache separado.
    """
    sufixo = "_espelhado" if espelhar else ""
    arquivo: Path = config.DATA_PROCESSED / "features" / f"{nome.split(' ')[0].lower()}{sufixo}.npz"
    if arquivo.exists() and not refazer:
        guardado = np.load(arquivo, allow_pickle=False)
        if np.array_equal(guardado["ids"], np.asarray(ids, str)):
            return guardado["features"]
    entrada = np.ascontiguousarray(imagens[:, :, ::-1]) if espelhar else imagens
    features = extrair(nome, entrada).astype(np.float32)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arquivo, ids=np.asarray(ids, str), features=features)
    return features


def sortear(y: np.ndarray, candidatos: np.ndarray, n_por_classe: int, semente: int) -> np.ndarray:
    """Índices de `n_por_classe` exames de cada classe, sorteados entre `candidatos` (bool)."""
    rng = np.random.default_rng(semente)
    escolhidos = []
    for classe in (0, 1):
        pool = np.flatnonzero(candidatos & (y == classe))
        if n_por_classe > len(pool):
            raise ValueError(f"só há {len(pool)} exames da classe {classe}")
        escolhidos.append(rng.choice(pool, n_por_classe, replace=False))
    return np.sort(np.concatenate(escolhidos))


def cabeca(n_por_classe: int):
    """Padronização + regressão logística L2, com C escolhido por validação cruzada na amostra.

    Até 5 dobras estratificadas, nunca mais dobras que exames por classe; a
    escolha é pela AUC.
    """
    dobras = int(min(5, n_por_classe))
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=[1e-3, 1e-2, 1e-1, 1.0],
            cv=dobras,
            scoring="roc_auc",
            l1_ratios=(0.0,),
            use_legacy_attributes=False,
            max_iter=3000,
        ),
    )
