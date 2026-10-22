"""Detecção do coração: caixas, formato YOLO, IoU e o dataset de treino.

Aulas "A caixa e o IoU", "Confiança e a briga das caixas" (NMS) e o curso de
YOLO do professor (aula 4, "Treino com dataset próprio"). A caixa é a
"moldura" que a segmentação do notebook 08 refinava; aqui ela volta como alvo
de um detector.

**Formato YOLO:** um `.txt` por imagem, uma linha por objeto:
`classe x_centro y_centro largura altura`, as quatro medidas divididas pelo
tamanho da imagem (entre 0 e 1). Imagem sem objeto = arquivo vazio.

**Pseudo-rótulos:** a base não tem caixas do coração. As de treino saem das
bordas que o perfil do ICT (notebook 04) mede em cada linha: um método
clássico rotula, a rede aprende. Os rótulos herdam os defeitos do método (o
perfil só mede as linhas 50 a 90, então o topo do coração fica de fora); o
notebook 09 mede isso contra os contornos marcados à mão.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src import config
from src.visao.ict import MedidaICT

Caixa = tuple[float, float, float, float]  # (x0, y0, x1, y1) em pixels


def caixa_do_ict(medida: MedidaICT, quantil: float = 10) -> Caixa | None:
    """Caixa do coração a partir das bordas por linha do perfil do ICT.

    Na horizontal, do percentil `quantil` da borda direita do coração ao
    percentil 100 − `quantil` da borda esquerda (robusto a uma linha ruim);
    na vertical, da primeira à última linha medida. None se não houve medida.
    """
    if not medida.ok:
        return None
    b = medida.bordas
    return (
        float(np.percentile(b[:, 2], quantil)),
        float(b[:, 0].min()),
        float(np.percentile(b[:, 3], 100 - quantil)) + 1,
        float(b[:, 0].max()) + 1,
    )


def caixa_do_poligono(pontos) -> Caixa:
    """Menor caixa alinhada aos eixos que contém o polígono."""
    p = np.asarray(pontos, float)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def para_yolo(caixa: Caixa, lado: int = config.TAMANHO) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) em pixels → (x_centro, y_centro, largura, altura) normalizados."""
    x0, y0, x1, y1 = caixa
    return ((x0 + x1) / 2 / lado, (y0 + y1) / 2 / lado, (x1 - x0) / lado, (y1 - y0) / lado)


def de_yolo(xc: float, yc: float, w: float, h: float, lado: int = config.TAMANHO) -> Caixa:
    """O caminho de volta: formato YOLO normalizado → (x0, y0, x1, y1) em pixels."""
    return ((xc - w / 2) * lado, (yc - h / 2) * lado, (xc + w / 2) * lado, (yc + h / 2) * lado)


def iou_caixas(a: Caixa, b: Caixa) -> float:
    """Interseção sobre união de duas caixas (x0, y0, x1, y1)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def gravar_exemplo(pasta: Path, nome: str, imagem: np.ndarray, caixas: list[Caixa]) -> None:
    """Grava `images/<nome>.png` e `labels/<nome>.txt` (classe 0 = coração) em `pasta`."""
    (pasta / "images").mkdir(parents=True, exist_ok=True)
    (pasta / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pasta / "images" / f"{nome}.png"), imagem)
    linhas = ["0 " + " ".join(f"{v:.6f}" for v in para_yolo(c, imagem.shape[1])) for c in caixas]
    (pasta / "labels" / f"{nome}.txt").write_text("\n".join(linhas) + ("\n" if linhas else ""))


def gravar_yaml(arquivo: Path, raiz: Path, treino: str, validacao: str) -> Path:
    """O arquivo de dados que o YOLO lê: caminhos e o nome da única classe."""
    arquivo.write_text(
        f"path: {raiz}\ntrain: {treino}\nval: {validacao}\nnames:\n  0: coracao\n",
        encoding="utf-8",
    )
    return arquivo
