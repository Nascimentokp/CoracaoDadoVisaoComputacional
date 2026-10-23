"""Concordância entre máscaras: Dice e IoU, e a distância entre as bordas.

Aula de segmentação: "Dice: acordo de área, não de caixa". As duas medidas
comparam duas máscaras binárias pixel a pixel e vão de 0 (nada em comum) a 1
(idênticas). O Dice dá peso dobrado à interseção e por isso é sempre ≥ IoU.

"O pixel da borda pesa mais": num coração grande, errar a borda por 3 pixels
quase não mexe no Dice, mas muda o ICT. As distâncias de borda medem isso em
pixels: para cada ponto do contorno de uma máscara, a distância até o contorno
da outra. A **média** resume o erro típico; o **Hausdorff 95** (percentil 95,
nos dois sentidos) pega o pedaço que mais escapou sem ser refém de um pixel.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """2·|A ∩ B| / (|A| + |B|). Duas máscaras vazias concordam totalmente (1)."""
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    soma = a.sum() + b.sum()
    return 1.0 if soma == 0 else float(2 * (a & b).sum() / soma)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """|A ∩ B| / |A ∪ B|. Duas máscaras vazias concordam totalmente (1)."""
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    uniao = (a | b).sum()
    return 1.0 if uniao == 0 else float((a & b).sum() / uniao)


def borda(mascara: np.ndarray) -> np.ndarray:
    """Pixels da máscara que tocam o fundo (vizinhança 4)."""
    m = np.asarray(mascara) > 0
    return m & ~ndi.binary_erosion(m, border_value=0)


def distancias_de_borda(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distância (px) de cada ponto da borda de A à borda de B e vice-versa, concatenadas.

    Vazia se alguma das máscaras for vazia: não há borda para comparar.
    """
    ba, bb = borda(a), borda(b)
    if not ba.any() or not bb.any():
        return np.array([])

    def ate(alvo):  # distância de cada pixel ao pixel de borda mais próximo de `alvo`
        return cv2.distanceTransform((~alvo).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    return np.concatenate([ate(bb)[ba], ate(ba)[bb]])


def distancia_media_borda(a: np.ndarray, b: np.ndarray) -> float:
    """Erro típico do contorno, em pixels. NaN se alguma máscara for vazia."""
    d = distancias_de_borda(a, b)
    return float(d.mean()) if d.size else float("nan")


def hausdorff95(a: np.ndarray, b: np.ndarray) -> float:
    """Percentil 95 das distâncias de borda nos dois sentidos, em pixels. NaN se vazia."""
    d = distancias_de_borda(a, b)
    return float(np.percentile(d, 95)) if d.size else float("nan")
