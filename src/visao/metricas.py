"""Concordância entre máscaras: Dice e IoU.

Aula de segmentação: "Dice: acordo de área, não de caixa". As duas medidas
comparam duas máscaras binárias pixel a pixel e vão de 0 (nada em comum) a 1
(idênticas). O Dice dá peso dobrado à interseção e por isso é sempre ≥ IoU.
"""

from __future__ import annotations

import numpy as np


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
