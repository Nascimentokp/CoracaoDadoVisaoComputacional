"""Dice, IoU e distâncias de borda em casos de resposta conhecida (MET-01, MET-02)."""

import numpy as np
import pytest

from src.visao.metricas import borda, dice, distancia_media_borda, hausdorff95, iou


def test_identicas_e_disjuntas():
    a = np.zeros((10, 10), bool)
    a[2:5, 2:5] = True
    b = np.zeros((10, 10), bool)
    b[6:9, 6:9] = True
    assert dice(a, a) == iou(a, a) == 1.0
    assert dice(a, b) == iou(a, b) == 0.0


def test_metade_sobreposta():
    a = np.zeros((4, 4), bool)
    a[:, :2] = True  # 8 pixels
    b = np.zeros((4, 4), bool)
    b[:, 1:3] = True  # 8 pixels, 4 em comum
    assert dice(a, b) == pytest.approx(0.5)
    assert iou(a, b) == pytest.approx(4 / 12)


def test_dice_nunca_menor_que_iou():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.random((16, 16)) > 0.5, rng.random((16, 16)) > 0.5
        assert dice(a, b) >= iou(a, b)


def test_vazias_concordam():
    vazia = np.zeros((5, 5), np.uint8)
    assert dice(vazia, vazia) == iou(vazia, vazia) == 1.0


def test_distancias_de_borda():
    a = np.zeros((40, 40), bool)
    a[10:30, 10:30] = True
    b = np.zeros((40, 40), bool)
    b[10:30, 13:33] = True  # o mesmo quadrado, 3 px para a direita
    assert distancia_media_borda(a, a) == hausdorff95(a, a) == 0.0
    assert hausdorff95(a, b) == pytest.approx(3.0, abs=0.01)
    assert 0 < distancia_media_borda(a, b) < hausdorff95(a, b)
    assert np.isnan(hausdorff95(a, np.zeros_like(a)))


def test_borda_e_o_anel_externo():
    a = np.zeros((10, 10), bool)
    a[2:8, 2:8] = True
    assert borda(a).sum() == 6 * 4 - 4
    assert not borda(a)[4, 4]
