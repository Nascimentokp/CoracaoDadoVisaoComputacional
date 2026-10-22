"""Caixas, formato YOLO e pseudo-rótulos (DET-01)."""

import numpy as np
import pytest

from src.visao import deteccao, ict
from tests.test_ict import torax_sintetico


def test_ida_e_volta_do_formato_yolo():
    caixa = (36.0, 46.0, 98.0, 94.0)
    xc, yc, w, h = deteccao.para_yolo(caixa, 128)
    assert 0 < xc < 1 and 0 < w < 1
    np.testing.assert_allclose(deteccao.de_yolo(xc, yc, w, h, 128), caixa)


def test_iou_de_caixas():
    a = (0.0, 0.0, 10.0, 10.0)
    assert deteccao.iou_caixas(a, a) == 1.0
    assert deteccao.iou_caixas(a, (20.0, 20.0, 30.0, 30.0)) == 0.0
    assert deteccao.iou_caixas(a, (5.0, 0.0, 15.0, 10.0)) == pytest.approx(50 / 150)


def test_caixa_do_poligono():
    assert deteccao.caixa_do_poligono([[3, 8], [9, 2], [6, 12]]) == (3.0, 2.0, 9.0, 12.0)


def test_pseudo_rotulo_cobre_o_coracao_sintetico():
    medida = ict.medir_ict(torax_sintetico(coracao=40), linhas=(60, 100))
    caixa = deteccao.caixa_do_ict(medida)
    assert caixa is not None
    assert deteccao.iou_caixas(caixa, (44.0, 60.0, 84.0, 100.0)) > 0.8


def test_sem_medida_sem_caixa():
    assert deteccao.caixa_do_ict(ict.medir_ict(np.full((128, 128), 120, np.uint8))) is None


def test_grava_imagem_rotulo_e_yaml(tmp_path):
    img = np.zeros((128, 128), np.uint8)
    deteccao.gravar_exemplo(tmp_path / "train", "a", img, [(32.0, 32.0, 96.0, 64.0)])
    deteccao.gravar_exemplo(tmp_path / "train", "vazio", img, [])
    linha = (tmp_path / "train" / "labels" / "a.txt").read_text().split()
    assert linha[0] == "0" and [float(v) for v in linha[1:]] == [0.5, 0.375, 0.5, 0.25]
    assert (tmp_path / "train" / "labels" / "vazio.txt").read_text() == ""
    yaml = deteccao.gravar_yaml(tmp_path / "d.yaml", tmp_path, "train/images", "val/images")
    assert "0: coracao" in yaml.read_text()
