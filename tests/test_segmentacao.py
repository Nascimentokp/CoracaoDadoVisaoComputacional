"""Máscaras do coração e leitura das marcações (SEG-01)."""

import json

import numpy as np
import pytest

from src.visao import ict, segmentacao
from src.visao.metricas import dice
from tests.test_ict import torax_sintetico


def test_poligono_quadrado_tem_a_area_certa():
    mascara = segmentacao.poligono_para_mascara([[10, 10], [30, 10], [30, 30], [10, 30]])
    assert mascara.sum() == pytest.approx(21 * 21, rel=0.05)
    assert mascara[20, 20] and not mascara[5, 5]


def test_largura_maxima():
    mascara = np.zeros((128, 128), bool)
    mascara[40, 30:70] = True
    mascara[50, 20:90] = True
    assert segmentacao.largura_maxima(mascara) == 70
    assert segmentacao.largura_maxima(np.zeros((8, 8), bool)) == 0


def test_metodos_acham_o_coracao_sintetico():
    img = torax_sintetico(coracao=40, torax=100, costelas=False)
    gabarito = np.zeros((128, 128), bool)
    gabarito[55:105, 44:84] = True  # o coração do tórax sintético
    medida = ict.medir_ict(img, linhas=(60, 100))
    pelo_ict = segmentacao.mascara_pelo_ict(medida)
    por_limiar = segmentacao.mascara_por_limiar(img, regiao=(24, 50, 104, 110))
    assert dice(gabarito, pelo_ict) > 0.7
    assert dice(gabarito, por_limiar) > 0.7


def test_carregar_marcacoes(tmp_path):
    (tmp_path / "u_a.json").write_text(
        json.dumps({"poligonos": {"test/true/1": [[1, 1], [5, 1], [5, 5]], "x": [[1, 1]]}})
    )
    (tmp_path / "u_b.json").write_text(
        json.dumps({"data": {"poligonos": {"test/true/1": [[2, 2], [6, 2], [6, 6]]}}})
    )
    tabela = segmentacao.carregar_marcacoes(tmp_path)
    assert len(tabela) == 2  # o polígono de 1 ponto fica de fora
    assert set(tabela["anotador"]) == {"u_a", "u_b"}


def test_maior_componente_fica_com_o_maior_e_tapa_buracos():
    mascara = np.zeros((60, 60), bool)
    mascara[10:40, 10:40] = True
    mascara[20:25, 20:25] = False  # buraco
    mascara[50:55, 50:55] = True  # ilha solta
    so_um = segmentacao.maior_componente(mascara)
    assert so_um[22, 22] and not so_um[52, 52]
    assert so_um.sum() == 30 * 30
    assert not segmentacao.maior_componente(np.zeros((8, 8), bool)).any()
