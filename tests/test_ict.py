"""Índice cardiotorácico num tórax sintético de medidas conhecidas (ICT-01)."""

import cv2
import numpy as np
import pytest

from src.visao import ict


def torax_sintetico(coracao=40, torax=100, costelas=True, semente=0):
    """Raio-X de brinquedo 128×128: corpo claro, dois pulmões escuros e o coração
    claro entre eles, com largura de coração e tórax conhecidas na faixa de baixo."""
    img = np.full((128, 128), 30, np.float32)  # fundo
    centro, esq, dir_ = 64, 64 - torax // 2, 64 + torax // 2
    img[8:124, esq - 6 : dir_ + 6] = 200  # corpo e parede torácica
    img[20:110, esq:dir_] = 70  # campos pulmonares
    img[20:110, centro - 6 : centro + 6] = 210  # mediastino superior
    img[55:105, centro - coracao // 2 : centro + coracao // 2] = 205  # coração
    if costelas:
        for y in range(24, 108, 9):
            img[y : y + 3, esq:dir_] += 45
    rng = np.random.default_rng(semente)
    img += rng.normal(0, 4, img.shape)
    return np.clip(cv2.GaussianBlur(img, (3, 3), 0), 0, 255).astype(np.uint8)


@pytest.mark.parametrize(("coracao", "torax"), [(40, 100), (56, 100), (50, 90)])
def test_perfil_recupera_ict(coracao, torax):
    medida = ict.medir_ict(torax_sintetico(coracao, torax), linhas=(60, 100))
    assert medida.ok
    assert medida.ict == pytest.approx(coracao / torax, abs=0.03)


def test_perfil_aguenta_costelas():
    com = ict.medir_ict(torax_sintetico(costelas=True), linhas=(60, 100)).ict
    sem = ict.medir_ict(torax_sintetico(costelas=False), linhas=(60, 100)).ict
    assert com == pytest.approx(sem, abs=0.02)


def test_imagem_sem_estrutura_nao_mede():
    lisa = np.full((128, 128), 120, np.uint8)
    medida = ict.medir_ict(lisa)
    assert not medida.ok
    assert medida.bordas.shape == (0, 5)


def test_segmentacao_acha_dois_pulmoes_sem_costelas():
    mascara = ict.segmentar_pulmoes(torax_sintetico(costelas=False))
    n, _ = cv2.connectedComponents(mascara)
    assert n - 1 == 2
    assert ict.ict_pela_mascara(mascara) == pytest.approx(0.40, abs=0.05)


def test_mascara_com_um_pulmao_so_nao_mede():
    mascara = np.zeros((128, 128), np.uint8)
    mascara[30:100, 20:50] = 255
    assert np.isnan(ict.ict_pela_mascara(mascara))


def test_caixas_de_referencia_cobrem_coracao_e_torax():
    medidas = [ict.medir_ict(torax_sintetico(semente=k), linhas=(60, 100)) for k in range(5)]
    coracao, torax = ict.caixas_de_referencia(medidas, folga=2)
    x0, y0, x1, y1 = coracao
    assert x0 <= 44 and x1 >= 84  # coração sintético: colunas 44 a 84
    assert torax[0] <= 15 and torax[2] >= 113 and torax[1] < y0
