"""Filtros à mão conferidos contra o OpenCV e comportamento esperado (FIL-01)."""

import cv2
import numpy as np
import pytest

from src.visao import filtros


@pytest.fixture
def imagem():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(40, 50), dtype=np.uint8)


@pytest.fixture
def suave():
    """Gradiente liso: o que sobrar de variação depois do ruído é culpa do filtro."""
    x = np.linspace(40, 200, 64, dtype=np.float32)
    return np.tile(x, (64, 1)).astype(np.uint8)


@pytest.mark.parametrize("nome", list(filtros.KERNELS))
def test_correlacao_a_mao_igual_ao_opencv(imagem, nome):
    kernel = filtros.KERNELS[nome]
    a_mao = filtros.correlacionar(imagem, kernel)
    opencv = cv2.filter2D(imagem.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    np.testing.assert_allclose(a_mao, opencv, atol=1e-3)


def test_kernel_assimetrico_tambem_confere(imagem):
    kernel = np.arange(15, dtype=np.float32).reshape(3, 5)
    opencv = cv2.filter2D(imagem.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    np.testing.assert_allclose(filtros.correlacionar(imagem, kernel), opencv, rtol=1e-5)


def test_suavizadores_preservam_imagem_lisa(suave):
    for aplicar in filtros.SUAVIZADORES.values():
        miolo = (slice(4, -4), slice(4, -4))
        assert np.abs(aplicar(suave)[miolo].astype(int) - suave[miolo]).max() <= 1


def test_mediana_vence_media_no_sal_e_pimenta(suave):
    sujo = filtros.ruido_sal_pimenta(suave, 0.05, np.random.default_rng(1))
    mediana = filtros.psnr(suave, filtros.SUAVIZADORES["Mediana 3×3"](sujo))
    media = filtros.psnr(suave, filtros.SUAVIZADORES["Média 3×3"](sujo))
    assert mediana > media > filtros.psnr(suave, sujo)


def test_sobel_responde_so_na_direcao_da_borda():
    degrau = np.zeros((20, 20), dtype=np.uint8)
    degrau[:, 10:] = 200
    gx, gy, magnitude = filtros.sobel(degrau)
    assert np.abs(gx).max() > 0
    assert np.abs(gy).max() == 0
    assert magnitude[:, 9:11].min() > 0


def test_canny_auto_binario_e_limiares_ordenados(imagem):
    bordas, (baixo, alto) = filtros.canny_auto(imagem)
    assert set(np.unique(bordas)) <= {0, 255}
    assert 0 <= baixo < alto <= 255


def test_clahe_aumenta_contraste_de_imagem_apagada():
    rng = np.random.default_rng(2)
    apagada = rng.integers(100, 130, size=(64, 64), dtype=np.uint8)
    realcada = filtros.clahe(apagada)
    assert realcada.shape == apagada.shape and realcada.dtype == np.uint8
    assert realcada.std() > apagada.std()


def test_psnr_identica_e_infinita(imagem):
    assert filtros.psnr(imagem, imagem) == float("inf")
