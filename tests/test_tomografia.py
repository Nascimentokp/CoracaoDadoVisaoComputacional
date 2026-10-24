"""Tomografia: unidades Hounsfield, pulmão por densidade e projeção (TC-01)."""

import numpy as np
import pytest

from src import config
from src.visao import tomografia


def fantoma(gas_no_abdome: bool = True) -> np.ndarray:
    """Volume LPS (x, y, z) com corpo de água, dois pulmões de ar e, embaixo, uma bolha de gás."""
    x, y, z = np.mgrid[:64, :64, :40]
    volume = np.full((64, 64, 40), -1000.0)  # ar de fora
    volume[((x - 32) / 28) ** 2 + ((y - 32) / 20) ** 2 < 1] = 40.0  # corpo
    for cx in (20, 44):  # pulmão direito (x pequeno) e esquerdo
        pulmao = ((x - cx) / 8) ** 2 + ((y - 32) / 10) ** 2 + ((z - 26) / 12) ** 2 < 1
        volume[pulmao] = -800.0
    if gas_no_abdome:
        # maior que 10 % de um pulmão: só a regra de subir ao alto do tórax o tira
        volume[((x - 38) / 7) ** 2 + ((y - 30) / 7) ** 2 + ((z - 7) / 5) ** 2 < 1] = -700.0
    return volume


def test_janela_vai_de_zero_a_um():
    hu = np.array([-2000.0, -600.0, 1000.0])
    assert tomografia.janela(hu, -600, 1500).tolist() == [0.0, 0.5, 1.0]


def test_pulmao_por_densidade_ignora_ar_de_fora_e_gas():
    volume = fantoma()
    corte = tomografia.pulmao_no_corte(volume[:, :, 26])
    assert corte[20, 32] and corte[44, 32] and not corte[0, 0]  # pulmões sim, ar de fora não
    com_gas = tomografia.pulmao_no_corte(volume[:, :, 7])
    assert com_gas[38, 30]  # corte a corte, o gás parece pulmão...
    pulmoes = tomografia.pulmao_no_volume(volume)
    assert pulmoes[20, 32, 26] and pulmoes[44, 32, 26]
    assert not pulmoes[:, :, :10].any()  # ...em 3D ele sai, porque não sobe ao alto do tórax


def test_atenuacao_relativa_a_agua():
    assert tomografia.atenuacao(np.array([-1000.0, 0.0, 1000.0, -3024.0])).tolist() == [
        0.0,
        1.0,
        2.0,
        0.0,
    ]


def test_projecao_na_convencao_radiologica():
    volume = np.zeros((10, 6, 8))
    volume[8, :, 7] = 1.0  # do lado esquerdo do paciente (x grande), no alto (z grande)
    imagem = tomografia.projetar(volume, (1.0, 2.0, 1.0))
    assert imagem.shape == (8, 10)
    assert imagem[0, 8] == pytest.approx(12.0)  # 6 voxels × 2 mm, em cima e à direita de quem olha
    assert imagem.sum() == pytest.approx(12.0)


def test_para_128_tem_pixel_quadrado_e_ar_em_volta():
    imagem = np.ones((40, 100), np.float32)  # 80 mm de altura (2 mm) × 100 mm de largura (1 mm)
    saida, mm_por_px = tomografia.para_128(imagem, 2.0, 1.0, margem_mm=10)
    assert saida.shape == (config.TAMANHO, config.TAMANHO)
    assert mm_por_px == pytest.approx(120 / config.TAMANHO)  # 100 mm + 2 × 10 mm de ar
    corpo = saida > 0.5
    linhas, colunas = np.flatnonzero(corpo.any(axis=1)), np.flatnonzero(corpo.any(axis=0))
    assert len(linhas) * mm_por_px == pytest.approx(80, abs=2)  # a altura não é esticada
    assert len(colunas) * mm_por_px == pytest.approx(100, abs=2)
    assert saida[0, 0] == 0 and saida[64, 0] == 0  # ar em cima e dos lados


def test_nitidez_deixa_o_corpo_acima_de_zero():
    imagem = np.zeros((64, 64), np.float32)
    imagem[8:56, 8:56] = 1.0
    imagem[20:44, 20:44] = 0.2  # "pulmão" escuro cercado de tecido claro
    realcada = tomografia.nitidez(imagem, sigma=6, forca=0.9)
    assert (realcada[imagem > 0] > 0).all() and (realcada[imagem == 0] == 0).all()
    u8 = tomografia.para_uint8(realcada)
    assert u8[32, 32] > 0  # o pulmão não vira o preto do ar de fora


def test_ler_ct_recusa_eixo_invertido(tmp_path):
    nrrd = pytest.importorskip("nrrd")
    dados = np.zeros((4, 4, 3), np.int16)
    certo, invertido = tmp_path / "certo.nrrd", tmp_path / "invertido.nrrd"
    base = {"space": "left-posterior-superior"}
    nrrd.write(str(certo), dados, {**base, "space directions": np.diag([0.5, 0.5, 2.0])})
    nrrd.write(str(invertido), dados, {**base, "space directions": np.diag([0.5, 0.5, -2.0])})
    volume, espacamento = tomografia.ler_ct(certo)
    assert volume.dtype == np.float32 and espacamento == (0.5, 0.5, 2.0)
    with pytest.raises(ValueError):
        tomografia.ler_ct(invertido)


def test_para_uint8():
    imagem = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    u8 = tomografia.para_uint8(imagem)
    assert u8.dtype == np.uint8 and u8.max() == 255 and u8[0, 0] == 0
