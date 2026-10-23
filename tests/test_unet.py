"""U-Net e perda Dice (SEG-02)."""

import keras
import numpy as np
import pytest

from src import config
from src.model import unet
from src.visao.metricas import dice


def test_dice_suave_bate_com_o_dice_em_mascaras_binarias():
    rng = np.random.default_rng(0)
    a = rng.random((2, 16, 16, 1)) > 0.5
    b = rng.random((2, 16, 16, 1)) > 0.5
    esperado = np.mean([dice(a[k], b[k]) for k in range(2)])
    obtido = float(unet.dice_suave(a.astype("float32"), b.astype("float32"), suavizacao=0.0))
    assert obtido == pytest.approx(esperado, abs=1e-5)
    assert float(unet.perda_bce_dice(a.astype("float32"), a.astype("float32"))) < 0.01


def test_unet_devolve_um_mapa_do_tamanho_da_imagem():
    modelo = unet.construir_unet(filtros=(4, 8, 16))
    saida = unet.prever(modelo, np.zeros((2, config.TAMANHO, config.TAMANHO, 1), np.float32))
    assert saida.shape == (2, config.TAMANHO, config.TAMANHO)
    assert ((0 <= saida) & (saida <= 1)).all()
    assert {"desce1", "fundo", "sobe1"} <= {c.name.rsplit("_", 1)[0] for c in modelo.layers}


def test_unet_aprende_a_recortar_um_disco():
    """Disco claro em posição sorteada sobre ruído: a rede tem de achar onde ele está."""
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[: config.TAMANHO, : config.TAMANHO]
    imagens, mascaras = [], []
    for _ in range(48):
        cy, cx, r = rng.integers(35, 93), rng.integers(35, 93), rng.integers(15, 28)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 < r**2
        imagens.append(rng.normal(0.3, 0.1, m.shape) + 0.4 * m)
        mascaras.append(m)
    X = np.clip(np.stack(imagens), 0, 1).astype(np.float32)[..., None]
    M = np.stack(mascaras)
    modelo = unet.construir_unet(filtros=(8, 16, 32), taxa=3e-3)
    unet.treinar(modelo, X[:40], M[:40], X[40:], M[40:], epocas=15, paciencia=15, lote=8)
    p = unet.prever(modelo, X[40:]) > 0.5
    assert np.mean([dice(M[40 + k], p[k]) for k in range(len(p))]) > 0.8
