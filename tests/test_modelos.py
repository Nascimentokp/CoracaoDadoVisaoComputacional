"""CNNs e avaliação: formato, aprendizado mínimo e comparação pareada (MOD-01, MOD-02)."""

import keras
import numpy as np
import pytest

from src import config
from src.model import avaliacao, cnn


@pytest.fixture(scope="module")
def brinquedo():
    """Classe 1 = quadrado claro largo no centro; classe 0 = estreito. Dá para aprender."""
    rng = np.random.default_rng(0)
    imagens, y = [], []
    for k in range(64):
        im = rng.integers(0, 60, (config.TAMANHO, config.TAMANHO)).astype(np.uint8)
        meia = 30 if k % 2 else 12
        im[40:90, 64 - meia : 64 + meia] = 220
        imagens.append(im)
        y.append(k % 2)
    return cnn.tensores(np.stack(imagens)), np.array(y)


def test_tensores_formato_e_faixa():
    imagens = np.full((3, config.TAMANHO, config.TAMANHO), 255, np.uint8)
    for pre in (False, True):
        X = cnn.tensores(imagens, preprocessar=pre)
        assert X.shape == (3, config.TAMANHO, config.TAMANHO, 1)
        assert X.dtype == np.float32 and 0 <= X.min() and X.max() <= 1


def test_aumento_nao_espelha():
    nomes = [type(c).__name__ for c in cnn.aumento().layers]
    assert "RandomFlip" not in nomes


def test_cnn_pequena_aprende_o_brinquedo(brinquedo):
    keras.utils.set_random_seed(0)
    X, y = brinquedo
    modelo = cnn.construir_cnn(com_aumento=False, taxa=3e-3)
    modelo.fit(X, y, epochs=25, batch_size=16, verbose=0)
    p = cnn.prever(modelo, X)
    assert p.shape == (len(y),) and ((0 <= p) & (p <= 1)).all()
    assert avaliacao.auc_bootstrap(y, p, n=50)["auc"] > 0.9


def test_camadas_nomeadas_para_visualizacao():
    modelo = cnn.construir_cnn()
    assert {"conv1", "conv2", "conv3"} <= {c.name for c in modelo.layers}


def test_auc_bootstrap_contem_a_auc():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    escore = y + rng.normal(0, 1, 300)
    r = avaliacao.auc_bootstrap(y, escore, n=300)
    assert r["ic_baixo"] < r["auc"] < r["ic_alto"]


def test_comparacao_pareada_detecta_o_melhor():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    bom = y + rng.normal(0, 0.5, 400)
    ruim = y + rng.normal(0, 2.0, 400)
    r = avaliacao.comparar_auc(y, bom, ruim, n=300)
    assert r["diferenca"] > 0 and r["ic_baixo"] > 0 and r["p_valor"] < 0.05
    igual = avaliacao.comparar_auc(y, bom, bom, n=100)
    assert igual["diferenca"] == 0


def test_limiar_youden_e_metricas():
    y = np.array([0, 0, 0, 1, 1, 1])
    escore = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    limiar = avaliacao.limiar_youden(y, escore)
    assert 0.3 < limiar <= 0.7
    m = avaliacao.metricas_no_limiar(y, escore, limiar)
    assert m["acuracia"] == m["sensibilidade"] == m["especificidade"] == 1.0


def test_escores_nan_ficam_de_fora():
    y = np.array([0, 1, 0, 1, 1])
    escore = np.array([0.1, 0.9, np.nan, 0.8, 0.7])
    assert avaliacao.auc_bootstrap(y, escore, n=50)["auc"] == 1.0


def test_gradcam_aponta_para_o_que_decide(brinquedo):
    from src.model import gradcam

    keras.utils.set_random_seed(0)
    X, y = brinquedo
    modelo = cnn.construir_cnn(com_aumento=False, taxa=3e-3)
    modelo.fit(X, y, epochs=25, batch_size=16, verbose=0)
    mapas = gradcam.gradcam(modelo, X[y == 1][:8])
    assert mapas.shape == (8, config.TAMANHO, config.TAMANHO)
    assert 0 <= mapas.min() and mapas.max() <= 1
    regiao = np.zeros((config.TAMANHO, config.TAMANHO), bool)
    regiao[30:100, 24:104] = True  # o quadrado claro com folga
    assert gradcam.fracao_na_regiao(mapas, regiao).mean() > regiao.mean()


def test_gradcam_no_transfer_learning_tem_formato_certo():
    from src.model import gradcam

    modelo = cnn.construir_transferencia()
    X = np.random.default_rng(0).random((2, config.TAMANHO, config.TAMANHO, 1)).astype("float32")
    mapas = gradcam.gradcam(modelo, X)
    assert mapas.shape == (2, config.TAMANHO, config.TAMANHO)


def test_gradcam_da_classe_prevista_nao_fica_vazio(brinquedo):
    from src.model import gradcam

    keras.utils.set_random_seed(0)
    X, y = brinquedo
    modelo = cnn.construir_cnn(com_aumento=False, taxa=3e-3)
    modelo.fit(X, y, epochs=25, batch_size=16, verbose=0)
    normais = X[y == 0][:8]
    previstos = (cnn.prever(modelo, normais) >= 0.5).astype(int)
    mapas = gradcam.gradcam(modelo, normais, classe=previstos)
    assert (mapas.sum(axis=(1, 2)) > 0).mean() >= 0.75


def test_fracoes_das_regioes_somam_um_e_mapa_vazio_e_nan():
    from src.model import gradcam

    rng = np.random.default_rng(3)
    mapas = rng.random((4, 16, 16))
    mapas[3] = 0
    metade = np.zeros((16, 16), bool)
    metade[:8] = True
    a, b = gradcam.fracao_na_regiao(mapas, metade), gradcam.fracao_na_regiao(mapas, ~metade)
    np.testing.assert_allclose((a + b)[:3], 1.0)
    assert np.isnan(a[3]) and np.isnan(b[3])


@pytest.mark.parametrize(
    ("construir", "lado"), [(cnn.construir_cnn, 32), (cnn.construir_transferencia, 4)]
)
def test_gradcam_usa_os_mapas_da_convolucao_mesmo_com_aumento(construir, lado):
    import tensorflow as tf

    from src.model import gradcam

    modelo = construir()
    assert any(c.name == "aumento" for c in modelo.layers)
    X = tf.convert_to_tensor(np.zeros((1, config.TAMANHO, config.TAMANHO, 1), np.float32))
    with tf.GradientTape() as fita:
        mapas, _ = gradcam._mapas_e_saida(modelo, X, None, fita)
    assert mapas.shape[1] == mapas.shape[2] == lado
    assert mapas.shape[-1] > 1
