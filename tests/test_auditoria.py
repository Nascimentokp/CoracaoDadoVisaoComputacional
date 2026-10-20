"""Medidas de auditoria do lote com imagens sintéticas (AUD-01)."""

import numpy as np

from src.visao import auditoria


def _imagens(n=6, semente=0):
    rng = np.random.default_rng(semente)
    return rng.integers(0, 256, size=(n, 64, 64), dtype=np.uint8)


def test_qualidade_brilho_contraste_nitidez():
    lisa = np.full((1, 32, 32), 100, dtype=np.uint8)
    xadrez = (np.indices((32, 32)).sum(axis=0) % 2 * 255).astype(np.uint8)[None]
    tabela = auditoria.qualidade(np.concatenate([lisa, xadrez]))
    assert tabela.loc[0, "brilho"] == 100
    assert tabela.loc[0, "contraste"] == 0
    assert tabela.loc[0, "nitidez"] == 0
    assert tabela.loc[1, "nitidez"] > 0


def test_duplicatas_exatas():
    imagens = _imagens()
    imagens[4] = imagens[1]
    assert auditoria.duplicatas_exatas(imagens) == [[1, 4]]


def test_assinatura_ignora_brilho_e_contraste():
    img = _imagens(1)
    clareada = np.clip(img.astype(int) // 2 + 60, 0, 255).astype(np.uint8)
    a = auditoria.assinatura(np.concatenate([img, clareada]))
    assert a[0] @ a[1] > 0.99


def test_vizinho_mais_proximo_acha_a_copia():
    imagens = _imagens(8)
    a = auditoria.assinatura(imagens)
    indices, correlacoes = auditoria.vizinho_mais_proximo(a[[5, 2]], a)
    assert list(indices) == [5, 2]
    assert np.allclose(correlacoes, 1, atol=1e-5)


def test_grupos_parecidos_liga_em_cadeia():
    imagens = _imagens(5)
    imagens[3] = imagens[0]
    imagens[4] = imagens[3]
    grupos = auditoria.grupos_parecidos(auditoria.assinatura(imagens), limiar=0.99)
    assert grupos[0] == grupos[3] == grupos[4]
    assert len(set(grupos)) == 3


def test_tapar_e_isolar_sao_complementares():
    imagens = _imagens(3)
    caixa = (10, 20, 40, 50)
    tapada = auditoria.tapar(imagens, caixa)
    isolada = auditoria.tapar(imagens, caixa, isolar=True)
    medias = imagens.reshape(3, -1).mean(axis=1).astype(np.uint8)
    for k in range(3):
        assert (tapada[k, 20:50, 10:40] == medias[k]).all()
        np.testing.assert_array_equal(isolada[k, 20:50, 10:40], imagens[k, 20:50, 10:40])
        assert (isolada[k, :20] == medias[k]).all()
        np.testing.assert_array_equal(tapada[k, :20], imagens[k, :20])
