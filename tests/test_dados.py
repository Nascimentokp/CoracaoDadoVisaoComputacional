"""Catálogo e divisão da base: contagens, equilíbrio e teste intocado (DADOS-01)."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.dados import (
    carregar_imagens,
    dividir,
    dividir_aleatorio,
    dividir_por_grupo,
    listar_imagens,
)

pytestmark = pytest.mark.base


@pytest.fixture(scope="module")
def catalogo():
    return dividir(listar_imagens())


def test_contagens_da_base(catalogo):
    assert len(catalogo) == 5552
    assert (catalogo["particao"] == "train").sum() == 4438
    assert (catalogo["particao"] == "test").sum() == 1114
    assert catalogo["id"].is_unique


def test_classes_equilibradas_em_cada_conjunto(catalogo):
    for _, grupo in catalogo.groupby("conjunto"):
        assert grupo["classe"].mean() == pytest.approx(0.5, abs=0.01)


def test_validacao_sai_do_treino_e_teste_fica_intocado(catalogo):
    validacao = catalogo[catalogo["conjunto"] == "validacao"]
    teste = catalogo[catalogo["conjunto"] == "teste"]
    assert (validacao["particao"] == "train").all()
    assert (teste["particao"] == "test").all()
    assert len(validacao) / (catalogo["particao"] == "train").sum() == pytest.approx(
        config.PROPORCAO_VALIDACAO, abs=0.005
    )


def test_divisao_reprodutivel():
    a = dividir(listar_imagens())["conjunto"]
    b = dividir(listar_imagens())["conjunto"]
    assert a.equals(b)


def test_imagens_128_tons_de_cinza(catalogo):
    imagens = carregar_imagens(catalogo["caminho"].head(20))
    assert imagens.shape == (20, config.TAMANHO, config.TAMANHO)
    assert imagens.dtype == np.uint8


def test_dividir_por_grupo_nunca_parte_um_grupo(catalogo):
    rng = np.random.default_rng(0)
    grupos = np.arange(len(catalogo))
    # 300 pares artificiais de "mesmo paciente"
    pares = rng.choice(len(catalogo), (300, 2), replace=False)
    grupos[pares[:, 1]] = grupos[pares[:, 0]]
    conjunto = dividir_por_grupo(catalogo, grupos)
    por_grupo = pd.Series(conjunto.to_numpy()).groupby(grupos).nunique()
    assert (por_grupo == 1).all()
    proporcoes = conjunto.value_counts(normalize=True)
    assert proporcoes["treino"] == pytest.approx(0.6, abs=0.02)
    assert proporcoes["teste"] == pytest.approx(0.2, abs=0.02)
    for nome in ("treino", "validacao", "teste"):
        assert catalogo["classe"][conjunto == nome].mean() == pytest.approx(0.5, abs=0.03)


def test_dividir_aleatorio_60_20_20(catalogo):
    conjunto = dividir_aleatorio(catalogo)
    proporcoes = conjunto.value_counts(normalize=True)
    assert proporcoes["treino"] == pytest.approx(0.6, abs=0.01)
    assert proporcoes["validacao"] == pytest.approx(0.2, abs=0.01)
    assert proporcoes["teste"] == pytest.approx(0.2, abs=0.01)


def test_download_conferido_apaga_arquivo_ruim(tmp_path):
    from src.data import dados

    origem = tmp_path / "origem.bin"
    origem.write_bytes(b"conteudo")
    destino = dados._baixar(origem.as_uri(), tmp_path / "sub" / "destino.bin")
    assert destino.read_bytes() == b"conteudo"
    assert not (tmp_path / "sub" / "destino.bin.part").exists()
    with pytest.raises(ValueError):
        dados._conferir(destino, "0" * 64, "sha256")
    assert not destino.exists()  # a próxima chamada baixa de novo
