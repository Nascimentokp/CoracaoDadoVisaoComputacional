"""Sorteio balanceado e cabeça do transfer learning (MOD-03)."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.model import transferencia


def test_sortear_e_balanceado_reprodutivel_e_respeita_candidatos():
    y = np.array([0, 1] * 50)
    candidatos = np.arange(100) < 80
    a = transferencia.sortear(y, candidatos, 10, semente=1)
    assert len(a) == 20 and (y[a] == 1).sum() == 10
    assert candidatos[a].all() and len(set(a)) == 20
    assert np.array_equal(a, transferencia.sortear(y, candidatos, 10, semente=1))
    assert not np.array_equal(a, transferencia.sortear(y, candidatos, 10, semente=2))
    with pytest.raises(ValueError):
        transferencia.sortear(y, candidatos, 41, semente=1)


def test_cabeca_aprende_com_poucos_exemplos():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 200)
    X = rng.normal(0, 1, (400, 10))
    X[:, 0] += 3 * y  # só a primeira feature informa
    treino = transferencia.sortear(y, np.arange(400) < 200, 10, semente=0)
    modelo = transferencia.cabeca(10).fit(X[treino], y[treino])
    assert roc_auc_score(y[200:], modelo.predict_proba(X[200:])[:, 1]) > 0.85


def test_base_desconhecida():
    with pytest.raises(ValueError):
        transferencia.extrair("VGG16", np.zeros((1, 128, 128), np.uint8))
