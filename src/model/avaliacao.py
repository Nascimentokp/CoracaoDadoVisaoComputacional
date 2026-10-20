"""Avaliação de classificadores binários: AUC com intervalo, limiar e comparação pareada.

Todo modelo do projeto — o ICT clássico, as CNNs — produz um escore por exame
(maior = mais cardiomegalia). As funções aqui comparam esses escores do mesmo
jeito para todos.

* `auc_bootstrap` — AUC e intervalo de 95 % por reamostragem dos exames;
* `comparar_auc` — diferença de AUC entre dois escores nos **mesmos** exames,
  com bootstrap **pareado** (a mesma reamostragem para os dois), que desconta a
  variação comum e é mais justo que comparar dois intervalos soltos;
* `limiar_youden` e `metricas_no_limiar` — o limiar é escolhido na validação e
  só aplicado no teste.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src import config


def _validos(y: np.ndarray, *escores: np.ndarray) -> tuple[np.ndarray, ...]:
    ok = np.ones(len(y), bool)
    for e in escores:
        ok &= np.isfinite(e)
    return (np.asarray(y)[ok], *(np.asarray(e)[ok] for e in escores))


def auc_bootstrap(
    y: np.ndarray, escore: np.ndarray, n: int = 2000, semente: int = config.SEMENTE
) -> dict:
    """AUC e IC 95 % por bootstrap. Exames sem escore (NaN) ficam de fora."""
    y, escore = _validos(y, escore)
    rng = np.random.default_rng(semente)
    amostras = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            amostras.append(roc_auc_score(y[i], escore[i]))
    baixo, alto = np.percentile(amostras, [2.5, 97.5])
    return {"auc": roc_auc_score(y, escore), "ic_baixo": baixo, "ic_alto": alto}


def comparar_auc(
    y: np.ndarray,
    escore_a: np.ndarray,
    escore_b: np.ndarray,
    n: int = 2000,
    semente: int = config.SEMENTE,
) -> dict:
    """Diferença AUC(a) − AUC(b) com bootstrap pareado.

    `p_valor` é a fração das reamostragens em que a diferença troca de sinal
    (bilateral): perto de 0 = diferença consistente.
    """
    y, a, b = _validos(y, escore_a, escore_b)
    rng = np.random.default_rng(semente)
    difs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            difs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    difs = np.array(difs)
    diferenca = roc_auc_score(y, a) - roc_auc_score(y, b)
    p = 2 * min((difs <= 0).mean(), (difs >= 0).mean())
    baixo, alto = np.percentile(difs, [2.5, 97.5])
    return {"diferenca": diferenca, "ic_baixo": baixo, "ic_alto": alto, "p_valor": min(p, 1.0)}


def limiar_youden(y: np.ndarray, escore: np.ndarray) -> float:
    """Limiar que maximiza sensibilidade + especificidade."""
    y, escore = _validos(y, escore)
    fpr, tpr, cortes = roc_curve(y, escore)
    return float(cortes[np.argmax(tpr - fpr)])


def metricas_no_limiar(y: np.ndarray, escore: np.ndarray, limiar: float) -> dict:
    """Acurácia, sensibilidade e especificidade com `escore >= limiar` = cardiomegalia."""
    y, escore = _validos(y, escore)
    previsto, real = escore >= limiar, y == 1
    return {
        "limiar": limiar,
        "acuracia": float((previsto == real).mean()),
        "sensibilidade": float(previsto[real].mean()),
        "especificidade": float((~previsto[~real]).mean()),
    }
