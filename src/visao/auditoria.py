"""Auditoria do lote: qualidade das imagens, duplicatas e quase duplicatas.

Aula "o lote que ninguém abriu" e "o dado que mente": antes de treinar,
olhar o dado. Aqui ficam as medidas usadas no notebook 01.

* **Qualidade** — brilho (média), contraste (desvio) e nitidez (variância das
  diferenças entre pixels vizinhos), como no exemplo do MNIST da aula.
* **Duplicatas exatas** — mesmo conteúdo de pixels, por hash.
* **Quase duplicatas** — correlação entre as imagens reduzidas a 32×32. Exames
  do mesmo paciente em datas diferentes ficam muito parecidos; se um cai no
  treino e outro no teste, a acurácia de teste fica inflada.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np
import pandas as pd


def qualidade(imagens: np.ndarray) -> pd.DataFrame:
    """Brilho, contraste e nitidez de cada imagem (n, h, w)."""
    x = imagens.astype(np.float32)
    dx = np.diff(x, axis=2)
    dy = np.diff(x, axis=1)
    return pd.DataFrame(
        {
            "brilho": x.mean(axis=(1, 2)),
            "contraste": x.std(axis=(1, 2)),
            "nitidez": dx.var(axis=(1, 2)) + dy.var(axis=(1, 2)),
        }
    )


def duplicatas_exatas(imagens: np.ndarray) -> list[list[int]]:
    """Grupos de índices com pixels idênticos (só grupos com 2 ou mais)."""
    grupos: dict[str, list[int]] = {}
    for i, img in enumerate(imagens):
        grupos.setdefault(hashlib.md5(img.tobytes()).hexdigest(), []).append(i)
    return [g for g in grupos.values() if len(g) > 1]


def assinatura(imagens: np.ndarray, lado: int = 32) -> np.ndarray:
    """Vetor de cada imagem reduzida a `lado`×`lado`, centrado e com norma 1.

    O produto escalar entre duas assinaturas é a correlação de Pearson entre
    as imagens reduzidas: 1 = idênticas a menos de brilho e contraste.
    """
    reduzidas = np.stack(
        [cv2.resize(img, (lado, lado), interpolation=cv2.INTER_AREA) for img in imagens]
    ).reshape(len(imagens), -1)
    v = reduzidas.astype(np.float32)
    v -= v.mean(axis=1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    return v


def vizinho_mais_proximo(
    consulta: np.ndarray, referencia: np.ndarray, lote: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Para cada assinatura de `consulta`, o índice e a correlação da mais parecida em
    `referencia`."""
    indices = np.empty(len(consulta), dtype=int)
    correlacoes = np.empty(len(consulta), dtype=np.float32)
    for inicio in range(0, len(consulta), lote):
        c = consulta[inicio : inicio + lote] @ referencia.T
        indices[inicio : inicio + lote] = c.argmax(axis=1)
        correlacoes[inicio : inicio + lote] = c.max(axis=1)
    return indices, correlacoes


def grupos_parecidos(assinaturas: np.ndarray, limiar: float, lote: int = 1024) -> np.ndarray:
    """Rótulo de grupo por imagem: imagens ligadas por correlação ≥ `limiar` (direta ou
    em cadeia) recebem o mesmo grupo.

    Serve de substituto para o ID de paciente que a base não traz: um split por
    esse grupo nunca põe dois exames quase iguais em lados opostos.
    """
    n = len(assinaturas)
    pai = np.arange(n)

    def raiz(i: int) -> int:
        while pai[i] != i:
            pai[i] = pai[pai[i]]
            i = pai[i]
        return i

    for inicio in range(0, n, lote):
        c = assinaturas[inicio : inicio + lote] @ assinaturas.T
        linhas, colunas = np.nonzero(c >= limiar)
        for a, b in zip(linhas + inicio, colunas):
            if a < b:
                ra, rb = raiz(a), raiz(b)
                if ra != rb:
                    pai[rb] = ra
    return np.array([raiz(i) for i in range(n)])


def tapar(
    imagens: np.ndarray, caixa: tuple[int, int, int, int], isolar: bool = False
) -> np.ndarray:
    """Esconde uma região retangular `caixa` = (x0, y0, x1, y1) de cada imagem.

    Com `isolar=False`, pinta a caixa; com `isolar=True`, pinta tudo **menos** a
    caixa. A tinta é a média da própria imagem, para não criar um contraste
    artificial diferente entre exames claros e escuros.

    Serve para perguntar ao modelo de onde vem a decisão: se ele acerta com o
    coração tapado, está usando outra coisa. A caixa precisa ser **a mesma para
    todos os exames**: uma caixa do tamanho de cada coração entregaria o rótulo
    pela própria forma do retângulo.
    """
    x0, y0, x1, y1 = caixa
    saida = imagens.copy()
    medias = imagens.reshape(len(imagens), -1).mean(axis=1).astype(imagens.dtype)
    dentro = np.zeros(imagens.shape[1:], bool)
    dentro[y0:y1, x0:x1] = True
    regiao = ~dentro if isolar else dentro
    saida[:, regiao] = medias[:, None]
    return saida
