"""Grad-CAM: para onde a rede olhou ao decidir.

Aula "Mapa de ativação: entendendo a auditoria" e "o que ainda não é": o mapa
mostra **onde** a decisão se apoiou, não **por quê**. Receita (Selvaraju et al.,
2017):

1. pegar os mapas de uma camada convolucional (A, um por filtro);
2. derivar a saída (probabilidade de cardiomegalia) em relação a cada mapa;
3. o peso de cada filtro é a média espacial desse gradiente;
4. somar os mapas ponderados, ficar só com a parte positiva (ReLU) e
   redimensionar para o tamanho da imagem.
"""

from __future__ import annotations

import cv2
import keras
import numpy as np
import tensorflow as tf


def _modelo_base(modelo: keras.Model) -> keras.Model | None:
    """A base pré-treinada aninhada (transfer learning), se houver.

    O bloco de aumento de dados também é um `keras.Model` (um `Sequential`) e
    vem antes da base: ele precisa ficar de fora, ou o "mapa" vira o gradiente
    em relação à própria imagem de entrada.
    """
    return next(
        (c for c in modelo.layers if isinstance(c, keras.Model) and c.name != "aumento"),
        None,
    )


def _mapas_e_saida(modelo: keras.Model, X: tf.Tensor, camada: str | None, fita: tf.GradientTape):
    base = _modelo_base(modelo)
    if base is None:
        alvo = modelo.get_layer(camada or "conv3")
        sub = keras.Model(modelo.inputs, [alvo.output, modelo.output])
        mapas, saida = sub(X, training=False)
        return mapas, saida
    # Transfer learning: a última ativação da base é o alvo; a cabeça é refeita à mão.
    x, mapas = X, None
    for c in modelo.layers[1:]:
        if c is base:
            mapas = c(x, training=False)
            fita.watch(mapas)
            x = mapas
        elif c.name == "aumento":
            continue  # só age no treino
        elif isinstance(c, keras.layers.Concatenate):
            x = c([x, x, x])
        else:
            x = c(x, training=False)
    return mapas, x


def gradcam(
    modelo: keras.Model,
    X: np.ndarray,
    classe: np.ndarray | None = None,
    camada: str | None = None,
    lote: int = 64,
) -> np.ndarray:
    """Mapas Grad-CAM (n, h, w) em [0, 1], no tamanho de `X`.

    `classe` diz, por imagem, que decisão explicar: 1 = cardiomegalia (a
    saída p), 0 = normal (1 − p). O padrão (None) explica cardiomegalia em
    todas. Para auditar uma decisão, passe a classe **prevista**: explicar
    "cardiomegalia" num exame que a rede acha normal dá gradientes quase todos
    negativos e, depois do ReLU, um mapa vazio.

    `camada` é o nome da convolução alvo na CNN pequena (padrão `conv3`); no
    modelo de transfer learning o alvo é sempre a última ativação da base.
    """
    altura, largura = X.shape[1:3]
    classe = np.ones(len(X)) if classe is None else np.asarray(classe, dtype=float)
    saida = []
    for inicio in range(0, len(X), lote):
        bloco = tf.convert_to_tensor(X[inicio : inicio + lote])
        sinal = tf.constant(classe[inicio : inicio + lote], dtype=tf.float32)
        with tf.GradientTape() as fita:
            mapas, pred = _mapas_e_saida(modelo, bloco, camada, fita)
            alvo = sinal * pred[:, 0] + (1 - sinal) * (1 - pred[:, 0])
        grad = fita.gradient(alvo, mapas)
        pesos = tf.reduce_mean(grad, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(pesos * mapas, axis=-1)).numpy()
        for m in cam:
            m = cv2.resize(m, (largura, altura), interpolation=cv2.INTER_LINEAR)
            saida.append(m / m.max() if m.max() > 0 else m)
    return np.stack(saida)


def fracao_na_regiao(mapas: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Fração da "atenção" (soma do mapa) que cai dentro da `mascara` booleana (h, w).

    Mapas vazios (soma zero) não têm atenção para repartir: devolvem NaN, para
    não entrarem como zero nas médias.
    """
    total = mapas.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, mapas[:, mascara].sum(axis=1) / total, np.nan)
