"""CNNs em Keras para cardiomegalia: a pequena, do zero, e a de transfer learning.

Aula "Do filtro escrito ao filtro aprendido": no notebook 02 os kernels eram
escritos à mão (Sobel, gaussiano); aqui a rede aprende os próprios kernels a
partir dos rótulos. A CNN pequena segue a receita dos exemplos da aula
(convolução → pooling, três vezes, e camadas densas); a de transfer learning
reaproveita os filtros de uma MobileNetV2 treinada no ImageNet.

**Aumento de dados sem espelhar:** espelhar na horizontal põe o coração do
lado direito, uma anatomia que não existe (fora a rara dextrocardia). Os
aumentos aqui são só pequenas rotações, deslocamentos, zoom e contraste, que
imitam variações reais de posicionamento e exposição.
"""

from __future__ import annotations

import keras
import numpy as np
from keras import layers

from src import config
from src.visao import filtros


def tensores(imagens: np.ndarray, preprocessar: bool = False) -> np.ndarray:
    """Imagens `uint8` (n, 128, 128) → `float32` (n, 128, 128, 1) em [0, 1].

    Com `preprocessar=True`, aplica antes o padrão do notebook 02 (gaussiano 3×3
    + CLAHE).
    """
    if preprocessar:
        imagens = np.stack([filtros.preprocessar(im) for im in imagens])
    return (imagens.astype(np.float32) / 255.0)[..., None]


def aumento() -> keras.Sequential:
    """Variações de posicionamento e exposição, sem espelhamento."""
    return keras.Sequential(
        [
            layers.RandomRotation(0.03, fill_mode="nearest"),  # ±~11°
            layers.RandomTranslation(0.05, 0.05, fill_mode="nearest"),
            layers.RandomZoom(0.1, fill_mode="nearest"),
            layers.RandomContrast(0.2),
        ],
        name="aumento",
    )


def _compilar(modelo: keras.Model, taxa: float) -> keras.Model:
    modelo.compile(
        optimizer=keras.optimizers.Adam(taxa),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return modelo


def construir_cnn(
    filtros_por_bloco: tuple[int, ...] = (16, 32, 64),
    densa: int = 64,
    dropout: float = 0.3,
    com_aumento: bool = True,
    taxa: float = 1e-3,
) -> keras.Model:
    """CNN pequena: blocos convolução 3×3 + ReLU → max pooling 2×2, depois densa.

    As convoluções se chamam `conv1`, `conv2`, ... para os notebooks acharem as
    camadas ao desenhar filtros e mapas de ativação.
    """
    entrada = keras.Input((config.TAMANHO, config.TAMANHO, 1), name="raio_x")
    x = aumento()(entrada) if com_aumento else entrada
    for i, n in enumerate(filtros_por_bloco, start=1):
        x = layers.Conv2D(n, 3, padding="same", activation="relu", name=f"conv{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="media_global")(x)
    x = layers.Dense(densa, activation="relu", name="densa")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    saida = layers.Dense(1, activation="sigmoid", name="cardiomegalia")(x)
    return _compilar(keras.Model(entrada, saida, name="cnn_pequena"), taxa)


def construir_transferencia(dropout: float = 0.3, taxa: float = 1e-3) -> keras.Model:
    """MobileNetV2 do ImageNet, congelada, com uma cabeça nova para cardiomegalia.

    O raio-X tem 1 canal e a MobileNet espera 3 em [−1, 1]: o canal é repetido
    três vezes e reescalado. A base fica congelada; `liberar_topo` descongela as
    últimas camadas para o ajuste fino.
    """
    entrada = keras.Input((config.TAMANHO, config.TAMANHO, 1), name="raio_x")
    x = aumento()(entrada)
    x = layers.Concatenate(name="tres_canais")([x, x, x])
    x = layers.Rescaling(2.0, offset=-1.0, name="escala_mobilenet")(x)
    base = keras.applications.MobileNetV2(
        input_shape=(config.TAMANHO, config.TAMANHO, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="media_global")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    saida = layers.Dense(1, activation="sigmoid", name="cardiomegalia")(x)
    return _compilar(keras.Model(entrada, saida, name="mobilenet_transferencia"), taxa)


def liberar_topo(modelo: keras.Model, n_camadas: int = 30, taxa: float = 1e-5) -> keras.Model:
    """Descongela as últimas `n_camadas` da base (exceto BatchNorm) e recompila
    com taxa baixa, para ajustar os filtros do ImageNet sem destruí-los."""
    base = next(c for c in modelo.layers if isinstance(c, keras.Model) and c.name != "aumento")
    base.trainable = True
    for camada in base.layers[:-n_camadas]:
        camada.trainable = False
    for camada in base.layers[-n_camadas:]:
        if isinstance(camada, layers.BatchNormalization):
            camada.trainable = False
    return _compilar(modelo, taxa)


def treinar(
    modelo: keras.Model,
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epocas: int = 40,
    paciencia: int = 8,
    lote: int = 32,
    verbose: int = 0,
) -> keras.callbacks.History:
    """Treina com parada antecipada pela AUC de validação, voltando aos melhores pesos."""
    parada = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=paciencia, restore_best_weights=True
    )
    return modelo.fit(
        X_treino,
        y_treino,
        validation_data=(X_val, y_val),
        epochs=epocas,
        batch_size=lote,
        callbacks=[parada],
        verbose=verbose,
    )


def prever(modelo: keras.Model, X: np.ndarray) -> np.ndarray:
    """Probabilidade de cardiomegalia, vetor 1-D."""
    return modelo.predict(X, verbose=0).ravel()
