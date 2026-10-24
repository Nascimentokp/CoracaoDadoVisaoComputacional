"""U-Net pequena em Keras para recortar o coração, e a perda Dice.

Aula de segmentação: a CNN do notebook 05 resume a imagem num número; a U-Net
devolve **uma probabilidade por pixel**. O caminho de descida (convolução →
pooling) entende *o que* há na imagem; o de subida (convolução transposta)
devolve *onde*, e as **conexões de atalho** trazem de volta os detalhes de
borda que o pooling perdeu.

**A perda Dice:** com entropia cruzada pura, o fundo (≈ 90 % dos pixels) domina
e a rede aprende a dizer "fundo" em quase tudo. O Dice "suave", calculado com
as probabilidades em vez da máscara binária, mede direto a sobreposição e não
liga para quantos pixels de fundo há. A perda usada é a soma das duas.

Sem aumento de dados: as camadas aleatórias do Keras mexeriam na imagem e não
na máscara. Os 3.550 raio-X de treino bastam para uma rede deste tamanho.
"""

from __future__ import annotations

import keras
import numpy as np
from keras import layers, ops

from src import config


def dice_suave(y_true, y_pred, suavizacao: float = 1.0):
    """Dice com probabilidades, por imagem e depois a média do lote."""
    eixos = (1, 2, 3)
    inter = ops.sum(y_true * y_pred, axis=eixos)
    soma = ops.sum(y_true, axis=eixos) + ops.sum(y_pred, axis=eixos)
    return ops.mean((2.0 * inter + suavizacao) / (soma + suavizacao))


def perda_bce_dice(y_true, y_pred):
    """Entropia cruzada (acerta pixel a pixel) + (1 − Dice) (acerta a área)."""
    bce = ops.mean(keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + 1.0 - dice_suave(y_true, y_pred)


def _bloco(x, n: int, nome: str):
    x = layers.Conv2D(n, 3, padding="same", activation="relu", name=f"{nome}_a")(x)
    return layers.Conv2D(n, 3, padding="same", activation="relu", name=f"{nome}_b")(x)


def construir_unet(filtros: tuple[int, ...] = (16, 32, 64, 128), taxa: float = 1e-3) -> keras.Model:
    """U-Net: `len(filtros) − 1` descidas, o fundo e as subidas com atalhos.

    Os blocos se chamam `desce1`…, `fundo` e `sobe1`… para os notebooks acharem
    as camadas.
    """
    entrada = keras.Input((config.TAMANHO, config.TAMANHO, 1), name="raio_x")
    x, atalhos = entrada, []
    for i, n in enumerate(filtros[:-1], start=1):
        x = _bloco(x, n, f"desce{i}")
        atalhos.append(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = _bloco(x, filtros[-1], "fundo")
    for i, (n, atalho) in enumerate(zip(filtros[-2::-1], atalhos[::-1]), start=1):
        x = layers.Conv2DTranspose(n, 2, strides=2, padding="same", name=f"sobe{i}_up")(x)
        x = layers.Concatenate(name=f"atalho{i}")([x, atalho])
        x = _bloco(x, n, f"sobe{i}")
    saida = layers.Conv2D(1, 1, activation="sigmoid", name="coracao")(x)
    modelo = keras.Model(entrada, saida, name="unet")
    modelo.compile(
        optimizer=keras.optimizers.Adam(taxa),
        loss=perda_bce_dice,
        metrics=[keras.metrics.MeanMetricWrapper(dice_suave, name="dice")],
    )
    return modelo


def treinar(
    modelo: keras.Model,
    X_treino: np.ndarray,
    M_treino: np.ndarray,
    X_val: np.ndarray,
    M_val: np.ndarray,
    epocas: int = 30,
    paciencia: int = 5,
    lote: int = 32,
    verbose: int = 0,
) -> keras.callbacks.History:
    """Treina com parada antecipada pelo Dice de validação, voltando aos melhores pesos.

    As máscaras entram como (n, 128, 128) booleanas ou 0/1.
    """
    parada = keras.callbacks.EarlyStopping(
        monitor="val_dice", mode="max", patience=paciencia, restore_best_weights=True
    )
    return modelo.fit(
        X_treino,
        np.asarray(M_treino, np.float32)[..., None],
        validation_data=(X_val, np.asarray(M_val, np.float32)[..., None]),
        epochs=epocas,
        batch_size=lote,
        callbacks=[parada],
        verbose=verbose,
    )


def treinar_ou_carregar(
    arquivo,
    X: np.ndarray,
    rotulo: np.ndarray,
    treino: np.ndarray,
    validacao: np.ndarray,
    retreinar: bool = False,
    verbose: int = 0,
) -> tuple[keras.Model, dict]:
    """U-Net de um professor: carrega `arquivo` se existir, senão treina e grava.

    Só entram no treino os exames em que o professor marcou algum coração.
    O histórico fica ao lado, em `<arquivo>.historico.json`. É a receita única
    do notebook 10 e de `scripts/treinar_unets.py`.
    """
    from pathlib import Path

    from src.utils.io import ler_json, salvar_json

    arquivo = Path(arquivo)
    historico = arquivo.with_suffix(".historico.json")
    if retreinar or not arquivo.exists():
        tem_coracao = rotulo.any(axis=(1, 2))
        keras.utils.set_random_seed(config.SEMENTE)
        modelo = construir_unet()
        tr, va = treino & tem_coracao, validacao & tem_coracao
        h = treinar(modelo, X[tr], rotulo[tr], X[va], rotulo[va], verbose=verbose)
        modelo.save(arquivo)
        salvar_json(h.history, historico)
    modelo = keras.models.load_model(
        arquivo, custom_objects={"perda_bce_dice": perda_bce_dice, "dice_suave": dice_suave}
    )
    return modelo, ler_json(historico)


def prever(modelo: keras.Model, X: np.ndarray, lote: int = 64) -> np.ndarray:
    """Probabilidade de coração por pixel, (n, 128, 128)."""
    return modelo.predict(X, batch_size=lote, verbose=0)[..., 0]
