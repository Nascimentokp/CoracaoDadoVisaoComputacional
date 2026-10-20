"""Geometria do ECG com papel sintético de régua conhecida (GEO-01)."""

import cv2
import numpy as np
import pytest

from src.visao import geometria

PX_MM = 6  # pixels por milímetro do papel sintético


def papel_ecg(bpm=75.0, largura_mm=250, altura_mm=100):
    """Papel milimetrado com uma tira de ritmo: picos R a cada 60/bpm segundos."""
    h, w = altura_mm * PX_MM, largura_mm * PX_MM
    img = np.full((h, w, 3), 250, np.uint8)
    for mm in range(0, largura_mm + 1):
        cor = (150, 150, 235) if mm % 5 == 0 else (205, 205, 245)
        cv2.line(img, (mm * PX_MM, 0), (mm * PX_MM, h), cor, 1)
    for mm in range(0, altura_mm + 1):
        cor = (150, 150, 235) if mm % 5 == 0 else (205, 205, 245)
        cv2.line(img, (0, mm * PX_MM), (w, mm * PX_MM), cor, 1)
    base = int(0.85 * h)
    rr_px = 60 / bpm * geometria.VELOCIDADE_MM_S * PX_MM
    x, pontos = 10.0, [(0, base)]
    while x + 2 * PX_MM < w:
        pontos += [(int(x), base), (int(x + PX_MM), base - 12 * PX_MM), (int(x + 2 * PX_MM), base)]
        x += rr_px
    pontos.append((w - 1, base))
    cv2.polylines(img, [np.array(pontos, np.int32)], False, (20, 20, 20), 2)
    return img


def test_ordenar_cantos_em_qualquer_ordem():
    certos = np.array([[10, 5], [90, 12], [95, 70], [3, 60]], np.float32)
    embaralhados = certos[[2, 0, 3, 1]]
    np.testing.assert_array_equal(geometria.ordenar_cantos(embaralhados), certos)


def test_passo_da_grade_recupera_px_por_mm():
    horizontal, vertical = geometria.passo_da_grade(papel_ecg())
    assert horizontal == pytest.approx(PX_MM, abs=0.05)
    assert vertical == pytest.approx(PX_MM, abs=0.05)


@pytest.mark.parametrize("bpm", [60.0, 75.0, 110.0])
def test_frequencia_cardiaca_no_papel_reto(bpm):
    papel = papel_ecg(bpm)
    mascara = geometria.mascara_tracado(papel)
    resultado = geometria.frequencia_cardiaca(mascara, PX_MM)
    assert resultado["bpm"] == pytest.approx(bpm, rel=0.03)


def test_mascara_ignora_grade_vermelha_e_sombra():
    papel = papel_ecg().astype(np.float32)
    sombra = papel * np.linspace(0.6, 1.0, papel.shape[1])[None, :, None]
    mascara = geometria.mascara_tracado(sombra.astype(np.uint8))
    tracado = cv2.cvtColor(papel_ecg(), cv2.COLOR_BGR2GRAY) < 60
    acerto = (mascara > 0)[tracado].mean()
    falso = (mascara > 0)[~tracado].mean()
    assert acerto > 0.9 and falso < 0.01


def test_foto_simulada_e_endireitada_volta_ao_papel():
    rng = np.random.default_rng(0)
    papel = papel_ecg()
    h, w = papel.shape[:2]
    cantos = np.array([[180, 120], [1500, 60], [1560, 760], [140, 690]], np.float32)
    foto = geometria.simular_foto(papel, cantos, (860, 1700), rng)

    achados = geometria.achar_papel(foto)
    assert np.abs(achados - cantos).max() < 6

    reta = geometria.retificar(foto, achados, w, h)
    horizontal, vertical = geometria.passo_da_grade(reta)
    assert horizontal == pytest.approx(PX_MM, abs=0.1)
    assert vertical == pytest.approx(PX_MM, abs=0.1)
