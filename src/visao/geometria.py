"""Geometria no ECG em papel: achar a folha, endireitar e medir com a régua da grade.

Aulas "Visão: Geometria" e "quando a régua chega". O papel do ECG é uma régua
impressa: quadradinho de 1 mm, quadrado grande de 5 mm, e o papel corre a
25 mm/s. Se a foto distorce a folha, a régua distorce junto e qualquer medida
(intervalo entre batimentos, frequência cardíaca) sai errada.

Pipeline de "scanner de documento", tudo com o conteúdo das aulas:

1. `achar_papel` — cinza → suavização → Otsu → fechamento → maior contorno →
   polígono de 4 cantos;
2. `retificar` — homografia dos 4 cantos para um retângulo (`warpPerspective`),
   com a interpolação escolhida;
3. `passo_da_grade` — quantos pixels vale 1 mm, pela autocorrelação do perfil
   da grade (as linhas se repetem a cada quadrado grande);
4. `mascara_tracado` + `frequencia_cardiaca` — separa o traçado da grade e mede
   a distância entre os picos R na tira de ritmo.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import find_peaks

#: Velocidade padrão do papel de ECG.
VELOCIDADE_MM_S = 25.0

INTERPOLACOES = {
    "Vizinho mais próximo": cv2.INTER_NEAREST,
    "Bilinear": cv2.INTER_LINEAR,
    "Bicúbica": cv2.INTER_CUBIC,
    "Lanczos4": cv2.INTER_LANCZOS4,
}

# --------------------------------------------------------------------------- #
# Cantos e homografia
# --------------------------------------------------------------------------- #


def ordenar_cantos(pontos: np.ndarray) -> np.ndarray:
    """Ordena 4 pontos como sup. esquerdo, sup. direito, inf. direito, inf. esquerdo.

    Os pontos são postos em volta do centro, no sentido horário (o eixo y da
    imagem cresce para baixo), e a volta começa no de menor soma x + y. Com a
    folha girada a 45°, dois pontos empatam na soma, mas a ordem horária não
    repete nenhum: cada canto sai uma vez só.
    """
    p = np.asarray(pontos, dtype=np.float32).reshape(4, 2)
    centro = p.mean(axis=0)
    horario = p[np.argsort(np.arctan2(p[:, 1] - centro[1], p[:, 0] - centro[0]))]
    return np.roll(horario, -int(horario.sum(axis=1).argmin()), axis=0)


def cantos_do_retangulo(largura: int, altura: int) -> np.ndarray:
    return np.array(
        [[0, 0], [largura - 1, 0], [largura - 1, altura - 1], [0, altura - 1]], dtype=np.float32
    )


def achar_papel(foto: np.ndarray, fechamento: int = 25) -> np.ndarray:
    """Os 4 cantos da folha clara sobre fundo escuro, ordenados.

    O fechamento (dilatação seguida de erosão) tapa os "buracos" que traçado e
    grade fazem na máscara do papel, para o contorno externo ser só a borda.
    """
    cinza = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY) if foto.ndim == 3 else foto
    suave = cv2.GaussianBlur(cinza, (5, 5), 0)
    _, mascara = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (fechamento, fechamento))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, elemento)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    folha = max(contornos, key=cv2.contourArea)
    perimetro = cv2.arcLength(folha, True)
    for tolerancia in (0.01, 0.02, 0.03, 0.05):
        poligono = cv2.approxPolyDP(folha, tolerancia * perimetro, True)
        if len(poligono) == 4:
            return ordenar_cantos(poligono)
    return ordenar_cantos(cv2.boxPoints(cv2.minAreaRect(folha)))


def retificar(
    foto: np.ndarray,
    cantos: np.ndarray,
    largura: int,
    altura: int,
    interpolacao: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Leva os 4 cantos da folha para um retângulo `largura` × `altura`."""
    matriz = cv2.getPerspectiveTransform(
        ordenar_cantos(cantos), cantos_do_retangulo(largura, altura)
    )
    return cv2.warpPerspective(foto, matriz, (largura, altura), flags=interpolacao)


def simular_foto(
    documento: np.ndarray,
    cantos_na_foto: np.ndarray,
    tamanho: tuple[int, int],
    rng: np.random.Generator,
    sombra: float = 0.35,
    ruido: float = 4.0,
) -> np.ndarray:
    """Fotografa o documento: folha em perspectiva sobre uma mesa escura, com
    sombra de um lado e chiado do sensor.

    `cantos_na_foto` diz onde os cantos da folha caem na foto (sup. esq., sup.
    dir., inf. dir., inf. esq.) e `tamanho` é (altura, largura) da foto. Como os
    cantos são conhecidos, a foto serve de gabarito para medir a correção.
    """
    altura, largura = tamanho
    h, w = documento.shape[:2]
    matriz = cv2.getPerspectiveTransform(
        cantos_do_retangulo(w, h), np.asarray(cantos_na_foto, np.float32)
    )
    folha = cv2.warpPerspective(documento, matriz, (largura, altura), flags=cv2.INTER_AREA)
    cobertura = cv2.warpPerspective(
        np.ones((h, w), np.float32), matriz, (largura, altura), flags=cv2.INTER_LINEAR
    )[..., None]

    # Mesa: marrom escuro com veios suaves.
    veios = cv2.GaussianBlur(rng.normal(0, 1, (altura, largura)).astype(np.float32), (0, 0), 9)
    mesa = np.stack([45 + 25 * veios, 60 + 25 * veios, 80 + 25 * veios], axis=-1)

    foto = cobertura * folha.astype(np.float32) + (1 - cobertura) * mesa
    luz = np.linspace(1 - sombra, 1.0, largura, dtype=np.float32)[None, :, None]
    foto = foto * luz + rng.normal(0, ruido, foto.shape)
    return np.clip(foto, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# A régua: passo da grade
# --------------------------------------------------------------------------- #


def _periodo(perfil: np.ndarray, menor: int, maior: int) -> float:
    """Período dominante (em pixels, com fração) de um perfil 1-D, pela autocorrelação."""
    p = perfil.astype(np.float32)
    p = p - cv2.GaussianBlur(p.reshape(1, -1), (0, 0), 15).ravel()
    p -= p.mean()
    ac = np.correlate(p, p, "full")[len(p) - 1 :]
    ac /= ac[0] + 1e-12
    maior = min(maior, len(ac) - 2)
    k = menor + int(np.argmax(ac[menor:maior]))
    # Interpolação parabólica entre os três pontos ao redor do pico.
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    desvio = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    return k + float(desvio)


def passo_da_grade(imagem: np.ndarray, menor: int = 15, maior: int = 200) -> tuple[float, float]:
    """Pixels por milímetro na horizontal e na vertical.

    A grade se repete a cada quadrado grande (5 mm); o período procurado é o do
    quadrado grande, que tem a linha mais forte, dividido por 5. O perfil usa a
    **mediana** de cada coluna (ou linha), que ignora o traçado: ele ocupa
    poucos pixels de cada coluna, enquanto a linha da grade ocupa a coluna toda.
    """
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if imagem.ndim == 3 else imagem
    horizontal = _periodo(np.median(cinza, axis=0), menor, maior) / 5
    vertical = _periodo(np.median(cinza, axis=1), menor, maior) / 5
    return horizontal, vertical


# --------------------------------------------------------------------------- #
# Traçado e frequência cardíaca
# --------------------------------------------------------------------------- #


def canal_sem_grade(imagem: np.ndarray) -> np.ndarray:
    """Canal vermelho de uma imagem BGR (ou a própria imagem, se já for 1 canal).

    A grade do ECG é vermelha ou rosa: no canal R ela fica quase tão clara quanto
    o papel e praticamente some, enquanto o traçado preto continua escuro.
    """
    return imagem[..., 2] if imagem.ndim == 3 else imagem


def realcar_tracado(imagem: np.ndarray, janela: int = 31) -> np.ndarray:
    """Fundo menos imagem, no canal sem grade: o que é mais escuro que a vizinhança
    larga fica claro.

    A mediana de janela grande estima o papel (com a sombra junto) e ignora o
    traçado fino; a diferença tira a iluminação desigual de graça.
    """
    canal = canal_sem_grade(imagem)
    return cv2.subtract(cv2.medianBlur(canal, janela), canal)


def mascara_tracado(imagem: np.ndarray, limiar: int = 60, janela: int = 31) -> np.ndarray:
    """Máscara binária (0/255) do traçado: só o que é bem mais escuro que o papel."""
    return np.where(realcar_tracado(imagem, janela) > limiar, 255, 0).astype(np.uint8)


def frequencia_cardiaca(
    mascara: np.ndarray,
    px_por_mm: float,
    velocidade: float = VELOCIDADE_MM_S,
    inicio_faixa: float = 0.7,
) -> dict:
    """Frequência cardíaca pela tira de ritmo (a derivação longa na base da folha).

    A linha de base é a linha da máscara com mais traçado no terço de baixo. Em
    cada coluna, a altura do ponto mais alto do traçado acima da base forma um
    sinal 1-D; os picos R são os picos estreitos e altos desse sinal. A mediana
    dos intervalos R-R (pixels → mm → segundos) vira batimentos por minuto.
    """
    altura, largura = mascara.shape
    inicio = int(inicio_faixa * altura)
    linha_base = inicio + int(np.argmax((mascara[inicio:] > 0).sum(axis=1)))
    topo_faixa = max(0, linha_base - int(15 * px_por_mm))
    faixa = mascara[topo_faixa : linha_base + int(5 * px_por_mm)] > 0
    tem = faixa.any(axis=0)
    topo = np.where(tem, faixa.argmax(axis=0), linha_base - topo_faixa)
    sinal = (linha_base - topo_faixa - topo).astype(np.float32)
    picos, _ = find_peaks(
        sinal,
        prominence=4 * px_por_mm,
        distance=max(1, int(0.3 * velocidade * px_por_mm)),
        width=(None, 2.5 * px_por_mm),
    )
    rr = np.diff(picos)
    rr_mm = float(np.median(rr)) / px_por_mm if len(rr) else float("nan")
    return {
        "bpm": 60 * velocidade / rr_mm if len(rr) else float("nan"),
        "rr_mm": rr_mm,
        "picos": picos,
        "linha_base": linha_base,
        "sinal": sinal,
    }
