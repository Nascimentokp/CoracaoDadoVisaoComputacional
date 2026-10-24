"""Configuração central do projeto.

Concentra caminhos, fonte dos dados, semente aleatória e paleta que precisam
ser idênticos entre notebooks, scripts e testes. Nenhum outro módulo deve
montar caminho na mão.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #

RAIZ = Path(__file__).resolve().parent.parent

DATA = RAIZ / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"

MODELS = RAIZ / "models"

REPORTS = RAIZ / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

# --------------------------------------------------------------------------- #
# Fonte dos dados
# --------------------------------------------------------------------------- #

#: Cardiomegaly Chest X-Ray Image Dataset (Zenodo, CC BY 4.0), recorte do
#: ChestX-ray14 do NIH Clinical Center. 5.552 raios-X em tons de cinza, 128×128.
URL_BASE = "https://zenodo.org/records/17937122/files/cardiomegaly_set.zip?download=1"
MD5_BASE = "dfd9a9441b87f6bcf177f315a8b0465b"
ZIP_BASE = DATA_RAW / "cardiomegaly_set.zip"
PASTA_BASE = DATA_RAW / "cardiomegaly_set"

#: Pastas da base: `<particao>/<classe>/<n>.png`. A divisão treino/teste já vem
#: pronta (80/20) e é mantida; a validação sai do treino.
PARTICOES = ["train", "test"]
CLASSES = {"true": 1, "false": 0}

#: Nomes legíveis para gráficos e tabelas.
ROTULOS = {0: "Normal", 1: "Cardiomegalia"}
NOMES_PARTICAO = {"train": "treino", "test": "teste"}

TAMANHO = 128

#: Amostras de ECG em papel do ECG-Image-Kit (Alphanumerics Lab, licença BSD-3),
#: do mesmo grupo que montou a ECG-Image-Database. Só as usadas no notebook 03.
URL_ECG = (
    "https://raw.githubusercontent.com/alphanumericslab/ecg-image-kit/main/sample-data/ecg-images"
)
PASTA_ECG = DATA_RAW / "ecg-image-kit"
#: Escaneado limpo (grade rosa, tira de ritmo DII embaixo): gabarito da geometria.
ECG_ESCANEADO = "ecg00012.jpg"
#: Foto de celular real, de frente, em baixa resolução (≈ 4,5 px/mm).
ECG_FOTO = "ecg00046.jpg"

#: Tomografia de tórax CTChest, amostra pública do 3D Slicer (a mesma dos exemplos de
#: tomografia do curso de segmentação). ≈ 40 MB, conferida pelo SHA-256 do endereço.
SHA256_CT = "4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e"
URL_CT = f"https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/{SHA256_CT}"
ARQUIVO_CT = DATA_RAW / "ct-torax.nrrd"

# --------------------------------------------------------------------------- #
# Reprodutibilidade e protocolo de avaliação
# --------------------------------------------------------------------------- #

SEMENTE = 42
DPI = 150

#: Fração do treino original separada, estratificada, para validação.
PROPORCAO_VALIDACAO = 0.20

#: Parâmetros do perfil por linha do ICT escolhidos pela grade de validação do
#: notebook 04 (AUC de validação, depois menos falhas). Todos os notebooks que
#: medem o ICT depois do 04 usam estes; se a grade mudar de escolha, mude aqui.
PARAMS_ICT = dict(sigma_vertical=3.0, linhas=(50, 90), quantil=75.0)

#: Correlação (imagens reduzidas a 32×32) a partir da qual dois raios-X são
#: tratados como prováveis exames do mesmo paciente. Calibrada olhando os pares
#: no notebook 01: entre 0,93 e 0,96 quase todos são a mesma pessoa; perto de
#: 0,92, metade; em 0,90 já são pessoas diferentes.
LIMIAR_QUASE_DUPLICATA = 0.92

# --------------------------------------------------------------------------- #
# Identidade visual (a mesma dos projetos Coração de Dados)
# --------------------------------------------------------------------------- #

AZUL_PROFUNDO = "#1B4965"
AZUL_COBALTO = "#2A6F97"
AZUL_CLARO = "#62B6CB"
VERDE = "#74C69D"
AMARELO = "#F4D35E"
CORAL = "#EE6C4D"
ROSA = "#F2A6A6"
LARANJA = "#F49D6E"
ROXO = "#8E7DBE"

#: Sequência categórica para gráficos com múltiplos grupos.
PALETA = [AZUL_COBALTO, CORAL, VERDE, AMARELO, ROXO, LARANJA, ROSA, AZUL_CLARO]

#: Cor fixa de cada classe em todos os gráficos.
CORES_CLASSE = {0: AZUL_COBALTO, 1: CORAL}


def garantir_diretorios() -> None:
    """Cria os diretórios de saída caso ainda não existam."""
    for diretorio in (DATA_RAW, DATA_PROCESSED, MODELS, FIGURES, TABLES):
        diretorio.mkdir(parents=True, exist_ok=True)
