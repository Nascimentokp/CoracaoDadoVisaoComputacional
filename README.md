# Coração de Dados — Visão Computacional
# EQUIPE: Adriano, Leonardo e João Paulo

Visão computacional para **detectar cardiomegalia** (coração aumentado) em raio-X de tórax.
É o terceiro projeto da série Coração de Dados: o Tema 6 agrupava pacientes pelo perfil
clínico, o projeto de Redes Neurais previa doença cardiovascular a partir desse perfil, e
aqui o mesmo coração passa a ser visto **pela imagem**.

O trabalho segue a ordem das aulas: abrir o lote, filtros, geometria, limiar e morfologia,
CNN, o dado que mente, mapas de ativação e segmentação. A visão clássica **mede** o coração
(índice cardiotorácico), a CNN **aprende** sozinha, e a auditoria mostra onde cada uma engana.

## Dados

**Cardiomegaly Chest X-Ray Image Dataset** ([Zenodo 17937122](https://zenodo.org/records/17937122),
CC BY 4.0), recorte do ChestX-ray14 do NIH Clinical Center.

| | Treino | Teste | Total |
|:--|--:|--:|--:|
| Cardiomegalia | 2.219 | 557 | 2.776 |
| Normal | 2.219 | 557 | 2.776 |
| **Total** | **4.438** | **1.114** | **5.552** |

Raio-X em tons de cinza, 128 × 128. O teste é o da própria base; 20 % do treino é separado,
estratificado, para validação. A base não vai para o git: `uv run invoke dados` baixa o zip,
confere o MD5 e extrai em `data/raw/`.

O notebook 03 usa duas amostras de ECG em papel do
[ECG-Image-Kit](https://github.com/alphanumericslab/ecg-image-kit) (licença BSD-3), do mesmo grupo
que montou a ECG-Image-Database: um escaneado limpo e uma foto de celular. Também são baixadas por
`uv run invoke dados`.

**Limitações conhecidas** (medidas no notebook 01):

* os arquivos se chamam só `<n>.png`, sem ID de paciente. 76 raio-X de teste (6,8 %) têm um
  quase gêmeo no treino (correlação ≥ 0,92), em geral a mesma pessoa em outro exame;
* os rótulos vêm de laudos lidos por processamento de texto: mesmo entre pares quase idênticos,
  só 60–70 % têm o mesmo rótulo.

## Resultados até aqui

| Abordagem | AUC no teste | Observação |
|:--|--:|:--|
| CNN pequena, raio-X pré-processado (notebook 05) | **0,809** (0,783–0,833) | 27 mil pesos; empata com a crua |
| CNN pequena, raio-X cru (notebook 05) | 0,802 (0,776–0,827) | +0,064 sobre o ICT (IC 0,030–0,097, bootstrap pareado) |
| ICT clássico (perfil por linha, notebook 04) | 0,739 (0,710–0,769) | Corte clínico 0,5: sensibilidade 54 %, especificidade 82 % |
| MobileNetV2 ImageNet + ajuste fino (notebook 05) | 0,795 (0,768–0,819) | Empata com a CNN pequena, com 80× mais pesos |

**O que a rede usa** (notebooks 06 e 07): com a região do coração tapada a AUC cai só de 0,695 para
0,657; o Grad-CAM mostra a borda esquerda do coração nas cardiomegalias, mas também atenção fora do
tórax. O vazamento por paciente não inflou a AUC (split aleatório 0,680 × por grupo 0,694).

## Roteiro

| Notebook | Aula | O que faz | Situação |
|:--|:--|:--|:--|
| `01-abrir-o-lote` | A imagem como dado | Shape, dtype, histograma por classe, brilho/contraste/nitidez, duplicatas e quase duplicatas entre treino e teste | ✅ |
| `02-filtros` | Filtros à mão | Kernel na mão conferido com o OpenCV, ruído × suavização (PSNR), Sobel no perfil do coração, Canny, CLAHE | ✅ |
| `03-geometria-ecg` | Geometria | ECG em papel: canais de cor e HSV, régua da grade (px/mm), frequência cardíaca na imagem, foto torta → Otsu + morfologia + 4 cantos + homografia, interpolações | ✅ |
| `04-limiar-morfologia` | Limiar e morfologia | **Índice cardiotorácico** sem rede: a receita Otsu + morfologia + componentes (e por que falha) contra o perfil por linha com limiar à meia altura. AUC 0,739 no teste | ✅ |
| `05-cnn` | CNN | CNN pequena em Keras (cru × pré-processado) contra MobileNetV2, filtros aprendidos comparados ao Sobel, mapas de ativação, teste pareado contra o ICT | ✅ |
| `06-o-dado-que-mente` | O dado que mente | Exames com gêmeo no treino, split aleatório × por grupo (3 sementes), coração tapado × só o coração | ✅ |
| `07-mapas-de-ativacao` | Mapa de ativação | Grad-CAM da classe prevista em acertos e erros; fração da atenção no coração, no tórax e fora dele | ✅ |
| `08-segmentacao` | Segmentação | Contorno do coração pelo ICT e por limiar contra o marcado à mão pela equipe: Dice e IoU | ⏳ aguardando as marcações |

## Como rodar

```bash
uv sync                    # ambiente (Python 3.12)
uv run invoke dados        # baixa e confere a base
uv run invoke testes       # critérios de aceitação (specs/01-cardiomegalia.md)
uv run invoke notebooks    # executa os notebooks e regrava reports/
uv run invoke lab          # JupyterLab
```

## Estrutura

```
src/
  config.py          caminhos, fonte dos dados, semente, paleta
  data/dados.py      download, catálogo, carga e divisão
  visao/auditoria.py qualidade, duplicatas e quase duplicatas
  visao/filtros.py   kernels à mão, ruído, Sobel, Canny, CLAHE e pré-processamento
  visao/geometria.py achar a folha, homografia, régua da grade, traçado e frequência cardíaca
  visao/metricas.py  Dice e IoU
  visao/ict.py       índice cardiotorácico: segmentação 2D e perfil por linha
  model/cnn.py       CNN pequena, MobileNetV2, aumento sem espelhar, treino
  model/avaliacao.py AUC com bootstrap, comparação pareada, limiar de Youden
  model/gradcam.py   Grad-CAM e atenção por região
  utils/io.py        gravação padronizada de figuras e tabelas
notebooks/           um notebook por aula
reports/             figuras e as tabelas que as originaram
specs/               critérios de aceitação verificados pelos testes
tests/
```
