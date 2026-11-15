# SEntinela: previsão de dengue nos municípios de São Paulo

*E SE a dengue chegar? A gente avisa antes.*

Notebook que monta uma base semanal de dengue para os 645 municípios do estado de São Paulo (2014–2025) a partir das fontes primárias, estima o número reprodutivo efetivo (Rt) e limiares epidêmicos por município, e treina modelos de aprendizado de máquina para prever a incidência e a ultrapassagem do limiar epidêmico 1 e 4 semanas à frente.

**Autores:** _preencher_ · **Curso:** _preencher_

## Fontes de dados

| Fonte | Conteúdo |
|---|---|
| SINAN — Ministério da Saúde (Portal de Dados Abertos do SUS) | notificações de dengue |
| ERA5-Land (Copernicus/ECMWF), via Google Earth Engine | temperatura e ponto de orvalho diários |
| IBGE — SIDRA, malhas e localidades | população, polígonos e nomes dos municípios |

Os dados não ficam no repositório: o notebook baixa tudo na primeira execução.

## Como executar

1. Abra `Previsao_dengue_SP_v2.ipynb` (versão mais recente) no Google Colab.
2. Cadastre-se gratuitamente no [Google Earth Engine](https://code.earthengine.google.com/register) para uso não comercial e coloque o ID do seu projeto do Google Cloud em `GEE_PROJETO`, no início da célula de clima (seção 4.5).
3. Execute todas as células. Os dados ficam em cache no Google Drive (`MyDrive/dengue_sp/dados` e `MyDrive/dengue_sp_gee`); se a execução for interrompida, basta rodar de novo.

Para rodar localmente: `pip install -r requirements.txt` e abra o notebook no Jupyter.

## Estrutura do notebook

1. Introdução, objetivos, perguntas de pesquisa e variáveis-alvo
2. Dados e fontes
3. Configuração
4. Extração (IBGE, SINAN, ERA5-Land via Earth Engine)
5. Base município × semana
6. Análise exploratória
7. Número reprodutivo efetivo (Rt)
8. Limiares epidêmicos
9. Formulação dos modelos (variáveis, vizinhança, alvo de crescimento, janelas móveis)
10. Regressão da incidência: regressão linear, modelos de árvore, faixas de incerteza (WIS) e ensemble
11. Classificação: regressão logística e Gradient Boosting, matrizes de confusão
12. Cenário sazonal com climatologia
13. Recomendações para a vigilância (ações, impactos e riscos)
14. Conclusões
15. Reprodutibilidade
16. Exportação dos dados para o painel interativo

## Painel interativo (Streamlit)

A pasta `painel/` tem o painel web do SEntinela. Ele só lê os arquivos exportados pelo notebook, sem treinar modelos, e tem:

- **Município:** casos reais e previstos com faixas de 50% e 90%, limiares, Rt, clima, nível de alerta, plano de ação para o nível previsto e o histórico dos alertas ("o SEntinela teria avisado?").
- **Mapa:** nível de alerta, probabilidade de semana epidêmica e incidência dos 645 municípios, semana a semana.
- **Estado:** total do estado e quantas entradas de municípios em epidemia foram avisadas com antecedência no período de teste.
- **Simulador:** cenários de temperatura e de ações de controle, com a estimativa de internações e leitos.
- **Seus dados:** o município envia um CSV com os casos da semana ou o índice de infestação (LIRAa) e vê esses dados ao lado da previsão.
- **Modelos:** métricas de regressão e classificação.

**Atualizar os dados do painel:** rode a seção 16 do notebook, baixe o `painel_arquivos.zip` e descompacte o conteúdo em `painel/arquivos/`.

**Rodar no computador:**

```bash
pip install -r painel/requirements.txt
streamlit run painel/streamlit_app.py
```

**Publicar na internet (grátis):** em [share.streamlit.io](https://share.streamlit.io), entre com a conta do GitHub, clique em *Create app*, escolha este repositório, a branch `main` e o arquivo `painel/streamlit_app.py`.
