# SEntinela

*E SE a dengue chegar? A gente avisa antes.*

| Campo | Conteúdo |
|---|---|
| **Nome** | SEntinela (provisório: SentinelaDengue; alternativas: VigiAedes, Aedes Radar) |
| **Cidade e bairro** | Osasco/SP, bairro a definir pela equipe |
| **Cidadão beneficiado** | Moradores de áreas com histórico de dengue, com foco em idosos, gestantes e crianças; agentes de saúde e gestores da prefeitura |
| **Problema** | Os surtos pegam moradores e serviços de saúde de surpresa, porque o aviso só chega depois que os casos já subiram |
| **Objetivo** | Avisar o surto antes dele acontecer: nível de alerta por município, 1 e 4 semanas à frente, a tempo de a vigilância agir |
| **Escopo** | Dispositivos no bairro detectam vestígios do mosquito e medem temperatura e umidade; o sistema cruza essas leituras com o clima e os casos notificados, calcula o risco de cada área e entrega o resultado num painel para a prefeitura e num app com assistente para o cidadão |
| **O que cada parte faz** | Dispositivo coleta · banco armazena · API entrega os dados · app mostra risco e alertas · IA de previsão apoia o gestor a priorizar · IA assistente orienta o cidadão |
| **Funcionalidades** | Nível de alerta (🟢 Baixo, 🟡 Atenção, 🔴 Epidemia) 1 e 4 semanas à frente<br>Previsão de casos da semana, com faixa de incerteza<br>Chance de semana epidêmica, calibrada com o histórico<br>Mapa do estado, semana a semana<br>Plano de ação para cada nível<br>Prestação de contas: quantos alertas se confirmaram |
| **Pronto neste repositório** | Base semanal dos 645 municípios de SP (2014–2025), Rt, limiares epidêmicos, modelo de previsão e painel web |
| **Ainda proposta** | Dispositivos de coleta, app do cidadão e assistente. Hoje o modelo usa SINAN e ERA5-Land no lugar das leituras; o sensoriamento entra depois como mais uma variável, medindo o bairro e não o município |
| **Autores / Curso** | _preencher_ |
| **Repositório** | _preencher_ |

## Como a previsão é feita

| Etapa | O que é feito |
|---|---|
| Dados | notificações de dengue (SINAN/OpenDataSUS), população e malhas (IBGE), temperatura e umidade (ERA5-Land via Google Earth Engine) |
| Indicadores | incidência semanal, Rt (método de Cori) e limiares pré-epidêmico e epidêmico por município (método MEM) |
| Variáveis | só casos já digitados na semana da previsão, incidência dos municípios vizinhos, Rt, sazonalidade e clima com 1 a 12 semanas de defasagem |
| Modelo | Random Forest prevendo o crescimento da incidência, e não o nível |
| Validação | janelas móveis: cada ano de 2023 a 2026 é previsto por modelos treinados só com os anos anteriores |
| Resultado (4 semanas antes) | 8 em cada 10 alertas se confirmaram · erro 24% menor que repetir as últimas semanas · em 83% das entradas em epidemia o painel já marcava Atenção ou Epidemia |

## Como executar

**Notebook** (`Previsao_dengue_SP_v2.ipynb`, no Google Colab): cadastre-se no [Google Earth Engine](https://code.earthengine.google.com/register), coloque o ID do seu projeto em `GEE_PROJETO` (seção 4.5) e execute as células. Os dados ficam em cache no Drive; o notebook baixa tudo na primeira execução.

**Painel** (`painel/streamlit_app.py`): lê os arquivos exportados pela seção 16 do notebook, sem treinar modelos.

```bash
pip install -r painel/requirements.txt
streamlit run painel/streamlit_app.py
```

Para atualizar os dados do painel, rode a seção 16, baixe o `painel_arquivos.zip` e descompacte em `painel/arquivos/`. Para publicar de graça, use [share.streamlit.io](https://share.streamlit.io) apontando para `painel/streamlit_app.py`.

## Limitações

- O modelo subestima epidemias fora do padrão, como a de 2024, e o começo de cada epidemia.
- Casos notificados incluem casos depois descartados, e as semanas mais recentes ainda estão incompletas.
- Ferramenta acadêmica de apoio: não substitui a vigilância epidemiológica oficial.
