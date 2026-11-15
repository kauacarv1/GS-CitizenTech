"""Painel de previsão de dengue — Estado de São Paulo.

Lê os arquivos gerados pela seção 16 do notebook `Previsao_dengue_SP_v2.ipynb` (pasta `painel/arquivos`).
Rodar localmente:  streamlit run painel/streamlit_app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Dengue SP — Previsão de casos", page_icon="🦟", layout="wide")

DIR_ARQ = Path(__file__).parent / "arquivos"
AZUL, AZUL_50, AZUL_90 = "#3b82f6", "rgba(59,130,246,0.35)", "rgba(59,130,246,0.15)"
CINZA, VERDE, AMARELO, VERMELHO = "#e5e7eb", "#22c55e", "#eab308", "#ef4444"
CORES_NIVEL = {"Baixo": VERDE, "Atenção": AMARELO, "Epidemia": VERMELHO}
LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15), hovermode="x unified",
              separators=",.", xaxis=dict(tickformat="%m/%Y"))

st.markdown("""
<style>
.cartao {background: #161a23; border: 1px solid #262b36; border-radius: 14px; padding: 14px 18px; height: 100%;}
.cartao .rot {font-size: 0.80rem; color: #9ca3af; margin-bottom: 4px;}
.cartao .val {font-size: 1.65rem; font-weight: 700; line-height: 1.2;}
.cartao .sub {font-size: 0.78rem; color: #9ca3af; margin-top: 4px;}
h1 {font-size: 1.9rem !important;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ dados
def num(x, casas=0):
    """Número no formato brasileiro."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data
def carregar():
    arquivos = ["municipios.parquet", "serie.parquet", "previsoes_teste.parquet", "previsao_futura.parquet",
                "malha.geojson", "metricas.json"]
    faltando = [a for a in arquivos if not (DIR_ARQ / a).exists()]
    if faltando:
        return None, faltando
    mun = pd.read_parquet(DIR_ARQ / "municipios.parquet")
    serie = pd.read_parquet(DIR_ARQ / "serie.parquet")
    prev = pd.read_parquet(DIR_ARQ / "previsoes_teste.parquet")
    fut = pd.read_parquet(DIR_ARQ / "previsao_futura.parquet")
    malha = orientar_d3(json.loads((DIR_ARQ / "malha.geojson").read_text(encoding="utf-8")))
    meta = json.loads((DIR_ARQ / "metricas.json").read_text(encoding="utf-8"))
    for df, col in [(serie, "semana"), (prev, "semana_alvo"), (fut, "semana_alvo"), (fut, "semana_base")]:
        df[col] = pd.to_datetime(df[col])
    # incidência -> casos
    fut["pop"] = mun.set_index("m").loc[fut.m, "pop"].to_numpy()
    for df in (prev, fut):
        fator = df["pop"].to_numpy() / 1e5
        for c in ["y", "ensemble", "rf", "linear", "q05", "q25", "q50", "q75", "q95"]:
            if c in df:
                df[f"{c}_casos"] = df[c] * fator
    return dict(mun=mun, serie=serie, prev=prev, fut=fut, malha=malha, meta=meta), []


def cartoes(itens):
    """Linha de cartões: (rótulo, valor, texto de apoio, cor do valor)."""
    for col, (rot, val, sub, cor) in zip(st.columns(len(itens)), itens):
        col.markdown(f'<div class="cartao"><div class="rot">{rot}</div><div class="val" style="color:{cor}">{val}</div>'
                     f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def grafico(fig):
    st.plotly_chart(fig, theme=None, config={"displaylogo": False, "displayModeBar": False})


def orientar_d3(malha):
    """O mapa do plotly (d3-geo) espera o anel externo em sentido horário e os buracos em anti-horário."""
    def area(anel):
        x, y = np.asarray(anel, float).T
        return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    def poligono(aneis):
        return [anel if (area(anel) < 0) == (i == 0) else anel[::-1] for i, anel in enumerate(aneis)]

    for f in malha["features"]:
        g = f["geometry"]
        if g["type"] == "Polygon":
            g["coordinates"] = poligono(g["coordinates"])
        elif g["type"] == "MultiPolygon":
            g["coordinates"] = [poligono(p) for p in g["coordinates"]]
    return malha


def nivel(inc_prevista, lim_pre, lim_epi):
    return np.where(inc_prevista >= lim_epi, "Epidemia", np.where(inc_prevista >= lim_pre, "Atenção", "Baixo"))


dados, faltando = carregar()
if dados is None:
    st.title("🦟 Dengue SP — Previsão de casos")
    st.error("Arquivos de dados não encontrados em `painel/arquivos/`: " + ", ".join(faltando))
    st.markdown("Rode a seção **16. Exportar dados para o painel interativo** do notebook e descompacte o "
                "`painel_arquivos.zip` nesta pasta.")
    st.stop()

MUN, SERIE, PREV, FUT, MALHA, META = (dados[k] for k in ["mun", "serie", "prev", "fut", "malha", "meta"])
NOMES = MUN.sort_values("nome").nome.tolist()
ULTIMA = SERIE.semana.max()

# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.markdown("## 🦟 Dengue SP")
    st.caption("Previsão de casos de dengue nos municípios de São Paulo com dados do SINAN, IBGE e ERA5-Land.")
    padrao = "Campinas" if "Campinas" in NOMES else MUN.loc[MUN["pop"].idxmax(), "nome"]
    nome_mun = st.selectbox("Município", NOMES, index=NOMES.index(padrao))
    h = st.radio("Antecedência da previsão", META["horizontes"], horizontal=True,
                 format_func=lambda x: f"{x} semana" + ("s" if x > 1 else ""))
    hist = st.slider("Histórico (semanas)", 12, 312, 52, step=4)
    st.divider()
    st.caption(f"Dados de {pd.Timestamp(META['periodo'][0]):%d/%m/%Y} a {pd.Timestamp(META['periodo'][1]):%d/%m/%Y} · "
               f"{num(META['n_casos'])} notificações · {META['n_municipios']} municípios.\n\n"
               f"Arquivos gerados em {META['gerado_em']}.")

LINHA_MUN = MUN[MUN.nome == nome_mun].iloc[0]
M_SEL = int(LINHA_MUN.m)
POP_SEL = float(LINHA_MUN["pop"])
LIM_PRE_C, LIM_EPI_C = LINHA_MUN.limiar_pre * POP_SEL / 1e5, LINHA_MUN.limiar_epidemico * POP_SEL / 1e5

st.markdown(f"# 🦟 Previsão de dengue — Estado de {'São Paulo' if META['uf'] == 'SP' else META['uf']}")
aba_mun, aba_mapa, aba_estado, aba_modelos, aba_sobre = st.tabs(
    ["📍 Município", "🗺️ Mapa", "📈 Estado", "🧪 Modelos", "ℹ️ Sobre"])

# ------------------------------------------------------------------ município
with aba_mun:
    s_mun = SERIE[SERIE.m == M_SEL].sort_values("semana")
    p_mun = PREV[(PREV.m == M_SEL) & (PREV.h == h)].sort_values("semana_alvo")
    f_mun = FUT[FUT.m == M_SEL].sort_values("h")
    ult = s_mun.iloc[-1]
    st.markdown(f"### Previsão de casos de dengue: {META['uf']} – {nome_mun}")

    f_h = f_mun[f_mun.h == h].iloc[0]
    nv = nivel(f_h.ensemble, LINHA_MUN.limiar_pre, LINHA_MUN.limiar_epidemico).item()
    itens = [(f"Casos na semana de {ULTIMA:%d/%m/%Y}", num(ult.casos), f"{num(ult.inc, 1)} por 100 mil hab.", CINZA)]
    for _, f in f_mun.iterrows():
        itens.append((f"Previsão para {f.semana_alvo:%d/%m/%Y} (+{int(f.h)} sem.)", num(f.ensemble_casos),
                      f"faixa de 90%: {num(f.q05_casos)} a {num(f.q95_casos)}", AZUL))
    itens.append((f"Nível previsto (+{h} sem.)", nv, f"P(semana epidêmica) = {f_h.p_epidemia:.0%}", CORES_NIVEL[nv]))
    itens.append(("Rt na última semana", num(ult.rt, 2), f"P(Rt > 1) = {ult.p_rt:.0%}",
                  VERMELHO if ult.p_rt > 0.95 else CINZA))
    cartoes(itens)
    st.write("")

    # gráfico principal
    inicio = ULTIMA - pd.Timedelta(weeks=hist)
    s_jan = s_mun[s_mun.semana >= inicio]
    p_jan = p_mun[p_mun.semana_alvo >= inicio]
    fig = go.Figure()
    if len(p_jan):
        x = pd.concat([p_jan.semana_alvo, p_jan.semana_alvo[::-1]])
        fig.add_trace(go.Scatter(x=x, y=pd.concat([p_jan.q95_casos, p_jan.q05_casos[::-1]]), fill="toself",
                                 fillcolor=AZUL_90, line=dict(width=0), name="faixa 90%", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=pd.concat([p_jan.q75_casos, p_jan.q25_casos[::-1]]), fill="toself",
                                 fillcolor=AZUL_50, line=dict(width=0), name="faixa 50%", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=p_jan.semana_alvo, y=p_jan.ensemble_casos, name=f"previsão ({h} sem. antes)",
                                 line=dict(color=AZUL, width=2.5), hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=s_jan.semana, y=s_jan.casos, name="casos reais", line=dict(color=CINZA, width=2),
                             hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=f_mun.semana_alvo, y=f_mun.ensemble_casos, mode="markers+lines",
                             name="previsão futura", line=dict(color=AZUL, dash="dot"),
                             marker=dict(size=10, symbol="diamond"),
                             error_y=dict(type="data", symmetric=False,
                                          array=f_mun.q95_casos - f_mun.ensemble_casos,
                                          arrayminus=f_mun.ensemble_casos - f_mun.q05_casos),
                             hovertemplate="%{y:,.0f}"))
    fig.add_hline(y=LIM_EPI_C, line=dict(color=VERMELHO, dash="dash", width=1),
                  annotation_text="limiar epidêmico", annotation_font_color=VERMELHO)
    fig.add_hline(y=LIM_PRE_C, line=dict(color=AMARELO, dash="dash", width=1),
                  annotation_text="limiar pré-epidêmico", annotation_font_color=AMARELO,
                  annotation_position="bottom right")
    fig.update_layout(**LAYOUT, height=440, title="Casos por semana: reais e previstos", yaxis_title="casos")
    grafico(fig)
    st.caption("A previsão de cada semana foi feita com os dados disponíveis "
               f"{h} semana(s) antes, por um modelo treinado só com os anos anteriores "
               f"(teste de {META['anos_teste'][0]} a {META['anos_teste'][-1]}). "
               "Os losangos são a previsão para depois da última semana da base.")

    c1, c2, c3 = st.columns(3)
    with c1:
        f2 = go.Figure(go.Scatter(x=s_jan.semana, y=s_jan.rt, line=dict(color="#a78bfa"), name="Rt"))
        f2.add_hline(y=1, line=dict(color=VERMELHO, dash="dash"))
        f2.update_layout(**{**LAYOUT, "legend": dict(orientation="h", y=-0.35)}, height=300, title="Número reprodutivo (Rt)", showlegend=False)
        grafico(f2)
    with c2:
        f3 = go.Figure(go.Bar(x=p_jan.semana_alvo, y=p_jan.p_epidemia * 100,
                              marker_color=np.where(p_jan.p_epidemia >= 0.5, VERMELHO, AZUL)))
        f3.update_layout(**{**LAYOUT, "legend": dict(orientation="h", y=-0.35)}, height=300, title=f"P(semana epidêmica), {h} sem. antes",
                         yaxis=dict(range=[0, 100], ticksuffix="%"), showlegend=False)
        grafico(f3)
    with c3:
        f4 = go.Figure()
        f4.add_trace(go.Scatter(x=s_jan.semana, y=s_jan.temp_min, name="temp. mínima (°C)", line=dict(color="#f97316")))
        f4.add_trace(go.Scatter(x=s_jan.semana, y=s_jan.umid_med, name="umidade (%)", yaxis="y2",
                                line=dict(color="#38bdf8")))
        f4.update_layout(**{**LAYOUT, "legend": dict(orientation="h", y=-0.35), "xaxis": dict(tickformat="%m/%Y", nticks=4)},
                         height=300, title="Clima (ERA5-Land)",
                         yaxis=dict(title="°C"), yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False, tickformat=".0f"))
        grafico(f4)

    with st.expander("Comparar com os modelos individuais"):
        f5 = go.Figure()
        f5.add_trace(go.Scatter(x=s_jan.semana, y=s_jan.casos, name="casos reais", line=dict(color=CINZA, width=2)))
        for col_, nome_, cor in [("ensemble_casos", "Ensemble", AZUL), ("rf_casos", "Random Forest", "#22c55e"),
                                 ("linear_casos", "Regressão linear", "#f472b6")]:
            f5.add_trace(go.Scatter(x=p_jan.semana_alvo, y=p_jan[col_], name=nome_, line=dict(color=cor)))
        f5.update_layout(**LAYOUT, height=360)
        grafico(f5)

# ------------------------------------------------------------------ mapa
with aba_mapa:
    semanas_mapa = sorted(PREV[PREV.h == h].semana_alvo.unique()) + sorted(FUT[FUT.h == h].semana_alvo.unique())
    semanas_mapa = [pd.Timestamp(s) for s in semanas_mapa]
    c1, c2 = st.columns([2, 1])
    sem = c1.select_slider("Semana prevista", semanas_mapa, value=semanas_mapa[-1],
                           format_func=lambda s: f"{s:%d/%m/%Y}" + (" (futura)" if s > ULTIMA else ""))
    variavel = c2.radio("Mostrar", ["Nível previsto", "Probabilidade de epidemia", "Incidência prevista",
                                    "Incidência real"], horizontal=False)
    base = FUT if sem > ULTIMA else PREV
    d = base[(base.h == h) & (base.semana_alvo == sem)].merge(MUN[["m", "geocodigo", "nome", "limiar_pre",
                                                                    "limiar_epidemico"]], on="m")
    d["nivel"] = nivel(d.ensemble, d.limiar_pre, d.limiar_epidemico)
    if variavel == "Incidência real" and "y" not in d:
        st.info("A incidência real ainda não existe para a semana futura; mostrando a incidência prevista.")
        variavel = "Incidência prevista"
    opcoes = dict(geojson=MALHA, locations="geocodigo", featureidkey="id", hover_name="nome", fitbounds="locations",
                  height=560)
    if variavel == "Nível previsto":
        figm = px.choropleth(d, color="nivel", color_discrete_map=CORES_NIVEL,
                                 category_orders={"nivel": list(CORES_NIVEL)}, **opcoes)
    elif variavel == "Probabilidade de epidemia":
        figm = px.choropleth(d.assign(p=d.p_epidemia * 100), color="p", range_color=(0, 100),
                                 color_continuous_scale="YlOrRd", labels={"p": "P(epidemia) %"}, **opcoes)
    else:
        col = "ensemble" if variavel == "Incidência prevista" else "y"
        teto = max(float(np.nanquantile(d[col], 0.98)), 1.0)
        figm = px.choropleth(d, color=col, range_color=(0, teto), color_continuous_scale="YlOrRd",
                                 labels={col: "casos / 100 mil"}, **opcoes)
    figm.update_geos(visible=False, bgcolor="rgba(0,0,0,0)")
    figm.update_traces(marker_line_color="#0b0d12", marker_line_width=0.4)
    figm.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", template="plotly_dark",
                       separators=",.",
                       legend=dict(title="", orientation="h", y=0.02, x=0.02, bgcolor="rgba(0,0,0,0.5)"))
    grafico(figm)

    contagem = d.nivel.value_counts().reindex(list(CORES_NIVEL), fill_value=0)
    cartoes([(f"Municípios no nível {nv_}", num(n_), f"semana de {sem:%d/%m/%Y}", CORES_NIVEL[nv_])
             for nv_, n_ in contagem.items()])
    st.write("")
    st.markdown("**Municípios com maior probabilidade de semana epidêmica**")
    top = (d.sort_values("p_epidemia", ascending=False).head(15)
           .assign(**{"Casos previstos": lambda x: x.ensemble_casos.round(),
                      "Incidência prevista": lambda x: x.ensemble.round(1),
                      "P(epidemia)": lambda x: (x.p_epidemia * 100).round().astype(int).astype(str) + "%"}))
    st.dataframe(top[["nome", "Casos previstos", "Incidência prevista", "P(epidemia)", "nivel"]]
                 .rename(columns={"nome": "Município", "nivel": "Nível"}), hide_index=True)

# ------------------------------------------------------------------ estado
with aba_estado:
    tot = SERIE.groupby("semana").casos.sum()
    pe = PREV[PREV.h == h].groupby("semana_alvo")[["y_casos", "ensemble_casos"]].sum()
    fe = FUT[FUT.h == h].groupby("semana_alvo").ensemble_casos.sum()
    anos = SERIE.assign(ano=SERIE.semana.dt.year).groupby("ano").casos.sum()
    acima = (SERIE[SERIE.semana == ULTIMA].merge(MUN[["m", "limiar_epidemico"]], on="m")
             .pipe(lambda x: ((x.inc > x.limiar_epidemico) & (x.casos > 10)).sum()))
    ano_ult = int(anos.index[-1])
    parcial = ULTIMA < pd.Timestamp(f"{ano_ult}-12-20")
    if parcial:   # ano em andamento: compara com o mesmo período do ano anterior
        mesmo_periodo = SERIE[(SERIE.semana.dt.year == ano_ult - 1)
                              & (SERIE.semana <= ULTIMA - pd.DateOffset(years=1))].casos.sum()
        comparacao = f"{anos.iloc[-1] / max(mesmo_periodo, 1) - 1:+.0%} em relação ao mesmo período de {ano_ult - 1}"
    else:
        comparacao = f"{anos.iloc[-1] / anos.iloc[-2] - 1:+.0%} em relação a {ano_ult - 1}"
    rot_ano = f"Notificações em {ano_ult}" + (f" (até {ULTIMA:%d/%m})" if parcial else "")
    cartoes([(rot_ano, num(anos.iloc[-1]), comparacao, CINZA),
             (f"Casos na semana de {ULTIMA:%d/%m/%Y}", num(tot.iloc[-1]), "total do estado", CINZA),
             (f"Previsão para {fe.index[0]:%d/%m/%Y}", num(fe.iloc[0]), f"+{h} semana(s), soma dos municípios", AZUL),
             ("Municípios acima do limiar epidêmico", num(acima), f"semana de {ULTIMA:%d/%m/%Y}", VERMELHO)])
    st.write("")

    fig = go.Figure()
    jan = tot[tot.index >= ULTIMA - pd.Timedelta(weeks=max(hist, 52))]
    fig.add_trace(go.Scatter(x=jan.index, y=jan.values, name="casos reais", line=dict(color=CINZA, width=2),
                             hovertemplate="%{y:,.0f}"))
    pj = pe[pe.index >= jan.index[0]]
    fig.add_trace(go.Scatter(x=pj.index, y=pj.ensemble_casos, name=f"previsão ({h} sem. antes)",
                             line=dict(color=AZUL, width=2.5), hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=fe.index, y=fe.values, name="previsão futura", mode="markers",
                             marker=dict(size=11, symbol="diamond", color=AZUL), hovertemplate="%{y:,.0f}"))
    fig.update_layout(**LAYOUT, height=440, title=f"Total do estado de {META['uf']}", yaxis_title="casos por semana")
    grafico(fig)

    c1, c2 = st.columns(2)
    with c1:
        fb = go.Figure(go.Bar(x=anos.index.astype(str), y=anos.values, marker_color=AZUL,
                              hovertemplate="%{y:,.0f}"))
        fb.update_layout(**LAYOUT, height=320, title="Notificações por ano", showlegend=False)
        grafico(fb)
    with c2:
        sx = SERIE.merge(MUN[["m", "limiar_epidemico"]], on="m")
        n_ac = sx.assign(a=(sx.inc > sx.limiar_epidemico) & (sx.casos > 10)).groupby("semana").a.sum()
        fa = go.Figure(go.Scatter(x=n_ac.index, y=n_ac.values, fill="tozeroy", line=dict(color=VERMELHO)))
        fa.update_layout(**LAYOUT, height=320, title="Municípios acima do limiar epidêmico", showlegend=False)
        grafico(fa)

# ------------------------------------------------------------------ modelos
with aba_modelos:
    st.markdown(f"Desempenho no teste de **{META['anos_teste'][0]} a {META['anos_teste'][-1]}**: cada ano foi previsto "
                "por modelos treinados só com os anos anteriores (janelas móveis), sem vazamento de dados.")
    reg = pd.DataFrame(META["regressao"][str(h)]).set_index("modelo").astype(float)
    st.markdown(f"#### Regressão: casos {h} semana(s) à frente")
    st.dataframe(reg.style.format(precision=3, na_rep="—")
                 .highlight_max(subset=["R²", "R² estado"], color="#14532d")
                 .highlight_min(subset=["MAE", "RMSE", "MAE relativo"], color="#14532d"))
    cob = META["cobertura"][str(h)]
    st.caption(f"MAE e RMSE em casos por semana e município; MAE relativo = MAE ÷ MAE da persistência "
               f"(abaixo de 1 é melhor que repetir as últimas semanas). Cobertura das faixas: 50% → "
               f"{cob['faixa 50%']:.0%}, 90% → {cob['faixa 90%']:.0%}.")
    clf = pd.DataFrame(META["classificacao"][str(h)]).set_index("modelo").astype(float)
    st.markdown(f"#### Classificação: a semana {h} semana(s) à frente será epidêmica?")
    st.dataframe(clf.style.format(precision=3).highlight_max(color="#14532d"))
    st.caption("Precisão: fração dos alertas que se confirmaram. Sensibilidade (recall): fração das semanas "
               "epidêmicas que foram antecipadas. F1: média harmônica das duas. AP: precisão média em todos os cortes.")

# ------------------------------------------------------------------ sobre
with aba_sobre:
    st.markdown(f"""
### Como funciona

Este painel mostra os resultados do notebook **`Previsao_dengue_SP_v2.ipynb`**, que monta uma base semanal
de dengue para os {META['n_municipios']} municípios de {META['uf']} a partir das fontes primárias e treina os modelos de previsão.

| Etapa | O que é feito |
|---|---|
| **Dados** | notificações de dengue do SINAN (OpenDataSUS), população do IBGE e clima do ERA5-Land (Google Earth Engine) |
| **Indicadores** | incidência semanal, número reprodutivo **Rt** (método de Cori) e **limiares** pré-epidêmico e epidêmico por município (método MEM) |
| **Variáveis** | só casos já digitados na semana da previsão, incidência dos municípios vizinhos, Rt, sazonalidade e clima com 1 a 12 semanas de defasagem |
| **Modelos** | Gradient Boosting e Random Forest prevendo o crescimento da incidência, combinados num *ensemble*; Gradient Boosting quantílico para as faixas; regressão linear e logística como modelos interpretáveis |
| **Validação** | janelas móveis: cada ano de {META['anos_teste'][0]} a {META['anos_teste'][-1]} é previsto por modelos treinados só com os anos anteriores |

**Níveis de alerta** (pela incidência prevista): 🟢 **Baixo**, abaixo do limiar pré-epidêmico; 🟡 **Atenção**,
entre os limiares; 🔴 **Epidemia**, acima do limiar epidêmico. A **probabilidade de semana epidêmica** usa a
distribuição dos erros do modelo no período de teste.

### Limitações

- A base termina em {pd.Timestamp(META['periodo'][1]):%d/%m/%Y}; a "previsão futura" é para as semanas logo depois dessa data.
- As semanas mais recentes ainda recebem notificações digitadas com atraso, então os casos reais delas tendem a subir nas próximas atualizações.
- As previsões tendem a **subestimar epidemias fora do padrão**, como a de 2024 (reintrodução do sorotipo DENV-3).
- Casos notificados incluem casos depois descartados, e o atraso de digitação não é corrigido (*nowcasting*).
- É uma ferramenta acadêmica de apoio, não substitui a vigilância epidemiológica oficial.

### Fontes
- SINAN — Dengue: OpenDataSUS, Ministério da Saúde.
- IBGE: estimativas de população (tabela 6579), Censo 2022 (tabela 4709) e malhas municipais.
- ERA5-Land (ECMWF/Copernicus), via Google Earth Engine.
""")
