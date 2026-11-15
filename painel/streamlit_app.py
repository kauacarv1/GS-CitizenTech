"""SEntinela: painel de previsão de dengue nos municípios de São Paulo.

Lê os arquivos gerados pela seção 16 do notebook `Previsao_dengue_SP_v2.ipynb` (pasta `painel/arquivos`).
Rodar localmente:  streamlit run painel/streamlit_app.py

Um modelo só (Random Forest sobre o crescimento da incidência), uma regra de alerta e uma probabilidade
calibrada com o histórico de acertos. Tudo o que aparece na tela sai dessas três coisas.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="SEntinela | Previsão de dengue em SP", page_icon="🦟", layout="wide")

DIR_ARQ = Path(__file__).parent / "arquivos"
MIN_CASOS = 10                 # semana epidêmica: incidência acima do limiar E mais de 10 casos (definição do notebook)
SEMANAS_INCOMPLETAS = 8        # semanas finais que ainda recebem notificações digitadas com atraso
AZUL, AZUL_50, AZUL_90 = "#3b82f6", "rgba(59,130,246,0.35)", "rgba(59,130,246,0.15)"
CINZA, VERDE, AMARELO, VERMELHO = "#e5e7eb", "#22c55e", "#eab308", "#ef4444"
CORES_NIVEL = {"Baixo": VERDE, "Atenção": AMARELO, "Epidemia": VERMELHO}
LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15), hovermode="x unified",
              separators=",.", xaxis=dict(tickformat="%m/%Y"))

st.markdown("""
<style>
.cartao {background: #161a23; color: #e5e7eb; border: 1px solid #262b36; border-radius: 14px; padding: 14px 18px; height: 100%;}
.cartao .rot {font-size: 0.80rem; color: #9ca3af; margin-bottom: 4px;}
.cartao .val {font-size: 1.65rem; font-weight: 700; line-height: 1.2;}
.cartao .sub {font-size: 0.78rem; color: #9ca3af; margin-top: 4px;}
.marca {font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px; line-height: 1.1;}
.marca .se {color: #3b82f6;}
.slogan {color: #9ca3af; font-size: 0.95rem; margin-top: 2px;}
.alerta {background: #161a23; border: 1px solid #262b36; border-left: 8px solid; border-radius: 14px;
         padding: 18px 22px; margin: 4px 0 14px 0;}
.alerta .rot {font-size: 0.85rem; color: #9ca3af;}
.alerta .nivel {font-size: 2.3rem; font-weight: 800; line-height: 1.15;}
.alerta .texto {color: #e5e7eb; font-size: 1.02rem; margin-top: 6px;}
.plano {background: #161a23; color: #e5e7eb; border-left: 4px solid; border-radius: 10px; padding: 12px 16px; height: 100%;}
.plano li {margin-bottom: 4px;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ funções
def num(x, casas=0):
    """Número no formato brasileiro."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x):
    return f"{100 * x:.0f}%"


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


def escore(prev, prev_casos, lim_epi):
    """Distância até a semana epidêmica: passa de 1 quando a previsão supera o limiar E os 10 casos."""
    return np.minimum(np.asarray(prev, float) / np.asarray(lim_epi, float), np.asarray(prev_casos, float) / MIN_CASOS)


def nivel(prev, prev_casos, lim_pre, lim_epi):
    epi = escore(prev, prev_casos, lim_epi) > 1
    return np.where(epi, "Epidemia", np.where(np.asarray(prev) >= np.asarray(lim_pre), "Atenção", "Baixo"))


def ajustar_calibracao(s, evento, n_faixas=30):
    """Frequência observada de semana epidêmica para cada valor do escore (curva crescente, PAV)."""
    ls = np.log(np.clip(s, 0.02, 50))
    bordas = np.linspace(np.log(0.02), np.log(50), n_faixas + 1)
    idx = np.clip(np.digitize(ls, bordas) - 1, 0, n_faixas - 1)
    n = np.bincount(idx, minlength=n_faixas).astype(float)
    k = np.bincount(idx, weights=evento.astype(float), minlength=n_faixas)
    blocos = []                                                  # [soma dos eventos, n, centros]
    for i in np.where(n > 0)[0]:
        blocos.append([k[i], n[i], [(bordas[i] + bordas[i + 1]) / 2]])
        while len(blocos) > 1 and blocos[-2][0] / blocos[-2][1] > blocos[-1][0] / blocos[-1][1]:
            b = blocos.pop()
            blocos[-1] = [blocos[-1][0] + b[0], blocos[-1][1] + b[1], blocos[-1][2] + b[2]]
    x = np.array([c for b in blocos for c in b[2]])
    y = np.array([b[0] / b[1] for b in blocos for _ in b[2]])
    return x, y


def prob_epidemia(s, calib):
    return np.interp(np.log(np.clip(s, 0.02, 50)), *calib)


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
    por_m = mun.set_index("m")
    fut["pop"] = por_m.loc[fut.m, "pop"].to_numpy()
    # previsão central: Random Forest, o melhor modelo no teste (arquivos antigos só têm o ensemble no futuro)
    prev["prev"] = prev["rf"]
    fut["prev"] = fut["rf"] if "rf" in fut else fut["ensemble"]
    calib = {}
    for df in (prev, fut):
        fator = df["pop"].to_numpy() / 1e5
        for c in ["y", "prev", "persistencia", "q05", "q25", "q75", "q95"]:
            if c in df:
                df[f"{c}_casos"] = df[c] * fator
        df["lim_pre"] = por_m.loc[df.m, "limiar_pre"].to_numpy()
        df["lim_epi"] = por_m.loc[df.m, "limiar_epidemico"].to_numpy()
        df["escore"] = escore(df.prev, df.prev_casos, df.lim_epi)
        df["nivel"] = nivel(df.prev, df.prev_casos, df.lim_pre, df.lim_epi)
    prev["evento"] = (prev.y > prev.lim_epi) & (prev.y_casos > MIN_CASOS)
    prev["alerta"] = prev.escore > 1
    prev["alerta_pers"] = escore(prev.persistencia, prev.persistencia_casos, prev.lim_epi) > 1
    prev["sinal"] = prev.nivel != "Baixo"                                   # Atenção ou Epidemia
    prev["sinal_pers"] = nivel(prev.persistencia, prev.persistencia_casos, prev.lim_pre, prev.lim_epi) != "Baixo"
    for hh, g in prev.groupby("h"):
        calib[int(hh)] = ajustar_calibracao(g.escore.to_numpy(), g.evento.to_numpy())
    for df in (prev, fut):
        df["chance"] = 0.0
        for hh in calib:
            sel = (df.h == hh).to_numpy()
            df.loc[sel, "chance"] = prob_epidemia(df.loc[sel, "escore"].to_numpy(), calib[hh])
    return dict(mun=mun, serie=serie, prev=prev, fut=fut, malha=malha, meta=meta), []


@st.cache_data
def desempenho(prev):
    """Acertos do alerta e do número previsto no período de teste, por antecedência."""
    out = {}
    for hh, g in prev.groupby("h"):
        erro, erro_pers = (g.prev_casos - g.y_casos).abs().mean(), (g.persistencia_casos - g.y_casos).abs().mean()
        out[int(hh)] = dict(
            precisao=g.evento[g.alerta].mean(), sensibilidade=g.alerta[g.evento].mean(),
            precisao_pers=g.evento[g.alerta_pers].mean(), sensibilidade_pers=g.alerta_pers[g.evento].mean(),
            n_eventos=int(g.evento.sum()), mae=erro, mae_pers=erro_pers,
            freq_nivel=g.groupby("nivel").evento.mean().to_dict(),
            cob90=((g.y >= g.q05) & (g.y <= g.q95)).mean())
    return out


@st.cache_data
def calcular_antecedencia(prev):
    """Para cada entrada de um município em epidemia no teste, com quantas semanas veio o sinal (Atenção ou Epidemia)."""
    base = prev[prev.h == prev.h.min()][["m", "semana_alvo", "evento"]]
    ano, mes = base.semana_alvo.dt.year, base.semana_alvo.dt.month
    base = base.assign(temporada=np.where(mes >= 10, ano, ano - 1))
    primeiro_ano = int(prev.semana_alvo.dt.year.min())
    inicios = (base[base.evento & (base.temporada >= primeiro_ano)].sort_values("semana_alvo")
               .groupby(["m", "temporada"]).head(1)[["m", "temporada", "semana_alvo"]])
    hs = sorted((int(x) for x in prev.h.unique()), reverse=True)
    for hh in hs:
        p_h = prev[prev.h == hh][["m", "semana_alvo", "sinal", "sinal_pers", "alerta"]].rename(
            columns={"sinal": f"sinal_{hh}", "sinal_pers": f"sinal_pers_{hh}", "alerta": f"epi_{hh}"})
        inicios = inicios.merge(p_h, on=["m", "semana_alvo"], how="left")
    cols = lambda pref: [inicios[f"{pref}_{hh}"].fillna(False).astype(bool) for hh in hs]
    inicios["semanas_antes"] = np.select(cols("sinal"), hs, 0).astype(int)
    inicios["semanas_antes_pers"] = np.select(cols("sinal_pers"), hs, 0).astype(int)
    inicios["ja_epidemia"] = np.any(cols("epi"), axis=0)
    inicios["rotulo"] = [f"{t}/{str(t + 1)[2:]}" for t in inicios.temporada]
    return inicios


def cartoes(itens):
    """Linha de cartões: (rótulo, valor, texto de apoio, cor do valor)."""
    for col, (rot, val, sub, cor) in zip(st.columns(len(itens)), itens):
        col.markdown(f'<div class="cartao"><div class="rot">{rot}</div><div class="val" style="color:{cor}">{val}</div>'
                     f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def grafico(fig):
    st.plotly_chart(fig, theme=None, config={"displaylogo": False, "displayModeBar": False})


def semanas_txt(n):
    return f"{n} semana" + ("s" if n > 1 else "")


PLANO_ACAO = {
    "Baixo": ["Manter a vigilância de rotina e a notificação oportuna dos casos suspeitos.",
              "Fazer o levantamento de infestação (LIRAa) e as visitas de rotina para eliminar criadouros.",
              "Conferir os estoques de soro, testes rápidos e analgésicos para a temporada."],
    "Atenção": ["Intensificar as visitas casa a casa nos bairros com mais casos e fazer mutirões de limpeza.",
                "Fazer o bloqueio de transmissão em volta dos casos confirmados.",
                "Avisar as UBS para aumentar a suspeição clínica e a classificação de risco.",
                "Revisar o plano de contingência e a escala das equipes para as próximas semanas."],
    "Epidemia": ["Ativar o plano de contingência e a sala de situação da dengue.",
                 "Abrir ou ampliar salas de hidratação e estender o horário das unidades de saúde.",
                 "Remanejar equipes e garantir insumos (soro, testes, leitos de observação).",
                 "Fazer campanha de comunicação: sinais de alarme e onde procurar atendimento."],
}

dados, faltando = carregar()
if dados is None:
    st.title("🦟 SEntinela")
    st.error("Arquivos de dados não encontrados em `painel/arquivos/`: " + ", ".join(faltando))
    st.markdown("Rode a seção **16. Exportar dados para o painel interativo** do notebook e descompacte o "
                "`painel_arquivos.zip` nesta pasta.")
    st.stop()

MUN, SERIE, PREV, FUT, MALHA, META = (dados[k] for k in ["mun", "serie", "prev", "fut", "malha", "meta"])
NOMES = MUN.sort_values("nome").nome.tolist()
ULTIMA = SERIE.semana.max()
COMPLETA_ATE = ULTIMA - pd.Timedelta(weeks=SEMANAS_INCOMPLETAS)
DESEMP = desempenho(PREV)
ANTEC = calcular_antecedencia(PREV)
ANOS = f"{META['anos_teste'][0]} a {META['anos_teste'][-1]}"

# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.markdown('<div class="marca">🦟 <span class="se">SE</span>ntinela</div>'
                '<div class="slogan">E SE a dengue chegar? A gente avisa antes.</div>', unsafe_allow_html=True)
    st.write("")
    padrao = "Campinas" if "Campinas" in NOMES else MUN.loc[MUN["pop"].idxmax(), "nome"]
    nome_mun = st.selectbox("Município", NOMES, index=NOMES.index(padrao))
    h = st.radio("Antecedência da previsão", META["horizontes"], index=len(META["horizontes"]) - 1, horizontal=True,
                 format_func=semanas_txt)
    st.divider()
    st.caption(f"Dados do SINAN, IBGE e ERA5-Land até {ULTIMA:%d/%m/%Y}, para os {META['n_municipios']} municípios "
               f"de {META['uf']}. Ferramenta acadêmica de apoio; não substitui a vigilância oficial.")

LINHA_MUN = MUN[MUN.nome == nome_mun].iloc[0]
M_SEL = int(LINHA_MUN.m)
POP_SEL = float(LINHA_MUN["pop"])
LIM_PRE_C, LIM_EPI_C = LINHA_MUN.limiar_pre * POP_SEL / 1e5, LINHA_MUN.limiar_epidemico * POP_SEL / 1e5
D = DESEMP[h]

aba_mun, aba_mapa, aba_conf = st.tabs(["📍 Município", "🗺️ Mapa", "✅ Dá para confiar?"])

# ------------------------------------------------------------------ município
with aba_mun:
    s_mun = SERIE[SERIE.m == M_SEL].sort_values("semana")
    p_mun = PREV[(PREV.m == M_SEL) & (PREV.h == h)].sort_values("semana_alvo")
    f_mun = FUT[FUT.m == M_SEL].sort_values("h")
    f_h = f_mun[f_mun.h == h].iloc[0]
    nv, cor = f_h.nivel, CORES_NIVEL[f_h.nivel]

    st.markdown(
        f'<div class="alerta" style="border-left-color:{cor}">'
        f'<div class="rot">{nome_mun} · semana de {f_h.semana_alvo:%d/%m/%Y} (daqui a {semanas_txt(h)})</div>'
        f'<div class="nivel" style="color:{cor}">{nv}</div>'
        f'<div class="texto">Previsão de <b>{num(f_h.prev_casos)} casos</b> na semana '
        f'(mais provável entre {num(f_h.q25_casos)} e {num(f_h.q75_casos)}). '
        f'Chance de semana epidêmica: <b>{pct(f_h.chance)}</b>.</div></div>', unsafe_allow_html=True)

    c_plano, c_hist = st.columns([3, 2])
    acoes = "".join(f"<li>{a}</li>" for a in PLANO_ACAO[nv])
    c_plano.markdown(f'<div class="plano" style="border-color:{cor}"><b>O que fazer no nível '
                     f'<span style="color:{cor}">{nv}</span></b><ul style="margin-top:8px">{acoes}</ul></div>',
                     unsafe_allow_html=True)
    hist_m = ANTEC[ANTEC.m == M_SEL]
    if len(hist_m):
        linhas = "".join(
            f"<li>{r.rotulo}: epidemia a partir de {r.semana_alvo:%d/%m/%Y}. "
            + (f"<b style='color:{VERDE}'>Sinal {semanas_txt(r.semanas_antes)} antes.</b>"
               if r.semanas_antes else f"<b style='color:{VERMELHO}'>Sem sinal antes.</b>") + "</li>"
            for r in hist_m.itertuples())
    else:
        linhas = f"<li>O município não entrou em epidemia de {ANOS}.</li>"
    c_hist.markdown(f'<div class="plano" style="border-color:{AZUL}"><b>O SEntinela teria avisado?</b>'
                    f'<ul style="margin-top:8px">{linhas}</ul></div>', unsafe_allow_html=True)
    st.write("")

    periodo = st.radio("Período do gráfico", ["1 ano", "2 anos", f"Desde {META['anos_teste'][0]}"], index=1,
                       horizontal=True, label_visibility="collapsed")
    inicio = {"1 ano": ULTIMA - pd.Timedelta(weeks=52), "2 anos": ULTIMA - pd.Timedelta(weeks=104)}.get(
        periodo, pd.Timestamp(f"{META['anos_teste'][0]}-01-01"))
    s_jan = s_mun[s_mun.semana >= inicio]
    p_jan = p_mun[p_mun.semana_alvo >= inicio]
    fig = go.Figure()
    if len(p_jan):
        x = pd.concat([p_jan.semana_alvo, p_jan.semana_alvo[::-1]])
        fig.add_trace(go.Scatter(x=x, y=pd.concat([p_jan.q95_casos, p_jan.q05_casos[::-1]]), fill="toself",
                                 fillcolor=AZUL_90, line=dict(width=0), name="faixa de 90%", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=p_jan.semana_alvo, y=p_jan.prev_casos, name=f"previsto {semanas_txt(h)} antes",
                                 line=dict(color=AZUL, width=2.5), hovertemplate="%{y:,.0f}"))
    completo, recente = s_jan[s_jan.semana <= COMPLETA_ATE], s_jan[s_jan.semana >= COMPLETA_ATE]
    fig.add_trace(go.Scatter(x=completo.semana, y=completo.casos, name="casos notificados",
                             line=dict(color=CINZA, width=2), hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=recente.semana, y=recente.casos, name="ainda incompleto",
                             line=dict(color=CINZA, width=2, dash="dot"), hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=f_mun.semana_alvo, y=f_mun.prev_casos, mode="markers", name="próximas semanas",
                             marker=dict(size=11, symbol="diamond", color=AZUL),
                             error_y=dict(type="data", symmetric=False, color=AZUL,
                                          array=np.clip(f_mun.q95_casos - f_mun.prev_casos, 0, None),
                                          arrayminus=np.clip(f_mun.prev_casos - f_mun.q05_casos, 0, None)),
                             hovertemplate="%{y:,.0f}"))
    fig.add_hline(y=LIM_EPI_C, line=dict(color=VERMELHO, dash="dash", width=1),
                  annotation_text="limiar epidêmico", annotation_font_color=VERMELHO)
    fig.add_hline(y=LIM_PRE_C, line=dict(color=AMARELO, dash="dash", width=1),
                  annotation_text="atenção", annotation_font_color=AMARELO, annotation_position="bottom right")
    fig.update_layout(**LAYOUT, height=430, title=f"{nome_mun}: casos por semana", yaxis_title="casos")
    grafico(fig)
    st.caption(f"A linha azul é o que o modelo previu {semanas_txt(h)} antes de cada semana, treinado só com os anos "
               "anteriores. Os losangos são a previsão para as próximas semanas. As últimas "
               f"{SEMANAS_INCOMPLETAS} semanas (pontilhado) ainda recebem notificações digitadas com atraso.")

    with st.expander("Transmissão (Rt) e clima"):
        c1, c2 = st.columns(2)
        ult = s_mun.iloc[-1]
        f2 = go.Figure(go.Scatter(x=s_jan.semana, y=s_jan.rt, line=dict(color="#a78bfa")))
        f2.add_hline(y=1, line=dict(color=VERMELHO, dash="dash"))
        f2.update_layout(**LAYOUT, height=280, title=f"Número reprodutivo (Rt): {num(ult.rt, 2)} na última semana",
                         showlegend=False)
        with c1:
            grafico(f2)
        f4 = go.Figure()
        f4.add_trace(go.Scatter(x=s_jan.semana, y=s_jan.temp_min, name="temp. mínima (°C)", line=dict(color="#f97316")))
        f4.add_trace(go.Scatter(x=s_jan.semana, y=s_jan.umid_med, name="umidade (%)", yaxis="y2",
                                line=dict(color="#38bdf8")))
        f4.update_layout(**{**LAYOUT, "legend": dict(orientation="h", y=-0.3)}, height=280, title="Clima (ERA5-Land)",
                         yaxis=dict(title="°C"), yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False))
        with c2:
            grafico(f4)
        st.caption("Rt acima de 1 indica transmissão crescendo. O clima entra no modelo com 1 a 12 semanas de defasagem.")

# ------------------------------------------------------------------ mapa
with aba_mapa:
    semanas_mapa = [pd.Timestamp(s) for s in sorted(PREV[PREV.h == h].semana_alvo.unique())
                    + sorted(FUT[FUT.h == h].semana_alvo.unique())]
    c1, c2 = st.columns([3, 1])
    sem = c1.select_slider("Semana prevista", semanas_mapa, value=semanas_mapa[-1],
                           format_func=lambda s: f"{s:%d/%m/%Y}" + (" (próxima)" if s > ULTIMA else ""))
    variavel = c2.radio("Mostrar", ["Nível previsto", "Chance de epidemia"])
    base = FUT if sem > ULTIMA else PREV
    d = base[(base.h == h) & (base.semana_alvo == sem)].merge(MUN[["m", "geocodigo", "nome"]], on="m")
    d = d.assign(chance_pct=(d.chance * 100).round(), casos_prev=d.prev_casos.round())
    opcoes = dict(geojson=MALHA, locations="geocodigo", featureidkey="id", hover_name="nome", fitbounds="locations",
                  height=540, hover_data={"geocodigo": False, "nivel": True, "casos_prev": True, "chance_pct": True},
                  labels={"nivel": "nível", "casos_prev": "casos previstos", "chance_pct": "chance de epidemia (%)"})
    if variavel == "Nível previsto":
        figm = px.choropleth(d, color="nivel", color_discrete_map=CORES_NIVEL,
                             category_orders={"nivel": list(CORES_NIVEL)}, **opcoes)
    else:
        figm = px.choropleth(d, color="chance_pct", range_color=(0, 100), color_continuous_scale="YlOrRd", **opcoes)
    figm.update_geos(visible=False, bgcolor="rgba(0,0,0,0)")
    figm.update_traces(marker_line_color="#0b0d12", marker_line_width=0.4)
    figm.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", template="plotly_dark",
                       separators=",.", legend=dict(title="", orientation="h", y=0.02, x=0.02, bgcolor="rgba(0,0,0,0.5)"))
    grafico(figm)

    contagem = d.nivel.value_counts().reindex(list(CORES_NIVEL), fill_value=0)
    cartoes([(f"Municípios no nível {nv_}", num(n_), f"semana de {sem:%d/%m/%Y}", CORES_NIVEL[nv_])
             for nv_, n_ in contagem.items()])
    st.write("")
    st.markdown("**Onde a chance de semana epidêmica é maior**")
    top = d.sort_values(["chance", "prev_casos"], ascending=False).head(10)
    st.dataframe(pd.DataFrame({"Município": top.nome, "Nível": top.nivel,
                               "Chance de epidemia": top.chance_pct.astype(int).astype(str) + "%",
                               "Casos previstos": top.casos_prev.astype(int)}), hide_index=True)

# ------------------------------------------------------------------ confiança
with aba_conf:
    st.markdown(f"#### Como o SEntinela se saiu de {ANOS}, prevendo com {semanas_txt(h)} de antecedência")
    st.caption("Cada ano foi previsto por um modelo treinado só com os anos anteriores, como aconteceria no uso real.")
    cartoes([
        ("Alertas que se confirmaram", f"{10 * D['precisao']:.0f} em cada 10",
         f"repetindo as últimas semanas: {10 * D['precisao_pers']:.0f} em 10", VERDE),
        ("Semanas epidêmicas avisadas", f"{10 * D['sensibilidade']:.0f} em cada 10",
         f"repetindo as últimas semanas: {10 * D['sensibilidade_pers']:.0f} em 10", VERDE),
        ("Erro no número de casos", f"{pct(1 - D['mae'] / D['mae_pers'])} menor",
         "do que repetir as últimas semanas", AZUL),
        ("Valor real dentro da faixa de 90%", pct(D["cob90"]), "o ideal é 90%", AZUL)])
    fq = D["freq_nivel"]
    st.markdown("**O que cada nível significou:** " + " · ".join(
        f"{e} **{n}**: semana epidêmica em {num(100 * fq.get(n, 0))} de cada 100"
        for e, n in [("🟢", "Baixo"), ("🟡", "Atenção"), ("🔴", "Epidemia")]))

    if len(ANTEC):
        h_max = int(PREV.h.max())
        resumo = ANTEC.groupby("rotulo").agg(entradas=("m", "size"),
                                              com_h=("semanas_antes", lambda x: int((x >= h_max).sum())),
                                              algum=("semanas_antes", lambda x: int((x > 0).sum())),
                                              pers=("semanas_antes_pers", lambda x: int((x > 0).sum())),
                                              epi=("ja_epidemia", "sum")).astype(int)
        r = resumo.sum()
        st.markdown(
            f"**O aviso chega antes?** De {ANOS}, municípios entraram em epidemia **{num(r.entradas)}** vezes. Em "
            f"**{pct(r.algum / r.entradas)}** delas o painel já mostrava Atenção ou Epidemia para aquela semana "
            f"({pct(r.com_h / r.entradas)} com {h_max} semanas de antecedência); repetindo as últimas semanas, seriam "
            f"{pct(r.pers / r.entradas)}. O nível Epidemia, sozinho, só antecipou {pct(r.epi / r.entradas)}: o modelo "
            "subestima a arrancada, então **Atenção é o aviso antecipado** e Epidemia confirma.")

    c1, c2 = st.columns([3, 2])
    with c1:
        tot = SERIE[SERIE.semana <= COMPLETA_ATE].groupby("semana").casos.sum()
        pe = PREV[PREV.h == h].groupby("semana_alvo").prev_casos.sum()
        tot = tot[tot.index >= pe.index.min()]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tot.index, y=tot.values, name="casos notificados", line=dict(color=CINZA, width=2),
                                 hovertemplate="%{y:,.0f}"))
        fig.add_trace(go.Scatter(x=pe.index, y=pe.values, name=f"previsto {semanas_txt(h)} antes",
                                 line=dict(color=AZUL, width=2.5), hovertemplate="%{y:,.0f}"))
        fig.update_layout(**LAYOUT, height=340, title=f"Total do estado de {META['uf']}", yaxis_title="casos por semana")
        grafico(fig)
    with c2:
        g = PREV[PREV.h == h]
        faixas = pd.cut(g.chance, [-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                        labels=["0–10%", "10–30%", "30–50%", "50–70%", "70–90%", "90–100%"])
        obs = g.groupby(faixas, observed=True).evento.mean() * 100
        fc = go.Figure(go.Bar(x=obs.index.astype(str), y=obs.values, marker_color=AZUL, hovertemplate="%{y:.0f}%"))
        fc.update_layout(**{**LAYOUT, "xaxis": dict(title="chance informada pelo painel")}, height=340,
                         title="Quando o painel diz X%, acontece em...", yaxis=dict(range=[0, 100], ticksuffix="%"),
                         showlegend=False)
        grafico(fc)

    with st.expander("Como funciona"):
        st.markdown(f"""
- **Previsão de casos:** um Random Forest prevê quanto a incidência vai crescer ou cair em relação às últimas 4 semanas,
  usando casos já digitados, municípios vizinhos, Rt, sazonalidade e clima defasado. Foi o modelo de menor erro no teste;
  a média com outros modelos (ensemble) não melhorou o resultado.
- **Nível:** 🔴 **Epidemia** quando a previsão passa do limiar epidêmico do município e de {MIN_CASOS} casos na semana;
  🟡 **Atenção** quando passa do limiar pré-epidêmico (é o aviso antecipado); 🟢 **Baixo** nos demais casos. Os limiares vêm do histórico de
  cada município até {pd.Timestamp(META['limiares_treino_ate']).year} (método MEM).
- **Chance de semana epidêmica:** frequência com que previsões parecidas viraram semana epidêmica de {ANOS}.
- **Faixa:** intervalo em que o número real deve cair em 9 de cada 10 semanas.
""")
    with st.expander("Limitações"):
        st.markdown(f"""
- O modelo tende a **subestimar epidemias fora do padrão**, como a de 2024, e o começo de cada epidemia: na semana de
  entrada, a previsão costuma ficar bem abaixo do número real.
- Em municípios pequenos, poucos casos mudam muito a incidência; por isso o alerta exige mais de {MIN_CASOS} casos.
- Casos notificados incluem casos depois descartados, e as últimas semanas ainda estão incompletas.
- Os acertos acima foram medidos no mesmo período usado para calcular a chance de semana epidêmica.
- Fontes: SINAN/OpenDataSUS (notificações), IBGE (população e malhas), ERA5-Land via Google Earth Engine (clima).
""")
    with st.expander("Comparação entre os modelos do notebook"):
        reg = pd.DataFrame(META["regressao"][str(h)]).set_index("modelo")[["MAE", "R²", "MAE relativo"]].astype(float)
        st.dataframe(reg.round(2))
        st.caption("MAE: erro médio em casos por município e semana. MAE relativo: MAE ÷ MAE da persistência "
                   "(abaixo de 1 é melhor que repetir as últimas semanas).")
        if len(ANTEC):
            st.dataframe(pd.DataFrame({
                "municípios que entraram em epidemia": resumo.entradas,
                "com sinal antes (Atenção ou Epidemia)": resumo.algum, f"com sinal {h_max} semanas antes": resumo.com_h,
                "já no nível Epidemia": resumo.epi,
                "com sinal só repetindo as últimas semanas": resumo.pers}).rename_axis("temporada"))
