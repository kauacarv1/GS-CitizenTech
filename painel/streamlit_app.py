"""SEntinela: painel de previsão de dengue nos municípios de São Paulo.

Lê os arquivos gerados pela seção 16 do notebook `Previsao_dengue_SP_v2.ipynb` (pasta `painel/arquivos`).
Rodar localmente:  streamlit run painel/streamlit_app.py
"""
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="SEntinela | Previsão de dengue em SP", page_icon="🦟", layout="wide")

DIR_ARQ = Path(__file__).parent / "arquivos"
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
h1 {font-size: 1.9rem !important;}
.marca {font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px; line-height: 1.1;}
.marca .se {color: #3b82f6;}
.slogan {color: #9ca3af; font-size: 0.95rem; margin-top: 2px;}
.destaque {background: linear-gradient(90deg, #172554, #161a23); border: 1px solid #1e3a8a; border-radius: 14px;
           padding: 16px 20px; margin: 6px 0 14px 0;}
.destaque .grande {font-size: 2.2rem; font-weight: 800; color: #60a5fa; line-height: 1.1;}
.destaque .texto {color: #e5e7eb; font-size: 0.98rem; margin-top: 4px;}
.plano {background: #161a23; color: #e5e7eb; border-left: 4px solid; border-radius: 10px; padding: 12px 16px; height: 100%;}
.plano li {margin-bottom: 4px;}
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


PLANO_ACAO = {
    "Baixo": ["Manter a vigilância de rotina e a notificação oportuna dos casos suspeitos.",
              "Fazer o levantamento de infestação (LIRAa) e as visitas de rotina para eliminar criadouros.",
              "Conferir os estoques de soro, testes rápidos e analgésicos para a temporada.",
              "Manter a comunicação com a população sobre água parada."],
    "Atenção": ["Intensificar as visitas casa a casa nos bairros com mais casos e fazer mutirões de limpeza.",
                "Fazer o bloqueio de transmissão em volta dos casos confirmados.",
                "Avisar as UBS para aumentar a suspeição clínica e a classificação de risco.",
                "Revisar o plano de contingência e a escala das equipes para as próximas semanas.",
                "Acompanhar os municípios vizinhos e combinar ações na região de saúde."],
    "Epidemia": ["Ativar o plano de contingência e a sala de situação da dengue.",
                 "Abrir ou ampliar salas de hidratação e estender o horário das unidades de saúde.",
                 "Remanejar equipes e garantir insumos (soro, testes, leitos de observação).",
                 "Fazer campanha intensa de comunicação: sinais de alarme e onde procurar atendimento.",
                 "Coordenar com o estado e os municípios vizinhos o apoio de equipes e insumos."],
}


@st.cache_data
def calcular_antecedencia(prev, mun):
    """Para cada entrada de um município em epidemia no período de teste, com quantas semanas o alerta chegou."""
    lim = mun.set_index("m").limiar_epidemico
    base = prev[prev.h == prev.h.min()][["m", "semana_alvo", "y", "y_casos"]].copy()
    base["epidemica"] = (base.y.to_numpy() > lim.loc[base.m].to_numpy()) & (base.y_casos.to_numpy() > 10)
    ano, mes = base.semana_alvo.dt.year, base.semana_alvo.dt.month
    base["temporada"] = np.where(mes >= 10, ano, ano - 1)
    primeiro_ano = int(prev.semana_alvo.dt.year.min())
    inicios = (base[base.epidemica & (base.temporada >= primeiro_ano)].sort_values("semana_alvo")
               .groupby(["m", "temporada"]).head(1)[["m", "temporada", "semana_alvo"]])
    hs = sorted((int(x) for x in prev.h.unique()), reverse=True)
    usa_classificador = "escore_gb" in prev       # alerta = classificador treinado para o evento (corte 0,5)
    for hh in hs:
        cols = ["m", "semana_alvo", "ensemble", "persistencia"] + (["escore_gb"] if usa_classificador else [])
        p_h = prev[prev.h == hh][cols].rename(columns={"ensemble": f"prev_{hh}", "persistencia": f"pers_{hh}",
                                                       "escore_gb": f"prob_{hh}"})
        inicios = inicios.merge(p_h, on=["m", "semana_alvo"], how="left")
        lim_m = lim.loc[inicios.m].to_numpy()
        if usa_classificador:
            inicios[f"alerta_{hh}"] = inicios[f"prob_{hh}"].to_numpy() >= 0.5
        else:
            inicios[f"alerta_{hh}"] = inicios[f"prev_{hh}"].to_numpy() >= lim_m
        inicios[f"alerta_pers_{hh}"] = inicios[f"pers_{hh}"].to_numpy() >= lim_m
    inicios["semanas_antes"] = np.select([inicios[f"alerta_{hh}"] for hh in hs], hs, 0).astype(int)
    inicios["semanas_antes_pers"] = np.select([inicios[f"alerta_pers_{hh}"] for hh in hs], hs, 0).astype(int)
    inicios["rotulo"] = [f"{t}/{str(t + 1)[2:]}" for t in inicios.temporada]
    return inicios


@st.cache_data
def precisao_alertas(prev, mun):
    """Fração dos alertas (classificador >= 0,5) que se confirmaram, em todas as semanas do teste, por horizonte."""
    if "escore_gb" not in prev:
        return {}
    lim = mun.set_index("m").limiar_epidemico.loc[prev.m].to_numpy()
    ev = (prev.y.to_numpy() > lim) & (prev.y_casos.to_numpy() > 10)
    al = prev.escore_gb.to_numpy() >= 0.5
    return {int(hh): float((al & ev & (prev.h == hh)).sum() / max((al & (prev.h == hh)).sum(), 1))
            for hh in prev.h.unique()}


def normalizar(texto):
    return unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode().lower().strip()


def classe_iip(v):
    """Classificação do índice de infestação predial (LIRAa)."""
    return "Satisfatório" if v < 1 else ("Alerta" if v < 4 else "Risco")


def ler_importados(arquivo, mun):
    """Lê o CSV enviado pelo município. Devolve (tabela, mensagens)."""
    if arquivo is None:
        return None, []
    try:
        arquivo.seek(0)
        df = pd.read_csv(arquivo, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    except Exception as e:
        return None, [f"Não foi possível ler o arquivo: {e}"]
    df.columns = [normalizar(c).replace(" ", "_") for c in df.columns]
    if not {"data", "indicador", "valor"} <= set(df.columns) or not ({"codigo_ibge", "municipio"} & set(df.columns)):
        return None, ["O arquivo precisa das colunas data, indicador, valor e codigo_ibge ou municipio (veja o modelo)."]
    m = pd.Series(np.nan, index=df.index)
    if "codigo_ibge" in df:
        cod = pd.to_numeric(df.codigo_ibge.astype(str).str.replace(r"\D", "", regex=True), errors="coerce")
        m = cod.map(mun.set_index("geocodigo").m).fillna(cod.map(mun.assign(g6=mun.geocodigo // 10).set_index("g6").m))
    if "municipio" in df:
        por_nome = mun.assign(n=mun.nome.map(normalizar)).drop_duplicates("n").set_index("n").m
        m = m.fillna(df.municipio.map(normalizar).map(por_nome))
    txt = df.data.astype(str).str.strip()
    data = pd.to_datetime(txt, format="%Y-%m-%d", errors="coerce").fillna(
        pd.to_datetime(txt, format="%d/%m/%Y", errors="coerce"))
    v = df.valor.astype(str).str.strip()
    milhar = v.str.fullmatch(r"\d{1,3}(\.\d{3})+")                         # 1.234 -> mil duzentos e trinta e quatro
    v = v.where(~milhar, v.str.replace(".", "", regex=False))
    v = v.where(~v.str.contains(","), v.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    valor = pd.to_numeric(v, errors="coerce")
    indicador = df.indicador.map(normalizar).replace({"liraa": "iip", "indice de infestacao predial": "iip"})
    ok = m.notna() & data.notna() & valor.notna() & indicador.isin(["casos", "iip"])
    msgs = []
    if (~ok).any():
        msgs.append(f"{int((~ok).sum())} linha(s) ignorada(s): município, data, indicador ou valor não reconhecido.")
    out = pd.DataFrame({"m": m[ok].astype(int), "data": data[ok], "indicador": indicador[ok], "valor": valor[ok]})
    out["semana"] = out.data - pd.to_timedelta((out.data.dt.weekday + 1) % 7, unit="D")
    chaves = ["m", "semana", "indicador"]
    out = pd.concat([out[out.indicador == "casos"].groupby(chaves).valor.sum().reset_index(),      # casos: soma
                     out[out.indicador == "iip"].groupby(chaves).valor.mean().reset_index()])      # índice: média
    return out.merge(mun[["m", "nome", "geocodigo"]], on="m"), msgs


def modelo_csv(nome, codigo, ultima):
    linhas = ["municipio;codigo_ibge;data;indicador;valor"]
    for k, n in [(14, 95), (7, 120), (0, 160)]:
        linhas.append(f"{nome};{codigo};{(ultima - pd.Timedelta(days=k)):%d/%m/%Y};casos;{n}")
    linhas.append(f"{nome};{codigo};{(ultima - pd.Timedelta(days=21)):%d/%m/%Y};iip;2,4")
    return ("\n".join(linhas) + "\n").encode("utf-8-sig")


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
ANTEC = calcular_antecedencia(PREV, MUN)
IMPORT, MSGS_IMPORT = ler_importados(st.session_state.get("upload"), MUN)

# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.markdown('<div class="marca">🦟 <span class="se">SE</span>ntinela</div>'
                '<div class="slogan">E SE a dengue chegar? A gente avisa antes.</div>', unsafe_allow_html=True)
    st.caption("Previsão de dengue nos 645 municípios de São Paulo com dados do SINAN, IBGE e ERA5-Land.")
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

st.markdown('<div class="marca" style="font-size:2.4rem">🦟 <span class="se">SE</span>ntinela</div>'
            '<div class="slogan">E SE a dengue chegar? A gente avisa antes. Previsão de dengue para os municípios '
            'de São Paulo.</div>', unsafe_allow_html=True)
st.write("")
aba_mun, aba_mapa, aba_estado, aba_sim, aba_dados, aba_modelos, aba_sobre = st.tabs(
    ["📍 Município", "🗺️ Mapa", "📈 Estado", "🧮 Simulador", "📤 Seus dados", "📊 Modelos", "ℹ️ Sobre"])

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

    c_plano, c_hist = st.columns([3, 2])
    with c_plano:
        acoes = "".join(f"<li>{a}</li>" for a in PLANO_ACAO[nv])
        c_plano.markdown(f'<div class="plano" style="border-color:{CORES_NIVEL[nv]}"><b>Plano de ação: nível '
                         f'<span style="color:{CORES_NIVEL[nv]}">{nv}</span> previsto para {f_h.semana_alvo:%d/%m/%Y}</b>'
                         f'<ul style="margin-top:8px">{acoes}</ul></div>', unsafe_allow_html=True)
    with c_hist:
        hist_m = ANTEC[ANTEC.m == M_SEL]
        if len(hist_m):
            linhas = "".join(
                f"<li>Temporada {r.rotulo}: entrou em epidemia na semana de {r.semana_alvo:%d/%m/%Y}. "
                + (f"<b style='color:{VERDE}'>Alerta {r.semanas_antes} semana{'s' if r.semanas_antes > 1 else ''} antes"
                   + (f" (probabilidade de {getattr(r, f'prob_{r.semanas_antes}'):.0%})" if hasattr(r, f"prob_{r.semanas_antes}") else "")
                   + ".</b>" if r.semanas_antes else f"<b style='color:{VERMELHO}'>Sem alerta antecipado.</b>") + "</li>"
                for r in hist_m.itertuples())
        else:
            linhas = "<li>O município não passou do limiar epidêmico no período de teste.</li>"
        c_hist.markdown(f'<div class="plano" style="border-color:{AZUL}"><b>O SEntinela teria avisado?</b>'
                        f'<ul style="margin-top:8px">{linhas}</ul></div>', unsafe_allow_html=True)
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
    if IMPORT is not None:
        imp_c = IMPORT[(IMPORT.m == M_SEL) & (IMPORT.indicador == "casos") & (IMPORT.semana >= inicio)]
        if len(imp_c):
            fig.add_trace(go.Scatter(x=imp_c.semana, y=imp_c.valor, mode="markers", name="casos informados pelo município",
                                     marker=dict(size=10, color="#f97316", symbol="circle-open", line=dict(width=2)),
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
    if len(ANTEC):
        h_max = int(PREV.h.max())
        resumo = ANTEC.groupby("rotulo").agg(entradas=("m", "size"),
                                              com_h=("semanas_antes", lambda x: int((x >= h_max).sum())),
                                              algum=("semanas_antes", lambda x: int((x > 0).sum())),
                                              pers=("semanas_antes_pers", lambda x: int((x > 0).sum()))).astype(int)
        tot_e, tot_a, tot_h, tot_p = (int(resumo[c].sum()) for c in ["entradas", "algum", "com_h", "pers"])
        prec = precisao_alertas(PREV, MUN).get(h_max)
        custo = (f" Em troca, {prec:.0%} dos alertas com {h_max} semanas se confirmaram: cerca de um alarme falso para cada "
                 f"alerta certo, um custo menor que o de uma epidemia sem preparo." if prec else "")
        st.markdown(
            f'<div class="destaque"><div class="grande">{tot_a / tot_e:.0%} das epidemias avisadas antes de começar</div>'
            f'<div class="texto">De {META["anos_teste"][0]} a {META["anos_teste"][-1]}, municípios passaram do limiar '
            f'epidêmico {num(tot_e)} vezes. O alerta do SEntinela chegou antes em {num(tot_a)} delas, e com {h_max} semanas '
            f'de antecedência em {num(tot_h)} ({tot_h / tot_e:.0%}). Repetindo as últimas semanas (persistência), o aviso '
            f'chegaria antes em só {num(tot_p)} ({num(100 * tot_p / tot_e, 1)}%).{custo}</div></div>',
            unsafe_allow_html=True)
        with st.expander("Antecedência dos alertas em cada temporada"):
            st.dataframe(resumo.assign(**{f"avisados {h_max} semanas antes": lambda x: (100 * x.com_h / x.entradas).round().astype(int).astype(str) + "%",
                                          "avisados com alguma antecedência": lambda x: (100 * x.algum / x.entradas).round().astype(int).astype(str) + "%",
                                          "persistência (alguma antecedência)": lambda x: (100 * x.pers / x.entradas).round().astype(int).astype(str) + "%"})
                         .drop(columns=["com_h", "algum", "pers"])
                         .rename(columns={"entradas": "municípios que entraram em epidemia"})
                         .rename_axis("temporada"))
            st.caption("Entrada em epidemia: primeira semana da temporada (outubro a setembro) com incidência acima do "
                       "limiar epidêmico do município e mais de 10 casos. O alerta é o do classificador (Gradient Boosting "
                       "treinado para prever semana epidêmica), quando a probabilidade calculada 1 ou 4 semanas antes "
                       "passou de 50%. Os números vêm do período de teste, com modelos treinados só com os anos anteriores.")
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

# ------------------------------------------------------------------ simulador
with aba_sim:
    st.markdown("### E SE...? Simulador de cenários")
    st.markdown("Ajuste o clima e as ações de controle para ver como a previsão das próximas semanas muda, e quantos "
                "leitos a rede precisaria.")
    escopo = st.radio("Simular para", [nome_mun, "Estado inteiro"], horizontal=True)
    c1, c2, c3, c4 = st.columns(4)
    d_temp = c1.slider("Variação da temperatura mínima (°C)", -2.0, 3.0, 0.0, 0.5)
    controle = c2.slider("Redução da transmissão pelas ações de controle (%)", 0, 60, 0, 5)
    tx_int = c3.slider("Casos que precisam de internação (%)", 0.5, 10.0, 3.0, 0.5,
                       help="Valor de exemplo. Use a proporção observada no seu município.")
    dias_int = c4.slider("Dias de internação, em média", 1, 10, 4,
                         help="Valor de exemplo. Use a média observada nos seus hospitais.")
    base_sim = FUT if escopo == "Estado inteiro" else FUT[FUT.m == M_SEL]
    linhas_sim = []
    for hh, g in base_sim.groupby("h"):
        lin = META.get("linear", {}).get(str(hh))
        efeito_t = 0.0
        if lin:
            efeito_t = sum(c / e * d_temp for v, c, e in zip(lin["variaveis"], lin["coef"], lin["escala"])
                           if v.startswith("temp_min") and e > 0)
        mult = np.exp(efeito_t) * (1 - controle / 100) ** (hh / 3)
        casos_b = g.ensemble_casos.sum()
        casos_c = casos_b * mult
        linhas_sim.append({"h": int(hh), "semana": g.semana_alvo.iloc[0], "base": casos_b, "cenario": casos_c,
                           "intern": casos_c * tx_int / 100, "leitos": casos_c * tx_int / 100 * dias_int / 7})
    sim = pd.DataFrame(linhas_sim)
    itens = []
    for r in sim.itertuples():
        var = r.cenario / max(r.base, 1e-9) - 1
        itens.append((f"Casos na semana de {r.semana:%d/%m/%Y} (+{r.h} sem.)", num(r.cenario),
                      f"sem mudanças: {num(r.base)} ({var:+.0%})".replace(".", ","), AZUL))
    r4 = sim.iloc[-1]
    casas = lambda x: 1 if x < 10 else 0
    itens.append((f"Internações por semana (+{r4.h} sem.)", num(r4.intern, casas(r4.intern)),
                  f"{num(tx_int, 1)}% dos casos", AMARELO))
    itens.append((f"Leitos ocupados em média (+{r4.h} sem.)", num(r4.leitos, casas(r4.leitos)),
                  f"{dias_int} dias por internação", VERMELHO))
    cartoes(itens)
    st.write("")
    fs = go.Figure()
    fs.add_trace(go.Bar(x=[f"+{r.h} sem. ({r.semana:%d/%m})" for r in sim.itertuples()], y=sim.base,
                        name="sem mudanças", marker_color="#475569"))
    fs.add_trace(go.Bar(x=[f"+{r.h} sem. ({r.semana:%d/%m})" for r in sim.itertuples()], y=sim.cenario,
                        name="cenário", marker_color=AZUL))
    fs.update_layout(**{**LAYOUT, "xaxis": {}, "hovermode": "x"}, height=340, barmode="group",
                     title=f"Casos previstos: {escopo}", yaxis_title="casos por semana")
    grafico(fs)
    st.caption(
        "Como o cenário é calculado: o efeito da temperatura mínima vem dos coeficientes da regressão linear do "
        "notebook (temperatura das últimas 4 e de 9 a 12 semanas). A redução da transmissão age sobre o crescimento "
        "da epidemia, considerando um intervalo de cerca de 3 semanas entre infecções sucessivas. As internações e os "
        "leitos usam os valores escolhidos acima. São aproximações para comparar cenários, não previsões exatas."
        + ("" if META.get("linear") else " (Os coeficientes da regressão linear não estão nos arquivos; rode a seção "
                                         "16 do notebook de novo para ativar o efeito da temperatura.)"))

# ------------------------------------------------------------------ dados do município
with aba_dados:
    st.markdown("### Importe os dados do seu município")
    st.markdown("Envie um CSV com os casos da semana ou com o índice de infestação predial (IIP) do LIRAa. "
                "Os casos aparecem no gráfico do município ao lado da previsão, e o IIP aparece no mapa abaixo.")
    c1, c2 = st.columns([1, 2])
    c1.download_button("Baixar modelo de CSV", modelo_csv(nome_mun, int(LINHA_MUN.geocodigo), ULTIMA),
                       file_name="modelo_sentinela.csv", mime="text/csv")
    c1.markdown("Colunas: `municipio` ou `codigo_ibge`, `data` (dd/mm/aaaa), `indicador` (`casos` ou `iip`) e `valor`. "
                "Separador `;` ou `,`.")
    c2.file_uploader("Arquivo CSV", type=["csv"], key="upload")
    for msg in MSGS_IMPORT:
        st.warning(msg)
    if IMPORT is not None and len(IMPORT):
        n_mun = IMPORT.m.nunique()
        st.success(f"{len(IMPORT)} registro(s) de {n_mun} município(s) carregado(s).")
        casos_imp = IMPORT[IMPORT.indicador == "casos"]
        if len(casos_imp):
            st.markdown("**Casos informados × base oficial e previsão**")
            comp = casos_imp.merge(SERIE[["m", "semana", "casos"]], on=["m", "semana"], how="left")
            comp = comp.merge(FUT[["m", "semana_alvo", "ensemble_casos", "q05_casos", "q95_casos"]]
                              .groupby(["m", "semana_alvo"]).first().reset_index()
                              .rename(columns={"semana_alvo": "semana"}), on=["m", "semana"], how="left")
            comp["situação"] = np.select(
                [comp.q95_casos.notna() & (comp.valor > comp.q95_casos), comp.q05_casos.notna() & (comp.valor < comp.q05_casos),
                 comp.q95_casos.notna()], ["acima da faixa prevista", "abaixo da faixa prevista", "dentro da faixa prevista"], "")
            tabela = comp.assign(semana=comp.semana.dt.strftime("%d/%m/%Y"))
            for c in ["valor", "casos", "ensemble_casos"]:
                tabela[c] = tabela[c].map(lambda x: "—" if pd.isna(x) else num(x))
            tabela["situação"] = tabela["situação"].replace("", "—")
            st.dataframe(tabela[["nome", "semana", "valor", "casos", "ensemble_casos", "situação"]]
                         .rename(columns={"nome": "Município", "semana": "Semana", "valor": "Casos informados",
                                          "casos": "Casos no SINAN", "ensemble_casos": "Casos previstos",
                                          "situação": "Situação"}), hide_index=True)
            st.caption("Casos informados acima da faixa prevista indicam que a transmissão pode estar acelerando mais "
                       "rápido do que o modelo esperava.")
        iip = IMPORT[IMPORT.indicador == "iip"].sort_values("semana").groupby("m").tail(1)
        if len(iip):
            st.markdown("**Índice de infestação predial (último valor de cada município)**")
            iip = iip.assign(classe=iip.valor.map(classe_iip))
            cores_iip = {"Satisfatório": VERDE, "Alerta": AMARELO, "Risco": VERMELHO}
            fi = px.choropleth(iip, geojson=MALHA, locations="geocodigo", featureidkey="id", hover_name="nome",
                               color="classe", color_discrete_map=cores_iip, category_orders={"classe": list(cores_iip)},
                               hover_data={"valor": ":.1f", "geocodigo": False}, height=480)
            todos = px.choropleth(MUN, geojson=MALHA, locations="geocodigo", featureidkey="id").data[0]
            todos.update(marker=dict(line=dict(color="#0b0d12", width=0.3)), colorscale=[[0, "#1f2937"], [1, "#1f2937"]],
                         z=np.zeros(len(MUN)), showscale=False, hoverinfo="skip", showlegend=False)
            fi.add_trace(todos)
            fi.data = (fi.data[-1],) + fi.data[:-1]
            fi.update_geos(visible=False, bgcolor="rgba(0,0,0,0)", fitbounds="locations")
            fi.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", template="plotly_dark",
                             legend=dict(title="IIP", orientation="h", y=0.02, x=0.02))
            grafico(fi)
            st.caption("Classificação do LIRAa: menos de 1% satisfatório, de 1% a 3,9% alerta, 4% ou mais risco de surto.")
    else:
        st.info("Nenhum arquivo enviado ainda. Baixe o modelo, preencha com os dados do município e envie.")

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

O **SEntinela** mostra os resultados do notebook **`Previsao_dengue_SP_v2.ipynb`**, que monta uma base semanal
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

O **alerta antecipado** (aba Estado e quadro "O SEntinela teria avisado?") vem do classificador Gradient Boosting,
treinado para responder diretamente se a semana será epidêmica. Ele avisa mais cedo que a previsão de casos, ao custo
de mais alarmes falsos.

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
