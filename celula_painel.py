# SEntinela — Seção 16: exportar os dados do painel
# Cole este código na célula da seção 16 do notebook (no Colab) e execute. Não retreina modelos:
# usa os resultados já em cache e leva segundos. Depois baixe o painel_arquivos.zip e descompacte
# o conteúdo em painel/arquivos/.

import shapely
from shapely.geometry import mapping

DIR_PAINEL = DIR_DADOS.parent / "painel_arquivos"
DIR_PAINEL.mkdir(parents=True, exist_ok=True)
LIM_EPI_M = LIM.limiar_epidemico.to_numpy()
f32 = lambda x: np.asarray(x, dtype=np.float32)

# 1. Municípios (limiares, população, centroide) e malha simplificada
centro = np.array([[geoms[g].centroid.y, geoms[g].centroid.x] for g in muns.geocodigo])
pd.DataFrame({
    "m": np.arange(M, dtype=np.int16), "geocodigo": muns.geocodigo.astype(np.int32), "nome": muns.nome,
    "pop": np.round(POP[-1]).astype(np.int64),
    "limiar_pre": f32(LIM.limiar_pre), "limiar_pos": f32(LIM.limiar_pos), "limiar_epidemico": f32(LIM.limiar_epidemico),
    "limiar_proprio": LIM.limiar_proprio.to_numpy(), "lat": f32(centro[:, 0]), "lon": f32(centro[:, 1]),
}).to_parquet(DIR_PAINEL / "municipios.parquet", index=False)

feicoes = [{"type": "Feature", "id": int(g), "properties": {"nome": n},
            "geometry": mapping(shapely.set_precision(geoms[g].simplify(0.003, preserve_topology=True), 1e-4))}
           for g, n in zip(muns.geocodigo, muns.nome)]
(DIR_PAINEL / "malha.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feicoes},
                                                     separators=(",", ":")), encoding="utf-8")

# 2. Séries semanais por município: casos, incidência, Rt e clima
pd.DataFrame({
    "semana": np.repeat(semanas.values, M), "m": np.tile(np.arange(M, dtype=np.int16), T),
    "casos": CASOS.ravel().astype(np.int32), "inc": f32(INC.ravel()),
    "rt": f32(np.clip(RT, 0, 10).ravel()), "p_rt": f32(P1.ravel()),
    "temp_min": f32(CLIMA["temp_min"].ravel()), "umid_med": f32(CLIMA["umid_med"].ravel()),
}).to_parquet(DIR_PAINEL / "serie.parquet", index=False, compression="zstd")

# 3. Previsões do período de teste (incidência por 100 mil hab.)
NIVEIS_Q = [0.05, 0.25, 0.5, 0.75, 0.95]
GRADE = np.linspace(0.005, 0.995, 100)
PRINCIPAL = "RF (crescimento)"     # modelo de menor erro no teste; é a previsão que o painel mostra
MULT = {}          # erros relativos do modelo principal no teste: (real + c) / (previsto + c)


def prob_epidemia(prev, lim, mult):
    """P(incidência > limiar epidêmico) pela distribuição empírica dos erros relativos do ensemble."""
    return ((prev[:, None] + C_INC) * mult[None, :] - C_INC > lim[:, None]).mean(axis=1)


partes = []
for h in HORIZONTES:
    d = RES[h]
    MULT[h] = np.quantile((d.y.to_numpy() + C_INC) / (d[PRINCIPAL].to_numpy() + C_INC), GRADE)
    mm = d.m.to_numpy()
    partes.append(pd.DataFrame({
        "h": np.int8(h), "semana_alvo": d.semana_alvo.to_numpy(), "m": mm.astype(np.int16),
        "ano_teste": d.ano_teste.to_numpy().astype(np.int16), "pop": f32(d.pop_alvo), "y": f32(d.y),
        "ensemble": f32(d.Ensemble), "rf": f32(d["RF (crescimento)"]), "linear": f32(d["Regressão linear"]),
        "persistencia": f32(d["Persistência"]),
        **{f"q{int(q * 100):02d}": f32(d[f"q{int(q * 100):02d}"]) for q in NIVEIS_Q},
        "p_epidemia": f32(prob_epidemia(d[PRINCIPAL].to_numpy(), LIM_EPI_M[mm], MULT[h])),
        "escore_logistica": f32(CLF[h]["escores"]["Regressão logística"]),
        "escore_gb": f32(CLF[h]["escores"]["Gradient Boosting (classif.)"]),
    }))
pd.concat(partes, ignore_index=True).to_parquet(DIR_PAINEL / "previsoes_teste.parquet", index=False, compression="zstd")

# 4. Previsão a partir da última semana da base (modelos do último ano de teste)
futuro = []
for h in HORIZONTES:
    df = montar_painel(h)
    linha = df.iloc[(T - 1) * M: T * M].copy()
    alvo = semanas[-1] + pd.Timedelta(weeks=h)
    se_alvo = ano_se([alvo])[1][0]
    linha["se_sin"], linha["se_cos"] = np.sin(2 * np.pi * se_alvo / 52.18), np.cos(2 * np.pi * se_alvo / 52.18)
    ref = PAINEIS[h].semana_alvo <= TREINO_ATE
    for c in [c for c in FEATS if c.startswith(("temp", "umid"))]:
        linha[c] = linha[c].fillna(PAINEIS[h].loc[ref, c].mean())
    modelos, base = ULTIMOS_MODELOS[h][0], linha.inc_recente_media4.to_numpy()
    prev = [np.clip(modelos[n].predict(linha[FEATS]), 0, None) if "nível" in n
            else de_razao(modelos[n].predict(linha[FEATS]), base) for n in MODELOS_PONTO]
    ens = np.mean(prev, axis=0)
    rf = prev[MODELOS_PONTO.index(PRINCIPAL)]
    qmult = np.quantile(MULT[h], NIVEIS_Q)
    futuro.append(pd.DataFrame({
        "h": np.int8(h), "semana_base": semanas[-1], "semana_alvo": alvo, "m": np.arange(M, dtype=np.int16),
        "rf": f32(rf), "ensemble": f32(ens),
        **{f"q{int(q * 100):02d}": f32(np.clip((rf + C_INC) * k - C_INC, 0, None)) for q, k in zip(NIVEIS_Q, qmult)},
        "p_epidemia": f32(prob_epidemia(rf, LIM_EPI_M, MULT[h])),
    }))
pd.concat(futuro, ignore_index=True).to_parquet(DIR_PAINEL / "previsao_futura.parquet", index=False)

# 5. Métricas e informações gerais
def tabela_json(tab):
    return json.loads(tab.round(4).reset_index(names="modelo").to_json(orient="records"))


meta = {
    "gerado_em": f"{pd.Timestamp.now():%d/%m/%Y %H:%M}", "uf": UF_SIGLA,
    "periodo": [str(semanas[0].date()), str(semanas[-1].date())], "n_municipios": int(M),
    "n_casos": int(CASOS.sum()), "horizontes": HORIZONTES, "anos_teste": ANOS_TESTE,
    "limiares_treino_ate": str(TREINO_ATE.date()),
    "regressao": {str(h): tabela_json(metricas_regressao(RES[h])) for h in HORIZONTES},
    "cobertura": {str(h): {k: round(float(v), 4) for k, v in cobertura(RES[h]).items()} for h in HORIZONTES},
    "classificacao": {str(h): tabela_json(pd.DataFrame({
        nome: metricas_classif(CLF[h]["y"], s, CLF[h]["cortes"][nome]) for nome, s in CLF[h]["escores"].items()}).T)
        for h in HORIZONTES},
}
(DIR_PAINEL / "metricas.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

for f in sorted(DIR_PAINEL.iterdir()):
    print(f"{f.name:28s} {f.stat().st_size / 1e6:6.2f} MB")
arq_zip = shutil.make_archive(str(DIR_PAINEL), "zip", DIR_PAINEL)
print(f"\nArquivos do painel em {DIR_PAINEL} e compactados em {arq_zip}")
if NO_COLAB:
    try:
        from google.colab import files
        files.download(arq_zip)
    except Exception as e:
        print(f"Download automático não funcionou ({e}); baixe o .zip pela pasta do Drive.")
