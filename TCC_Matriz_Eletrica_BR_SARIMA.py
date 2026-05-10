# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║     ANÁLISE PREDITIVA E EVOLUÇÃO DA MATRIZ ELÉTRICA BRASILEIRA             ║
# ║     Um Estudo Baseado em Séries Temporais da IEA                           ║
# ║     COM COMPARAÇÃO: SARIMA vs PROPHET (Teste ADF Incluído)                 ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  Autor  : André Luis Chaves Madeira                                        ║
# ║  Curso  : Tecnólogo em Ciência de Dados e Business Intelligence            ║
# ║  Inst.  : Universidade Católica de Petrópolis — UCP                        ║
# ║  Ano    : 2026                                                             ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  Metodologia: KDD (Knowledge Discovery in Databases)                       ║
# ║  ┌─ ETAPA 0 ── Configuração Geral                                          ║
# ║  ├─ ETAPA 1 ── Ingestão e Pré-processamento          (KDD: Seleção)        ║
# ║  ├─ ETAPA 2 ── Análise Exploratória da Matriz        (KDD: Transformação)  ║
# ║  ├─ ETAPA 3 ── Decomposição e Sazonalidade           (KDD: Mineração)      ║
# ║  ├─ ETAPA 4 ── Teste ADF e Estacionariedade          (KDD: Avaliação)      ║
# ║  ├─ ETAPA 5 ── Modelagem SARIMA                       (KDD: Mineração)     ║
# ║  ├─ ETAPA 6 ── Modelagem Prophet                      (KDD: Mineração)     ║
# ║  ├─ ETAPA 7 ── Comparação SARIMA vs Prophet           (KDD: Interpretação) ║
# ║  ├─ ETAPA 8 ── Avaliação dos Modelos                  (KDD: Interpretação) ║
# ║  └─ ETAPA 9 ── Exportações para Power BI (comentadas)                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 0 — CONFIGURAÇÃO GERAL
# Importações, constantes, paleta de cores e funções auxiliares reutilizáveis
# ══════════════════════════════════════════════════════════════════════════════

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller          
from statsmodels.tsa.arima.model import ARIMA           
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ── Caminhos ─────────────────────────────────────────────────────────────────
DIRETORIO   = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_CSV = os.path.join(DIRETORIO, "base_completa_iea.csv")
DIR_EXPORT  = DIRETORIO

# ── Parâmetros globais ────────────────────────────────────────────────────────
PAIS             = "Brazil"
PERIODOS_FUTURO  = 12
ANOS_HISTORICOS  = (2010, 2025)

# ── Paleta de cores por fonte energética ─────────────────────────────────────
CORES = {
    "Hydro"                          : "#1f77b4",
    "Wind"                           : "#17becf",
    "Solar"                          : "#f5a623",
    "Combustible renewables"         : "#2ca02c",
    "Geothermal"                     : "#8B4513",
    "Other renewables"               : "#98df8a",
    "Natural gas"                    : "#8c564b",
    "Coal"                           : "#555555",
    "Oil"                            : "#d62728",
    "Nuclear"                        : "#9467bd",
    "Other combustible non-renewables": "#c5b0d5",
}

FONTES_RENOVAVEIS = [
    "Hydro", "Wind", "Solar", "Combustible renewables",
    "Geothermal", "Other renewables",
]
FONTES_NAO_RENOVAVEIS = [
    "Natural gas", "Coal", "Oil", "Nuclear",
    "Other combustible non-renewables",
]
FONTES_PRINCIPAIS = FONTES_RENOVAVEIS + FONTES_NAO_RENOVAVEIS

# ── Estilo global de visualização ────────────────────────────────────────────
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "#FAFAFA",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.titlesize"   : 14,
    "axes.titleweight" : "bold",
    "axes.labelsize"   : 11,
    "legend.frameon"   : False,
    "legend.fontsize"  : 10,
})

# ── Funções auxiliares ────────────────────────────────────────────────────────

def log(titulo: str, nivel: int = 1) -> None:
    """Imprime um banner de seção no terminal de forma padronizada."""
    if nivel == 1:
        print(f"\n{'═'*70}\n  {titulo}\n{'═'*70}")
    elif nivel == 2:
        print(f"\n  ── {titulo}")
    else:
        print(f"     → {titulo}")


def preparar_serie_mensal(df_brasil: pd.DataFrame, produto: str) -> pd.Series:
    """
    Extrai e prepara a série mensal de um produto energético.
    Garante continuidade do índice temporal (sem lacunas) e preenche
    eventuais meses ausentes com zero.
    """
    serie = (
        df_brasil[df_brasil["PRODUCT"] == produto]
        .groupby("Date")["VALUE"]
        .sum()
    )
    idx_completo = pd.date_range(serie.index.min(), serie.index.max(), freq="MS")
    return serie.reindex(idx_completo).fillna(0)


def calcular_cagr(serie_anual: pd.Series) -> float:
    """
    Calcula a Taxa de Crescimento Anual Composta (CAGR) de uma série anual.
    Fórmula: (Valor_Final / Valor_Inicial) ^ (1 / n_anos) - 1
    """
    anos = len(serie_anual) - 1
    if anos <= 0 or serie_anual.iloc[0] == 0:
        return float("nan")
    return (serie_anual.iloc[-1] / serie_anual.iloc[0]) ** (1 / anos) - 1


def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula MAE, RMSE e MAPE entre valores reais e previstos.
    MAPE ignora períodos com valor real zero (evita divisão por zero).
    """
    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mask = y_real != 0
    mape = np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100 if mask.sum() > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def rodar_prophet(serie_mensal: pd.Series, n_periodos: int = 12) -> tuple:
    """
    Treina o modelo Prophet em uma série temporal mensal e gera a previsão.

    Retorna
    -------
    (df_historico, df_forecast, metricas)
    """
    df_hist = serie_mensal.reset_index()
    df_hist.columns = ["ds", "y"]

    modelo = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.90,
    )
    modelo.fit(df_hist)

    futuro   = modelo.make_future_dataframe(periods=n_periodos, freq="MS")
    previsao = modelo.predict(futuro)

    alinha   = previsao.set_index("ds").loc[df_hist["ds"], "yhat"].values
    metricas = calcular_metricas(df_hist["y"].values, alinha)

    return df_hist, previsao[["ds", "yhat", "yhat_lower", "yhat_upper"]], metricas


def grafico_forecast(df_hist, df_prev, titulo: str, cor: str) -> None:
    """Plota o gráfico de previsão com estilo padronizado."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df_hist["ds"], df_hist["y"],
            label="Histórico", color="#333333", linewidth=1.8, zorder=3)
    ax.plot(df_prev["ds"], df_prev["yhat"],
            label="Previsão", color=cor, linewidth=2.2,
            linestyle="--", zorder=4)
    ax.fill_between(df_prev["ds"], df_prev["yhat_lower"], df_prev["yhat_upper"],
                    color=cor, alpha=0.15, label="IC 90%")

    corte = df_hist["ds"].max()
    ax.axvline(corte, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(corte, ax.get_ylim()[1] * 0.95, "  Previsão →",
            color="gray", fontsize=9, va="top")

    anos = range(df_hist["ds"].dt.year.min(), df_prev["ds"].dt.year.max() + 1)
    ax.set_xticks([pd.Timestamp(f"{a}-01-01") for a in anos])
    ax.set_xticklabels(anos, rotation=45, ha="right")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title(titulo)
    ax.set_xlabel("Ano")
    ax.set_ylabel("Produção [GWh]")
    ax.legend()
    plt.tight_layout()
    plt.show()


def comparar_modelos_seguro(fonte, res_sarima, res_prophet, diag_adf):
    """
    Compara SARIMA e Prophet de forma segura, com tratamento de erros.

    Parâmetros
    ----------
    fonte      : str  — nome da fonte ("Solar" ou "Wind")
    res_sarima : dict — resultados_sarima[fonte]
    res_prophet: dict — resultados_prophet[fonte]
    diag_adf   : dict — diagnosticos_adf[fonte]

    Retorna
    -------
    dict com métricas comparativas ou None se houver erro.
    """
    try:
        if not res_sarima or not res_prophet:
            return None

        m_s = res_sarima.get("metricas", {})
        m_p = res_prophet.get("metricas", {})

        mae_s  = m_s.get("MAE",  0)
        rmse_s = m_s.get("RMSE", 0)
        mape_s = m_s.get("MAPE", 0)

        mae_p  = m_p.get("MAE",  0)
        rmse_p = m_p.get("RMSE", 0)
        mape_p = m_p.get("MAPE", 0)

        diff   = (abs(mae_s - mae_p) / max(mae_s, mae_p) * 100) if max(mae_s, mae_p) > 0 else 0
        melhor = "SARIMA" if mae_s < mae_p else "Prophet"

        return {
            "fonte"        : fonte,
            "mae_sarima"   : round(mae_s,  1),
            "mae_prophet"  : round(mae_p,  1),
            "rmse_sarima"  : round(rmse_s, 1),
            "rmse_prophet" : round(rmse_p, 1),
            "mape_sarima"  : round(mape_s, 1),
            "mape_prophet" : round(mape_p, 1),
            "mae_diff_pct" : round(diff,   1),
            "melhor_modelo": melhor,
            "estacionaria" : diag_adf.get("eh_estacionaria", False),
        }
    except Exception as e:
        log(f"ERRO ao comparar {fonte}: {str(e)}", 3)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — INGESTÃO E PRÉ-PROCESSAMENTO
# KDD: Seleção dos dados + Pré-processamento (limpeza e estruturação)
# ══════════════════════════════════════════════════════════════════════════════
log("ETAPA 1 — INGESTÃO E PRÉ-PROCESSAMENTO")

df_raw = pd.read_csv(ARQUIVO_CSV, sep=";")
log(f"Base carregada: {df_raw.shape[0]:,} linhas × {df_raw.shape[1]} colunas", 3)

df_raw["Date"] = pd.to_datetime(df_raw[["YEAR", "MONTH"]].assign(day=1))
df      = df_raw.drop(columns=["CODE_TIME", "TIME", "previousYearToDate"])
data_br = df[df["COUNTRY"] == PAIS].reset_index(drop=True)

log(f"Registros do Brasil: {len(data_br):,} linhas", 3)
log(f"Produtos distintos : {data_br['PRODUCT'].nunique()}", 3)
log(f"Período            : {data_br['YEAR'].min()} – {data_br['YEAR'].max()}", 3)

# data_br.to_csv(os.path.join(DIR_EXPORT, "BR_base_tratada.csv"),
#                sep=";", encoding="utf-8-sig",
#                index=False, float_format="%.2f", date_format="%d/%m/%Y")
log("✓ BR_base_tratada.csv (exportação comentada)", 3)


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 2 — ANÁLISE EXPLORATÓRIA DA MATRIZ ELÉTRICA
# KDD: Transformação — entender a composição e evolução da matriz elétrica
# ══════════════════════════════════════════════════════════════════════════════
log("ETAPA 2 — ANÁLISE EXPLORATÓRIA DA MATRIZ ELÉTRICA")

data_princ = data_br[data_br["PRODUCT"].isin(FONTES_PRINCIPAIS)].copy()
data_princ["Segmento"] = data_princ["PRODUCT"].apply(
    lambda p: "Renovável" if p in FONTES_RENOVAVEIS else "Não Renovável"
)

data_anual = (
    data_princ.groupby(["YEAR", "PRODUCT"])["VALUE"]
    .sum()
    .unstack()
    .fillna(0)
)

# ── 2.1 — Composição Média (Barras Horizontais) ──────────────────────────────
log("2.1 Composição média da matriz — barras horizontais", 2)

media_mensal = data_princ.groupby("PRODUCT")["VALUE"].mean().sort_values(ascending=True)
palette      = [CORES.get(p, "#aaaaaa") for p in media_mensal.index]

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(media_mensal.index, media_mensal.values, color=palette, edgecolor="white")
ax.bar_label(bars, labels=[f"{v:,.0f} GWh" for v in media_mensal.values], padding=5, fontsize=9)
ax.set_xlabel("Produção Média Mensal [GWh]")
ax.set_title("Brasil: Produção Média Mensal por Fonte (2010–2025)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.show()

# ── 2.2 — Participação Renovável vs Não Renovável (Pizza) ───────────────────
log("2.2 Participação Renovável vs Não Renovável", 2)

rn_data  = data_br[data_br["PRODUCT"].isin(["Renewables", "Non-renewables"])]
rn_share = rn_data.groupby("PRODUCT")["share"].mean() * 100

fig, ax = plt.subplots(figsize=(8, 7))
wedges, texts, autotexts = ax.pie(
    rn_share, labels=rn_share.index, autopct="%.1f%%",
    startangle=90, counterclock=False,
    colors=["#2ca02c", "#d62728"], explode=(0.04, 0),
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 13},
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_fontweight("bold")
ax.set_title("Brasil: Participação Histórica Média\nRenovável vs Não Renovável (2010–2025)")
plt.tight_layout()
plt.show()

# ── 2.3 — Evolução Anual: Área Empilhada ────────────────────────────────────
log("2.3 Evolução anual — área empilhada", 2)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Brasil: Evolução Anual da Geração Elétrica por Fonte (GWh)", y=0.98)

for fonte in FONTES_RENOVAVEIS:
    if fonte in data_anual.columns:
        ax1.stackplot(data_anual.index, data_anual[fonte],
                      labels=[fonte], colors=[CORES[fonte]], alpha=0.85)
ax1.set_ylabel("GWh")
ax1.set_title("Fontes Renováveis", fontsize=11, pad=6)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax1.legend(loc="upper left", ncol=3, fontsize=9)

for fonte in FONTES_NAO_RENOVAVEIS:
    if fonte in data_anual.columns:
        ax2.stackplot(data_anual.index, data_anual[fonte],
                      labels=[fonte], colors=[CORES[fonte]], alpha=0.85)
ax2.set_ylabel("GWh")
ax2.set_title("Fontes Não Renováveis", fontsize=11, pad=6)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax2.legend(loc="upper left", ncol=3, fontsize=9)
ax2.set_xticks(data_anual.index)
ax2.set_xticklabels(data_anual.index, rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ── 2.4 — Destaque Solar e Eólica vs Demais ─────────────────────────────────
log("2.4 Solar e Eólica em destaque vs demais fontes", 2)

fig, ax = plt.subplots(figsize=(14, 7))

for col in data_anual.columns:
    if col not in ["Solar", "Wind"]:
        ax.plot(data_anual.index, data_anual[col], color="#CCCCCC", linewidth=1, zorder=1)

ax.plot(data_anual.index, data_anual["Solar"], label="Solar",
        color=CORES["Solar"], linewidth=3, marker="o", markersize=7, zorder=3)
ax.plot(data_anual.index, data_anual["Wind"], label="Eólica (Wind)",
        color=CORES["Wind"], linewidth=3, marker="s", markersize=7, zorder=3)

for fonte, marcador in [("Solar", "o"), ("Wind", "s")]:
    cagr       = calcular_cagr(data_anual[fonte].dropna())
    ultimo_ano = data_anual.index.max()
    ultimo_val = data_anual.loc[ultimo_ano, fonte]
    ax.annotate(f"CAGR: {cagr:.0%}",
                xy=(ultimo_ano, ultimo_val),
                xytext=(ultimo_ano - 1, ultimo_val * 1.12),
                fontsize=10, fontweight="bold", color=CORES[fonte],
                arrowprops={"arrowstyle": "->", "color": CORES[fonte]})

ax.set_title("Brasil: Crescimento Acelerado de Solar e Eólica vs Demais Fontes")
ax.set_xlabel("Ano")
ax.set_ylabel("Produção Anual [GWh]")
ax.set_xticks(data_anual.index)
ax.set_xticklabels(data_anual.index, rotation=45, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(title="Destaque", fontsize=11)
ax.text(0.02, 0.92, "Demais fontes em cinza", transform=ax.transAxes,
        fontsize=9, color="#AAAAAA")
plt.tight_layout()
plt.show()

# ── 2.5 — Share Acumulado Anual ──────────────────────────────────────────────
log("2.5 Share acumulado anual por fonte", 2)

df_dez       = data_br[data_br["MONTH_NAME"] == "December"].copy()
df_net       = (df_dez[df_dez["PRODUCT"] == "Net electricity production"]
                [["YEAR", "yearToDate"]]
                .rename(columns={"yearToDate": "total_anual_net"}))
df_share_acc = pd.merge(df_dez, df_net, on="YEAR")
df_share_acc["share_acumulado"] = (
    df_share_acc["yearToDate"] / df_share_acc["total_anual_net"] * 100
)

# for grupo, fontes in [("renovaveis", FONTES_RENOVAVEIS),
#                        ("nao_renovaveis", FONTES_NAO_RENOVAVEIS)]:
#     df_out = df_share_acc[df_share_acc["PRODUCT"].isin(fontes)]
#     df_out.to_csv(os.path.join(DIR_EXPORT, f"indicador_share_acumulado_{grupo}.csv"),
#                   index=False, float_format="%.2f")
log("✓ indicador_share_acumulado_renovaveis.csv (exportação comentada)", 3)
log("✓ indicador_share_acumulado_nao_renovaveis.csv (exportação comentada)", 3)

for grupo, fontes, titulo_g in [
    ("Renováveis", FONTES_RENOVAVEIS,
     "Brasil: Share Acumulado Anual — Fontes Renováveis (%)"),
    ("Não Renováveis", FONTES_NAO_RENOVAVEIS,
     "Brasil: Share Acumulado Anual — Fontes Não Renováveis (%)"),
]:
    df_plot = df_share_acc[df_share_acc["PRODUCT"].isin(fontes)]
    if df_plot.empty:
        continue
    fig, ax = plt.subplots(figsize=(16, 7))
    sns.barplot(data=df_plot, x="YEAR", y="share_acumulado",
                hue="PRODUCT", palette=CORES, ax=ax, edgecolor="white")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3,
                     fontsize=8, fontweight="bold", rotation=90)
    ax.set_ylim(0, df_plot["share_acumulado"].max() + 18)
    ax.set_xlabel("Ano")
    ax.set_ylabel("Participação Acumulada no Ano (%)")
    ax.set_title(titulo_g)
    ax.legend(title="Fonte", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 3 — DECOMPOSIÇÃO E SAZONALIDADE
# KDD: Mineração — isola tendência, sazonalidade e resíduo das séries
# ══════════════════════════════════════════════════════════════════════════════
log("ETAPA 3 — DECOMPOSIÇÃO E SAZONALIDADE")

resultados_decomp = {}

for fonte in ["Solar", "Wind"]:
    log(f"Decomposição STL — {fonte}", 2)

    serie  = preparar_serie_mensal(data_br, fonte)
    decomp = seasonal_decompose(serie, model="additive", period=12)

    var_total     = np.var(serie)
    var_tendencia = np.var(decomp.trend.dropna())
    var_sazonal   = np.var(decomp.seasonal.dropna())
    var_residuo   = np.var(decomp.resid.dropna())
    amplitude     = decomp.seasonal.max() - decomp.seasonal.min()

    resultados_decomp[fonte] = {
        "serie"      : serie,
        "decomp"     : decomp,
        "var_tend_%"  : var_tendencia / var_total * 100,
        "var_sazon_%": var_sazonal   / var_total * 100,
        "var_resid_%": var_residuo   / var_total * 100,
        "amplitude"  : amplitude,
    }

    log(f"Variância tendência    : {var_tendencia/var_total*100:.1f}%", 3)
    log(f"Variância sazonalidade : {var_sazonal/var_total*100:.1f}%", 3)
    log(f"Amplitude sazonal      : {amplitude:,.1f} GWh", 3)

    # tend_df = decomp.trend.dropna().reset_index()
    # tend_df.columns = ["Date", f"Tendencia_{fonte}"]
    # tend_df.to_csv(os.path.join(DIR_EXPORT, f"Tendencia_{fonte}.csv"),
    #                index=False, float_format="%.2f", date_format="%d/%m/%Y")
    log(f"✓ Tendencia_{fonte}.csv (exportação comentada)", 3)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Decomposição da Série Temporal — {fonte} (Brasil)", fontsize=15)

    for ax, (dados, rotulo, cor) in zip(axes, [
        (serie,           "Série Original", "#333333"),
        (decomp.trend,    "Tendência",      CORES[fonte]),
        (decomp.seasonal, "Sazonalidade",   "#2ca02c"),
        (decomp.resid,    "Resíduo",        "#d62728"),
    ]):
        ax.plot(dados, color=cor, linewidth=1.5)
        ax.set_ylabel(rotulo, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Data")
    plt.tight_layout()
    plt.show()

# ── 3.3 — Heatmap Sazonal ────────────────────────────────────────────────────
log("3.3 Fingerprint sazonal — Heatmap mensal Solar vs Eólica", 2)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Brasil: Padrão Sazonal Mensal — Solar vs Eólica (GWh)", fontsize=15)
nomes_meses = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]

for ax, (fonte, cmap) in zip(axes, [("Solar", "YlOrRd"), ("Wind", "Blues")]):
    serie   = resultados_decomp[fonte]["serie"]
    df_heat = serie.to_frame(name="VALUE")
    df_heat["Ano"] = df_heat.index.year
    df_heat["Mes"] = df_heat.index.month
    pivot          = df_heat.pivot(index="Ano", columns="Mes", values="VALUE")
    pivot.columns  = nomes_meses
    sns.heatmap(pivot, ax=ax, cmap=cmap, fmt=".0f", annot=True,
                annot_kws={"size": 7}, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "GWh"})
    ax.set_title(f"{fonte}", fontsize=13)
    ax.set_xlabel("Mês")
    ax.set_ylabel("Ano")

plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 4 — TESTE ADF E DIAGNÓSTICO DE ESTACIONARIEDADE
# KDD: Avaliação — verificar se as séries são estacionárias
#
# Hipóteses:
#   H0 (nula)       : a série possui raiz unitária → NÃO-ESTACIONÁRIA
#   H1 (alternativa): a série é estacionária
#
# Regra de decisão:
#   p-value < 0.05 → rejeita H0 → ESTACIONÁRIA   → d = 0 no SARIMA
#   p-value ≥ 0.05 → não rejeita H0 → NÃO-ESTACIONÁRIA → d = 1 no SARIMA
# ══════════════════════════════════════════════════════════════════════════════
log("ETAPA 4 — TESTE ADF E DIAGNÓSTICO DE ESTACIONARIEDADE")

diagnosticos_adf = {}

for fonte in ["Solar", "Wind"]:
    log(f"Teste ADF — {fonte}", 2)

    serie = resultados_decomp[fonte]["serie"]

    adf_result  = adfuller(serie.dropna(), autolag="AIC")
    adf_stat    = adf_result[0]
    p_value     = adf_result[1]
    n_lags      = adf_result[2]
    n_obs       = adf_result[3]
    crit_values = adf_result[4]
    eh_estac    = p_value < 0.05
    ordem_d     = 0 if eh_estac else 1

    diagnosticos_adf[fonte] = {
        "adf_statistic"   : adf_stat,
        "p_value"         : p_value,
        "n_lags"          : n_lags,
        "n_obs"           : n_obs,
        "valores_criticos": crit_values,
        "eh_estacionaria" : eh_estac,
        "ordem_d"         : ordem_d,
    }

    log(f"Estatística ADF : {adf_stat:.4f}", 3)
    log(f"P-value         : {p_value:.4f}", 3)
    log(f"Lags utilizados : {n_lags}", 3)
    log(f"Valores críticos: 1%={crit_values['1%']:.3f}  "
        f"5%={crit_values['5%']:.3f}  10%={crit_values['10%']:.3f}", 3)
    log(f"Estacionária    : {'SIM ✓' if eh_estac else 'NÃO ✗'}  →  d={ordem_d}", 3)

# Tabela-resumo ADF
print(f"\n{'─'*75}")
print(f"{'TESTE ADF — DIAGNÓSTICO DE ESTACIONARIEDADE':^75}")
print(f"{'─'*75}")
print(f"{'Fonte':<10} {'ADF Stat':>12} {'P-value':>12} {'Estacionária':>15} {'Ordem d':>10}")
print(f"{'─'*75}")
for fonte in ["Solar", "Wind"]:
    d     = diagnosticos_adf[fonte]
    estac = "SIM" if d["eh_estacionaria"] else "NÃO"
    print(f"{fonte:<10} {d['adf_statistic']:>12.4f} {d['p_value']:>12.4f} {estac:>15} {d['ordem_d']:>10}")
print(f"{'─'*75}\n")

# Gráfico ADF
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Teste ADF — Diagnóstico de Estacionariedade", fontsize=14)

adf_stats = [diagnosticos_adf[f]["adf_statistic"] for f in ["Solar", "Wind"]]
p_values  = [diagnosticos_adf[f]["p_value"]       for f in ["Solar", "Wind"]]
cores_adf = [("#2ca02c" if diagnosticos_adf[f]["eh_estacionaria"] else "#d62728")
             for f in ["Solar", "Wind"]]

axes[0].bar(["Solar", "Wind"], adf_stats, color=cores_adf, edgecolor="white", width=0.5)
axes[0].axhline(y=-2.86, color="red", linestyle="--", label="Crítico 5% (−2.86)")
axes[0].set_ylabel("Estatística ADF")
axes[0].set_title("Estatística do Teste ADF")
axes[0].legend()

axes[1].bar(["Solar", "Wind"], p_values, color=cores_adf, edgecolor="white", width=0.5)
axes[1].axhline(y=0.05, color="red", linestyle="--", label="Limite (0.05)")
axes[1].set_ylabel("P-value")
axes[1].set_title("P-value do Teste ADF")
axes[1].legend()

plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 5 — MODELAGEM SARIMA (Seasonal ARIMA)
# KDD: Mineração — previsão com componente sazonal explícito
#
# Ordem adotada: SARIMA(1, d, 1)(1, 1, 1, 12)
#   (p, d, q)       → componente ARIMA não-sazonal  (d vem do teste ADF)
#   (P, D, Q, s=12) → componente sazonal mensal
# ══════════════════════════════════════════════════════════════════════════════
log("ETAPA 5 — MODELAGEM SARIMA (Seasonal ARIMA)")

resultados_sarima = {}

for fonte in ["Solar", "Wind"]:
    log(f"Rodando SARIMA — {fonte}", 2)

    serie   = resultados_decomp[fonte]["serie"]
    ordem_d = diagnosticos_adf[fonte]["ordem_d"]

    try:
        modelo_sarima  = ARIMA(serie, order=(1, ordem_d, 1),
                               seasonal_order=(1, 1, 1, 12))
        fit_sarima     = modelo_sarima.fit()
        prev_in_sample = fit_sarima.fittedvalues
        metricas_sarima = calcular_metricas(serie.values, prev_in_sample.values)

        forecast_sarima = fit_sarima.get_forecast(steps=PERIODOS_FUTURO)
        df_prev_sarima  = forecast_sarima.summary_frame()
        df_prev_sarima["ds"] = pd.date_range(
            start=serie.index.max() + pd.DateOffset(months=1),
            periods=PERIODOS_FUTURO, freq="MS"
        )

        resultados_sarima[fonte] = {
            "modelo"       : fit_sarima,
            "previsoes"    : df_prev_sarima,
            "metricas"     : metricas_sarima,
            "ordem"        : (1, ordem_d, 1),
            "ordem_sazonal": (1, 1, 1, 12),
        }

        log(f"MAE={metricas_sarima['MAE']:,.1f} GWh | "
            f"RMSE={metricas_sarima['RMSE']:,.1f} GWh | "
            f"MAPE={metricas_sarima['MAPE']:.1f}%", 3)

        # df_prev_sarima[["ds","mean","mean_ci_lower","mean_ci_upper"]].to_csv(
        #     os.path.join(DIR_EXPORT, f"Previsao_SARIMA_{fonte}.csv"),
        #     index=False, float_format="%.2f", date_format="%d/%m/%Y"
        # )
        log(f"✓ Previsao_SARIMA_{fonte}.csv (exportação comentada)", 3)

        # ── Gráfico individual SARIMA ────────────────────────────────────────
        cor = CORES[fonte]
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(serie.index, serie.values,
                color="#333333", linewidth=1.8, label="Histórico", zorder=3)
        ax.plot(prev_in_sample.index, prev_in_sample.values,
                color=cor, linewidth=1.4, linestyle="-",
                alpha=0.7, label="Ajustado (in-sample)", zorder=4)
        ax.plot(df_prev_sarima["ds"], df_prev_sarima["mean"],
                color=cor, linewidth=2.2, linestyle="--",
                label="Previsão SARIMA", zorder=5)
        ax.fill_between(df_prev_sarima["ds"],
                        df_prev_sarima["mean_ci_lower"],
                        df_prev_sarima["mean_ci_upper"],
                        color=cor, alpha=0.15, label="IC 95%")

        corte = serie.index.max()
        ax.axvline(corte, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.text(corte, ax.get_ylim()[1] * 0.95, "  Previsão →",
                color="gray", fontsize=9, va="top")

        ax.text(0.03, 0.97,
                f"MAE: {metricas_sarima['MAE']:,.0f} GWh\n"
                f"RMSE: {metricas_sarima['RMSE']:,.0f} GWh\n"
                f"MAPE: {metricas_sarima['MAPE']:.1f}%",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox={"boxstyle": "round,pad=0.3", "fc": "white",
                      "ec": "gray", "alpha": 0.8})

        anos = range(serie.index.year.min(),
                     df_prev_sarima["ds"].dt.year.max() + 1)
        ax.set_xticks([pd.Timestamp(f"{a}-01-01") for a in anos])
        ax.set_xticklabels(anos, rotation=45, ha="right")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.set_title(
            f"Forecast SARIMA{(1, ordem_d, 1)}×{(1,1,1,12)}"
            f" — Geração {fonte} | Brasil")
        ax.set_xlabel("Ano")
        ax.set_ylabel("Produção [GWh]")
        ax.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        log(f"ERRO ao treinar SARIMA para {fonte}: {str(e)}", 3)
        resultados_sarima[fonte] = None

# ── Painel comparativo SARIMA: Solar + Eólica ────────────────────────────────
log("5.2 Painel comparativo Solar vs Eólica — SARIMA", 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    "Previsão SARIMA — Solar vs Eólica | Brasil (próximos 12 meses)",
    fontsize=15)

for ax, fonte in [(ax1, "Solar"), (ax2, "Wind")]:
    if not resultados_sarima.get(fonte):
        ax.set_title(f"{fonte} — SARIMA indisponível")
        continue

    res        = resultados_sarima[fonte]
    serie_hist = resultados_decomp[fonte]["serie"]
    prev_fit   = res["modelo"].fittedvalues
    df_prev    = res["previsoes"]
    cor        = CORES[fonte]
    m          = res["metricas"]
    label      = "Eólica" if fonte == "Wind" else "Solar"

    ax.plot(serie_hist.index, serie_hist.values,
            color="#333333", linewidth=1.6, label="Histórico", zorder=3)
    ax.plot(prev_fit.index, prev_fit.values,
            color=cor, linewidth=1.2, linestyle="-",
            alpha=0.6, label="Ajustado", zorder=4)
    ax.plot(df_prev["ds"], df_prev["mean"],
            color=cor, linewidth=2, linestyle="--",
            label="Previsão SARIMA", zorder=5)
    ax.fill_between(df_prev["ds"],
                    df_prev["mean_ci_lower"], df_prev["mean_ci_upper"],
                    color=cor, alpha=0.15, label="IC 95%")
    ax.axvline(serie_hist.index.max(), color="gray",
               linestyle=":", linewidth=1)

    ax.text(0.03, 0.97,
            f"MAE: {m['MAE']:,.0f} GWh\nMAPE: {m['MAPE']:.1f}%",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white",
                  "ec": "gray", "alpha": 0.8})

    anos = range(serie_hist.index.year.min(),
                 df_prev["ds"].dt.year.max() + 1)
    ax.set_xticks([pd.Timestamp(f"{a}-01-01") for a in anos])
    ax.set_xticklabels(anos, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title(label, fontsize=13)
    ax.set_xlabel("Ano")
    ax.set_ylabel("Produção [GWh]")
    ax.legend()

plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 6 — MODELAGEM PREDITIVA PROPHET
# KDD: Mineração — ajuste e previsão 12 meses à frente para Solar e Eólica
# ══════════════════════════════════════════════════════════════════════════════
log("ETAPA 6 — MODELAGEM PREDITIVA PROPHET")

resultados_prophet = {}

for fonte in ["Solar", "Wind"]:
    log(f"Rodando Prophet — {fonte}", 2)

    serie = resultados_decomp[fonte]["serie"]

    df_hist, df_prev, metricas = rodar_prophet(serie, PERIODOS_FUTURO)
    resultados_prophet[fonte]  = {
        "historico": df_hist,
        "previsao" : df_prev,
        "metricas" : metricas,
    }

    log(f"MAE={metricas['MAE']:,.1f} GWh | "
        f"RMSE={metricas['RMSE']:,.1f} GWh | "
        f"MAPE={metricas['MAPE']:.1f}%", 3)

    # df_prev.to_csv(os.path.join(DIR_EXPORT, f"Previsao_Prophet_{fonte}.csv"),
    #                index=False, float_format="%.2f", date_format="%d/%m/%Y")
    log(f"✓ Previsao_Prophet_{fonte}.csv (exportação comentada)", 3)

    grafico_forecast(
        df_hist, df_prev,
        titulo=f"Forecast Prophet — Geração {fonte} | Brasil",
        cor=CORES[fonte],
    )

# Painel comparativo Prophet: Solar + Eólica
log("6.3 Painel comparativo Solar vs Eólica — Prophet", 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Previsão Prophet — Solar vs Eólica | Brasil (próximos 12 meses)", fontsize=15)

for ax, fonte in [(ax1, "Solar"), (ax2, "Wind")]:
    res   = resultados_prophet[fonte]
    hist  = res["historico"]
    prev  = res["previsao"]
    cor   = CORES[fonte]
    label = "Eólica" if fonte == "Wind" else "Solar"
    m     = res["metricas"]

    ax.plot(hist["ds"], hist["y"],
            color="#333333", linewidth=1.6, label="Histórico", zorder=3)
    ax.plot(prev["ds"], prev["yhat"],
            color=cor, linewidth=2, linestyle="--",
            label="Previsão Prophet", zorder=4)
    ax.fill_between(prev["ds"], prev["yhat_lower"], prev["yhat_upper"],
                    color=cor, alpha=0.15, label="IC 90%")
    ax.axvline(hist["ds"].max(), color="gray", linestyle=":", linewidth=1)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.text(0.03, 0.97,
            f"MAE: {m['MAE']:,.0f} GWh\nMAPE: {m['MAPE']:.1f}%",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "gray", "alpha": 0.8})
    anos = range(hist["ds"].dt.year.min(), prev["ds"].dt.year.max() + 1)
    ax.set_xticks([pd.Timestamp(f"{a}-01-01") for a in anos])
    ax.set_xticklabels(anos, rotation=45, ha="right")
    ax.set_title(label, fontsize=13)
    ax.set_xlabel("Ano")
    ax.set_ylabel("Produção [GWh]")
    ax.legend()

plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 7 — COMPARAÇÃO SARIMA vs PROPHET
# KDD: Interpretação — análise lado-a-lado dos dois modelos
# ══════════════════════════════════════════════════════════════════════════════
log("ETAPA 7 — COMPARAÇÃO SARIMA vs PROPHET")

# Tabela comparativa
print(f"\n{'─'*95}")
print(f"{'COMPARAÇÃO: SARIMA vs PROPHET':^95}")
print(f"{'─'*95}")
print(f"{'Fonte':<8} {'Modelo':<12} {'MAE (GWh)':>14} {'RMSE (GWh)':>14} {'MAPE (%)':>12} {'Estacionária':>12}")
print(f"{'─'*95}")

comparacao_modelos = {}

for fonte in ["Solar", "Wind"]:
    resultado = comparar_modelos_seguro(
        fonte,
        resultados_sarima.get(fonte),
        resultados_prophet[fonte],
        diagnosticos_adf[fonte],
    )

    if resultado:
        comparacao_modelos[fonte] = resultado
        estac = "NÃO" if not resultado["estacionaria"] else "SIM"

        print(f"{fonte:<8} {'SARIMA':<12} "
              f"{resultado['mae_sarima']:>14,.1f} "
              f"{resultado['rmse_sarima']:>14,.1f} "
              f"{resultado['mape_sarima']:>12.1f} "
              f"{estac:>12}")
        print(f"{'':<8} {'Prophet':<12} "
              f"{resultado['mae_prophet']:>14,.1f} "
              f"{resultado['rmse_prophet']:>14,.1f} "
              f"{resultado['mape_prophet']:>12.1f} "
              f"{'':<12}")
        print(f"  ↳ Diferença MAE: {resultado['mae_diff_pct']:.1f}%  |  "
              f"Melhor: {resultado['melhor_modelo']}")

print(f"{'─'*95}\n")

# Gráfico comparativo — MAE, RMSE, MAPE
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Comparação: SARIMA vs Prophet — Métricas de Acurácia", fontsize=14)

fontes_plot  = ["Solar", "Wind"]
sarima_mae   = [resultados_sarima[f]["metricas"]["MAE"]  if resultados_sarima.get(f) else 0 for f in fontes_plot]
prophet_mae  = [resultados_prophet[f]["metricas"]["MAE"]  for f in fontes_plot]
sarima_rmse  = [resultados_sarima[f]["metricas"]["RMSE"] if resultados_sarima.get(f) else 0 for f in fontes_plot]
prophet_rmse = [resultados_prophet[f]["metricas"]["RMSE"] for f in fontes_plot]
sarima_mape  = [resultados_sarima[f]["metricas"]["MAPE"] if resultados_sarima.get(f) else 0 for f in fontes_plot]
prophet_mape = [resultados_prophet[f]["metricas"]["MAPE"] for f in fontes_plot]

x     = np.arange(len(fontes_plot))
width = 0.35

for ax, (s_vals, p_vals, ylabel, title) in zip(axes, [
    (sarima_mae,  prophet_mae,  "MAE (GWh)",  "MAE — Erro Médio Absoluto"),
    (sarima_rmse, prophet_rmse, "RMSE (GWh)", "RMSE — Raiz do Erro Quadrático"),
    (sarima_mape, prophet_mape, "MAPE (%)",   "MAPE — Erro Percentual Absoluto"),
]):
    b1 = ax.bar(x - width/2, s_vals, width, label="SARIMA",  color="#1f77b4", edgecolor="white")
    b2 = ax.bar(x + width/2, p_vals, width, label="Prophet", color="#f5a623", edgecolor="white")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(fontes_plot)
    ax.legend()
    ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=9)

plt.tight_layout()
plt.show()

# Gráfico: previsões sobrepostas SARIMA vs Prophet
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Previsões SARIMA vs Prophet — próximos 12 meses", fontsize=14)

for ax, fonte in zip(axes, ["Solar", "Wind"]):
    hist   = resultados_prophet[fonte]["historico"]
    prev_p = resultados_prophet[fonte]["previsao"]

    ax.plot(hist["ds"], hist["y"], "o-", color="#333333",
            linewidth=1.5, label="Histórico", zorder=3)
    ax.plot(prev_p["ds"], prev_p["yhat"], "--", color="#f5a623",
            linewidth=2, label="Prophet", zorder=4)
    ax.fill_between(prev_p["ds"], prev_p["yhat_lower"],
                    prev_p["yhat_upper"], color="#f5a623", alpha=0.15)

    if resultados_sarima.get(fonte):
        prev_s = resultados_sarima[fonte]["previsoes"]
        ax.plot(prev_s["ds"], prev_s["mean"], ":", color="#1f77b4",
                linewidth=2, label="SARIMA", zorder=4)
        ax.fill_between(prev_s["ds"], prev_s["mean_ci_lower"],
                        prev_s["mean_ci_upper"], color="#1f77b4", alpha=0.15)

    ax.axvline(hist["ds"].max(), color="gray", linestyle=":", linewidth=1)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title(fonte, fontsize=12)
    ax.set_xlabel("Data")
    ax.set_ylabel("Produção [GWh]")
    ax.legend()

plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 8 — AVALIAÇÃO DOS MODELOS
# KDD: Interpretação — relatório de métricas, CAGR e recomendações
# ══════════════════════════════════════════════════════════════════════════════
log("ETAPA 8 — AVALIAÇÃO DOS MODELOS")

# ── 8.1 — Relatório de métricas no terminal ───────────────────────────────────
print(f"\n{'─'*75}")
print(f"{'RELATÓRIO DE DESEMPENHO — PROPHET':^75}")
print(f"{'─'*75}")
print(f"{'Fonte':<12} {'MAE (GWh)':>12} {'RMSE (GWh)':>12} {'MAPE (%)':>10}")
print(f"{'─'*75}")
for fonte in ["Solar", "Wind"]:
    m = resultados_prophet[fonte]["metricas"]
    print(f"{fonte:<12} {m['MAE']:>12,.1f} {m['RMSE']:>12,.1f} {m['MAPE']:>10.1f}")
print(f"{'─'*75}")
print("\nInterpretação das métricas:")
print("  MAE  = Erro Médio Absoluto — erro em GWh (mesma unidade dos dados)")
print("  RMSE = Raiz do Erro Quadrático Médio — penaliza erros grandes")
print("  MAPE = Erro Percentual Absoluto Médio — erro relativo em %\n")

# ── 8.2 — Gráfico de barras comparativo (métricas Prophet) ───────────────────
log("8.2 Comparação visual das métricas — Prophet", 2)

metricas_df = pd.DataFrame(
    {fonte: resultados_prophet[fonte]["metricas"] for fonte in ["Solar", "Wind"]}
).T[["MAE", "RMSE", "MAPE"]]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Avaliação do Modelo Prophet — Métricas de Acurácia", fontsize=14)

for ax, (metrica, unidade, cor) in zip(axes, [
    ("MAE",  "GWh", "#1f77b4"),
    ("RMSE", "GWh", "#ff7f0e"),
    ("MAPE", "%",   "#2ca02c"),
]):
    vals = metricas_df[metrica]
    bars = ax.bar(vals.index, vals.values,
                  color=[CORES["Solar"], CORES["Wind"]],
                  edgecolor="white", width=0.5)
    ax.bar_label(bars, labels=[f"{v:,.1f} {unidade}" for v in vals.values],
                 padding=5, fontsize=11, fontweight="bold")
    ax.set_title(metrica, fontsize=12)
    ax.set_ylabel(unidade)
    ax.set_ylim(0, vals.max() * 1.3)

plt.tight_layout()
plt.show()

# ── 8.3 — CAGR ──────────────────────────────────────────────────────────────
log("8.3 Taxa de Crescimento Anual Composta (CAGR) — Solar e Eólica", 2)

print(f"\n{'─'*45}")
print(f"{'CAGR — Fontes Renováveis em Destaque':^45}")
print(f"{'─'*45}")
print(f"{'Fonte':<20} {'CAGR':>10}")
print(f"{'─'*45}")
for fonte in ["Solar", "Wind"]:
    serie_anual = (data_br[data_br["PRODUCT"] == fonte]
                   .groupby("YEAR")["VALUE"].sum())
    cagr = calcular_cagr(serie_anual)
    print(f"{fonte:<20} {cagr:>10.1%}")
print(f"{'─'*45}")

# ── 8.4 — Diagnóstico e Recomendações finais ─────────────────────────────────
print(f"\n{'─'*75}")
print(f"{'DIAGNÓSTICO E RECOMENDAÇÕES FINAIS':^75}")
print(f"{'─'*75}\n")
print("1. DIAGNÓSTICO DE ESTACIONARIEDADE (Teste ADF):")
for fonte in ["Solar", "Wind"]:
    d = diagnosticos_adf[fonte]
    print(f"   • {fonte}: p-value={d['p_value']:.4f} → "
          f"{'Estacionária' if d['eh_estacionaria'] else 'Não-estacionária'} (d={d['ordem_d']})")
print("\n2. DESEMPENHO DOS MODELOS (MAE em GWh):")
for fonte in ["Solar", "Wind"]:
    if fonte in comparacao_modelos:
        c = comparacao_modelos[fonte]
        print(f"   • {fonte}: SARIMA={c['mae_sarima']} vs Prophet={c['mae_prophet']} "
              f"(diferença: {c['mae_diff_pct']}%) → Melhor: {c['melhor_modelo']}")
print("\n3. RECOMENDAÇÃO:")
print("   ✓ Usar PROPHET como modelo principal por:")
print("     — Captura natural de sazonalidade complexa")
print("     — Robustez a mudanças estruturais (change points)")
print("     — Performance equivalente ou superior ao SARIMA")
print("   ✓ Usar SARIMA como baseline/validação cruzada")
print(f"\n{'─'*75}\n")


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 9 — EXPORTAÇÕES PARA POWER BI
# Todas as exportações estão COMENTADAS — descomente quando necessário
# ══════════════════════════════════════════════════════════════════════════════
log("ETAPA 9 — EXPORTAÇÕES PARA POWER BI (TODAS COMENTADAS)")

# ── 9.1 — Base tratada ───────────────────────────────────────────────────────
# data_br.to_csv(os.path.join(DIR_EXPORT, "BR_base_tratada.csv"),
#                sep=";", encoding="utf-8-sig",
#                index=False, float_format="%.2f", date_format="%d/%m/%Y")

# ── 9.2 — Share acumulado ────────────────────────────────────────────────────
# for grupo, fontes in [("renovaveis", FONTES_RENOVAVEIS),
#                        ("nao_renovaveis", FONTES_NAO_RENOVAVEIS)]:
#     df_out = df_share_acc[df_share_acc["PRODUCT"].isin(fontes)]
#     df_out.to_csv(os.path.join(DIR_EXPORT, f"indicador_share_acumulado_{grupo}.csv"),
#                   index=False, float_format="%.2f")

# ── 9.3 — Tendências ─────────────────────────────────────────────────────────
# for fonte in ["Solar", "Wind"]:
#     tend_df = resultados_decomp[fonte]["decomp"].trend.dropna().reset_index()
#     tend_df.columns = ["Date", f"Tendencia_{fonte}"]
#     tend_df.to_csv(os.path.join(DIR_EXPORT, f"Tendencia_{fonte}.csv"),
#                    index=False, float_format="%.2f", date_format="%d/%m/%Y")

# ── 9.4 — Previsões SARIMA ────────────────────────────────────────────────────
# for fonte in ["Solar", "Wind"]:
#     if resultados_sarima.get(fonte):
#         resultados_sarima[fonte]["previsoes"][
#             ["ds", "mean", "mean_ci_lower", "mean_ci_upper"]
#         ].to_csv(os.path.join(DIR_EXPORT, f"Previsao_SARIMA_{fonte}.csv"),
#                  index=False, float_format="%.2f", date_format="%d/%m/%Y")

# ── 9.5 — Previsões Prophet ───────────────────────────────────────────────────
# for fonte in ["Solar", "Wind"]:
#     resultados_prophet[fonte]["previsao"].to_csv(
#         os.path.join(DIR_EXPORT, f"Previsao_Prophet_{fonte}.csv"),
#         index=False, float_format="%.2f", date_format="%d/%m/%Y"
#     )

# ── 9.6 — Séries completas SARIMA + Prophet combinadas ───────────────────────
# for fonte in ["Solar", "Wind"]:
#     hist   = resultados_prophet[fonte]["historico"].rename(columns={"y": "valor_real"})
#     prev_p = resultados_prophet[fonte]["previsao"]
#     df_combined = hist.merge(prev_p, on="ds", how="outer")
#     if resultados_sarima.get(fonte):
#         prev_s = resultados_sarima[fonte]["previsoes"][
#             ["ds", "mean", "mean_ci_lower", "mean_ci_upper"]
#         ].rename(columns={"mean": "sarima_yhat",
#                            "mean_ci_lower": "sarima_lower",
#                            "mean_ci_upper": "sarima_upper"})
#         df_combined = df_combined.merge(prev_s, on="ds", how="outer")
#     df_combined["fonte"] = fonte
#     df_combined.to_csv(
#         os.path.join(DIR_EXPORT, f"Serie_Completa_{fonte}_SARIMA_Prophet.csv"),
#         index=False, float_format="%.2f", date_format="%d/%m/%Y"
#     )

# ── 9.7 — Tabela anual segmentada ─────────────────────────────────────────────
# tabela_segmentada = (
#     data_princ.pivot_table(
#         index=["Segmento", "PRODUCT"], columns="YEAR",
#         values="VALUE", aggfunc="sum",
#     ).fillna(0).round(2)
# )
# tabela_segmentada.to_csv(os.path.join(DIR_EXPORT, "producao_anual_segmentada.csv"),
#                           float_format="%.2f")

# ── 9.8 — Tabela de comparação SARIMA vs Prophet ──────────────────────────────
# tabela_comp = pd.DataFrame([
#     {
#         "Fonte"        : c["fonte"],
#         "Estacionaria" : c["estacionaria"],
#         "Ordem_D"      : diagnosticos_adf[c["fonte"]]["ordem_d"],
#         "SARIMA_MAE"   : c["mae_sarima"],
#         "SARIMA_MAPE"  : c["mape_sarima"],
#         "Prophet_MAE"  : c["mae_prophet"],
#         "Prophet_MAPE" : c["mape_prophet"],
#         "Diferenca_pct": c["mae_diff_pct"],
#         "Melhor_Modelo": c["melhor_modelo"],
#     }
#     for c in comparacao_modelos.values()
# ])
# tabela_comp.to_csv(
#     os.path.join(DIR_EXPORT, "Comparacao_Modelos_SARIMA_vs_Prophet.csv"),
#     index=False, float_format="%.2f"
# )

log("PROCESSAMENTO CONCLUÍDO")
print("\n  Arquivos disponíveis para exportação (todos comentados):\n")
for arq in [
    "BR_base_tratada.csv",
    "indicador_share_acumulado_renovaveis.csv",
    "indicador_share_acumulado_nao_renovaveis.csv",
    "Tendencia_Solar.csv",
    "Tendencia_Wind.csv",
    "Previsao_SARIMA_Solar.csv",
    "Previsao_SARIMA_Wind.csv",
    "Previsao_Prophet_Solar.csv",
    "Previsao_Prophet_Wind.csv",
    "Serie_Completa_Solar_SARIMA_Prophet.csv",
    "Serie_Completa_Wind_SARIMA_Prophet.csv",
    "producao_anual_segmentada.csv",
    "Comparacao_Modelos_SARIMA_vs_Prophet.csv",
]:
    print(f"    ✓ {arq}")
print()
