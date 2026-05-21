"""
tab_correlacao.py — Correlação Temporal entre Termos Tecnológicos

Módulo chamado pelo dashboard.py via tab_correlacao.render(df, selected_terms).

Implementa:
  1. Ranking de correlação de Pearson — quais termos evoluem junto com o(s) selecionado(s)
  2. Série temporal comparada — visualização direta das trajetórias paralelas
  3. Matriz de correlação entre os termos selecionados (quando 2 ou 3)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import pearsonr


COLORS = ["#636EFA", "#EF553B", "#00CC96"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_temporal_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["year_month", "term"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )


def _pearson_with_term(pivot: pd.DataFrame, selected_term: str) -> pd.DataFrame:
    if selected_term not in pivot.columns:
        return pd.DataFrame()
    x = pivot[selected_term].values
    if x.std() == 0:
        return pd.DataFrame()
    rows = []
    for term in pivot.columns:
        if term == selected_term:
            continue
        y = pivot[term].values
        if y.std() == 0:
            continue
        r, p = pearsonr(x, y)
        rows.append({"parceiro": term, "pearson_r": r, "p_value": p})
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False)


def _pearson_between(pivot: pd.DataFrame, term_a: str, term_b: str):
    if term_a not in pivot.columns or term_b not in pivot.columns:
        return None, None
    x, y = pivot[term_a].values, pivot[term_b].values
    if x.std() == 0 or y.std() == 0:
        return None, None
    return pearsonr(x, y)


# ---------------------------------------------------------------------------
# Render principal
# ---------------------------------------------------------------------------

def render(df: pd.DataFrame, selected_terms):
    # Aceita string (compatibilidade) ou lista
    if isinstance(selected_terms, str):
        selected_terms = [selected_terms]
    selected_terms = [t for t in selected_terms if t]
    if not selected_terms:
        st.warning("Nenhum termo selecionado.")
        return

    primary_term = selected_terms[0]

    st.subheader("📈 Correlação Temporal")
    st.markdown("Mede se termos crescem e decaem **juntos ao longo do tempo** usando correlação de Pearson sobre as séries mensais de frequência.")

    if df.empty:
        st.warning("DataFrame vazio.")
        return

    months = sorted(df["year_month"].unique())
    if len(months) < 3:
        st.warning(f"São necessários pelo menos 3 meses de dados. Encontrados: {len(months)}.")
        return

    freq_filter = df.groupby("term")["patent_id"].nunique()
    valid_terms = freq_filter[freq_filter >= 2].index
    df = df[df["term"].isin(valid_terms)].copy()

    N_patentes = df["patent_id"].nunique()
    N_termos   = df["term"].nunique()
    st.caption(f"Base: **{N_patentes}** patentes · **{N_termos}** termos · **{len(months)}** meses ({months[0]} → {months[-1]})")

    pivot = _build_temporal_matrix(df)

    # -----------------------------------------------------------------------
    # SEÇÃO 1: Séries temporais dos termos selecionados
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 1. Séries Temporais dos Termos Selecionados")

    fig_series = go.Figure()
    for i, term in enumerate(selected_terms):
        if term in pivot.columns:
            fig_series.add_trace(go.Scatter(
                x=pivot.index,
                y=pivot[term],
                name=term,
                mode="lines+markers",
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                marker=dict(size=6),
            ))
    fig_series.update_layout(
        template="plotly_dark",
        xaxis_title="Mês",
        yaxis_title="Ocorrências mensais",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_series, use_container_width=True, key="corr_series")

    # -----------------------------------------------------------------------
    # SEÇÃO 2: Matriz de correlação entre os termos selecionados
    # -----------------------------------------------------------------------
    if len(selected_terms) >= 2:
        st.markdown("---")
        st.markdown("#### 2. Correlação entre os Termos Selecionados")

        valid_sel = [t for t in selected_terms if t in pivot.columns]
        if len(valid_sel) >= 2:
            cols = st.columns(len(valid_sel) * (len(valid_sel) - 1) // 2)
            col_idx = 0
            from itertools import combinations
            for t_a, t_b in combinations(valid_sel, 2):
                r, p = _pearson_between(pivot, t_a, t_b)
                if r is not None:
                    sig = "✅ significativo" if p < 0.05 else "— não significativo"
                    cols[col_idx].metric(
                        f"{t_a[:15]} × {t_b[:15]}",
                        f"r = {r:+.4f}",
                        delta=sig,
                        delta_color="normal" if p < 0.05 else "off",
                    )
                    col_idx += 1

    # -----------------------------------------------------------------------
    # SEÇÃO 3: Ranking de correlação com o termo principal
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown(f"#### 3. Ranking de Correlação com '{primary_term}'")

    pearson_df = _pearson_with_term(pivot, primary_term)

    if pearson_df.empty:
        st.info(f"Sem dados suficientes para calcular correlação com '{primary_term}'.")
    else:
        col_a, col_b = st.columns([2, 1])
        top_n     = col_a.slider("Quantos termos exibir", 5, min(40, len(pearson_df)), 15, key="corr_topn")
        threshold = col_b.slider("Filtro mínimo |r|", 0.0, 0.9, 0.0, step=0.05, key="corr_threshold")

        df_filtered = pearson_df[pearson_df["pearson_r"].abs() >= threshold].head(top_n)

        if df_filtered.empty:
            st.info("Nenhum termo acima do filtro selecionado.")
        else:
            df_plot = df_filtered.sort_values("pearson_r")
            colors_bar = ["#e74c3c" if r < 0 else "#2ecc71" for r in df_plot["pearson_r"]]

            fig_bar = go.Figure(go.Bar(
                x=df_plot["pearson_r"],
                y=df_plot["parceiro"],
                orientation="h",
                marker_color=colors_bar,
                text=df_plot["pearson_r"].apply(lambda v: f"{v:+.3f}"),
                textposition="outside",
                customdata=df_plot["p_value"],
                hovertemplate="<b>%{y}</b><br>r = %{x:.4f}<br>p-value = %{customdata:.4f}<extra></extra>",
            ))
            fig_bar.update_layout(
                template="plotly_dark",
                title=f"Correlação de Pearson com '{primary_term}'",
                xaxis=dict(title="Pearson r", range=[-1.25, 1.25], zeroline=True, zerolinecolor="rgba(255,255,255,0.3)"),
                yaxis_title="",
                height=max(350, len(df_plot) * 32),
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="corr_bar")
            st.caption("🟢 Positivo = crescem juntos | 🔴 Negativo = anticorrelação")

    # -----------------------------------------------------------------------
    # SEÇÃO 4: Comparar com termo adicional
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 4. Comparar com Termo Adicional")

    all_terms = [t for t in pivot.columns if t not in selected_terms]
    if all_terms:
        default_partner = pearson_df.iloc[0]["parceiro"] if not pearson_df.empty and pearson_df.iloc[0]["parceiro"] in all_terms else all_terms[0]
        parceiro = st.selectbox(
            "Escolha um termo adicional para comparar:",
            options=all_terms,
            index=all_terms.index(default_partner) if default_partner in all_terms else 0,
            key="corr_partner_select",
        )

        fig_extra = go.Figure()
        all_plot_terms = selected_terms + [parceiro]
        styles = [dict(width=2), dict(width=2, dash="dot"), dict(width=2, dash="dash"), dict(width=2, dash="dashdot")]
        symbols = ["circle", "diamond", "square", "x"]
        for i, term in enumerate(all_plot_terms):
            if term in pivot.columns:
                fig_extra.add_trace(go.Scatter(
                    x=pivot.index,
                    y=pivot[term],
                    name=term,
                    mode="lines+markers",
                    line=dict(color=COLORS[i % len(COLORS)], **styles[i % len(styles)]),
                    marker=dict(size=6, symbol=symbols[i % len(symbols)]),
                ))
        fig_extra.update_layout(
            template="plotly_dark",
            xaxis_title="Mês",
            yaxis_title="Ocorrências mensais",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_extra, use_container_width=True, key="corr_extra")

    # -----------------------------------------------------------------------
    # Tabela completa + explicação
    # -----------------------------------------------------------------------
    with st.expander("📋 Ver tabela completa de correlações"):
        if not pearson_df.empty:
            display = pearson_df.copy()
            display.columns = ["Termo", "Pearson r", "p-value"]
            display["Pearson r"]    = display["Pearson r"].round(4)
            display["p-value"]      = display["p-value"].round(4)
            display["Significativo"] = display["p-value"].apply(lambda p: "✅" if p < 0.05 else "—")
            st.dataframe(display.reset_index(drop=True), use_container_width=True)

    with st.expander("📐 Sobre a correlação de Pearson"):
        st.markdown("""
        **r** mede a correlação linear entre as séries mensais de dois termos.

        | r | Interpretação |
        |---|--------------|
        | > 0.7 | Correlação forte — trajetórias muito similares |
        | 0.4 – 0.7 | Correlação moderada |
        | 0 – 0.4 | Correlação fraca |
        | < 0 | Anticorrelação — um cresce quando o outro cai |

        O **p-value** indica se o resultado é estatisticamente significativo.
        """)