"""
tab_correlacao.py — Correlação Temporal entre Termos Tecnológicos

Módulo chamado pelo dashboard.py via tab_correlacao.render(df, selected_term).

Implementa:
  1. Ranking de correlação de Pearson — quais termos evoluem junto com o selecionado
  2. Série temporal comparada — visualização direta das trajetórias paralelas
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_temporal_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna pivot (meses × termos) com contagem mensal de ocorrências."""
    return (
        df.groupby(['year_month', 'term'])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )


def _pearson_with_term(pivot: pd.DataFrame, selected_term: str, min_r: float = -1.0) -> pd.DataFrame:
    """
    Calcula correlação de Pearson entre o termo selecionado e todos os outros.
    Retorna DataFrame [parceiro, pearson_r, p_value] ordenado por |r|.
    """
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
        if r >= min_r:
            rows.append({'parceiro': term, 'pearson_r': r, 'p_value': p})

    return pd.DataFrame(rows).sort_values('pearson_r', ascending=False)


# ---------------------------------------------------------------------------
# Render principal
# ---------------------------------------------------------------------------

def render(df: pd.DataFrame, selected_term: str):
    st.subheader(f"📈 Correlação Temporal para '{selected_term}'")
    st.markdown("""
    Mede se dois termos crescem e decaem **juntos ao longo do tempo** usando
    correlação de Pearson sobre as séries mensais de frequência.
    """)

    if df.empty:
        st.warning("DataFrame vazio.")
        return

    months = sorted(df['year_month'].unique())
    if len(months) < 3:
        st.warning(f"São necessários pelo menos 3 meses de dados. Encontrados: {len(months)}.")
        return

    # Filtra termos com ao menos 2 patentes
    freq_filter = df.groupby('term')['patent_id'].nunique()
    valid_terms = freq_filter[freq_filter >= 2].index
    df = df[df['term'].isin(valid_terms)].copy()

    N_patentes = df['patent_id'].nunique()
    N_termos   = df['term'].nunique()
    st.caption(f"Base: **{N_patentes}** patentes · **{N_termos}** termos · **{len(months)}** meses ({months[0]} → {months[-1]})")

    # Monta pivot e calcula correlações
    pivot = _build_temporal_matrix(df)
    pearson_df = _pearson_with_term(pivot, selected_term)

    if pearson_df.empty:
        st.info(f"Sem dados suficientes para calcular correlação com '{selected_term}'.")
        return

    # -----------------------------------------------------------------------
    # SEÇÃO 1: Ranking de correlação
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 1. Ranking de Correlação com o Termo Selecionado")

    col_a, col_b = st.columns([2, 1])
    top_n = col_a.slider("Quantos termos exibir", 5, min(40, len(pearson_df)), 15, key="corr_topn")
    threshold = col_b.slider("Filtro mínimo |r|", 0.0, 0.9, 0.0, step=0.05, key="corr_threshold")

    df_filtered = pearson_df[pearson_df['pearson_r'].abs() >= threshold].head(top_n)

    if df_filtered.empty:
        st.info("Nenhum termo acima do filtro selecionado.")
    else:
        df_plot = df_filtered.sort_values('pearson_r')
        colors = ['#e74c3c' if r < 0 else '#2ecc71' for r in df_plot['pearson_r']]

        fig = go.Figure(go.Bar(
            x=df_plot['pearson_r'],
            y=df_plot['parceiro'],
            orientation='h',
            marker_color=colors,
            text=df_plot['pearson_r'].apply(lambda v: f"{v:+.3f}"),
            textposition='outside',
            customdata=df_plot['p_value'],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "r = %{x:.4f}<br>"
                "p-value = %{customdata:.4f}<extra></extra>"
            ),
        ))
        fig.update_layout(
            template='plotly_dark',
            title=f"Correlação de Pearson com '{selected_term}'",
            xaxis=dict(
                title='Pearson r',
                range=[-1.25, 1.25],
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.3)',
                zerolinewidth=1,
            ),
            yaxis_title='',
            height=max(350, len(df_plot) * 32),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🟢 Positivo = crescem juntos | 🔴 Negativo = anticorrelação (um cresce quando o outro cai)")

    # -----------------------------------------------------------------------
    # SEÇÃO 2: Série temporal comparada
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 2. Comparar Séries Temporais")

    all_terms = [t for t in pivot.columns if t != selected_term]
    default_partner = pearson_df.iloc[0]['parceiro'] if not pearson_df.empty else all_terms[0]

    parceiro = st.selectbox(
        "Escolha um termo para comparar:",
        options=all_terms,
        index=all_terms.index(default_partner) if default_partner in all_terms else 0,
        key="corr_partner_select"
    )

    if selected_term in pivot.columns and parceiro in pivot.columns:
        r_val = pearson_df[pearson_df['parceiro'] == parceiro]['pearson_r'].values
        p_val = pearson_df[pearson_df['parceiro'] == parceiro]['p_value'].values

        if len(r_val) > 0:
            col1, col2 = st.columns(2)
            col1.metric("Pearson r", f"{r_val[0]:+.4f}")
            col2.metric("p-value", f"{p_val[0]:.4f}",
                        delta="significativo" if p_val[0] < 0.05 else "não significativo",
                        delta_color="normal" if p_val[0] < 0.05 else "off")

        fig2 = go.Figure()

        # Série do termo selecionado
        fig2.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot[selected_term],
            name=selected_term,
            mode='lines+markers',
            line=dict(color='#636EFA', width=2),
            marker=dict(size=6),
        ))

        # Série do parceiro
        fig2.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot[parceiro],
            name=parceiro,
            mode='lines+markers',
            line=dict(color='#EF553B', width=2, dash='dot'),
            marker=dict(size=6, symbol='diamond'),
        ))

        fig2.update_layout(
            template='plotly_dark',
            title=f"'{selected_term}'  vs  '{parceiro}'",
            xaxis_title='Mês',
            yaxis_title='Ocorrências mensais',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified',
        )
        st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------------------------------------------
    # Tabela completa
    # -----------------------------------------------------------------------
    with st.expander("📋 Ver tabela completa de correlações"):
        display = pearson_df.copy()
        display.columns = ['Termo', 'Pearson r', 'p-value']
        display['Pearson r'] = display['Pearson r'].round(4)
        display['p-value']   = display['p-value'].round(4)
        display['Significativo'] = display['p-value'].apply(lambda p: '✅' if p < 0.05 else '—')
        st.dataframe(display.reset_index(drop=True), use_container_width=True)

    with st.expander("📐 Sobre a correlação de Pearson"):
        st.markdown(f"""
        **r** mede a correlação linear entre as séries mensais de dois termos.

        | r | Interpretação |
        |---|--------------|
        | > 0.7 | Correlação forte — trajetórias muito similares |
        | 0.4 – 0.7 | Correlação moderada |
        | 0 – 0.4 | Correlação fraca |
        | < 0 | Anticorrelação — um cresce quando o outro cai |

        O **p-value** indica se o resultado é estatisticamente significativo.
        """)
