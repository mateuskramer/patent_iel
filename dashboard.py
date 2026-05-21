import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as sp
import processador
import tab_correlacao
import tab_dicionario

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

load_dotenv()

# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     os.environ["DB_HOST"],
    "database": os.environ.get("DB_NAME", "postgres"),
    "user":     os.environ["DB_USER"],
    "password": os.environ["DB_PASS"],
    "port":     int(os.environ.get("DB_PORT", 5432)),
    "sslmode":  "require",
}

st.set_page_config(page_title="Patent AI Lab", layout="wide")

if "stop_processing" not in st.session_state:
    st.session_state.stop_processing = False


# ─────────────────────────────────────────────────────────────
# CONEXÃO E CARREGAMENTO
# ─────────────────────────────────────────────────────────────

def run_query(query, params=None):
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except Exception as e:
        st.error(f"Erro na query: {e}")
        return pd.DataFrame()
    finally:
        if conn is not None:
            conn.close()


@st.cache_data
def load_patents():
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        query = "SELECT id, title, abstract, year_month, embedding FROM patents WHERE embedding IS NOT NULL AND embedding <> ''"
        df = pd.read_sql_query(query, conn)
        df["embedding"] = df["embedding"].apply(lambda x: np.array(json.loads(x), dtype=np.float32))
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Erro ao carregar patentes: {e}")
        return pd.DataFrame()
    finally:
        if conn is not None:
            conn.close()


@st.cache_data
def load_terms():
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        query = """
            SELECT p.id AS patent_id, p.year_month, td.term
            FROM patent_terms pt
            JOIN patents p ON pt.patent_id::text = p.id::text
            JOIN term_dictionary td ON td.id = pt.term_id
            WHERE p.year_month IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar termos: {e}")
        return pd.DataFrame()
    finally:
        if conn is not None:
            conn.close()


@st.cache_data
def prepare_sparse_engine(df_terms_hash):
    df_terms = load_terms()
    df_terms = df_terms.copy()
    df_terms["term"]      = df_terms["term"].astype("category")
    df_terms["patent_id"] = df_terms["patent_id"].astype("category")

    categories  = df_terms["term"].cat.categories
    idx_to_term = {i: t for i, t in enumerate(categories)}
    term_to_idx = {t: i for i, t in enumerate(categories)}

    rows = df_terms["patent_id"].cat.codes
    cols = df_terms["term"].cat.codes
    data = np.ones(len(df_terms))

    A = sp.csr_matrix((data, (rows, cols)))
    C = A.T @ A
    C.setdiag(0)

    return C, term_to_idx, idx_to_term


# ─────────────────────────────────────────────────────────────
# FUNÇÕES ANALÍTICAS
# ─────────────────────────────────────────────────────────────

def similar_patents(idx, df, EMB, top_n=10):
    vec  = EMB[idx].reshape(1, -1)
    sims = cosine_similarity(vec, EMB)[0]
    out  = df.copy()
    out["similarity"] = sims
    return out[out.index != idx].sort_values("similarity", ascending=False).head(top_n)


def monthly_term_count(term, terms_df):
    return (
        terms_df[terms_df["term"] == term]
        .groupby("year_month")
        .size()
        .reset_index(name="count")
        .sort_values("year_month")
    )


def semantic_vector(term, month, terms_df):
    dfm     = terms_df[terms_df["year_month"] == month]
    patents = dfm[dfm["term"] == term]["patent_id"].unique()
    co      = dfm[dfm["patent_id"].isin(patents)]
    vec     = co["term"].value_counts()
    all_t   = terms_df["term"].unique()
    full    = pd.Series(0, index=all_t)
    full.update(vec)
    return full.values


def build_graph(root_term, terms_df, depth=3, top_n=5):
    G = nx.Graph()
    G.add_node(root_term, layer=0)
    frontier, visited = [(root_term, 0)], set()
    while frontier:
        curr, level = frontier.pop(0)
        if curr in visited or level >= depth:
            continue
        visited.add(curr)
        pats = terms_df[terms_df["term"] == curr]["patent_id"].unique()
        co_counts = (
            terms_df[(terms_df["patent_id"].isin(pats)) & (terms_df["term"] != curr)]["term"]
            .value_counts()
        )
        for t, weight in co_counts.head(top_n).items():
            if not G.has_node(t):
                G.add_node(t, layer=level + 1)
            G.add_edge(curr, t, weight=int(weight))
            frontier.append((t, level + 1))
    return G


def calc_growth(term, terms_df):
    m = monthly_term_count(term, terms_df)
    if len(m) < 2:
        return 0
    prev, curr = m.iloc[-2]["count"], m.iloc[-1]["count"]
    return ((curr - prev) / prev) * 100 if prev != 0 else 0


def calc_density(term, terms_df):
    return len(terms_df[terms_df["term"] == term]["patent_id"].unique())


def calc_fusion(term, terms_df):
    pats = terms_df[terms_df["term"] == term]["patent_id"].unique()
    return terms_df[(terms_df["patent_id"].isin(pats)) & (terms_df["term"] != term)]["term"].nunique()


def calc_shift(term, terms_df):
    months = sorted(terms_df["year_month"].unique().tolist())
    if len(months) < 2:
        return 0
    v1 = semantic_vector(term, months[0],  terms_df)
    v2 = semantic_vector(term, months[-1], terms_df)
    if v1.sum() == 0 or v2.sum() == 0:
        return 0
    return cosine(v1, v2) * 100


def calc_future_score(term, terms_df):
    g = min(max(calc_growth(term, terms_df), 0), 100)
    f = min(calc_fusion(term, terms_df) * 5,     100)
    s = min(calc_shift(term, terms_df),           100)
    d = min(calc_density(term, terms_df),         100)
    return round(0.35 * g + 0.25 * f + 0.20 * s + 0.20 * d, 2)


def term_correlations(term, terms_df):
    total     = terms_df["patent_id"].nunique()
    patents_a = set(terms_df[terms_df["term"] == term]["patent_id"].unique())
    others    = terms_df[terms_df["term"] != term]["term"].unique()
    rows = []
    for other in others:
        patents_b = set(terms_df[terms_df["term"] == other]["patent_id"].unique())
        inter = patents_a & patents_b
        if not inter:
            continue
        pa  = len(patents_a) / total
        pb  = len(patents_b) / total
        pab = len(inter) / total
        rows.append({
            "term":    other,
            "cooc":    len(inter),
            "lift":    round(pab / (pa * pb), 4),
            "jaccard": round(len(inter) / len(patents_a | patents_b), 4),
            "pmi":     round(np.log2(pab / (pa * pb)), 4),
        })
    return pd.DataFrame(rows).sort_values("lift", ascending=False) if rows else pd.DataFrame()


def get_sparse_opportunities(target_term, C, term_to_idx, idx_to_term, top_n=20):
    if target_term not in term_to_idx:
        return pd.DataFrame()
    idx          = term_to_idx[target_term]
    direct_vec   = C[idx].toarray().flatten()
    term_vector  = C[idx].T
    indirect_vec = (C @ term_vector).toarray().flatten()
    mask         = (indirect_vec > 0) & (direct_vec == 0)
    mask[idx]    = False
    potential_indices = np.where(mask)[0]

    if len(potential_indices) == 0:
        return pd.DataFrame()

    max_val = C.max() if C.max() > 0 else 1
    res = []
    for p_idx in potential_indices:
        res.append({
            "term":                   idx_to_term[p_idx],
            "bridge_strength":        int(indirect_vec[p_idx]),
            "common_neighbors_score": round(float(indirect_vec[p_idx] / max_val), 4),
        })

    if not res:
        return pd.DataFrame()

    return pd.DataFrame(res).sort_values("bridge_strength", ascending=False).head(top_n)


@st.cache_data
def ranking_table(terms_df):
    terms = terms_df["term"].value_counts().index.tolist()[:300]
    rows  = []
    for t in terms:
        rows.append({
            "term":         t,
            "growth_%":     round(calc_growth(t, terms_df), 2),
            "density":      calc_density(t, terms_df),
            "fusion":       calc_fusion(t, terms_df),
            "shift_%":      round(calc_shift(t, terms_df), 2),
            "future_score": calc_future_score(t, terms_df),
        })
    return pd.DataFrame(rows).sort_values("future_score", ascending=False)


# ─────────────────────────────────────────────────────────────
# CARREGAMENTO INICIAL
# ─────────────────────────────────────────────────────────────

df       = load_patents()
terms_df = load_terms()

if df.empty or terms_df.empty:
    st.error("❌ Falha ao conectar ao banco de dados. Verifique as configurações.")
    st.stop()

EMB = np.vstack(df["embedding"].values)
C_matrix, t_map, idx_map = prepare_sparse_engine(len(terms_df))


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

st.sidebar.title("🧠 Patent AI Control")

with st.sidebar.container(border=True):
    st.subheader("Processamento")
    counter_placeholder = st.empty()
    c1, c2 = st.columns(2)
    if c1.button("🚀 INICIAR", use_container_width=True):
        st.session_state.stop_processing = False
        processed = 0
        with st.status("Processando patentes...", expanded=True) as status:
            for msg in processador.processar_local():
                if "✅" in msg:
                    processed += 1
                counter_placeholder.metric("Patentes Processadas", processed)
                st.write(msg)
                if st.session_state.stop_processing:
                    break
            status.update(label="Concluído!", state="complete")
        st.cache_data.clear()
        st.rerun()
    if c2.button("🛑 PARAR", use_container_width=True):
        st.session_state.stop_processing = True

sel_idx = st.sidebar.selectbox(
    "Patente:",
    df.index,
    format_func=lambda x: f"{df.loc[x,'id']} - {df.loc[x,'title'][:40]}",
)
sel_term = st.sidebar.selectbox(
    "Termo:",
    terms_df["term"].value_counts().index.tolist(),
)
depth = st.sidebar.slider("Camadas rede", 1, 5, 3)

# ── Filtro de período para aba TFT ───────────────────────────

hist_df_full = monthly_term_count(sel_term, terms_df)
hist_df_full["year_month"] = pd.to_datetime(hist_df_full["year_month"])

all_months = sorted(hist_df_full["year_month"].dt.to_period("M").unique())

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Período — Predição TFT")

if len(all_months) >= 2:
    period_labels = [str(p) for p in all_months]

    shortcuts = {
        "Tudo":          (period_labels[0], period_labels[-1]),
        "Último ano":    (str((pd.Period(period_labels[-1], "M") - 11)),  period_labels[-1]),
        "Últimos 2 anos":(str((pd.Period(period_labels[-1], "M") - 23)),  period_labels[-1]),
        "Últimos 3 anos":(str((pd.Period(period_labels[-1], "M") - 35)),  period_labels[-1]),
    }
    for k, (s, e) in shortcuts.items():
        if s not in period_labels:
            shortcuts[k] = (period_labels[0], e)

    btn_cols = st.sidebar.columns(2)
    for i, (label, (s, e)) in enumerate(shortcuts.items()):
        if btn_cols[i % 2].button(label, key=f"shortcut_{label}", use_container_width=True):
            st.session_state["tft_start"] = s
            st.session_state["tft_end"]   = e

    default_start = st.session_state.get("tft_start", period_labels[0])
    default_end   = st.session_state.get("tft_end",   period_labels[-1])
    if default_start not in period_labels:
        default_start = period_labels[0]
    if default_end not in period_labels:
        default_end = period_labels[-1]

    col_s, col_e = st.sidebar.columns(2)
    range_start = col_s.selectbox("De",  period_labels, index=period_labels.index(default_start), key="tft_start_sel")
    range_end   = col_e.selectbox("Até", period_labels, index=period_labels.index(default_end),   key="tft_end_sel")

    if range_start > range_end:
        range_start, range_end = range_end, range_start

    st.session_state["tft_start"] = range_start
    st.session_state["tft_end"]   = range_end

    filter_start = pd.Period(range_start, "M").to_timestamp()
    filter_end   = pd.Period(range_end,   "M").to_timestamp(how="end")

    n_months = (pd.Period(range_end, "M") - pd.Period(range_start, "M")).n + 1
    st.sidebar.caption(f"🗓 {n_months} meses selecionados  ·  {range_start} → {range_end}")

else:
    filter_start = hist_df_full["year_month"].min() if not hist_df_full.empty else None
    filter_end   = hist_df_full["year_month"].max() if not hist_df_full.empty else None


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────

st.title("🔬 Patent AI Explorer")
tabs = st.tabs([
    "📐 Similaridade",
    "📈 Tendência",
    "🕸 Rede",
    "🧬 Evolução",
    "🚀 Indicadores",
    "🔥 Correlação",
    "📈 Correlação Temporal",
    "🏆 Ranking",
    "🌌 Esparsos",
    "🔮 Predição (TFT)",
    "📚 Dicionário",
])

# ── Tab 0: Similaridade ──────────────────────────────────────
with tabs[0]:
    st.markdown("### Patentes Semelhantes por Proximidade Vetorial")
    st.dataframe(
        similar_patents(sel_idx, df, EMB)[["id", "title", "year_month", "similarity"]],
        use_container_width=True,
    )

# ── Tab 1: Tendência ─────────────────────────────────────────
with tabs[1]:
    all_terms_list = terms_df["term"].value_counts().index.tolist()
    selected_terms = st.multiselect(
        "Selecione até 3 termos para comparar:",
        options=all_terms_list,
        default=[sel_term],
        max_selections=3,
        key="tendencia_multiselect",
    )

    if selected_terms:
        fig_tend = go.Figure()
        colors = ["#636EFA", "#EF553B", "#00CC96"]
        for i, term in enumerate(selected_terms):
            df_term = monthly_term_count(term, terms_df)
            fig_tend.add_trace(go.Scatter(
                x=df_term["year_month"],
                y=df_term["count"],
                name=term,
                mode="lines+markers",
                line=dict(color=colors[i], width=2),
                marker=dict(size=6),
            ))
        fig_tend.update_layout(
            template="plotly_dark",
            xaxis_title="Mês",
            yaxis_title="Ocorrências mensais",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_tend, use_container_width=True)
    else:
        st.info("Selecione ao menos um termo.")

# ── Tab 2: Rede ──────────────────────────────────────────────
with tabs[2]:
    st.markdown("### Mapa de Co-ocorrência Tecnológica")
    G   = build_graph(sel_term, terms_df, depth, 5)
    pos = nx.spring_layout(G, k=0.8, seed=42)

    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight  = edge[2].get("weight", 1)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=min(max(weight / 2, 1), 10), color="rgba(136,136,136,0.4)"),
            hoverinfo="none", mode="lines",
        ))

    node_x, node_y, labels, colors, node_sizes = [], [], [], [], []
    palette = ["#FF4B4B", "#FFA500", "#1E90FF", "#00FF7F", "#808080"]
    for n in G.nodes():
        node_x.append(pos[n][0])
        node_y.append(pos[n][1])
        labels.append(f"{n} (Conexões: {G.degree(n)})")
        layer = G.nodes[n].get("layer", 0)
        colors.append(palette[min(layer, 4)])
        node_sizes.append(15 + G.degree(n) * 3)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[n for n in G.nodes()],
        textposition="top center",
        hoverinfo="text", hovertext=labels,
        marker=dict(showscale=False, color=colors, size=node_sizes, line_width=2),
    )
    fig_net = go.Figure(data=edge_traces + [node_trace])
    fig_net.update_layout(template="plotly_dark", showlegend=False, height=700)
    st.plotly_chart(fig_net, use_container_width=True)

    st.info("""
    **Legenda de Cores:**
    * 🔴 Vermelho: Termo Raiz
    * 🟠 Laranja: Conexões Diretas
    * 🔵 Azul: 2ª Camada
    * 🟢 Verde: 3ª Camada
    * ⚪ Cinza: Camadas Profundas
    """)

# ── Tab 3: Evolução Semântica ────────────────────────────────
with tabs[3]:
    st.markdown("### Análise de Mutação Semântica")
    months = sorted(terms_df["year_month"].unique().tolist())
    if len(months) >= 2:
        m1 = st.selectbox("Mês A", months)
        m2 = st.selectbox("Mês B", months, index=len(months) - 1)
        v1 = semantic_vector(sel_term, m1, terms_df)
        v2 = semantic_vector(sel_term, m2, terms_df)
        if v1.sum() > 0 and v2.sum() > 0:
            sim_val = 1 - cosine(v1, v2)
            st.metric("Similaridade Contextual", f"{sim_val:.4f}")
            st.metric("Shift % (Mutação)", f"{(1 - sim_val) * 100:.2f}%")

# ── Tab 4: Indicadores ───────────────────────────────────────
with tabs[4]:
    st.markdown("### Métricas de Maturidade e Impacto")

    all_terms_list_4 = terms_df["term"].value_counts().index.tolist()
    terms_indicadores = st.multiselect(
        "Selecione até 3 termos:",
        options=all_terms_list_4,
        default=[sel_term],
        max_selections=3,
        key="indicadores_multiselect",
    )

    colors_ind = ["#636EFA", "#EF553B", "#00CC96"]
    indicadores = ["Growth %", "Density", "Fusion", "Shift %", "Future Score"]

    # Tabela comparativa
    rows_ind = []
    for term in terms_indicadores:
        rows_ind.append({
            "Termo":        term,
            "Growth %":     round(calc_growth(term, terms_df), 2),
            "Density":      calc_density(term, terms_df),
            "Fusion":       calc_fusion(term, terms_df),
            "Shift %":      round(calc_shift(term, terms_df), 2),
            "Future Score": calc_future_score(term, terms_df),
        })
    df_ind = pd.DataFrame(rows_ind)
    st.dataframe(df_ind.set_index("Termo"), use_container_width=True)

    # Gráfico radar comparativo
    fig_radar = go.Figure()
    for i, row in df_ind.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row["Growth %"], row["Density"], row["Fusion"], row["Shift %"], row["Future Score"]],
            theta=indicadores,
            fill="toself",
            name=row["Termo"],
            line=dict(color=colors_ind[i]),
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.info(f"""
    **Glossário de Indicadores:**
    * **Growth %:** Taxa de crescimento do termo no último mês em relação ao anterior.
    * **Density:** Quantidade de patentes únicas que utilizam este termo.
    * **Fusion:** Quantidade de outras tecnologias que se conectam a este termo.
    * **Shift %:** O quanto o contexto do termo mudou ao longo do tempo.
    * **Future Score:** `0.35·Growth + 0.25·Fusion·5 + 0.20·Shift + 0.20·Density`
    """)

# ── Tab 5: Correlação (Lift/Jaccard/PMI) ─────────────────────
with tabs[5]:
    st.markdown("### Análise Estatística de Correlação")

    all_terms_list_5 = terms_df["term"].value_counts().index.tolist()
    terms_corr = st.multiselect(
        "Selecione até 3 termos:",
        options=all_terms_list_5,
        default=[sel_term],
        max_selections=3,
        key="corr_lift_multiselect",
    )
    if not terms_corr:
        terms_corr = [sel_term]

    for i, term in enumerate(terms_corr):
        if len(terms_corr) > 1:
            st.markdown(f"#### {term}")
        corr = term_correlations(term, terms_df)
        if not corr.empty:
            st.dataframe(corr.head(20), use_container_width=True)
            st.plotly_chart(
                px.bar(
                    corr.head(15), x="term", y="lift", color="jaccard",
                    template="plotly_dark",
                    title=f"Lift — {term}",
                ),
                use_container_width=True,
                key=f"corr_lift_chart_{i}",
            )
        else:
            st.info(f"Sem correlações para '{term}'.")
        if i < len(terms_corr) - 1:
            st.markdown("---")

    st.info("""
    **O que significam estas métricas?**
    * **Cooc:** Quantas vezes os dois termos aparecem juntos na mesma patente.
    * **Lift:** Força da associação. Se > 1, a presença de um atrai o outro acima do aleatório.
    * **Jaccard:** Percentual de sobreposição entre os dois termos (0 a 1).
    * **PMI:** Dependência estatística — valores altos indicam forte ligação no vocabulário técnico.
    """)

# ── Tab 6: Correlação Temporal (Pearson) ─────────────────────
with tabs[6]:
    all_terms_list_6 = terms_df["term"].value_counts().index.tolist()
    terms_temporal = st.multiselect(
        "Selecione até 3 termos:",
        options=all_terms_list_6,
        default=[sel_term],
        max_selections=3,
        key="corr_temporal_multiselect",
    )
    tab_correlacao.render(terms_df, terms_temporal if terms_temporal else [sel_term])

# ── Tab 7: Ranking ───────────────────────────────────────────
with tabs[7]:
    st.markdown("### Ranking de Tecnologias Emergentes")
    rk = ranking_table(terms_df)
    st.dataframe(rk, use_container_width=True)

    st.success("""
    **O que este Ranking sugere?**
    Tecnologias no topo tendem a ser emergentes (alto Growth), versáteis (alto Fusion) e em transformação (alto Shift).
    """)

# ── Tab 8: Esparsos ──────────────────────────────────────────
with tabs[8]:
    st.markdown("### 🌌 Oportunidades Invisíveis (Associações Esparsas)")
    sparse_res = get_sparse_opportunities(sel_term, C_matrix, t_map, idx_map)
    if not sparse_res.empty:
        col_t, col_p = st.columns([1, 1])
        with col_t:
            st.dataframe(sparse_res, use_container_width=True)
        with col_p:
            fig_s = px.bar(
                sparse_res.head(15),
                x="bridge_strength", y="term",
                orientation="h", color="bridge_strength",
                template="plotly_dark",
            )
            st.plotly_chart(fig_s, use_container_width=True)

    st.warning("""
    **Análise de "Buracos" na Inovação:**
    Estes termos NUNCA apareceram na mesma patente que o termo selecionado, mas compartilham vizinhos tecnológicos.
    * **Bridge Strength:** Força da conexão indireta.
    * **Common Neighbors Score:** Similaridade de ecossistema normalizada.
    """)

# ── Tab 9: Predição TFT ──────────────────────────────────────
with tabs[9]:
    st.header(f"🔮 Inteligência Preditiva (TFT): {sel_term}")

    hist_filtered = hist_df_full.copy()
    if filter_start is not None:
        hist_filtered = hist_filtered[hist_filtered["year_month"] >= filter_start]
    if filter_end is not None:
        hist_filtered = hist_filtered[hist_filtered["year_month"] <= filter_end]

    pred_df = run_query(
        "SELECT * FROM patent_predictions WHERE term = %s ORDER BY target_year_month",
        (sel_term,),
    )
    if not pred_df.empty:
        pred_df["target_year_month"] = pd.to_datetime(pred_df["target_year_month"])
        if not hist_df_full.empty:
            cutoff = hist_df_full["year_month"].max()
            pred_df = pred_df[pred_df["target_year_month"] > cutoff]

    bt_df = pd.DataFrame()
    try:
        bt_df = run_query(
            "SELECT * FROM patent_backtest WHERE term = %s ORDER BY target_year_month",
            (sel_term,),
        )
        if not bt_df.empty:
            bt_df["target_year_month"] = pd.to_datetime(bt_df["target_year_month"])
            if filter_start is not None:
                bt_df = bt_df[bt_df["target_year_month"] >= filter_start]
            if filter_end is not None:
                bt_df = bt_df[bt_df["target_year_month"] <= filter_end]
    except Exception:
        bt_df = pd.DataFrame()

    fig_tft = go.Figure()

    if not hist_filtered.empty:
        fig_tft.add_trace(go.Scatter(
            x=hist_filtered["year_month"],
            y=hist_filtered["count"],
            name="Histórico Real",
            mode="lines+markers",
            line=dict(color="#3498db", width=3),
            marker=dict(size=7),
        ))

    if not bt_df.empty:
        fig_tft.add_trace(go.Scatter(
            x=bt_df["target_year_month"],
            y=bt_df["predicted_count"],
            name="IA — Backtest (previsto)",
            mode="lines+markers",
            line=dict(color="#e67e22", width=2, dash="dot"),
            marker=dict(size=6, symbol="x"),
        ))
        if "real_count" in bt_df.columns:
            fig_tft.add_trace(go.Scatter(
                x=bt_df["target_year_month"],
                y=bt_df["real_count"],
                name="Real — período backtest",
                mode="lines+markers",
                line=dict(color="#e74c3c", width=2),
                marker=dict(size=6),
            ))

    if not pred_df.empty:
        if "optimistic_count" in pred_df.columns and "pessimistic_count" in pred_df.columns:
            fig_tft.add_trace(go.Scatter(
                x=pd.concat([pred_df["target_year_month"], pred_df["target_year_month"].iloc[::-1]]),
                y=pd.concat([pred_df["optimistic_count"],  pred_df["pessimistic_count"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(46, 204, 113, 0.15)",
                line_color="rgba(255,255,255,0)",
                name="Incerteza (q10–q90)",
                hoverinfo="skip",
            ))

        fig_tft.add_trace(go.Scatter(
            x=pred_df["target_year_month"],
            y=pred_df["predicted_count"],
            name="Previsão Futura (q50)",
            mode="lines+markers",
            line=dict(color="#2ecc71", width=3, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
        ))

        if not hist_df_full.empty:
            cutoff_str = hist_df_full["year_month"].max().strftime("%Y-%m-%d")
            fig_tft.add_shape(
                type="line",
                x0=cutoff_str, x1=cutoff_str,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
            )
            fig_tft.add_annotation(
                x=cutoff_str, y=0.97,
                xref="x", yref="paper",
                text="→ futuro",
                showarrow=False,
                font=dict(color="rgba(255,255,255,0.5)", size=11),
                xanchor="left", yanchor="top",
            )

    if pred_df.empty:
        fig_tft.add_annotation(
            text="Sem predições — rode patent_tft_pipeline.py",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="rgba(255,255,255,0.4)", size=14),
        )

    fig_tft.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        height=520,
        xaxis_title="Mês",
        yaxis_title="Contagem de Patentes",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50),
    )
    st.plotly_chart(fig_tft, use_container_width=True)

    st.markdown("""
    <div style='display:flex; gap:24px; font-size:13px; margin-top:-8px; margin-bottom:12px;'>
        <span>🔵 Histórico real</span>
        <span>🟢 Previsão futura (q50)</span>
        <span>🟠 Backtest — o que a IA teria previsto</span>
        <span>🔴 Real no período de backtest</span>
    </div>
    """, unsafe_allow_html=True)

    if bt_df.empty:
        st.info("ℹ️ Sem dados de backtest para este termo.")

    if not pred_df.empty:
        last_real = hist_df_full["count"].iloc[-1] if not hist_df_full.empty else 0
        next_pred = pred_df["predicted_count"].iloc[0]
        delta_pct = f"{((next_pred - last_real) / last_real * 100):.1f}%" if last_real > 0 else "—"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Último real",       f"{last_real:.0f} pat.")
        col2.metric("Próximo mês (q50)", f"{next_pred:.1f}", delta_pct)
        col3.metric(
            "Pessimista (q10)",
            f"{pred_df['pessimistic_count'].iloc[0]:.1f}" if "pessimistic_count" in pred_df.columns else "—",
        )
        col4.metric(
            "Otimista (q90)",
            f"{pred_df['optimistic_count'].iloc[-1]:.1f}" if "optimistic_count" in pred_df.columns else "—",
        )

    if (
        not bt_df.empty
        and "real_count" in bt_df.columns
        and not bt_df["predicted_count"].isna().all()
    ):
        mae  = np.mean(np.abs(bt_df["real_count"] - bt_df["predicted_count"]))
        rmse = np.sqrt(np.mean((bt_df["real_count"] - bt_df["predicted_count"]) ** 2))
        col_a, col_b = st.columns(2)
        col_a.metric("MAE — Backtest",  f"{mae:.2f} pat.",  help="Erro médio absoluto")
        col_b.metric("RMSE — Backtest", f"{rmse:.2f} pat.", help="Raiz do erro quadrático médio")

    if not pred_df.empty:
        with st.expander("📋 Tabela de predições completa"):
            cols_available = [c for c in [
                "target_year_month", "pessimistic_count", "q25_count",
                "predicted_count", "q75_count", "optimistic_count",
            ] if c in pred_df.columns]
            rename_map = {
                "target_year_month": "Mês",
                "pessimistic_count": "q10 (pessimista)",
                "q25_count":         "q25",
                "predicted_count":   "q50 (mediana)",
                "q75_count":         "q75",
                "optimistic_count":  "q90 (otimista)",
            }
            st.dataframe(
                pred_df[cols_available].rename(columns=rename_map),
                use_container_width=True,
            )
    elif pred_df.empty:
        st.warning(
            "⚠️ Nenhuma predição encontrada. Execute o script `patent_tft_pipeline.py` para gerar as previsões."
        )

# ── Tab 10: Dicionário ───────────────────────────────────────
with tabs[10]:
    tab_dicionario.render(DB_CONFIG)