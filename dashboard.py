import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import json
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import os

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

import tab_correlacao

DB_CONFIG = {
    'host': os.environ['DB_HOST'],
    'database': os.environ.get('DB_NAME', 'postgres'),
    'user': os.environ['DB_USER'],
    'password': os.environ['DB_PASS'],
    'port': int(os.environ.get('DB_PORT', 5432)),
    'sslmode': 'require'
}

st.set_page_config(page_title="Patent AI Lab", layout="wide")

@st.cache_data
def load_patents():
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT id, title, abstract, year_month, embedding
        FROM patents
        WHERE embedding IS NOT NULL AND embedding <> ''
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    df["year_month"] = df["year_month"].astype(str)
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(json.loads(x), dtype=np.float32)
    )
    return df.reset_index(drop=True)

@st.cache_data
def load_terms():
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT p.id AS patent_id, p.year_month, td.term
        FROM patent_terms pt
        JOIN patents p ON pt.patent_id::text = p.id::text
        JOIN term_dictionary td ON td.id = pt.term_id
        WHERE p.year_month IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df["year_month"] = df["year_month"].astype(str)
    return df

df = load_patents()
terms_df = load_terms()

if df.empty:
    st.warning("Nenhum embedding encontrado.")
    st.stop()

EMB = np.vstack(df["embedding"].values)

def similar_patents(idx, top_n=10):
    vec = EMB[idx].reshape(1, -1)
    sims = cosine_similarity(vec, EMB)[0]
    out = df.copy()
    out["similarity"] = sims
    out = out[out.index != idx]
    return out.sort_values("similarity", ascending=False).head(top_n)

def monthly_term_count(term):
    x = terms_df[terms_df["term"] == term]
    return (
        x.groupby("year_month")
        .size()
        .reset_index(name="count")
        .sort_values("year_month")
    )

def semantic_vector(term, month):
    dfm = terms_df[terms_df["year_month"] == month]
    patents = dfm[dfm["term"] == term]["patent_id"].unique()
    co = dfm[dfm["patent_id"].isin(patents)]
    vec = co["term"].value_counts()
    all_terms = terms_df["term"].unique()
    full = pd.Series(0, index=all_terms)
    full.update(vec)
    return full.values

def build_graph(root_term, depth=3, top_n=5):
    G = nx.Graph()
    G.add_node(root_term, layer=0)
    frontier = [(root_term, 0)]
    visited = set()

    while frontier:
        current, level = frontier.pop(0)
        if current in visited:
            continue
        visited.add(current)
        if level >= depth:
            continue
        patents = terms_df[terms_df["term"] == current]["patent_id"].unique()
        co = terms_df[
            (terms_df["patent_id"].isin(patents)) &
            (terms_df["term"] != current)
        ]
        top_terms = co["term"].value_counts().head(top_n).index.tolist()
        for t in top_terms:
            if not G.has_node(t):
                G.add_node(t, layer=level + 1)
            G.add_edge(current, t)
            frontier.append((t, level + 1))
    return G

def calc_growth(term):
    m = monthly_term_count(term)
    if len(m) < 2:
        return 0
    prev = m.iloc[-2]["count"]
    curr = m.iloc[-1]["count"]
    if prev == 0:
        return 0
    return ((curr - prev) / prev) * 100

def calc_density(term):
    return len(terms_df[terms_df["term"] == term]["patent_id"].unique())

def calc_fusion(term):
    patents = terms_df[terms_df["term"] == term]["patent_id"].unique()
    co = terms_df[
        (terms_df["patent_id"].isin(patents)) &
        (terms_df["term"] != term)
    ]
    return co["term"].nunique()

def calc_shift(term):
    months = sorted(terms_df["year_month"].unique().tolist())
    if len(months) < 2:
        return 0
    v1 = semantic_vector(term, months[0])
    v2 = semantic_vector(term, months[-1])
    if v1.sum() == 0 or v2.sum() == 0:
        return 0
    sim = 1 - cosine(v1, v2)
    return (1 - sim) * 100

def calc_future_score(term):
    score = (
        0.35 * min(max(calc_growth(term), 0), 100) +
        0.25 * min(calc_fusion(term) * 5, 100) +
        0.20 * min(calc_shift(term), 100) +
        0.20 * min(calc_density(term), 100)
    )
    return round(score, 2)

def term_correlations(term):
    total = terms_df["patent_id"].nunique()
    patents_a = set(terms_df[terms_df["term"] == term]["patent_id"].unique())
    others = terms_df[terms_df["term"] != term]["term"].unique()
    rows = []
    for other in others:
        patents_b = set(terms_df[terms_df["term"] == other]["patent_id"].unique())
        inter = patents_a & patents_b
        if len(inter) == 0:
            continue
        union = len(patents_a | patents_b)
        pa = len(patents_a) / total
        pb = len(patents_b) / total
        pab = len(inter) / total
        lift = pab / (pa * pb) if pa * pb > 0 else 0
        jaccard = len(inter) / union if union > 0 else 0
        pmi = np.log2(pab / (pa * pb)) if pab > 0 else 0
        rows.append({
            "term": other,
            "cooc": len(inter),
            "lift": round(lift, 4),
            "jaccard": round(jaccard, 4),
            "pmi": round(pmi, 4)
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("lift", ascending=False)

def sparse_associations(term):
    patents_a = set(terms_df[terms_df["term"] == term]["patent_id"].unique())
    if not patents_a:
        return pd.DataFrame()
    neighbors_a = set(terms_df[terms_df["patent_id"].isin(patents_a)]["term"].unique())
    neighbors_a.discard(term)
    len_a = len(neighbors_a)
    if len_a == 0:
        return pd.DataFrame()
    patents_with_neighbors = terms_df[terms_df["term"].isin(neighbors_a)]["patent_id"].unique()
    potential_terms = set(terms_df[terms_df["patent_id"].isin(patents_with_neighbors)]["term"].unique())
    potential_terms.discard(term)
    rows = []
    for other in potential_terms:
        patents_b = set(terms_df[terms_df["term"] == other]["patent_id"].unique())
        if len(patents_a & patents_b) > 0:
            continue
        neighbors_b = set(terms_df[terms_df["patent_id"].isin(patents_b)]["term"].unique())
        neighbors_b.discard(other)
        len_b = len(neighbors_b)
        if len_b == 0:
            continue
        shared_neighbors = neighbors_a & neighbors_b
        len_shared = len(shared_neighbors)
        if len_shared == 0:
            continue
        union_neighbors = len_a + len_b - len_shared
        context_jaccard = len_shared / union_neighbors if union_neighbors > 0 else 0
        context_cosine = len_shared / (np.sqrt(len_a) * np.sqrt(len_b))
        rows.append({
            "term": other,
            "shared_neighbors": len_shared,
            "context_jaccard": round(context_jaccard, 4),
            "context_cosine": round(context_cosine, 4)
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("context_cosine", ascending=False)

@st.cache_data
def ranking_table():
    terms = terms_df["term"].value_counts().index.tolist()
    rows = []
    for t in terms[:300]:
        rows.append({
            "term": t,
            "growth_%": round(calc_growth(t), 2),
            "density": calc_density(t),
            "fusion": calc_fusion(t),
            "shift_%": round(calc_shift(t), 2),
            "future_score": calc_future_score(t)
        })
    return pd.DataFrame(rows).sort_values("future_score", ascending=False)

# --- SIDEBAR ---
st.sidebar.title("🧠 Patent AI")

selected_idx = st.sidebar.selectbox(
    "Escolha uma patente:",
    df.index,
    format_func=lambda x: f"{df.loc[x,'id']} - {df.loc[x,'title'][:55]}"
)

selected_term = st.sidebar.selectbox(
    "Escolha um termo:",
    terms_df["term"].value_counts().index.tolist()
)

depth = st.sidebar.slider("Camadas da rede", 1, 5, 3)

base = df.loc[selected_idx]

st.title("🧠 Patent AI Lab")
st.markdown(f"### {base['title']}")
st.write(f"Patente ID: {base['id']}")

with st.expander("Ver Abstract"):
    st.write(base["abstract"])

# --- ABAS ---
tabs = st.tabs([
    "📐 Similaridade",
    "📈 Tendência",
    "🕸 Rede",
    "🧬 Evolução",
    "🚀 Indicadores",
    "🔥 Correlação",
    "📈 Correlação Temporal",
    "🏆 Ranking",
    "🌌 Esparsos"
])

with tabs[0]:
    sim = similar_patents(selected_idx)
    st.dataframe(sim[["id", "title", "year_month", "similarity"]], use_container_width=True)

with tabs[1]:
    trend = monthly_term_count(selected_term)
    fig = px.line(trend, x="year_month", y="count", markers=True, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    G = build_graph(selected_term, depth, 5)
    pos = nx.spring_layout(G, k=0.8)
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1))
    node_x, node_y, labels, colors, sizes = [], [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x); node_y.append(y); labels.append(n)
        layer = G.nodes[n]["layer"]
        colors.append(["red","orange","lightblue","green"][min(layer,3)] if layer < 4 else "gray")
        sizes.append(max(12, 30 - layer * 3))
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=labels,
        textposition="top center", marker=dict(size=sizes, color=colors)
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.info("""
🔴 Vermelho = termo principal selecionado  
🟠 Laranja = conexões diretas (1ª camada)  
🔵 Azul = conexões secundárias (2ª camada)  
🟢 Verde = conexões terciárias (3ª camada)  
⚪ Cinza = camadas profundas  
Tamanho maior = nó mais central
""")

with tabs[3]:
    months = sorted(terms_df["year_month"].unique().tolist())
    if len(months) >= 2:
        c1, c2 = st.columns(2)
        m1 = c1.selectbox("Mês A", months)
        m2 = c2.selectbox("Mês B", months, index=len(months)-1)
        v1 = semantic_vector(selected_term, m1)
        v2 = semantic_vector(selected_term, m2)
        if v1.sum() > 0 and v2.sum() > 0:
            sim = 1 - cosine(v1, v2)
            st.metric("Similaridade", f"{sim:.4f}")
            st.metric("Shift %", f"{(1-sim)*100:.2f}")

with tabs[4]:
    c1, c2, c3 = st.columns(3)
    c4, c5 = st.columns(2)
    c1.metric("Growth %", f"{calc_growth(selected_term):.2f}")
    c2.metric("Density", calc_density(selected_term))
    c3.metric("Fusion", calc_fusion(selected_term))
    c4.metric("Shift %", f"{calc_shift(selected_term):.2f}")
    c5.metric("Future Score", calc_future_score(selected_term))
    st.info("""
**Growth %** → crescimento de uso do termo no último mês.  
**Density** → quantidade de patentes contendo o termo.  
**Fusion** → número de tecnologias conectadas ao termo.  
**Shift %** → mudança semântica ao longo do tempo.  
**Future Score** → score geral de potencial futuro.
""")

with tabs[5]:
    corr = term_correlations(selected_term)
    st.dataframe(corr.head(20), use_container_width=True)
    fig = px.bar(corr.head(15), x="term", y="lift", color="jaccard", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.info("""
**cooc** → número de patentes onde os dois termos aparecem juntos.  
**lift** → força real da associação. >1 indica forte relação.  
**jaccard** → percentual de sobreposição entre conjuntos.  
**pmi** → relevância estatística entre termos raros/específicos.
""")

with tabs[6]:
    tab_correlacao.render(terms_df, selected_term)

with tabs[7]:
    rank = ranking_table()
    st.dataframe(rank, use_container_width=True)
    fig = px.bar(rank.head(15), x="term", y="future_score", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tabs[8]:
    st.markdown("### Associações Indiretas (Termos que não coocorrem)")
    with st.spinner("Calculando conexões esparsas..."):
        sparse = sparse_associations(selected_term)
    if sparse.empty:
        st.warning("Nenhuma associação esparsa com contexto compartilhado encontrada para este termo.")
    else:
        st.dataframe(sparse.head(20), use_container_width=True)
        fig_sparse = px.bar(
            sparse.head(15), x="term", y="context_cosine",
            color="context_jaccard", template="plotly_dark",
            title="Top Termos Esparsos (por Similaridade de Cosseno)"
        )
        st.plotly_chart(fig_sparse, use_container_width=True)
        st.info("""
**shared_neighbors** → Quantidade de termos em comum que coocorrem com ambos (ecossistema compartilhado).  
**context_jaccard** → Similaridade de sobreposição entre os ecossistemas dos dois termos.  
**context_cosine** → Similaridade de cosseno baseada no contexto.
""")