import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import json
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import processador
import os
import tab_correlacao
from dotenv import load_dotenv
load_dotenv()

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

DB_CONFIG = {
    'host': os.environ['DB_HOST'],
    'database': os.environ.get('DB_NAME', 'postgres'),
    'user': os.environ['DB_USER'],
    'password': os.environ['DB_PASS'],
    'port': int(os.environ.get('DB_PORT', 5432)),
    'sslmode': 'require'
}


st.set_page_config(page_title="Patent AI Lab", layout="wide")

if 'stop_processing' not in st.session_state:
    st.session_state.stop_processing = False

@st.cache_data
def load_patents():
    conn = psycopg2.connect(**DB_CONFIG)
    query = "SELECT id, title, abstract, year_month, embedding FROM patents WHERE embedding IS NOT NULL AND embedding <> ''"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df["embedding"] = df["embedding"].apply(lambda x: np.array(json.loads(x), dtype=np.float32))
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
    return df

df = load_patents()
terms_df = load_terms()
EMB = np.vstack(df["embedding"].values)

def similar_patents(idx, top_n=10):
    vec = EMB[idx].reshape(1, -1)
    sims = cosine_similarity(vec, EMB)[0]
    out = df.copy(); out["similarity"] = sims
    return out[out.index != idx].sort_values("similarity", ascending=False).head(top_n)

def monthly_term_count(term):
    return terms_df[terms_df["term"] == term].groupby("year_month").size().reset_index(name="count").sort_values("year_month")

def semantic_vector(term, month):
    dfm = terms_df[terms_df["year_month"] == month]
    patents = dfm[dfm["term"] == term]["patent_id"].unique()
    co = dfm[dfm["patent_id"].isin(patents)]
    vec = co["term"].value_counts()
    all_terms = terms_df["term"].unique()
    full = pd.Series(0, index=all_terms); full.update(vec)
    return full.values

def build_graph(root_term, depth=3, top_n=5):
    G = nx.Graph(); G.add_node(root_term, layer=0)
    frontier, visited = [(root_term, 0)], set()
    while frontier:
        curr, level = frontier.pop(0)
        if curr in visited or level >= depth: continue
        visited.add(curr)
        pats = terms_df[terms_df["term"] == curr]["patent_id"].unique()
        top = terms_df[(terms_df["patent_id"].isin(pats)) & (terms_df["term"] != curr)]["term"].value_counts().head(top_n).index.tolist()
        for t in top:
            if not G.has_node(t): G.add_node(t, layer=level + 1)
            G.add_edge(curr, t); frontier.append((t, level+1))
    return G

def calc_growth(term):
    m = monthly_term_count(term)
    if len(m) < 2: return 0
    prev, curr = m.iloc[-2]["count"], m.iloc[-1]["count"]
    return ((curr - prev) / prev) * 100 if prev != 0 else 0

def calc_density(term): return len(terms_df[terms_df["term"] == term]["patent_id"].unique())

def calc_fusion(term):
    pats = terms_df[terms_df["term"] == term]["patent_id"].unique()
    return terms_df[(terms_df["patent_id"].isin(pats)) & (terms_df["term"] != term)]["term"].nunique()

def calc_shift(term):
    months = sorted(terms_df["year_month"].unique().tolist())
    if len(months) < 2: return 0
    v1, v2 = semantic_vector(term, months[0]), semantic_vector(term, months[-1])
    if v1.sum() == 0 or v2.sum() == 0: return 0
    return (1 - (1 - cosine(v1, v2))) * 100

def calc_future_score(term):
    return round(0.35 * min(max(calc_growth(term), 0), 100) + 0.25 * min(calc_fusion(term)*5, 100) + 0.20 * min(calc_shift(term), 100) + 0.20 * min(calc_density(term), 100), 2)

def term_correlations(term):
    total = terms_df["patent_id"].nunique()
    patents_a = set(terms_df[terms_df["term"] == term]["patent_id"].unique())
    others = terms_df[terms_df["term"] != term]["term"].unique(); rows = []
    for other in others:
        patents_b = set(terms_df[terms_df["term"] == other]["patent_id"].unique())
        inter = patents_a & patents_b
        if not inter: continue
        pa, pb, pab = len(patents_a)/total, len(patents_b)/total, len(inter)/total
        rows.append({"term": other, "cooc": len(inter), "lift": round(pab/(pa*pb), 4), "jaccard": round(len(inter)/len(patents_a|patents_b), 4), "pmi": round(np.log2(pab/(pa*pb)), 4)})
    return pd.DataFrame(rows).sort_values("lift", ascending=False) if rows else pd.DataFrame()

def sparse_associations(term):
    p_a = set(terms_df[terms_df["term"] == term]["patent_id"].unique())
    n_a = set(terms_df[terms_df["patent_id"].isin(p_a)]["term"].unique()); n_a.discard(term); rows = []
    potential = set(terms_df[terms_df["patent_id"].isin(terms_df[terms_df["term"].isin(n_a)]["patent_id"].unique())]["term"].unique()); potential.discard(term)
    for other in potential:
        p_b = set(terms_df[terms_df["term"] == other]["patent_id"].unique())
        if p_a & p_b: continue
        n_b = set(terms_df[terms_df["patent_id"].isin(p_b)]["term"].unique()); n_b.discard(other)
        inter = n_a & n_b
        if inter:
            cosine_v = len(inter) / (np.sqrt(len(n_a)) * np.sqrt(len(n_b)))
            rows.append({"term": other, "shared_neighbors": len(inter), "context_jaccard": round(len(inter)/len(n_a|n_b), 4), "context_cosine": round(cosine_v, 4)})
    return pd.DataFrame(rows).sort_values("context_cosine", ascending=False) if rows else pd.DataFrame()

@st.cache_data
def ranking_table():
    terms = terms_df["term"].value_counts().index.tolist()[:300]; rows = []
    for t in terms:
        rows.append({"term": t, "growth_%": round(calc_growth(t), 2), "density": calc_density(t), "fusion": calc_fusion(t), "shift_%": round(calc_shift(t), 2), "future_score": calc_future_score(t)})
    return pd.DataFrame(rows).sort_values("future_score", ascending=False)

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
                if "✅" in msg: processed += 1
                counter_placeholder.metric("Patentes Processadas", processed)
                st.write(msg)
                if st.session_state.stop_processing: break
            status.update(label="Concluído!", state="complete")
        st.cache_data.clear(); st.rerun()
    
    if c2.button("🛑 PARAR", use_container_width=True):
        st.session_state.stop_processing = True

sel_idx = st.sidebar.selectbox("Patente:", df.index, format_func=lambda x: f"{df.loc[x,'id']} - {df.loc[x,'title'][:40]}")
sel_term = st.sidebar.selectbox("Termo:", terms_df["term"].value_counts().index.tolist())
depth = st.sidebar.slider("Camadas rede", 1, 5, 3)

st.title("🔬 Patent AI Explorer")
tabs = st.tabs(["📐 Similaridade", "📈 Tendência", "🕸 Rede", "🧬 Evolução", "🚀 Indicadores", "🔥 Correlação", "📈 Correlação Temporal", "🏆 Ranking", "🌌 Esparsos"])

with tabs[0]:
    st.dataframe(similar_patents(sel_idx)[["id", "title", "year_month", "similarity"]], use_container_width=True)

with tabs[1]:
    st.plotly_chart(px.line(monthly_term_count(sel_term), x="year_month", y="count", markers=True, template="plotly_dark"), use_container_width=True)

with tabs[2]:
    st.markdown("### Mapa de Co-ocorrência Tecnológica")
    G = build_graph(sel_term, depth, 5); pos = nx.spring_layout(G, k=0.8)
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0, x1, y1 = pos[e[0]][0], pos[e[0]][1], pos[e[1]][0], pos[e[1]][1]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    node_x, node_y, labels, colors = [pos[n][0] for n in G.nodes()], [pos[n][1] for n in G.nodes()], list(G.nodes()), []
    for n in G.nodes(): colors.append(["red", "orange", "lightblue", "green", "gray"][min(G.nodes[n]["layer"], 4)])
    fig = go.Figure(data=[go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#888")), go.Scatter(x=node_x, y=node_y, mode="markers+text", text=labels, textposition="top center", marker=dict(size=18, color=colors))])
    fig.update_layout(template="plotly_dark", showlegend=False); st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Legenda de Cores (Hierarquia):**
    * 🔴 **Vermelho:** Termo Raiz (O ponto de partida da análise).
    * 🟠 **Laranja:** Conexões Diretas (Termos que aparecem nas mesmas patentes que a raiz).
    * 🔵 **Azul:** 2ª Camada (Termos conectados às conexões diretas).
    * 🟢 **Verde:** 3ª Camada (Expansão do ecossistema tecnológico).
    * ⚪ **Cinza:** Camadas Profundas (Periferia tecnológica).
    """)

with tabs[3]:
    months = sorted(terms_df["year_month"].unique().tolist())
    if len(months) >= 2:
        m1, m2 = st.selectbox("Mês A", months), st.selectbox("Mês B", months, index=len(months)-1)
        v1, v2 = semantic_vector(sel_term, m1), semantic_vector(sel_term, m2)
        if v1.sum() > 0 and v2.sum() > 0:
            sim_val = 1 - cosine(v1, v2)
            st.metric("Similaridade Contextual", f"{sim_val:.4f}"); st.metric("Shift % (Mutação)", f"{(1-sim_val)*100:.2f}%")

with tabs[4]:
    st.markdown("Métricas de Maturidade e Impacto")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Growth %", f"{calc_growth(sel_term):.2f}%")
    c2.metric("Density", calc_density(sel_term))
    c3.metric("Fusion", calc_fusion(sel_term))
    c4.metric("Shift %", f"{calc_shift(sel_term):.2f}%")
    c5.metric("Future Score", calc_future_score(sel_term))

    st.info(f"""
    **Glossário de Indicadores:**
    * **Growth %:** Taxa de crescimento do termo no último mês em relação ao anterior.
    * **Density:** Volume bruto (quantidade de patentes únicas) que utilizam este termo.
    * **Fusion:** Capacidade de hibridização. Quantidade de outras tecnologias diferentes que se conectam a este termo.
    * **Shift %:** O quanto o "contexto" do termo mudou. Um Shift alto indica que a tecnologia está sendo aplicada em novas áreas.
    * **Future Score:** Potencial disruptivo calculado pela fórmula:
    $$FutureScore = (0.35 \cdot Growth) + (0.25 \cdot Fusion \cdot 5) + (0.20 \cdot Shift) + (0.20 \cdot Density)$$
    """)

with tabs[5]:
    st.markdown("### Análise Estatística de Correlação")
    corr = term_correlations(sel_term)
    if not corr.empty:
        st.dataframe(corr.head(20), use_container_width=True)
        st.plotly_chart(px.bar(corr.head(15), x="term", y="lift", color="jaccard", template="plotly_dark"), use_container_width=True)
    
    st.info("""
    **O que significam estas métricas?**
    * **Cooc (Co-ocorrência):** Quantas vezes os dois termos aparecem juntos na mesma patente.
    * **Lift:** Força da associação. Se > 1, a presença de um termo "atrai" a do outro de forma mais que aleatória.
    * **Jaccard:** Percentual de sobreposição entre os dois termos (0 a 1).
    * **PMI (Pointwise Mutual Information):** Mede a dependência estatística. Valores altos indicam que os termos são fortemente ligados no vocabulário técnico.
    """)
    
with tabs[6]:
    tab_correlacao.render(terms_df, sel_term)

with tabs[7]:
    st.markdown("### Ranking de Tecnologias Emergentes")
    rk = ranking_table(); st.dataframe(rk, use_container_width=True)
    
    st.success("""
    **O que este Ranking sugere?**
    Este ranking identifica tecnologias com **alto Future Score**. Tecnologias no topo tendem a ser:
    1.  **Emergentes:** Alto crescimento (Growth).
    2.  **Versáteis:** Conectam-se com muitas outras áreas (Fusion).
    3.  **Em Transformação:** Estão mudando de significado ou aplicação (Shift).
    *Use este ranking para identificar para onde o investimento e a pesquisa estão migrando.*
    """)

with tabs[8]:
    st.markdown("### Associações Esparsas (Oportunidades Invisíveis)")
    sparse = sparse_associations(sel_term)
    if not sparse.empty:
        st.dataframe(sparse.head(20), use_container_width=True)
        st.plotly_chart(px.bar(sparse.head(15), x="term", y="context_cosine", color="context_jaccard", template="plotly_dark"), use_container_width=True)
    
    st.warning("""
    **Análise de "Buracos" na Inovação:**
    Estes termos **NUNCA** apareceram na mesma patente que o termo selecionado, mas compartilham vizinhos tecnológicos em comum.
    * **Shared Neighbors:** Quantidade de tecnologias "ponte" que ambos conhecem.
    * **Context Jaccard:** Similaridade de ecossistema (quão parecida é a "vizinhança").
    * **Context Cosine:** Similaridade vetorial baseada na vizinhança comum.
    *O que isso sugere? Se dois termos têm alta similaridade de contexto mas não coocorrem, existe uma **oportunidade de fusão inédita**.*
    """)