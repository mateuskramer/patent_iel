import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import psycopg2
from scipy.spatial.distance import cosine
import os
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
def load_data():
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT p.year_month, p.id as patent_id, td.term, td.id as term_id
        FROM patent_terms pt
        JOIN patents p ON pt.patent_id = p.id
        JOIN term_dictionary td ON pt.term_id = td.id
        WHERE p.year_month IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

df = load_data()

if df.empty:
    st.warning("O banco de dados está vazio. Rode o script de ingestão primeiro!")
    st.stop()

# --- SIDEBAR: Escolha do Termo ---
st.sidebar.title("🎛️ Controle de Análise")
term_counts = df['term'].value_counts()
st.sidebar.write("### 📊 Top Termos Mais Densos")
st.sidebar.dataframe(term_counts.head(15))

selected_term = st.sidebar.selectbox("🎯 Selecione um Termo para Analisar:", term_counts.index.tolist())

# --- ABAS DA INTERFACE ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análise Temporal", "🕸️ Grafo de Co-ocorrência (2 Camadas)", "🧬 Evolução Semântica (Cosine)", "📈 Correlação Temporal"])

# ABA 1: TENDÊNCIA TEMPORAL
with tab1:
    st.subheader(f"Evolução de '{selected_term}' ao longo do tempo")
    df_term = df[df['term'] == selected_term]
    temporal_data = df_term.groupby('year_month').size().reset_index(name='occurrences')
    temporal_data = temporal_data.sort_values('year_month')
    
    fig = px.line(temporal_data, x='year_month', y='occurrences', markers=True, 
                  title=f"Frequência de '{selected_term}' (2024+)",
                  line_shape='spline', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# ABA 2: GRAFO DE PROFUNDIDADE (2 CAMADAS)
with tab2:
    st.subheader(f"Ecossistema de Tecnologias ao redor de '{selected_term}'")
    depth_level = st.slider("Camadas de Profundidade", 1, 2, 2)
    top_n_connections = st.slider("Máx Conexões por Termo", 2, 10, 5)
    
    G = nx.Graph()
    G.add_node(selected_term, size=30, color='red')
    
    # Camada 1
    patents_with_term = df[df['term'] == selected_term]['patent_id'].unique()
    co_terms_l1 = df[(df['patent_id'].isin(patents_with_term)) & (df['term'] != selected_term)]
    top_l1 = co_terms_l1['term'].value_counts().head(top_n_connections).index.tolist()
    
    for t in top_l1:
        weight = len(co_terms_l1[co_terms_l1['term'] == t])
        G.add_node(t, size=20, color='orange')
        G.add_edge(selected_term, t, weight=weight)
        
        # Camada 2
        if depth_level == 2:
            patents_l2 = df[df['term'] == t]['patent_id'].unique()
            co_terms_l2 = df[(df['patent_id'].isin(patents_l2)) & (~df['term'].isin([selected_term, t]))]
            top_l2 = co_terms_l2['term'].value_counts().head(top_n_connections - 2).index.tolist()
            
            for t2 in top_l2:
                weight2 = len(co_terms_l2[co_terms_l2['term'] == t2])
                G.add_node(t2, size=10, color='lightblue')
                G.add_edge(t, t2, weight=weight2)

    pos = nx.spring_layout(G, k=0.5, iterations=50)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y); node_text.append(node)
        node_color.append(G.nodes[node]['color']); node_size.append(G.nodes[node]['size'])
        
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
        marker=dict(size=node_size, color=node_color, line=dict(width=2, color='white')))
    
    fig_net = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_net, use_container_width=True)

# ABA 3: EVOLUÇÃO SEMÂNTICA VIA COSINE SIMILARITY
with tab3:
    st.subheader(f"Evolução do Contexto Semântico de '{selected_term}'")
    st.markdown("""
    *Como o significado/uso do termo muda com o tempo?* Calculamos o **Vetor de Co-ocorrência** do termo em dois meses distintos e usamos Similaridade de Cosseno.
    1.0 = Contexto exato igual | 0.0 = Contextos completamente diferentes.
    """)
    
    meses_disponiveis = sorted(df['year_month'].unique().tolist())
    if len(meses_disponiveis) >= 2:
        col1, col2 = st.columns(2)
        m1 = col1.selectbox("Mês de Referência (A):", meses_disponiveis, index=0)
        m2 = col2.selectbox("Mês de Comparação (B):", meses_disponiveis, index=len(meses_disponiveis)-1)
        
        def get_semantic_vector(term, month, df_full):
            df_month = df_full[df_full['year_month'] == month]
            patents_w_term = df_month[df_month['term'] == term]['patent_id'].unique()
            co_occurrences = df_month[df_month['patent_id'].isin(patents_w_term)]
            vector_series = co_occurrences['term'].value_counts()
            all_terms = df_full['term'].unique()
            vector = pd.Series(0, index=all_terms)
            vector.update(vector_series)
            return vector.values

        vec1 = get_semantic_vector(selected_term, m1, df)
        vec2 = get_semantic_vector(selected_term, m2, df)
        
        if sum(vec1) == 0 or sum(vec2) == 0:
            st.warning(f"O termo '{selected_term}' não possui ocorrências suficientes em um dos meses selecionados para comparação.")
        else:
            sim = 1 - cosine(vec1, vec2)
            st.metric(label=f"Similaridade Semântica ({m1} vs {m2})", value=f"{sim:.4f}")
            
            if sim > 0.8:
                st.info("💡 Alta similaridade: O termo está sendo usado junto com as mesmas tecnologias de antes.")
            elif sim < 0.4:
                st.warning("⚠️ Baixa similaridade: O termo sofreu um Shift Semântico! Está sendo aplicado a novas tecnologias agora.")
            else:
                st.success("Estabilidade média.")
    else:
        st.info("Patentes em apenas 1 mês. Necessário mais dados temporais para calcular evolução.")


with tab4:
    tab_correlacao.render(df, selected_term)
