"""
tab_dicionario.py — Manutenção do Dicionário de Termos

Módulo chamado pelo dashboard.py via tab_dicionario.render(DB_CONFIG).

Operações disponíveis:
  - Ver e filtrar termos
  - Adicionar novo termo
  - Editar classe e status
  - Renomear termo (com aviso)
  - Deletar termo (com confirmação explícita)
"""

import streamlit as st
import psycopg2
from psycopg2 import extras
import pandas as pd

CLASSES_VALIDAS  = ["technology"]
STATUS_VALIDOS   = ["pending", "approved"]


# ─────────────────────────────────────────────────────────────
# Helpers de banco
# ─────────────────────────────────────────────────────────────

def _get_conn(db_config):
    return psycopg2.connect(**db_config)


def _load_dictionary(db_config) -> pd.DataFrame:
    conn = _get_conn(db_config)
    try:
        df = pd.read_sql_query(
            "SELECT id, term, class, status, created_at FROM term_dictionary ORDER BY term",
            conn,
        )
        return df
    finally:
        conn.close()


def _count_patents(db_config, term_id: int) -> int:
    conn = _get_conn(db_config)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM patent_terms WHERE term_id = %s", (term_id,))
        return cur.fetchone()[0]
    finally:
        conn.close()


def _add_term(db_config, term: str, classe: str, status: str) -> str:
    conn = _get_conn(db_config)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO term_dictionary (term, class, status)
            VALUES (%s, %s, %s)
            ON CONFLICT (term) DO NOTHING
            RETURNING id
            """,
            (term.strip().lower(), classe, status),
        )
        result = cur.fetchone()
        conn.commit()
        if result:
            return "ok"
        return "duplicate"
    except Exception as e:
        conn.rollback()
        return str(e)
    finally:
        conn.close()


def _update_term(db_config, term_id: int, new_term: str, classe: str, status: str) -> str:
    conn = _get_conn(db_config)
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE term_dictionary SET term = %s, class = %s, status = %s WHERE id = %s",
            (new_term.strip(), classe, status, term_id),
        )
        conn.commit()
        return "ok"
    except Exception as e:
        conn.rollback()
        return str(e)
    finally:
        conn.close()


def _delete_term(db_config, term_id: int) -> str:
    conn = _get_conn(db_config)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM patent_terms WHERE term_id = %s", (term_id,))
        cur.execute("DELETE FROM term_dictionary WHERE id = %s", (term_id,))
        conn.commit()
        return "ok"
    except Exception as e:
        conn.rollback()
        return str(e)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# Render principal
# ─────────────────────────────────────────────────────────────

def render(db_config: dict):
    st.subheader("📚 Manutenção do Dicionário de Termos")

    # ── Recarrega dados ──────────────────────────────────────
    if st.button("🔄 Recarregar", key="dict_reload"):
        st.rerun()

    df = _load_dictionary(db_config)

    # ── Filtros ──────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    busca  = col_f1.text_input("🔍 Buscar termo", key="dict_busca")
    f_status = col_f2.selectbox("Status", ["Todos"] + STATUS_VALIDOS, key="dict_status")
    f_classe = col_f3.selectbox("Classe", ["Todas"] + CLASSES_VALIDAS, key="dict_classe")

    df_view = df.copy()
    if busca:
        df_view = df_view[df_view["term"].str.contains(busca, case=False, na=False)]
    if f_status != "Todos":
        df_view = df_view[df_view["status"] == f_status]
    if f_classe != "Todas":
        df_view = df_view[df_view["class"] == f_classe]

    st.caption(f"{len(df_view)} termos encontrados de {len(df)} no total")
    st.dataframe(df_view[["id", "term", "class", "status", "created_at"]], use_container_width=True)

    st.markdown("---")

    # ── Seção: Adicionar ─────────────────────────────────────
    with st.expander("➕ Adicionar novo termo"):
        c1, c2, c3 = st.columns([3, 1, 1])
        new_term   = c1.text_input("Nome do termo", key="dict_new_term")
        new_class  = c2.selectbox("Classe",  CLASSES_VALIDAS,  key="dict_new_class")
        new_status = c3.selectbox("Status",  STATUS_VALIDOS,   key="dict_new_status")

        if st.button("Adicionar", key="dict_add_btn"):
            if not new_term.strip():
                st.warning("Digite um nome para o termo.")
            else:
                result = _add_term(db_config, new_term, new_class, new_status)
                if result == "ok":
                    st.success(f"✅ Termo '{new_term}' adicionado.")
                    st.rerun()
                elif result == "duplicate":
                    st.warning(f"⚠️ O termo '{new_term}' já existe no dicionário.")
                else:
                    st.error(f"Erro: {result}")

    # ── Seção: Editar ────────────────────────────────────────
    with st.expander("✏️ Editar termo"):
        term_options = df["term"].tolist()
        edit_term_name = st.selectbox("Selecione o termo para editar", term_options, key="dict_edit_sel")
        row = df[df["term"] == edit_term_name].iloc[0]

        c1, c2, c3 = st.columns([3, 1, 1])
        edited_name   = c1.text_input("Nome", value=row["term"], key="dict_edit_name")
        edited_class  = c2.selectbox(
            "Classe", CLASSES_VALIDAS,
            index=CLASSES_VALIDAS.index(row["class"]) if row["class"] in CLASSES_VALIDAS else 0,
            key="dict_edit_class",
        )
        edited_status = c3.selectbox(
            "Status", STATUS_VALIDOS,
            index=STATUS_VALIDOS.index(row["status"]) if row["status"] in STATUS_VALIDOS else 0,
            key="dict_edit_status",
        )

        name_changed = edited_name.strip() != row["term"]
        if name_changed:
            st.warning(
                "⚠️ Você está renomeando o termo. As associações com patentes são mantidas "
                "(usam o ID interno), mas o nome visível em todo o sistema será alterado."
            )

        if st.button("Salvar edição", key="dict_edit_btn"):
            result = _update_term(db_config, int(row["id"]), edited_name, edited_class, edited_status)
            if result == "ok":
                st.success("✅ Termo atualizado.")
                st.rerun()
            else:
                st.error(f"Erro: {result}")

    # ── Seção: Deletar ───────────────────────────────────────
    with st.expander("🗑️ Deletar termo"):
        st.error(
            "**Atenção:** deletar um termo remove também todas as associações com patentes em `patent_terms`. "
            "Essa operação não pode ser desfeita."
        )

        del_term_name = st.selectbox("Selecione o termo para deletar", term_options, key="dict_del_sel")
        del_row       = df[df["term"] == del_term_name].iloc[0]
        n_patents     = _count_patents(db_config, int(del_row["id"]))

        st.info(f"Este termo está associado a **{n_patents} patente(s)**.")

        confirmacao = st.text_input(
            f"Para confirmar, digite exatamente: `{del_term_name}`",
            key="dict_del_confirm",
        )

        if st.button("🗑️ Deletar permanentemente", key="dict_del_btn", type="primary"):
            if confirmacao.strip() == del_term_name:
                result = _delete_term(db_config, int(del_row["id"]))
                if result == "ok":
                    st.success(f"✅ Termo '{del_term_name}' e suas {n_patents} associações foram removidos.")
                    st.rerun()
                else:
                    st.error(f"Erro: {result}")
            else:
                st.warning("Nome digitado não confere. Operação cancelada.")