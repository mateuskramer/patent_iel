"""
tab_dicionario.py — Manutenção do Dicionário de Termos

Módulo chamado pelo dashboard.py via tab_dicionario.render(DB_CONFIG).

Operações disponíveis:
  - Ver e filtrar termos
  - Adicionar novo termo
  - Renomear termo (com aviso)
  - Deletar termo (com confirmação explícita)
"""

import streamlit as st
import psycopg2
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Helpers de banco
# ─────────────────────────────────────────────────────────────

def _get_conn(db_config):
    return psycopg2.connect(**db_config)


def _load_dictionary(db_config) -> pd.DataFrame:
    conn = _get_conn(db_config)
    try:
        df = pd.read_sql_query(
            "SELECT id, term, created_at FROM term_dictionary ORDER BY term",
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


def _add_term(db_config, term: str) -> str:
    conn = _get_conn(db_config)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO term_dictionary (term, class, status)
            VALUES (%s, 'technology', 'approved')
            ON CONFLICT (term) DO NOTHING
            RETURNING id
            """,
            (term.strip().lower(),),
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


def _update_term(db_config, term_id: int, new_term: str) -> str:
    conn = _get_conn(db_config)
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE term_dictionary SET term = %s WHERE id = %s",
            (new_term.strip(), term_id),
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

    if st.button("🔄 Recarregar", key="dict_reload"):
        st.rerun()

    df = _load_dictionary(db_config)

    # ── Filtro ───────────────────────────────────────────────
    busca = st.text_input("🔍 Buscar termo", key="dict_busca")
    df_view = df.copy()
    if busca:
        df_view = df_view[df_view["term"].str.contains(busca, case=False, na=False)]

    st.caption(f"{len(df_view)} termos encontrados de {len(df)} no total")
    st.dataframe(df_view[["id", "term", "created_at"]], use_container_width=True)

    st.markdown("---")

    # ── Seção: Adicionar ─────────────────────────────────────
    with st.expander("➕ Adicionar novo termo"):
        new_term = st.text_input("Nome do termo", key="dict_new_term")

        if st.button("Adicionar", key="dict_add_btn"):
            if not new_term.strip():
                st.warning("Digite um nome para o termo.")
            else:
                result = _add_term(db_config, new_term)
                if result == "ok":
                    st.success(f"✅ Termo '{new_term}' adicionado.")
                    st.rerun()
                elif result == "duplicate":
                    st.warning(f"⚠️ O termo '{new_term}' já existe no dicionário.")
                else:
                    st.error(f"Erro: {result}")

    # ── Seção: Editar ────────────────────────────────────────
    with st.expander("✏️ Renomear termo"):
        term_options = df["term"].tolist()
        edit_term_name = st.selectbox("Selecione o termo para renomear", term_options, key="dict_edit_sel")
        row = df[df["term"] == edit_term_name].iloc[0]
        edited_name = st.text_input("Novo nome", value=row["term"], key="dict_edit_name")

        if edited_name.strip() != row["term"]:
            st.warning(
                "⚠️ As associações com patentes são mantidas (usam o ID interno), "
                "mas o nome visível em todo o sistema será alterado."
            )

        if st.button("Salvar", key="dict_edit_btn"):
            result = _update_term(db_config, int(row["id"]), edited_name)
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
        del_row = df[df["term"] == del_term_name].iloc[0]
        n_patents = _count_patents(db_config, int(del_row["id"]))

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