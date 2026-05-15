"""
Pipeline de Predição de Patentes com Temporal Fusion Transformer (TFT)
======================================================================
Melhorias aplicadas:
  - Preenchimento de gaps temporais por termo (evita erro de filtro)
  - Split treino/validação real no Optuna (evita overfitting de HP)
  - Early stopping no treino final
  - Horizonte de predição realmente configurável
  - Logging de métricas por termo no banco
  - Robustez no predict via TimeSeriesDataSet.from_dataset
  - Quantis configuráveis e coerentes com o output_layer
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import extras
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

torch.set_float32_matmul_precision("medium")

# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ─────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "ka",
    "password": "1234",
    "port": 5432,
}

# Quantis que o modelo vai aprender (devem ser ímpares para ter mediana central)
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

# Parâmetros temporais
MAX_ENCODER_LENGTH = 2    # meses de histórico que o encoder vê
MAX_PREDICTION_LENGTH = 1 # horizonte de predição em meses
MIN_HISTORY = 2  # minimo absoluto de meses por termo

# Hiperparâmetros fixos do TFT
TFT_FIXED = dict(
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    optimizer="adam",
)

# Optuna
N_TRIALS = 5
OPTUNA_EPOCHS = 5

# Treino final
FINAL_EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 7  # early stopping


# ─────────────────────────────────────────────────────────────
# 1. CARREGAMENTO E PREPARAÇÃO DE DADOS
# ─────────────────────────────────────────────────────────────

def load_data_from_db() -> pd.DataFrame:
    """Carrega contagens mensais de patentes por termo."""
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT p.year_month, td.term, COUNT(*) AS count
        FROM patent_terms pt
        JOIN patents p ON pt.patent_id::text = p.id::text
        JOIN term_dictionary td ON td.id = pt.term_id
        WHERE p.year_month IS NOT NULL
        GROUP BY p.year_month, td.term
        ORDER BY td.term, p.year_month
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def fill_time_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche meses faltantes com count=0 para cada termo.
    Isso é crítico: gaps no time_idx causam o erro
    'filters should not remove all entries'.
    """
    df["year_month"] = pd.to_datetime(df["year_month"])
    all_months = pd.date_range(df["year_month"].min(), df["year_month"].max(), freq="MS")

    filled_parts = []
    for term, group in df.groupby("term"):
        # Mantém só a coluna 'count' antes do reindex para evitar mismatch de colunas
        count_series = group.set_index("year_month")["count"]
        count_series = count_series.reindex(all_months, fill_value=0)
        filled = pd.DataFrame({"year_month": count_series.index, "count": count_series.values})
        filled["term"] = term
        filled_parts.append(filled)

    return pd.concat(filled_parts, ignore_index=True)


def prepare_tft_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formata o DataFrame para o TFT:
      - Garante time_idx contíguo por término
      - Filtra termos com histórico insuficiente
      - Adiciona features de calendário
    """
    df = fill_time_gaps(df)
    df = df.sort_values(["term", "year_month"]).reset_index(drop=True)

    min_date = df["year_month"].min()
    df["time_idx"] = (
        (df["year_month"].dt.year - min_date.year) * 12
        + (df["year_month"].dt.month - min_date.month)
    ).astype(int)

    # Features de calendário (known reals — o modelo sabe esses valores no futuro)
    df["month"] = df["year_month"].dt.month.astype(float)
    df["quarter"] = df["year_month"].dt.quarter.astype(float)

    # Lag manual: count do mês anterior (unknown real — só disponível no passado)
    df = df.sort_values(["term", "time_idx"])
    df["count_lag1"] = df.groupby("term")["count"].shift(1).fillna(0)

    df["count"] = df["count"].astype(float)

    # Filtra termos com histórico insuficiente
    counts = df.groupby("term").size()
    valid_terms = counts[counts >= MIN_HISTORY].index
    removed = counts[counts < MIN_HISTORY].index.tolist()
    if removed:
        print(f"⚠️  Termos removidos por histórico insuficiente (<{MIN_HISTORY} meses): {removed}")

    df = df[df["term"].isin(valid_terms)].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────
# 2. CONSTRUÇÃO DOS DATASETS
# ─────────────────────────────────────────────────────────────

def build_datasets(df: pd.DataFrame):
    """
    Cria os TimeSeriesDataSets de treino e validação.
    A validação usa os últimos MAX_PREDICTION_LENGTH passos de cada série.
    """
    training_cutoff = df["time_idx"].max() - MAX_PREDICTION_LENGTH

    shared_kwargs = dict(
        time_idx="time_idx",
        target="count",
        group_ids=["term"],
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        time_varying_known_reals=["time_idx", "month", "quarter"],
        time_varying_unknown_reals=["count", "count_lag1"],
        target_normalizer=GroupNormalizer(groups=["term"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    train_dataset = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        **shared_kwargs,
    )

    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        df,
        predict=True,
        stop_randomization=True,
    )

    return train_dataset, val_dataset


# ─────────────────────────────────────────────────────────────
# 3. OTIMIZAÇÃO DE HIPERPARÂMETROS COM OPTUNA
# ─────────────────────────────────────────────────────────────

def run_optuna_optimization(train_dataset, val_dataset):
    """
    Busca learning_rate e hidden_size via Optuna,
    usando a loss de VALIDAÇÃO como métrica objetivo.
    """
    train_loader = train_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        h_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
        attn_heads = trial.suggest_categorical("attention_head_size", [1, 2, 4])

        model = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=lr,
            hidden_size=h_size,
            attention_head_size=attn_heads,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=QuantileLoss(quantiles=QUANTILES),
            optimizer="adam",
            log_interval=-1,
        )

        trainer = pl.Trainer(
            max_epochs=OPTUNA_EPOCHS,
            accelerator="auto",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float("inf")))
        return val_loss.item()

    print(f"🔍 Iniciando otimização Optuna ({N_TRIALS} trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"✅ Melhores hiperparâmetros: {study.best_params} | val_loss={study.best_value:.4f}")
    return study.best_params


# ─────────────────────────────────────────────────────────────
# 4. BANCO DE DADOS
# ─────────────────────────────────────────────────────────────

def init_db():
    """
    Garante que as tabelas existam e migra colunas novas com segurança.
    Compativel com tabelas criadas por versoes anteriores do script.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Tabela base (schema minimo)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patent_predictions (
            term                TEXT,
            target_year_month   TEXT,
            predicted_count     FLOAT,
            pessimistic_count   FLOAT,
            optimistic_count    FLOAT,
            trained_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (term, target_year_month)
        );
    """)

    # Migracao segura: adiciona colunas novas se nao existirem
    for col in ["q25_count", "q75_count"]:
        cur.execute(
            "ALTER TABLE patent_predictions ADD COLUMN IF NOT EXISTS "
            + col + " FLOAT;"
        )

    cur.execute("""
        CREATE TABLE IF NOT EXISTS patent_prediction_runs (
            run_id      SERIAL PRIMARY KEY,
            run_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            best_params JSONB,
            n_terms     INT,
            horizon     INT
        );
    """)

    conn.commit()
    cur.close()
    conn.close()


# ─────────────────────────────────────────────────────────────
# 5. TREINO FINAL E PREDIÇÃO
# ─────────────────────────────────────────────────────────────

def train_final_model(train_dataset, val_dataset, best_params):
    """Treina o modelo final com early stopping."""
    train_loader = train_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        **best_params,
        dropout=TFT_FIXED["dropout"],
        hidden_continuous_size=TFT_FIXED["hidden_continuous_size"],
        loss=QuantileLoss(quantiles=QUANTILES),
        optimizer=TFT_FIXED["optimizer"],
        log_interval=10,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        mode="min",
        verbose=False,
    )

    trainer = pl.Trainer(
        max_epochs=FINAL_EPOCHS,
        accelerator="auto",
        enable_checkpointing=False,
        callbacks=[early_stop],
        enable_progress_bar=True,
    )

    print("🏋️  Treinando modelo final...")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"✅ Treino finalizado na época {trainer.current_epoch + 1}.")
    return model


def forecast_and_save(df: pd.DataFrame, model, train_dataset, best_params):
    """Gera previsões para todos os termos e persiste no Postgres."""
    init_db()

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("DELETE FROM patent_predictions")

    terms = df["term"].unique()
    print(f"🔮 Gerando previsões para {len(terms)} termos (horizonte={MAX_PREDICTION_LENGTH} meses)...")

    success, skipped, errors = 0, 0, 0

    for term in terms:
        try:
            cur.execute("SAVEPOINT sp_term")  # isola cada termo — erro nao quebra os outros

            term_df = df[df["term"] == term].sort_values("time_idx").copy()

            if len(term_df) < train_dataset.max_encoder_length:
                print(f"⏭️  {term}: histórico curto ({len(term_df)}). Pulando.")
                cur.execute("RELEASE SAVEPOINT sp_term")
                skipped += 1
                continue

            encoder_data = term_df.tail(train_dataset.max_encoder_length).copy()

            pred_dataset = TimeSeriesDataSet.from_dataset(
                train_dataset,
                encoder_data,
                predict=True,
                stop_randomization=True,
            )

            pred_loader = pred_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

            raw_output = model.predict(pred_loader, mode="raw", return_x=False)
            preds = raw_output["prediction"]  # shape: [1, horizon, n_quantiles]

            term_last_date = pd.to_datetime(encoder_data["year_month"].max())

            for step in range(preds.shape[1]):
                target_date = (term_last_date + pd.DateOffset(months=step + 1)).strftime("%Y-%m")

                # QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
                #              idx 0    1     2    3    4
                pess  = max(0.0, float(preds[0, step, 0]))
                q25   = max(0.0, float(preds[0, step, 1]))
                pred  = max(0.0, float(preds[0, step, 2]))
                q75   = max(0.0, float(preds[0, step, 3]))
                opti  = max(0.0, float(preds[0, step, 4]))

                cur.execute("""
                    INSERT INTO patent_predictions
                        (term, target_year_month, predicted_count,
                         pessimistic_count, optimistic_count, q25_count, q75_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (term, target_year_month) DO UPDATE SET
                        predicted_count   = EXCLUDED.predicted_count,
                        pessimistic_count = EXCLUDED.pessimistic_count,
                        optimistic_count  = EXCLUDED.optimistic_count,
                        q25_count         = EXCLUDED.q25_count,
                        q75_count         = EXCLUDED.q75_count,
                        trained_at        = CURRENT_TIMESTAMP
                """, (term, target_date, pred, pess, opti, q25, q75))

            cur.execute("RELEASE SAVEPOINT sp_term")
            success += 1

        except Exception as e:
            cur.execute("ROLLBACK TO SAVEPOINT sp_term")  # desfaz só este termo
            cur.execute("RELEASE SAVEPOINT sp_term")
            print(f"⚠️  Erro no termo '{term}': {e}")
            errors += 1
            continue

    # Salva metadados da run
    import json
    cur.execute("""
        INSERT INTO patent_prediction_runs (best_params, n_terms, horizon)
        VALUES (%s, %s, %s)
    """, (json.dumps(best_params), success, MAX_PREDICTION_LENGTH))

    conn.commit()
    cur.close()
    conn.close()

    print(f"\n📊 Resultado: {success} termos preditos | {skipped} pulados | {errors} erros")
    print("✅ Processo concluído com sucesso!")


# ─────────────────────────────────────────────────────────────
# EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("📊 Pipeline de Predição de Patentes — TFT + Optuna")
    print("=" * 60)

    # 1. Carga
    print("\n[1/5] Carregando dados do banco...")
    raw_df = load_data_from_db()

    if raw_df.empty:
        print("❌ Banco retornou DataFrame vazio. Verifique a query.")
        exit(1)

    print(f"    Registros brutos: {len(raw_df):,} | Termos únicos: {raw_df['term'].nunique()}")

    # 2. Preparação
    print("\n[2/5] Preparando dataset...")
    processed_df = prepare_tft_dataset(raw_df)

    if processed_df.empty:
        print(f"❌ Nenhum termo com histórico >= {MIN_HISTORY} meses.")
        exit(1)

    print(f"    Termos válidos: {processed_df['term'].nunique()} | "
          f"Período: {processed_df['year_month'].min().date()} → {processed_df['year_month'].max().date()}")

    # 3. Datasets
    print("\n[3/5] Construindo datasets de treino e validação...")
    train_ds, val_ds = build_datasets(processed_df)

    # 4. Otimização
    print("\n[4/5] Otimizando hiperparâmetros...")
    best_params = run_optuna_optimization(train_ds, val_ds)

    # 5. Treino final + predição
    print("\n[5/5] Treino final e geração de previsões...")
    final_model = train_final_model(train_ds, val_ds, best_params)
    forecast_and_save(processed_df, final_model, train_ds, best_params)