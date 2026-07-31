"""Feature engineering pipeline for the credit default (inadimplência) prediction case.

The central entry point is :func:`build_features`, which takes the four raw bases
(already loaded as DataFrames) and returns one feature row per row of ``df_target``
(the "current transactions" table — either the development or the test payments base).

Data leakage guard
-------------------
Behavioral features (Section 3) summarize a client's payment history. For a target
row with ``SAFRA_REF = "2024-03"``, only historical records with ``SAFRA_REF`` strictly
before ``"2024-03"`` may be used. This is enforced with ``pandas.merge_asof(...,
allow_exact_matches=False)``, which — for every target row — looks up the most recent
historical snapshot with a safra strictly earlier than the target's, per client. Rows
that share the target's own safra (e.g. two invoices due in the same month) are
therefore never used as history for one another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Ordinal order reflects increasing company size, a reasonable business ordering
# for a "PORTE" (size) category — GRANDE tends to be the lowest-risk end in practice.
PORTE_ORDER = {"PEQUENO": 0, "MEDIO": 1, "GRANDE": 2}

# Categories with no recorded PORTE are encoded as -1 ("unknown"), distinguishable
# from any valid ordinal value instead of silently colliding with PEQUENO (0).
PORTE_UNKNOWN_CODE = -1

CSV_SEP = ";"


def read_csv_semicolon(path: str) -> pd.DataFrame:
    """Read one of the project's raw CSV files (";"-delimited)."""
    return pd.read_csv(path, sep=CSV_SEP)


def _to_safra_date(safra_ref: pd.Series) -> pd.Series:
    """Convert a 'YYYY-MM' safra string column to a Timestamp at the first day of month."""
    return pd.to_datetime(safra_ref + "-01", format="%Y-%m-%d")


def _cadastral_features(df_cadastral: pd.DataFrame) -> pd.DataFrame:
    """Build client-level (one row per ID_CLIENTE) features from base_cadastral."""
    df = df_cadastral.copy()

    # FLAG_PF only ever takes the value 'X' (present) or null (absent) in this base,
    # so it behaves as a boolean marker for "pessoa física" rather than a real category.
    df["flag_pf"] = (df["FLAG_PF"] == "X").astype(int)

    df["porte_encoded"] = df["PORTE"].map(PORTE_ORDER).fillna(PORTE_UNKNOWN_CODE).astype(int)

    # Frequency encoding is computed from the cadastral population itself: it is a
    # static, outcome-independent lookup table of client attributes, so this does not
    # leak target information across the temporal train/test split.
    for col, out_col in [
        ("SEGMENTO_INDUSTRIAL", "segmento_freq"),
        ("DOMINIO_EMAIL", "dominio_freq"),
        ("CEP_2_DIG", "cep_freq"),
        ("DDD", "ddd_freq"),
    ]:
        freq_map = df[col].value_counts(normalize=True, dropna=True)
        df[out_col] = df[col].map(freq_map).fillna(0.0)

    df["_data_cadastro"] = pd.to_datetime(df["DATA_CADASTRO"])

    return df[
        [
            "ID_CLIENTE",
            "flag_pf",
            "porte_encoded",
            "segmento_freq",
            "dominio_freq",
            "cep_freq",
            "ddd_freq",
            "_data_cadastro",
        ]
    ]


def _monthly_info_features(df_info: pd.DataFrame) -> pd.DataFrame:
    """Select the monthly (ID_CLIENTE, SAFRA_REF) info features from base_info."""
    return df_info[["ID_CLIENTE", "SAFRA_REF", "RENDA_MES_ANTERIOR", "NO_FUNCIONARIOS"]].rename(
        columns={"RENDA_MES_ANTERIOR": "renda_mes_anterior", "NO_FUNCIONARIOS": "no_funcionarios"}
    )


def _prepare_payment_history(df_pagamentos_hist: pd.DataFrame) -> pd.DataFrame:
    """Compute per-transaction delay/default fields from a payments-with-outcome table."""
    hist = df_pagamentos_hist.copy()
    hist["DATA_PAGAMENTO"] = pd.to_datetime(hist["DATA_PAGAMENTO"])
    hist["DATA_VENCIMENTO"] = pd.to_datetime(hist["DATA_VENCIMENTO"])
    hist["_safra_date"] = _to_safra_date(hist["SAFRA_REF"])

    hist["dias_atraso"] = (hist["DATA_PAGAMENTO"] - hist["DATA_VENCIMENTO"]).dt.days
    hist["inadimplente"] = ((hist["dias_atraso"] >= 5) | hist["DATA_PAGAMENTO"].isna()).astype(int)
    return hist


def _cumulative_monthly_history(hist: pd.DataFrame) -> pd.DataFrame:
    """Aggregate payment history to one row per (ID_CLIENTE, safra), with running-total
    statistics ("as of, and including, this safra") that can later be looked up with
    an as-of merge for any strictly-later target safra.
    """
    monthly = (
        hist.groupby(["ID_CLIENTE", "_safra_date"])
        .agg(
            n_transacoes=("inadimplente", "size"),
            n_inadimplencias=("inadimplente", "sum"),
            sum_atraso=("dias_atraso", "sum"),
            count_atraso=("dias_atraso", "count"),
            sumsq_atraso=("dias_atraso", lambda s: float(np.square(s).sum())),
            max_atraso=("dias_atraso", "max"),
            sum_valor=("VALOR_A_PAGAR", "sum"),
            count_valor=("VALOR_A_PAGAR", "count"),
            sum_taxa=("TAXA", "sum"),
            count_taxa=("TAXA", "count"),
        )
        .reset_index()
        .sort_values(["ID_CLIENTE", "_safra_date"])
    )

    cum_sum_cols = [
        "n_transacoes",
        "n_inadimplencias",
        "sum_atraso",
        "count_atraso",
        "sumsq_atraso",
        "sum_valor",
        "count_valor",
        "sum_taxa",
        "count_taxa",
    ]
    grouped = monthly.groupby("ID_CLIENTE")
    monthly[cum_sum_cols] = grouped[cum_sum_cols].cumsum()
    # max_atraso: pandas cummax skips NaN (keeps running max from prior non-null values).
    monthly["max_atraso"] = grouped["max_atraso"].cummax()

    return monthly


def _behavioral_features(df_pagamentos_hist: pd.DataFrame, target_keys: pd.DataFrame) -> pd.DataFrame:
    """For each (ID_CLIENTE, safra_date) row in target_keys, compute behavioral
    features using only historical transactions with a strictly earlier safra.

    target_keys must have columns ['ID_CLIENTE', 'safra_date'] and preserve the
    original row order/index of df_target (duplicated keys — e.g. two invoices due
    in the same month — are supported and each receive the same historical snapshot).
    """
    hist = _prepare_payment_history(df_pagamentos_hist)
    monthly_cum = _cumulative_monthly_history(hist)

    left = target_keys.reset_index().rename(columns={"index": "_orig_index"})
    left_sorted = left.sort_values("safra_date")

    right_sorted = monthly_cum.rename(columns={"_safra_date": "safra_date"}).sort_values("safra_date")
    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        on="safra_date",
        by="ID_CLIENTE",
        direction="backward",
        allow_exact_matches=False,  # strictly earlier safra only -> no leakage
    )
    merged = merged.sort_values("_orig_index").set_index("_orig_index")
    merged.index.name = None

    out = pd.DataFrame(index=merged.index)
    out["qtd_pagamentos_anteriores"] = merged["n_transacoes"].fillna(0).astype(int)
    out["qtd_inadimplencias_anteriores"] = merged["n_inadimplencias"].fillna(0).astype(int)
    out["flag_cliente_novo"] = (out["qtd_pagamentos_anteriores"] == 0).astype(int)

    with np.errstate(invalid="ignore", divide="ignore"):
        out["taxa_inadimplencia_hist"] = merged["n_inadimplencias"] / merged["n_transacoes"]
        media_atraso = merged["sum_atraso"] / merged["count_atraso"]
        # Sample variance from sum/sum-of-squares/count; clipped at 0 to absorb
        # floating point noise when the true variance is ~0.
        var_atraso = (merged["sumsq_atraso"] - merged["count_atraso"] * media_atraso**2) / (
            merged["count_atraso"] - 1
        )
    out["media_atraso_dias"] = media_atraso
    out["std_atraso_dias"] = np.sqrt(var_atraso.clip(lower=0))
    out.loc[merged["count_atraso"] <= 1, "std_atraso_dias"] = np.nan
    out["max_atraso_dias"] = merged["max_atraso"]

    with np.errstate(invalid="ignore", divide="ignore"):
        out["valor_medio_hist"] = merged["sum_valor"] / merged["count_valor"]
        out["taxa_media_hist"] = merged["sum_taxa"] / merged["count_taxa"]

    return out


def _current_transaction_features(df_target: pd.DataFrame, renda: pd.Series) -> pd.DataFrame:
    """Features derived directly from the current (target) transaction row."""
    out = pd.DataFrame(index=df_target.index)
    data_emissao = pd.to_datetime(df_target["DATA_EMISSAO_DOCUMENTO"])
    data_vencimento = pd.to_datetime(df_target["DATA_VENCIMENTO"])

    out["valor_a_pagar"] = df_target["VALOR_A_PAGAR"]
    out["taxa"] = df_target["TAXA"]
    out["dia_vencimento"] = data_vencimento.dt.day
    out["dias_emissao_vencimento"] = (data_vencimento - data_emissao).dt.days

    renda_safe = renda.replace(0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = out["valor_a_pagar"] / renda_safe
    out["ratio_valor_renda"] = ratio.replace([np.inf, -np.inf], np.nan)

    return out


def build_features(
    df_cadastral: pd.DataFrame,
    df_info: pd.DataFrame,
    df_pagamentos_hist: pd.DataFrame,
    df_target: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full feature matrix for the rows in ``df_target``.

    Parameters
    ----------
    df_cadastral:
        base_cadastral.csv loaded as a DataFrame (one row per ID_CLIENTE).
    df_info:
        base_info.csv loaded as a DataFrame (one row per ID_CLIENTE, SAFRA_REF).
    df_pagamentos_hist:
        The payments table used as the source of behavioral (historical) features.
        Must include a ``DATA_PAGAMENTO`` column (i.e. realized outcomes), since
        historical default/delay stats are computed from it. In practice this is
        always ``base_pagamentos_desenvolvimento`` — both when building features
        for the development set itself (each row's own past) and for the test set
        (whose entire history lies in the development period).
    df_target:
        The transactions to build features for — one output row per input row,
        in the same order. Only ``ID_CLIENTE``, ``SAFRA_REF``, ``VALOR_A_PAGAR``,
        ``TAXA``, ``DATA_EMISSAO_DOCUMENTO`` and ``DATA_VENCIMENTO`` are used, so the
        same function works whether or not ``df_target`` has a ``DATA_PAGAMENTO``
        column (the label itself is intentionally not touched here).

    Returns
    -------
    pd.DataFrame
        One row per row of ``df_target`` (same index), with all engineered features.
    """
    target_safra_date = _to_safra_date(df_target["SAFRA_REF"])

    features = pd.DataFrame(index=df_target.index)
    features["ID_CLIENTE"] = df_target["ID_CLIENTE"].values
    features["SAFRA_REF"] = df_target["SAFRA_REF"].values

    # --- 1. Cadastral (static, merged by ID_CLIENTE) ---------------------------------
    cad = _cadastral_features(df_cadastral)
    cad_merged = df_target[["ID_CLIENTE"]].merge(cad, on="ID_CLIENTE", how="left")
    for col in ["flag_pf", "porte_encoded", "segmento_freq", "dominio_freq", "cep_freq", "ddd_freq"]:
        features[col] = cad_merged[col].values
    meses_desde_cadastro = (
        (target_safra_date.dt.year.values - cad_merged["_data_cadastro"].dt.year.values) * 12
        + (target_safra_date.dt.month.values - cad_merged["_data_cadastro"].dt.month.values)
    )
    features["meses_desde_cadastro"] = meses_desde_cadastro

    # --- 2. Monthly info (merged by ID_CLIENTE + SAFRA_REF) --------------------------
    info_merged = df_target[["ID_CLIENTE", "SAFRA_REF"]].merge(
        _monthly_info_features(df_info), on=["ID_CLIENTE", "SAFRA_REF"], how="left"
    )
    features["renda_mes_anterior"] = info_merged["renda_mes_anterior"].values
    features["no_funcionarios"] = info_merged["no_funcionarios"].values

    # --- 3. Behavioral, strictly-past history (leakage-safe as-of merge) -------------
    target_keys = pd.DataFrame(
        {"ID_CLIENTE": df_target["ID_CLIENTE"].values, "safra_date": target_safra_date.values},
        index=df_target.index,
    )
    behavior = _behavioral_features(df_pagamentos_hist, target_keys)
    for col in behavior.columns:
        features[col] = behavior[col].values

    # --- 4. Current transaction ------------------------------------------------------
    current = _current_transaction_features(df_target, renda=info_merged["renda_mes_anterior"])
    for col in current.columns:
        features[col] = current[col].values

    return features
