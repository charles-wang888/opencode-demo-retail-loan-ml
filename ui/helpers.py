"""UI 展示与演示用辅助函数。"""
import json
from typing import List

import numpy as np
import pandas as pd

from app.model_training import (
    PropensityModel,
    describe_feature,
    top_feature_contributions,
)


def prepare_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """将 DataFrame 转为可安全展示的格式（列表/字典等转字符串）。"""
    if df is None or df.empty:
        return df

    def _clean_value(value):
        if isinstance(value, (np.generic,)):
            return value.item()
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return value.isoformat() if hasattr(value, "isoformat") else str(value)
        if isinstance(value, (list, tuple, set)):
            try:
                return json.dumps(list(value), ensure_ascii=False)
            except TypeError:
                return str(list(value))
        if isinstance(value, dict):
            try:
                return json.dumps(value, ensure_ascii=False)
            except TypeError:
                return str(value)
        if isinstance(value, np.ndarray):
            try:
                return json.dumps(value.tolist(), ensure_ascii=False)
            except TypeError:
                return str(value.tolist())
        if pd.isna(value):
            return ""
        return value

    prepared = df.copy()
    for col in prepared.columns:
        if prepared[col].dtype == object:
            prepared[col] = prepared[col].map(_clean_value)
        elif np.issubdtype(prepared[col].dtype, np.number):
            prepared[col] = prepared[col].apply(
                lambda v: v.item() if isinstance(v, np.generic) else v
            )
    return prepared


def compute_recency(interactions: pd.DataFrame) -> pd.DataFrame:
    max_ts = interactions["timestamp"].max()
    agg = interactions.groupby("user_id")["timestamp"].max().to_frame()
    agg["recent_interaction"] = agg["timestamp"] / max_ts
    return agg[["recent_interaction"]]


def build_feature_row(
    user_row: pd.Series,
    item_row: pd.Series,
    recency_lookup: pd.DataFrame,
) -> pd.Series:
    user_id = user_row["user_id"]
    recent_value = (
        recency_lookup.loc[user_id, "recent_interaction"]
        if user_id in recency_lookup.index
        else 0.5
    )
    overlap = len(
        set(user_row["top_tags"]).intersection(set(item_row["product_tags"]))
    )
    return pd.Series({
        "asset_score": user_row["asset_score"],
        "user_risk_score": user_row["user_risk_score"],
        "item_risk_score": item_row["item_risk_score"],
        "genre_overlap": overlap,
        "risk_gap": user_row["user_risk_score"] - item_row["item_risk_score"],
        "recent_interaction": recent_value,
        "asset_label": user_row["asset_label"],
        "user_risk_label": user_row["user_risk_label"],
        "item_risk_label": item_row["item_risk_label"],
    })


def explain_propensity(model: PropensityModel, features: pd.Series) -> List[str]:
    contribs = model.explain_instance(features)
    top = top_feature_contributions(contribs, top_k=5)
    return [describe_feature(n, v) for n, v in top]
