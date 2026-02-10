import hashlib
import itertools
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from recommenders.datasets import movielens


# 预设的标签字典，可按业务需求调整
GENRE_TO_PRODUCT_TAG = {
    "Action": "高额度周转贷",
    "Adventure": "创业扩张贷",
    "Animation": "家庭消费贷",
    "Children's": "教育分期贷",
    "Comedy": "生活消费贷",
    "Crime": "风控重点产品",
    "Documentary": "政策支持贷",
    "Drama": "综合经营贷",
    "Fantasy": "创新项目贷",
    "Film-Noir": "高收益专项贷",
    "Horror": "高风险特色贷",
    "Musical": "文化旅游贷",
    "Mystery": "调研定制贷",
    "Romance": "婚庆装修贷",
    "Sci-Fi": "科技研发贷",
    "Thriller": "应急备用贷",
    "War": "制造业升级贷",
    "Western": "传统实业贷",
}

GENRE_TO_RISK_SCORE = {
    "Documentary": 0.2,
    "Children's": 0.25,
    "Comedy": 0.3,
    "Drama": 0.35,
    "Romance": 0.4,
    "Animation": 0.4,
    "Musical": 0.45,
    "Fantasy": 0.5,
    "Mystery": 0.55,
    "Sci-Fi": 0.6,
    "Action": 0.65,
    "War": 0.65,
    "Crime": 0.7,
    "Thriller": 0.7,
    "Adventure": 0.75,
    "Western": 0.75,
    "Horror": 0.85,
    "Film-Noir": 0.85,
}

_FALLBACK_GENRES = list(GENRE_TO_PRODUCT_TAG.keys()) or ["Drama", "Comedy", "Action"]


def _normalize_scores(scores: List[float]) -> float:
    if not scores:
        return 0.5
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _risk_label(score: float) -> str:
    if score < 0.4:
        return "低风险偏好"
    if score < 0.6:
        return "中等风险偏好"
    return "高风险偏好"


def _hash_to_asset_level(user_id: str) -> Tuple[str, float]:
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:4], 16) % 3
    if bucket == 0:
        return "成长型客户", 0.35
    if bucket == 1:
        return "稳健型客户", 0.55
    return "高潜力客户", 0.8


@lru_cache()
def load_interactions(size: str = "100k") -> pd.DataFrame:
    """加载评分数据（交互行为）。"""
    df = movielens.load_pandas_df(size=size)
    df = df.rename(
        columns={"userID": "user_id", "itemID": "item_id", "rating": "rating", "timestamp": "timestamp"}
    )
    return df


@lru_cache()
def load_items(size: str = "100k") -> pd.DataFrame:
    """加载贷款产品（电影）信息，若未提供则构造占位数据。"""
    try:
        items = movielens.load_item_df(size=size)
    except Exception:
        items = None

    if items is None or items.empty:
        interactions = load_interactions(size=size)
        return _build_fallback_items(interactions)

    items = items.rename(columns={"movieID": "item_id", "title": "title", "genres": "genres"})
    return items[["item_id", "title", "genres"]]


def _split_genres(genres: str) -> List[str]:
    if pd.isna(genres):
        return []
    return [g.strip() for g in genres.split("|") if g]


def _build_fallback_items(interactions: pd.DataFrame) -> pd.DataFrame:
    """当外部电影元数据不可用时，基于交互生成占位贷款产品信息。"""
    unique_items = interactions["item_id"].dropna().unique()
    genre_cycle = itertools.cycle(_FALLBACK_GENRES)
    rows = []
    for idx, item_id in enumerate(unique_items, start=1):
        first_genre = next(genre_cycle)
        second_genre = next(genre_cycle)
        rows.append(
            {
                "item_id": item_id,
                "title": f"零售贷款产品 {idx:04d}",
                "genres": f"{first_genre}|{second_genre}",
            }
        )
    return pd.DataFrame(rows)


def build_item_profiles(items: pd.DataFrame) -> pd.DataFrame:
    """为贷款产品生成标签与风险偏好。"""
    rows = []
    for _, row in items.iterrows():
        genres = _split_genres(row["genres"])
        product_tags = sorted({GENRE_TO_PRODUCT_TAG.get(g, "通用偏好") for g in genres})
        risk_scores = [GENRE_TO_RISK_SCORE.get(g, 0.5) for g in genres]
        risk_score = _normalize_scores(risk_scores)
        rows.append(
            {
                "item_id": row["item_id"],
                "title": row["title"],
                "genres": genres,
                "product_tags": product_tags or ["通用偏好"],
                "item_risk_score": risk_score,
                "item_risk_label": _risk_label(risk_score),
            }
        )
    return pd.DataFrame(rows)


def build_user_profiles(interactions: pd.DataFrame, item_profiles: pd.DataFrame) -> pd.DataFrame:
    """根据客户历史行为生成贷款偏好与风险画像。"""
    merged = interactions.merge(item_profiles, on="item_id", how="left")

    aggregated: List[Dict[str, object]] = []
    for user_id, group in merged.groupby("user_id"):
        weighted = group.assign(weight=np.clip(group["rating"], 1.0, 5.0))
        genre_counts: Dict[str, float] = {}
        risk_scores: List[float] = []
        for _, row in weighted.iterrows():
            for tag in row["product_tags"]:
                genre_counts[tag] = genre_counts.get(tag, 0.0) + row["weight"]
            risk_scores.append(row["item_risk_score"])

        top_tags = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        tag_labels = [tag for tag, _ in top_tags] or ["通用偏好"]
        risk_score = _normalize_scores(risk_scores)
        risk_label = _risk_label(risk_score)
        asset_label, asset_score = _hash_to_asset_level(str(user_id))

        aggregated.append(
            {
                "user_id": user_id,
                "top_tags": tag_labels,
                "user_risk_score": risk_score,
                "user_risk_label": risk_label,
                "asset_label": asset_label,
                "asset_score": asset_score,
            }
        )

    profiles = pd.DataFrame(aggregated)
    return profiles


def prepare_datasets(size: str = "100k") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """准备交互数据、客户画像与贷款产品画像。"""
    interactions = load_interactions(size=size)
    items = load_items(size=size)
    item_profiles = build_item_profiles(items)
    user_profiles = build_user_profiles(interactions, item_profiles)
    return interactions, user_profiles, item_profiles
