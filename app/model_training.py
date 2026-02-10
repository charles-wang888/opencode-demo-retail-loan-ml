from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from recommenders.models.sar import SAR


USER_COL = "user_id"
ITEM_COL = "item_id"
RATING_COL = "rating"
TIMESTAMP_COL = "timestamp"

FEATURE_LABELS: Dict[str, str] = {
    "num__asset_score": "客户资质评分",
    "num__user_risk_score": "客户风险承受力",
    "num__item_risk_score": "贷款产品风险等级",
    "num__genre_overlap": "需求匹配标签数量",
    "num__risk_gap": "客户与产品风险差值",
    "num__recent_interaction": "近期贷款行为活跃度",
    "cat__asset_label_高潜力客户": "高潜力客户身份",
    "cat__asset_label_稳健型客户": "稳健型客户身份",
    "cat__asset_label_成长型客户": "成长型客户身份",
    "cat__user_risk_label_低风险偏好": "低风险偏好客户",
    "cat__user_risk_label_中等风险偏好": "中等风险偏好客户",
    "cat__user_risk_label_高风险偏好": "高风险偏好客户",
    "cat__item_risk_label_低风险偏好": "低风险贷款产品",
    "cat__item_risk_label_中等风险偏好": "中等风险贷款产品",
    "cat__item_risk_label_高风险偏好": "高风险贷款产品",
}


@dataclass
class PropensityModel:
    pipeline: Pipeline
    feature_names: List[str]
    roc_auc: float
    report: str
    training_summary: Dict[str, Any]

    def explain_instance(self, row: pd.Series) -> List[Tuple[str, float]]:
        """计算单个客户-贷款组合的特征贡献（线性模型）。"""
        model: LogisticRegression = self.pipeline.named_steps["model"]  # type: ignore[assignment]
        preprocessor: ColumnTransformer = self.pipeline.named_steps["preprocessor"]  # type: ignore[assignment]
        transformed = preprocessor.transform(pd.DataFrame([row[self.feature_names]]))
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()  # type: ignore[assignment]
        transformed = np.asarray(transformed).reshape(-1)
        coefs = model.coef_.reshape(-1)
        contributions = transformed * coefs
        feature_names = preprocessor.get_feature_names_out()
        return sorted(zip(feature_names, contributions), key=lambda x: abs(x[1]), reverse=True)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(features)[:, 1]


@dataclass
class RecommendationModel:
    model: SAR
    training_summary: Dict[str, Any]

    def recommend(self, user_id: str, top_k: int = 5) -> pd.DataFrame:
        return self.model.recommend_k_items(
            pd.DataFrame([{USER_COL: user_id}]),
            top_k=top_k,
            remove_seen=True,
        )

    def similar_items(self, item_id: str, top_k: int = 5) -> pd.DataFrame:
        if hasattr(self.model, "get_item_based_neighbors"):
            return self.model.get_item_based_neighbors(item_id, top_k=top_k)
        if hasattr(self.model, "get_item_based_recommendations"):
            return self.model.get_item_based_recommendations(item_id, top_k=top_k)
        return pd.DataFrame(columns=[ITEM_COL, "correlation"])


def _build_feature_frame(
    interactions: pd.DataFrame,
    user_profiles: pd.DataFrame,
    item_profiles: pd.DataFrame,
) -> pd.DataFrame:
    merged = (
        interactions.merge(user_profiles, on=USER_COL, how="left")
        .merge(item_profiles, on=ITEM_COL, how="left")
        .dropna(subset=["top_tags", "product_tags"])
    )

    merged["label"] = (merged[RATING_COL] >= 4).astype(int)
    merged["genre_overlap"] = merged.apply(
        lambda row: len(set(row["top_tags"]).intersection(set(row["product_tags"]))), axis=1
    )
    merged["risk_gap"] = merged["user_risk_score"] - merged["item_risk_score"]
    merged["recent_interaction"] = merged[TIMESTAMP_COL] / merged[TIMESTAMP_COL].max()
    return merged


def train_propensity_model(
    interactions: pd.DataFrame, user_profiles: pd.DataFrame, item_profiles: pd.DataFrame
) -> PropensityModel:
    features = _build_feature_frame(interactions, user_profiles, item_profiles)

    feature_cols = [
        "asset_score",
        "user_risk_score",
        "item_risk_score",
        "genre_overlap",
        "risk_gap",
        "recent_interaction",
        "asset_label",
        "user_risk_label",
        "item_risk_label",
    ]

    X = features[feature_cols]
    y = features["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numeric_cols = ["asset_score", "user_risk_score", "item_risk_score", "genre_overlap", "risk_gap", "recent_interaction"]
    categorical_cols = ["asset_label", "user_risk_label", "item_risk_label"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=500, solver="lbfgs")),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, digits=3)

    accuracy = float((y_pred == y_test).mean())
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

    fpr, tpr, _thresholds = roc_curve(y_test, y_pred_prob)
    roc_curve_df = pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr})

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_curve_df = pd.DataFrame({"recall": recall_curve, "precision": precision_curve})

    cm = confusion_matrix(y_test, y_pred)
    confusion_df = pd.DataFrame(
        cm,
        index=pd.Index(["实际负类", "实际正类"], name="实际"),
        columns=pd.Index(["预测负类", "预测正类"], name="预测"),
    )

    model: LogisticRegression = pipeline.named_steps["model"]  # type: ignore[assignment]
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]  # type: ignore[assignment]
    feature_names_out = preprocessor.get_feature_names_out()
    coefs = model.coef_.reshape(-1)
    feature_weights = (
        pd.DataFrame({"feature": feature_names_out, "weight": coefs})
        .assign(abs_weight=lambda df: df["weight"].abs())
        .sort_values("abs_weight", ascending=False)
    )
    feature_weights["label"] = feature_weights["feature"].apply(lambda name: FEATURE_LABELS.get(name, name))

    threshold_metrics = []
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        preds_thr = (y_pred_prob >= thr).astype(int)
        thr_precision = precision_score(y_test, preds_thr, zero_division=0)
        thr_recall = recall_score(y_test, preds_thr, zero_division=0)
        thr_accuracy = float((preds_thr == y_test).mean())
        thr_f1 = 0.0 if thr_precision + thr_recall == 0 else 2 * (thr_precision * thr_recall) / (thr_precision + thr_recall)
        threshold_metrics.append(
            {
                "threshold": float(thr),
                "precision": thr_precision,
                "recall": thr_recall,
                "accuracy": thr_accuracy,
                "f1": thr_f1,
            }
        )
    threshold_metrics_df = pd.DataFrame(threshold_metrics)

    test_results = X_test.reset_index(drop=True).copy()
    test_results = test_results.assign(
        label=y_test.reset_index(drop=True),
        probability=y_pred_prob,
    )
    test_results["prediction"] = (test_results["probability"] >= 0.5).astype(int)
    segment_rows: List[Dict[str, Any]] = []
    segment_columns = {
        "客户层级": "asset_label",
        "风险偏好": "user_risk_label",
    }
    for dimension, column in segment_columns.items():
        if column not in test_results:
            continue
        for segment_value, group in test_results.groupby(column):
            if group.empty:
                continue
            seg_precision = precision_score(group["label"], group["prediction"], zero_division=0)
            seg_recall = recall_score(group["label"], group["prediction"], zero_division=0)
            seg_accuracy = float((group["label"] == group["prediction"]).mean())
            seg_f1 = 0.0 if seg_precision + seg_recall == 0 else 2 * (seg_precision * seg_recall) / (seg_precision + seg_recall)
            segment_rows.append(
                {
                    "分群维度": dimension,
                    "分群标签": str(segment_value),
                    "样本数": int(len(group)),
                    "平均概率": float(group["probability"].mean()),
                    "准确率": seg_accuracy,
                    "精确率": seg_precision,
                    "召回率": seg_recall,
                    "F1 分数": seg_f1,
                }
            )
    segment_metrics_df = pd.DataFrame(segment_rows)

    dataset_info = {
        "样本总数": f"{len(features)}",
        "训练集大小": f"{len(X_train)}",
        "测试集大小": f"{len(X_test)}",
        "正样本占比": f"{y.mean():.1%}",
        "训练集正样本占比": f"{y_train.mean():.1%}",
    }
    training_steps = [
        {"title": "构建特征表", "detail": f"结合交互、用户画像与产品画像生成 {len(features)} 条样本"},
        {"title": "划分训练/测试集", "detail": f"按 8:2 划分，随机种子 42，保持标签分布稳定"},
        {"title": "特征预处理", "detail": "数值特征标准化，类别特征进行 One-Hot 编码（将分类变量拆分为0/1特征）"},
        {"title": "训练逻辑回归模型", "detail": "使用 lbfgs 求解器（拟牛顿法，收敛更稳定），最大迭代次数 500"},
        {"title": "性能评估", "detail": f"ROC-AUC（衡量模型区分正负样本能力）={roc_auc:.3f}，并生成分类报告"},
    ]

    training_summary = {
        "steps": training_steps,
        "dataset_info": dataset_info,
        "feature_weights": feature_weights.reset_index(drop=True),
        "performance": {
            "准确率": accuracy,
            "精确率": precision,
            "召回率": recall,
            "F1 分数": f1,
        },
        "roc_curve": roc_curve_df,
        "pr_curve": pr_curve_df,
        "confusion_matrix": confusion_df,
        "threshold_metrics": threshold_metrics_df,
        "segment_metrics": segment_metrics_df,
    }

    return PropensityModel(
        pipeline=pipeline,
        feature_names=feature_cols,
        roc_auc=roc_auc,
        report=report,
        training_summary=training_summary,
    )


def train_recommendation_model(interactions: pd.DataFrame) -> RecommendationModel:
    model = SAR(
        col_user=USER_COL,
        col_item=ITEM_COL,
        col_rating=RATING_COL,
        col_timestamp=TIMESTAMP_COL,
        similarity_type="jaccard",
    )
    model.fit(interactions)

    interaction_count = len(interactions)
    user_count = interactions[USER_COL].nunique()
    item_count = interactions[ITEM_COL].nunique()
    avg_unique_items = interactions.groupby(USER_COL)[ITEM_COL].nunique().mean()
    time_span = ""
    if TIMESTAMP_COL in interactions:
        try:
            timestamps = pd.to_datetime(interactions[TIMESTAMP_COL], unit="s")
            time_diff = (timestamps.max() - timestamps.min()).days
            base_date = pd.Timestamp("2025-01-01")
            min_date = base_date
            max_date = base_date + pd.Timedelta(days=time_diff)
            time_span = f"{min_date.date()} - {max_date.date()}"
        except Exception:
            time_span = "时间戳格式无法解析"

    top_items = (
        interactions.groupby(ITEM_COL)[RATING_COL]
        .count()
        .rename("交互次数")
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    training_steps = [
        {"title": "整理交互矩阵", "detail": f"包含 {interaction_count} 条评分，{user_count} 位客户与 {item_count} 个产品"},
        {"title": "配置 SAR 参数", "detail": "相似度度量选择 Jaccard（交互集合相似度），包含时间衰减因子"},
        {"title": "构建稀疏矩阵", "detail": "根据客户-产品评分生成稀疏矩阵用于相似度计算"},
        {"title": "模型训练", "detail": "拟合相似度矩阵，准备进行召回与相似产品检索"},
    ]

    stats = {
        "客户数量": f"{user_count}",
        "贷款产品数量": f"{item_count}",
        "交互行为数量": f"{interaction_count}",
        "平均客户触达产品数": f"{avg_unique_items:.1f}",
        "数据时间跨度": time_span or "未提供",
    }

    training_summary = {
        "steps": training_steps,
        "stats": stats,
        "top_items": top_items,
    }

    return RecommendationModel(model=model, training_summary=training_summary)


def top_feature_contributions(contribs: List[Tuple[str, float]], top_k: int = 5) -> List[Tuple[str, float]]:
    return contribs[:top_k]


def describe_feature(name: str, value: float) -> str:
    label = FEATURE_LABELS.get(name, name)
    return f"{label}: {value:.3f}"
