"""
零售贷款智能运营的 MCP 工具：供 Agent 调用加载数据、训练、推荐、预测等。
工具通过 MCP 服务器提供，需要在 opencode 配置中注册 MCP 服务器。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

# 全局会话状态：当前进程内一次对话共享
_state: dict[str, Any] = {
    "datasets": None,
    "propensity_model": None,
    "recommendation_model": None,
}


def _get_state():
    return _state


def set_ui_state(datasets=None, propensity_model=None, recommendation_model=None):
    """供 Gradio UI 在加载数据或训练后同步状态，使 Agent 与界面共享同一数据/模型。"""
    s = _get_state()
    if datasets is not None:
        s["datasets"] = datasets
    if propensity_model is not None:
        s["propensity_model"] = propensity_model
    if recommendation_model is not None:
        s["recommendation_model"] = recommendation_model


def _ensure_datasets():
    """若未加载数据则加载（同步，供工具内调用）。"""
    state = _get_state()
    if state["datasets"] is not None:
        return state["datasets"]
    os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
    from app.data_prep import prepare_datasets
    state["datasets"] = prepare_datasets(size="100k")
    return state["datasets"]


def _ensure_models():
    """若未训练则先加载数据再训练（同步）。"""
    state = _get_state()
    if state["propensity_model"] is not None and state["recommendation_model"] is not None:
        return state["propensity_model"], state["recommendation_model"]
    from app.data_prep import prepare_datasets
    from app.model_training import train_propensity_model, train_recommendation_model
    if state["datasets"] is None:
        state["datasets"] = prepare_datasets(size="100k")
    interactions, user_profiles, item_profiles = state["datasets"]
    state["propensity_model"] = train_propensity_model(interactions, user_profiles, item_profiles)
    state["recommendation_model"] = train_recommendation_model(interactions)
    return state["propensity_model"], state["recommendation_model"]


def _build_feature_row(user_row, item_row, recency_lookup):
    """构建单条特征用于意愿预测。"""
    import pandas as pd
    user_id = user_row["user_id"]
    recent_value = recency_lookup.loc[user_id, "recent_interaction"] if user_id in recency_lookup.index else 0.5
    overlap = len(set(user_row["top_tags"]).intersection(set(item_row["product_tags"])))
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


def _compute_recency(interactions):
    import pandas as pd
    max_ts = interactions["timestamp"].max()
    agg = interactions.groupby("user_id")["timestamp"].max().to_frame()
    agg["recent_interaction"] = agg["timestamp"] / max_ts
    return agg[["recent_interaction"]]


def _text_content(text: str) -> list:
    return [{"type": "text", "text": text}]


async def tool_load_data(args: dict) -> dict:
    """加载示例数据（MovieLens 100k 转贷款场景）。无需参数。"""
    try:
        datasets = _ensure_datasets()
        interactions, user_profiles, item_profiles = datasets
        n_inter = len(interactions)
        n_users = user_profiles["user_id"].nunique()
        n_items = item_profiles["item_id"].nunique()
        msg = (
            f"数据已加载。交互记录: {n_inter:,} 条，客户数: {n_users:,}，贷款产品数: {n_items:,}。"
            " 可继续使用「训练模型」或「推荐」等工具。"
        )
        return {"content": _text_content(msg)}
    except Exception as e:
        return {"content": _text_content(f"加载数据失败: {e}")}


async def tool_train_models(args: dict) -> dict:
    """基于当前已加载数据训练贷款意愿模型与推荐模型。无需参数。若未加载数据会先自动加载。"""
    try:
        propensity, recommender = _ensure_models()
        auc = propensity.roc_auc
        perf = propensity.training_summary.get("performance", {})
        msg = (
            f"训练完成。贷款意愿模型 ROC-AUC: {auc:.3f}；"
            f"准确率: {perf.get('准确率', 0):.1%}，精确率: {perf.get('精确率', 0):.1%}，"
            f"召回率: {perf.get('召回率', 0):.1%}，F1: {perf.get('F1 分数', 0):.1%}。"
            " 可使用「推荐」或「预测意愿」工具。"
        )
        return {"content": _text_content(msg)}
    except Exception as e:
        return {"content": _text_content(f"训练失败: {e}")}


async def tool_get_data_summary(args: dict) -> dict:
    """查看当前数据概况（交互数、客户数、产品数），并返回示例客户ID与产品ID供推荐使用。若无数据会先加载。"""
    try:
        datasets = _ensure_datasets()
        interactions, user_profiles, item_profiles = datasets
        n_inter = len(interactions)
        n_users = user_profiles["user_id"].nunique()
        n_items = item_profiles["item_id"].nunique()
        sample_size = min(20, n_users, n_items)
        user_ids = user_profiles["user_id"].drop_duplicates().head(sample_size).tolist()
        items = item_profiles[["item_id", "title"]].drop_duplicates("item_id").head(sample_size)
        item_ids = items["item_id"].tolist()
        item_titles = items["title"].fillna("").tolist()
        user_list = "、".join(str(u) for u in user_ids)
        product_list = "、".join(f"{i}({t})" if t else str(i) for i, t in zip(item_ids, item_titles))
        msg = (
            f"当前数据：交互记录 {n_inter:,} 条，客户 {n_users:,} 人，贷款产品 {n_items:,} 个。\n"
            f"示例客户ID（前{sample_size}个，用于推荐时填写）：{user_list}\n"
            f"示例产品ID及名称（前{sample_size}个）：{product_list}\n"
            "若需更多ID，可使用 list_customers 或 list_products 工具。"
        )
        return {"content": _text_content(msg)}
    except Exception as e:
        return {"content": _text_content(f"获取概况失败: {e}")}


async def tool_list_customers(args: dict) -> dict:
    """列出当前数据中的客户ID列表，便于做推荐时选择客户。参数：limit（返回数量，默认50）。"""
    try:
        datasets = _ensure_datasets()
        _, user_profiles, _ = datasets
        limit = int(args.get("limit", 50))
        limit = min(max(1, limit), 500)
        user_ids = user_profiles["user_id"].drop_duplicates().head(limit).tolist()
        msg = f"客户ID列表（共{len(user_ids)}个）：" + "、".join(str(u) for u in user_ids)
        return {"content": _text_content(msg)}
    except Exception as e:
        return {"content": _text_content(f"列出客户失败: {e}")}


async def tool_list_products(args: dict) -> dict:
    """列出当前数据中的贷款产品ID及名称，便于做推荐或预测时选择产品。参数：limit（返回数量，默认50）。"""
    try:
        datasets = _ensure_datasets()
        _, _, item_profiles = datasets
        limit = int(args.get("limit", 50))
        limit = min(max(1, limit), 500)
        items = item_profiles[["item_id", "title"]].drop_duplicates("item_id").head(limit)
        lines = [f"{row['item_id']}: {row.get('title', '') or '(无名称)'}" for _, row in items.iterrows()]
        msg = f"贷款产品列表（共{len(lines)}个）：\n" + "\n".join(lines)
        return {"content": _text_content(msg)}
    except Exception as e:
        return {"content": _text_content(f"列出产品失败: {e}")}


async def tool_recommend(args: dict) -> dict:
    """为指定客户做个性化贷款推荐。参数：user_id（客户ID，字符串或整数），top_k（推荐数量，默认5）。"""
    try:
        user_id_raw = args.get("user_id", "")
        top_k = int(args.get("top_k", 5))
        if not user_id_raw:
            return {"content": _text_content("请提供 user_id 参数。")}
        # 清理 user_id：去除前后空格，尝试提取数字
        user_id_raw = str(user_id_raw).strip()
        # 尝试将 user_id 转为整数（数据中 user_id 是整数类型）
        try:
            user_id_int = int(user_id_raw)
            user_id_str = str(user_id_int)  # 统一为字符串表示
        except (ValueError, TypeError):
            user_id_int = None
            user_id_str = user_id_raw
        # 先获取数据，确保与 list_customers 使用相同的数据源
        datasets = _ensure_datasets()
        interactions, user_profiles, item_profiles = datasets
        
        # 确保 user_profiles 的 user_id 列是整数类型（与数据一致）
        if user_profiles["user_id"].dtype != "int64":
            user_profiles = user_profiles.copy()
            user_profiles["user_id"] = user_profiles["user_id"].astype("int64")
        
        # 训练模型（如果需要）
        propensity, recommender = _ensure_models()
        recency_lookup = _compute_recency(interactions)
        
        # 调试：检查 user_id=5 是否真的存在
        all_user_ids = user_profiles["user_id"].unique().tolist()
        has_user_5 = 5 in all_user_ids if user_id_int == 5 else None
        
        # 先尝试用整数匹配（数据中的实际类型）
        if user_id_int is not None:
            user_rows = user_profiles[user_profiles["user_id"] == user_id_int]
        else:
            # 如果无法转为整数，尝试字符串匹配
            user_rows = user_profiles[user_profiles["user_id"].astype(str) == user_id_str]
        
        if user_rows.empty:
            # 提供更详细的错误信息，包括调试信息
            available_ids = user_profiles["user_id"].head(10).tolist()
            user_id_dtype = user_profiles["user_id"].dtype
            total_users = len(user_profiles)
            debug_info = f"总客户数: {total_users}, user_id类型: {user_id_dtype}"
            if has_user_5 is not None:
                debug_info += f", user_id=5是否存在: {has_user_5}"
            return {"content": _text_content(f"未找到客户 {user_id_raw}（尝试匹配: int={user_id_int}, str={user_id_str}）。{debug_info}。前10个客户ID: {available_ids}。请确保使用正确的客户ID。")}
        
        user_row = user_rows.iloc[0]
        actual_user_id = user_row["user_id"]
        
        # recommender.recommend 期望字符串，但 SAR 模型内部会在 interactions 中查找
        # 由于 interactions 中的 user_id 是整数，传字符串应该也能工作（pandas 会处理类型转换）
        # 但为了保险，我们传字符串（符合方法签名）
        rec_df = recommender.recommend(str(actual_user_id), top_k=top_k)
        rec_df = rec_df.merge(item_profiles, on="item_id", how="left")
        lines = []
        for idx, (_, item) in enumerate(rec_df.iterrows(), 1):
            feature_row = _build_feature_row(user_row, item, recency_lookup)
            prob = float(propensity.predict_proba(feature_row.to_frame().T)[0])
            lines.append(
                f"{idx}. {item.get('title', item['item_id'])} | 推荐得分: {item.get('prediction', 0):.3f} | 贷款意愿概率: {prob:.3f}"
            )
        msg = "推荐结果：\n" + "\n".join(lines) if lines else "暂无推荐结果。"
        return {"content": _text_content(msg)}
    except Exception as e:
        return {"content": _text_content(f"推荐失败: {e}")}


async def tool_predict_propensity(args: dict) -> dict:
    """预测某客户对某贷款产品的申请意愿概率。参数：user_id（客户ID，字符串或整数），item_id（产品ID，整数）。"""
    try:
        user_id_raw = args.get("user_id", "")
        item_id = args.get("item_id")
        if item_id is not None:
            item_id = int(item_id)
        if not user_id_raw or item_id is None:
            return {"content": _text_content("请提供 user_id 和 item_id 参数。")}
        # 尝试将 user_id 转为整数（数据中 user_id 是整数类型）
        try:
            user_id_int = int(user_id_raw)
            user_id_str = str(user_id_raw)
        except (ValueError, TypeError):
            user_id_int = None
            user_id_str = str(user_id_raw)
        propensity, _ = _ensure_models()
        state = _get_state()
        interactions, user_profiles, item_profiles = state["datasets"]
        recency_lookup = _compute_recency(interactions)
        # 先尝试用整数匹配（数据中的实际类型）
        if user_id_int is not None:
            user_rows = user_profiles[user_profiles["user_id"] == user_id_int]
        else:
            user_rows = user_profiles[user_profiles["user_id"].astype(str) == user_id_str]
        item_rows = item_profiles[item_profiles["item_id"] == item_id]
        if user_rows.empty or item_rows.empty:
            return {"content": _text_content("未找到该客户或产品，请使用数据中的有效ID。")}
        user_row = user_rows.iloc[0]
        item_row = item_rows.iloc[0]
        feature_row = _build_feature_row(user_row, item_row, recency_lookup)
        prob = float(propensity.predict_proba(feature_row.to_frame().T)[0])
        from app.model_training import describe_feature, top_feature_contributions
        contribs = propensity.explain_instance(feature_row)
        top = top_feature_contributions(contribs, top_k=5)
        expl = "; ".join(describe_feature(n, v) for n, v in top)
        user_id_display = user_id_int if user_id_int is not None else user_id_raw
        msg = f"客户 {user_id_display} 对产品 {item_id} 的贷款意愿概率: {prob:.3f}。主要影响因素: {expl}"
        return {"content": _text_content(msg)}
    except Exception as e:
        return {"content": _text_content(f"预测失败: {e}")}


async def tool_similar_items(args: dict) -> dict:
    """查找与某贷款产品相似的其他产品。参数：item_id（产品ID），top_k（数量，默认5）。"""
    try:
        item_id = args.get("item_id")
        if item_id is not None:
            item_id = int(item_id)
        top_k = int(args.get("top_k", 5))
        if item_id is None:
            return {"content": _text_content("请提供 item_id 参数。")}
        _, recommender = _ensure_models()
        state = _get_state()
        _, _, item_profiles = state["datasets"]
        similar_df = recommender.similar_items(item_id, top_k=top_k)
        similar_df = similar_df.merge(item_profiles, on="item_id", how="left")
        titles = similar_df["title"].fillna("").tolist() if not similar_df.empty else []
        msg = "相似产品: " + ", ".join(titles) if titles else "暂无相似产品。"
        return {"content": _text_content(msg)}
    except Exception as e:
        return {"content": _text_content(f"查询失败: {e}")}


def get_loan_agent_tools() -> List[Dict[str, Any]]:
    """
    返回工具定义列表，用于在 opencode 配置中注册 MCP 工具。
    这些工具需要通过 MCP 服务器提供，需要在 opencode.json 中配置 MCP 服务器。
    """
    return [
        {
            "name": "load_data",
            "description": "加载示例贷款场景数据（MovieLens 100k 转贷款）。无参数。",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "train_models",
            "description": "训练贷款意愿模型与推荐模型。无参数；若未加载数据会先自动加载。",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "get_data_summary",
            "description": "查看当前数据概况（交互数、客户数、产品数）及示例客户ID、产品ID。无参数。",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "list_customers",
            "description": "列出当前数据中的客户ID列表，便于做推荐时选择客户。参数: limit（返回数量，默认50）。",
            "inputSchema": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 50}},
            },
        },
        {
            "name": "list_products",
            "description": "列出当前数据中的贷款产品ID及名称，便于做推荐或预测时选择产品。参数: limit（返回数量，默认50）。",
            "inputSchema": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 50}},
            },
        },
        {
            "name": "recommend",
            "description": "为指定客户做个性化贷款推荐。参数: user_id（客户ID）, top_k（推荐数量，默认5）。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["user_id"],
            },
        },
        {
            "name": "predict_propensity",
            "description": "预测某客户对某贷款产品的申请意愿概率。参数: user_id, item_id。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "item_id": {"type": "integer"},
                },
                "required": ["user_id", "item_id"],
            },
        },
        {
            "name": "similar_items",
            "description": "查找与某贷款产品相似的其他产品。参数: item_id, top_k（默认5）。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "integer"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["item_id"],
            },
        },
    ]
