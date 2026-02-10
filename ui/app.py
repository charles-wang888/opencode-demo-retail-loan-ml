"""
Gradio 界面：数据管理与查看、模型训练、模型演示。
"""
import time
from typing import Any, List, Optional, Tuple

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from app.data_prep import prepare_datasets
from app.model_training import (
    PropensityModel,
    RecommendationModel,
    train_propensity_model,
    train_recommendation_model,
)
from tools.loan_agent_tools import set_ui_state
from ui.helpers import (
    build_feature_row,
    compute_recency,
    explain_propensity,
    prepare_for_display,
)

PAGE_SIZE = 20

# 三步向导 HTML（current_step 1/2/3）
def stepper_html(current_step: int) -> str:
    steps = [
        (1, "数据管理与查看"),
        (2, "模型训练"),
        (3, "模型演示"),
    ]
    n = len(steps)
    conn_width = round((current_step - 1) * 100 / (n - 1), 2) if current_step > 1 else 0
    circles = ""
    for num, title in steps:
        if num < current_step:
            c_class, t_class, disp = "completed", "completed", "✓"
        elif num == current_step:
            c_class, t_class, disp = "active", "active", str(num)
        else:
            c_class, t_class, disp = "pending", "pending", str(num)
        circles += f'<div class="wizard-step"><div class="step-circle {c_class}">{disp}</div><div class="step-title {t_class}">{title}</div></div>'
    return f"""
    <div class="wizard-container">
        <div class="wizard-steps">
            <div class="step-connector"><div class="step-connector-fill" style="width:{conn_width}%"></div></div>
            {circles}
        </div>
    </div>
    <style>
    .wizard-container {{ background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px 15px; border-radius: 15px; margin: 15px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    .wizard-steps {{ display: flex; justify-content: space-between; align-items: center; position: relative; padding: 0 15px 10px 15px; }}
    .wizard-step {{ flex: 1; display: flex; flex-direction: column; align-items: center; position: relative; z-index: 2; }}
    .step-connector {{ position: absolute; top: 28px; left: 25px; right: 25px; height: 4px; background: linear-gradient(90deg, #e0e0e0, #d0d0d0); z-index: 1; border-radius: 10px; }}
    .step-connector-fill {{ height: 100%; background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); border-radius: 10px; transition: width 0.5s ease; }}
    .step-circle {{ width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: bold; margin-bottom: 10px; border: 3px solid; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
    .step-circle.active {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-color: #667eea; box-shadow: 0 0 0 6px rgba(102,126,234,0.2); }}
    .step-circle.completed {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; border-color: #11998e; }}
    .step-circle.pending {{ color: #999; border-color: #d0d0d0; }}
    .step-title {{ font-size: 13px; font-weight: 600; text-align: center; color: #333; }}
    .step-title.active {{ color: #667eea; }}
    .step-title.completed {{ color: #11998e; }}
    .step-title.pending {{ color: #999; }}
    </style>
    """


def load_data_fn():
    """加载数据并同步到 Agent 状态。"""
    import os
    os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
    datasets = prepare_datasets(size="100k")
    set_ui_state(datasets=datasets)
    interactions, user_profiles, item_profiles = datasets
    n_inter, n_user, n_item = len(interactions), user_profiles["user_id"].nunique(), item_profiles["item_id"].nunique()
    msg = f"数据已加载。交互记录: {n_inter:,} 条，客户数: {n_user:,}，贷款产品数: {n_item:,}。"
    # 展示用：前若干行
    item_disp = prepare_for_display(item_profiles.head(100))
    user_disp = prepare_for_display(user_profiles.head(100))
    inter_disp = prepare_for_display(interactions.head(100))
    return datasets, msg, item_disp, user_disp, inter_disp, n_inter, n_user, n_item


def clear_data_fn():
    set_ui_state(datasets=None, propensity_model=None, recommendation_model=None)
    return None, "已清除会话内的数据与模型。", None, None, None, 0, 0, 0


def train_models_fn(datasets):
    """训练两个模型，返回模型与训练步骤 HTML、以及详情用于展示。"""
    if datasets is None:
        return None, None, "请先加载数据。", "", "", None, None, None, None, "", "", None
    interactions, user_profiles, item_profiles = datasets
    steps_list = [
        ("初始化训练环境...", 0.3),
        ("构建特征表...", 0.5),
        ("划分训练/测试集...", 0.4),
        ("训练贷款意愿模型...", None),
        ("训练推荐模型...", None),
        ("评估模型性能...", 0.6),
        ("生成训练报告...", 0.5),
    ]
    total = len(steps_list)
    all_html = ""
    for i, (msg, delay) in enumerate(steps_list):
        step_num = i + 1
        bar = f'<div style="margin-bottom:12px;padding:14px 20px;border-radius:10px;border-left:5px solid #4facfe;background:linear-gradient(120deg,#f6f9ff,#eef3ff);font-size:14px;color:#1f2a53;font-weight:600;"><span style="color:#4facfe;">步骤 {step_num}/{total}</span> {msg}</div>'
        all_html += bar
        if i == 3:
            propensity = train_propensity_model(interactions, user_profiles, item_profiles)
        elif i == 4:
            recommender = train_recommendation_model(interactions)
        elif delay is not None:
            time.sleep(delay)
    all_html += '<div style="margin-top:16px;padding:14px 20px;border-radius:10px;border-left:5px solid #4facfe;background:linear-gradient(120deg,#e8f5ff,#d6e8ff);font-size:14px;font-weight:600;"><span style="color:#4facfe;">✓</span> 训练完成！</div>'
    set_ui_state(propensity_model=propensity, recommendation_model=recommender)

    # 意愿模型详情
    summary = propensity.training_summary
    prop_steps = "\n".join(f"{i+1}. **{s.get('title','')}**：{s.get('detail','')}" for i, s in enumerate(summary.get("steps", [])))
    prop_info = "  \n".join(f"- **{k}**: {v}" for k, v in summary.get("dataset_info", {}).items())
    perf = summary.get("performance", {})
    prop_perf = "  \n".join(f"- **{k}**: {v:.1%}" for k, v in perf.items())
    roc_df = summary.get("roc_curve")
    roc_fig = None
    if roc_df is not None and not roc_df.empty:
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=roc_df["false_positive_rate"], y=roc_df["true_positive_rate"], mode="lines+markers", name="ROC"))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="基线"))
        roc_fig.update_layout(title="ROC 曲线", xaxis_title="假正率", yaxis_title="真正率", height=300)
    pr_df = summary.get("pr_curve")
    pr_fig = None
    if pr_df is not None and not pr_df.empty:
        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(x=pr_df["recall"], y=pr_df["precision"], mode="lines+markers", name="PR"))
        pr_fig.update_layout(title="Precision-Recall 曲线", xaxis_title="召回率", yaxis_title="精确率", height=300)
    thresh_df = summary.get("threshold_metrics")
    conf_df = summary.get("confusion_matrix")
    seg_df = summary.get("segment_metrics")
    weights_df = summary.get("feature_weights")
    if weights_df is not None and not weights_df.empty:
        weights_df = weights_df[["label", "weight"]].head(15).rename(columns={"label": "特征", "weight": "系数权重"})

    # 推荐模型详情
    rec_summary = recommender.training_summary
    rec_steps = "\n".join(f"{i+1}. **{s.get('title','')}**：{s.get('detail','')}" for i, s in enumerate(rec_summary.get("steps", [])))
    rec_stats = "  \n".join(f"- **{k}**: {v}" for k, v in rec_summary.get("stats", {}).items())
    rec_top = rec_summary.get("top_items")
    if rec_top is not None:
        rec_top = rec_top.rename(columns={"item_id": "产品ID"}).head(15)

    return (
        propensity, recommender, all_html,
        prop_steps, prop_info, prop_perf, roc_fig, pr_fig, thresh_df, conf_df, seg_df, weights_df,
        rec_steps, rec_stats, rec_top,
    )


def get_recommendations_fn(datasets, propensity_model, recommendation_model, user_id, top_k: int):
    """根据所选客户与 top_k 生成推荐列表与展示用 HTML。"""
    if datasets is None or propensity_model is None or recommendation_model is None:
        return "请先完成数据加载与模型训练。", None
    interactions, user_profiles, item_profiles = datasets
    try:
        uid = int(user_id) if user_id is not None else None
    except (TypeError, ValueError):
        uid = user_id
    if uid is None:
        return "请选择客户。", None
    user_rows = user_profiles[user_profiles["user_id"] == uid]
    if user_rows.empty:
        return "未找到该客户。", None
    user_row = user_rows.iloc[0]
    recency_lookup = compute_recency(interactions)
    rec_df = recommendation_model.recommend(uid, top_k=int(top_k))
    rec_df = rec_df.merge(item_profiles, on="item_id", how="left")
    lines = []
    for idx, (_, item) in enumerate(rec_df.iterrows(), 1):
        fr = build_feature_row(user_row, item, recency_lookup)
        prob = float(propensity_model.predict_proba(pd.DataFrame([fr]))[0])
        expl = explain_propensity(propensity_model, fr)
        similar = recommendation_model.similar_items(item["item_id"], top_k=3)
        sim_titles = similar.merge(item_profiles, on="item_id", how="left")["title"].fillna("").tolist() if not similar.empty else []
        lines.append({
            "序号": idx,
            "产品": item.get("title", item["item_id"]),
            "推荐得分": round(item.get("prediction", 0), 3),
            "贷款意愿概率": round(prob, 3),
            "产品标签": " / ".join(item.get("product_tags", [])),
            "风险": item.get("item_risk_label", ""),
            "相似产品": ", ".join(sim_titles[:3]),
            "意愿影响因子": "; ".join(expl[:3]),
        })
    df = pd.DataFrame(lines)
    return None, df


def create_app() -> gr.Blocks:
    with gr.Blocks(title="零售贷款智能运营演示") as app:
        gr.Markdown("# 🧠 零售贷款智能运营演示")
        gr.Markdown("基于 recommenders 框架 + 机器学习模型的客户贷款意愿评估与产品推荐示例。")

        # 顶部三步向导（默认第一步）
        step_html = gr.HTML(stepper_html(1), elem_id="stepper")

        # 共享状态
        state_datasets = gr.State(value=None)
        state_propensity = gr.State(value=None)
        state_recommendation = gr.State(value=None)

        with gr.Tabs(selected=0) as tabs:
            # ========== Tab 1: 数据管理与查看 ==========
            with gr.Tab(id=0, label="1. 数据管理与查看"):
                gr.Markdown("### 📦 数据管理与查看")
                gr.Markdown("管理示例数据的加载，并通过数据查看器快速浏览交互行为、客户画像与贷款产品。")
                with gr.Row():
                    btn_load = gr.Button("加载/刷新示例数据", variant="primary")
                    btn_clear = gr.Button("清除会话数据")
                msg_data = gr.Markdown("暂未加载数据。点击「加载/刷新示例数据」开始准备数据。")
                gr.Markdown("#### 数据概况")
                with gr.Row():
                    metric_inter = gr.Number(label="交互记录数", value=0)
                    metric_user = gr.Number(label="客户数量", value=0)
                    metric_item = gr.Number(label="贷款产品数量", value=0)
                gr.Markdown("#### 贷款产品")
                df_items = gr.Dataframe(label="贷款产品", interactive=False)
                gr.Markdown("#### 客户画像")
                df_users = gr.Dataframe(label="客户画像", interactive=False)
                gr.Markdown("#### 交互行为")
                df_inter = gr.Dataframe(label="交互行为", interactive=False)
                with gr.Row():
                    btn_next1 = gr.Button("下一步：模型训练", variant="primary")

            # ========== Tab 2: 模型训练 ==========
            with gr.Tab(id=1, label="2. 模型训练"):
                gr.Markdown("### 模型训练看板")
                gr.Markdown("该看板用于手动触发训练流程，并查看训练过程、性能指标与分群分析。")
                btn_train = gr.Button("开始训练模型", variant="primary")
                train_steps_html = gr.HTML(visible=True)
                with gr.Row():
                    btn_show_detail = gr.Button("查看训练细节")
                    btn_hide_detail = gr.Button("隐藏训练细节", visible=False)
                detail_container = gr.Column(visible=False)
                with detail_container:
                    with gr.Tabs():
                        with gr.Tab(id="prop", label="贷款意愿模型"):
                            prop_steps_md = gr.Markdown("")
                            prop_info_md = gr.Markdown("")
                            prop_perf_md = gr.Markdown("")
                            roc_plot = gr.Plot(label="ROC 曲线")
                            pr_plot = gr.Plot(label="PR 曲线")
                            thresh_df = gr.Dataframe(label="阈值对比", interactive=False)
                            conf_df = gr.Dataframe(label="混淆矩阵", interactive=False)
                            seg_df = gr.Dataframe(label="分群指标", interactive=False)
                            weights_df = gr.Dataframe(label="关键特征权重", interactive=False)
                        with gr.Tab(id="rec", label="推荐模型"):
                            rec_steps_md = gr.Markdown("")
                            rec_stats_md = gr.Markdown("")
                            rec_top_df = gr.Dataframe(label="高频贷款产品", interactive=False)
                with gr.Row():
                    btn_prev2 = gr.Button("上一步：数据管理与查看")
                    btn_next2 = gr.Button("下一步：模型演示", variant="primary")

            # ========== Tab 3: 模型演示 ==========
            with gr.Tab(id=2, label="3. 模型演示"):
                gr.Markdown("### 🧠 零售贷款智能运营演示")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 客户筛选")
                        demo_user_id = gr.Dropdown(label="示例客户", choices=[], value=None)
                        demo_top_k = gr.Slider(3, 10, value=5, step=1, label="推荐贷款数量")
                        btn_recommend = gr.Button("获取推荐", variant="primary")
                    with gr.Column(scale=2):
                        gr.Markdown("#### 客户画像")
                        profile_md = gr.Markdown("请先加载数据并选择客户。")
                demo_msg = gr.Markdown("")
                demo_result_df = gr.Dataframe(label="贷款推荐与意愿评估", interactive=False)
                gr.Markdown("#### 贷款产品标签总览")
                demo_product_table = gr.Dataframe(label="产品标签", interactive=False)
                btn_prev3 = gr.Button("上一步：模型训练")

        # ---------- Tab 1 事件 ----------
        def on_load():
            out = load_data_fn()
            interactions, user_profiles, item_profiles = out[0]
            choices = sorted(user_profiles["user_id"].sample(n=min(30, len(user_profiles)), random_state=42).tolist())
            value = str(choices[0]) if choices else None
            pt = item_profiles[["title", "product_tags", "item_risk_label"]].head(20).copy()
            pt["product_tags"] = pt["product_tags"].apply(lambda t: " / ".join(t))
            return (
                out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                gr.update(choices=[str(c) for c in choices], value=value), pt,
            )

        def on_clear():
            out = clear_data_fn()
            return (
                out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                gr.update(choices=[], value=None), None,
            )

        btn_load.click(
            fn=on_load,
            outputs=[state_datasets, msg_data, df_items, df_users, df_inter, metric_inter, metric_user, metric_item, demo_user_id, demo_product_table],
        )
        btn_clear.click(
            fn=on_clear,
            outputs=[state_datasets, msg_data, df_items, df_users, df_inter, metric_inter, metric_user, metric_item, demo_user_id, demo_product_table],
        )

        btn_next1.click(lambda: gr.Tabs(selected=1), None, [tabs])

        # ---------- Tab 2 事件 ----------
        def on_train(datasets):
            if datasets is None:
                return (
                    None, None,
                    "请先加载数据。", "", "", None, None, None, None, None, None, None,
                    "", "", None,
                    gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                )
            out = train_models_fn(datasets)
            return (
                out[0], out[1], out[2],
                out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11],
                out[12], out[13], out[14],
                gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
            )

        train_outputs = [
            state_propensity, state_recommendation, train_steps_html,
            prop_steps_md, prop_info_md, prop_perf_md, roc_plot, pr_plot, thresh_df, conf_df, seg_df, weights_df,
            rec_steps_md, rec_stats_md, rec_top_df,
            detail_container, btn_show_detail, btn_hide_detail,
        ]
        btn_train.click(fn=on_train, inputs=[state_datasets], outputs=train_outputs)

        def show_detail():
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
        def hide_detail():
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        btn_show_detail.click(show_detail, None, [detail_container, btn_show_detail, btn_hide_detail])
        btn_hide_detail.click(hide_detail, None, [detail_container, btn_show_detail, btn_hide_detail])

        btn_prev2.click(lambda: gr.Tabs(selected=0), None, [tabs])
        btn_next2.click(lambda: gr.Tabs(selected=2), None, [tabs])

        # ---------- Tab 3 事件 ----------
        def on_user_change(datasets, user_id):
            if datasets is None or not user_id:
                return "请先加载数据并选择客户。"
            try:
                uid = int(user_id)
            except (TypeError, ValueError):
                uid = user_id
            _, user_profiles, _ = datasets
            rows = user_profiles[user_profiles["user_id"] == uid]
            if rows.empty:
                return "未找到该客户。"
            row = rows.iloc[0]
            return f"""**客户层级**: {row['asset_label']}  
**风险承受力**: {row['user_risk_label']} · {row['user_risk_score']:.2f}  
**贷款需求标签**: {' / '.join(row['top_tags'])}"""

        demo_user_id.change(
            fn=on_user_change,
            inputs=[state_datasets, demo_user_id],
            outputs=[profile_md],
        )

        def on_recommend(datasets, p_model, r_model, user_id, top_k):
            err, df = get_recommendations_fn(datasets, p_model, r_model, user_id, top_k)
            if err:
                return err, None
            return None, df

        btn_recommend.click(
            fn=on_recommend,
            inputs=[state_datasets, state_propensity, state_recommendation, demo_user_id, demo_top_k],
            outputs=[demo_msg, demo_result_df],
        )

        btn_prev3.click(lambda: gr.Tabs(selected=1), None, [tabs])

        gr.Markdown("---\n**说明**：步骤 1～3 为数据与模型操作。")
    return app
