"""
零售贷款智能运营 MCP 服务器（stdio 传输）。
供 OpenCode 等客户端通过 stdio 调用 load_data、train_models、recommend 等工具。

运行方式（需在项目根目录）：
  python -m tools.loan_agent_mcp_server

日志输出到 stderr，可通过环境变量 LOAN_AGENT_MCP_LOG=DEBUG 开启详细日志。

OpenCode 配置示例（opencode.json）：
  "mcp": {
    "loan_agent": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "tools.loan_agent_mcp_server"],
      "cwd": "/path/to/customer-behavior-ml-3"
    }
  }
"""
from __future__ import annotations

import logging
import os
import sys

# 日志仅输出到 stderr，避免干扰 MCP 协议（stdout）
def _setup_logging() -> logging.Logger:
    level = os.environ.get("LOAN_AGENT_MCP_LOG", "INFO").upper()
    level = getattr(logging, level, logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] loan_agent_mcp: %(message)s"
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    log = logging.getLogger("loan_agent_mcp")
    log.setLevel(level)
    log.handlers.clear()
    log.addHandler(h)
    return log

log = _setup_logging()

# 确保以 python -m tools.loan_agent_mcp_server 运行时，项目根在 path 中
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
log.info("项目根目录: %s", _PROJECT_ROOT)

# 抑制 recommenders 等库的警告
os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

log.debug("正在导入 tools.loan_agent_tools ...")
from tools.loan_agent_tools import (
    tool_get_data_summary,
    tool_list_customers,
    tool_list_products,
    tool_load_data,
    tool_predict_propensity,
    tool_recommend,
    tool_similar_items,
    tool_train_models,
)
log.info("已加载 loan_agent_tools 共 8 个工具")


def _text_from_result(result: dict) -> str:
    """从工具返回的 dict 中取出首段文本。"""
    content = result.get("content") or []
    if not content:
        return ""
    first = content[0] if isinstance(content[0], dict) else {}
    return first.get("text", str(first))


def _create_mcp_server():
    """创建并配置 MCP 服务器（兼容 FastMCP / MCPServer）。"""
    log.info("正在创建 MCP 服务器 ...")
    try:
        from mcp.server.fastmcp import FastMCP
        mcp = FastMCP("loan_agent", json_response=True)
        log.info("使用 FastMCP 后端")
    except ImportError:
        try:
            from mcp.server.mcpserver import MCPServer
            mcp = MCPServer("loan_agent")
            log.info("使用 MCPServer 后端")
        except ImportError as e:
            log.error("未找到 MCP SDK: %s", e)
            raise RuntimeError(
                "未安装 MCP SDK。请执行: pip install mcp"
            ) from e

    # --- 无参工具 ---
    @mcp.tool()
    async def load_data() -> str:
        """加载示例贷款场景数据（MovieLens 100k 转贷款）。无参数。"""
        log.info("工具调用: load_data")
        result = await tool_load_data({})
        return _text_from_result(result)

    @mcp.tool()
    async def train_models() -> str:
        """训练贷款意愿模型与推荐模型。无参数；若未加载数据会先自动加载。"""
        log.info("工具调用: train_models")
        result = await tool_train_models({})
        return _text_from_result(result)

    @mcp.tool()
    async def get_data_summary() -> str:
        """查看当前数据概况（交互数、客户数、产品数）及示例客户ID、产品ID。无参数。"""
        log.info("工具调用: get_data_summary")
        result = await tool_get_data_summary({})
        return _text_from_result(result)

    @mcp.tool()
    async def list_customers(limit: int = 50) -> str:
        """列出当前数据中的客户ID列表，便于做推荐时选择客户。参数: limit（返回数量，默认50）。"""
        log.info("工具调用: list_customers(limit=%s)", limit)
        result = await tool_list_customers({"limit": limit})
        return _text_from_result(result)

    @mcp.tool()
    async def list_products(limit: int = 50) -> str:
        """列出当前数据中的贷款产品ID及名称，便于做推荐或预测时选择产品。参数: limit（返回数量，默认50）。"""
        log.info("工具调用: list_products(limit=%s)", limit)
        result = await tool_list_products({"limit": limit})
        return _text_from_result(result)

    # --- 有参工具 ---
    @mcp.tool()
    async def recommend(user_id: str, top_k: int = 5) -> str:
        """为指定客户做个性化贷款推荐。参数: user_id（客户ID）, top_k（推荐数量，默认5）。"""
        log.info("工具调用: recommend(user_id=%s, top_k=%s)", user_id, top_k)
        result = await tool_recommend({"user_id": user_id, "top_k": top_k})
        return _text_from_result(result)

    @mcp.tool()
    async def predict_propensity(user_id: str, item_id: int) -> str:
        """预测某客户对某贷款产品的申请意愿概率。参数: user_id, item_id。"""
        log.info("工具调用: predict_propensity(user_id=%s, item_id=%s)", user_id, item_id)
        result = await tool_predict_propensity({"user_id": user_id, "item_id": item_id})
        return _text_from_result(result)

    @mcp.tool()
    async def similar_items(item_id: int, top_k: int = 5) -> str:
        """查找与某贷款产品相似的其他产品。参数: item_id, top_k（默认5）。"""
        log.info("工具调用: similar_items(item_id=%s, top_k=%s)", item_id, top_k)
        result = await tool_similar_items({"item_id": item_id, "top_k": top_k})
        return _text_from_result(result)

    log.info("已注册 8 个工具: load_data, train_models, get_data_summary, list_customers, list_products, recommend, predict_propensity, similar_items")
    return mcp


def main() -> None:
    log.info("loan_agent MCP 服务器启动中 ...")
    mcp = _create_mcp_server()
    run = getattr(mcp, "run", None)
    if not callable(run):
        log.error("MCP server 未提供 run 方法")
        raise RuntimeError("MCP server 未提供 run 方法")
    log.info("正在启动 stdio 传输（stdin/stdout 用于 MCP 协议，日志在 stderr）...")
    try:
        run(transport="stdio")
    except TypeError as te:
        log.debug("run(transport='stdio') 不支持，尝试 run(): %s", te)
        try:
            run()
        except Exception as e:
            log.exception("MCP server run 失败: %s", e)
            raise
    except Exception as e:
        log.exception("MCP server 异常: %s", e)
        raise
    log.info("MCP 服务器已退出")


if __name__ == "__main__":
    main()
