"""
消费 Agent 流式消息，格式化为 UI 可展示的文本。
适配 opencode-sdk 的消息格式。
"""
from typing import Any


def format_message_for_ui(message: Any) -> tuple[str, str]:
    """
    将 SDK / runner 返回的 message 格式化为 (文本, 类型)。
    类型: "text" | "tool_use" | "result" | "error" | "other"
    """
    if isinstance(message, dict):
        if message.get("type") == "error":
            return f"错误: {message.get('message', '')}", "error"
        if message.get("type") == "text":
            return str(message.get("content", "")), "text"
        if message.get("type") == "tool_use":
            tool_name = message.get("name", "")
            tool_input = message.get("input", {})
            return f"调用工具: {tool_name}({tool_input})", "tool_use"
        if message.get("type") == "tool_result":
            return str(message.get("content", "")), "result"
        if message.get("type") == "complete":
            return "", "other"
        if "content" in message:
            return str(message.get("content", "")), "text"
        if "result" in message:
            result_val = message.get("result")
            if result_val is None or str(result_val).strip() == "None":
                return "", "other"
            return str(result_val), "result"
    if hasattr(message, "content"):
        parts = []
        for block in getattr(message, "content", []) or []:
            if getattr(block, "type", None) == "text" or hasattr(block, "text"):
                parts.append(getattr(block, "text", str(block)))
        if parts:
            return "\n".join(parts), "text"
    if hasattr(message, "result"):
        result_val = getattr(message, "result", None)
        if result_val is None or str(result_val).strip() == "None":
            return "", "other"
        return str(result_val), "result"
    if getattr(message, "type", None) == "result":
        raw = str(message)
        if raw.strip() == "None":
            return "", "other"
        return raw, "result"
    return "", "other"
