"""
命令行对话示例：使用 opencode-sdk 调用 opencode 服务器执行 Agent 对话。
从项目根目录运行：python -m scripts.run_agent

交互方式：
- 输入一句话回车，等待 Agent 回复（流式输出）
- 输入 exit 或 quit 或 空回车 结束对话
"""
import anyio
from agents.runner import run_agent


def _print_message(msg: dict) -> None:
    """把 run_agent  yield 的消息打成可读输出；区分委派子 Agent 与 MCP 工具便于日志查看。"""
    if msg.get("type") == "error":
        print(f"[错误] {msg.get('message', msg)}")
    elif msg.get("type") == "text" and msg.get("content"):
        print(msg["content"], end="", flush=True)
    elif msg.get("type") == "tool_use":
        name = msg.get("name", "?")
        inp = msg.get("input") or {}
        if name == "task":
            subagent = inp.get("subagent_type") or inp.get("subagentType") or "?"
            print(f"\n[委派子Agent: {subagent}] ", end="", flush=True)
        else:
            # MCP 工具名可能带前缀，如 mcp__loan_agent__load_data，只取最后一段更易读
            display_name = name.split("__")[-1] if "__" in name else name
            print(f"\n[MCP 工具: {display_name}] ", end="", flush=True)
    elif msg.get("type") == "tool_result":
        # 简短打印工具结果，便于确认子 Agent / MCP 已返回
        content = msg.get("content")
        if isinstance(content, str) and len(content) > 0:
            preview = content[:80] + "..." if len(content) > 80 else content
            print(f"\n  → 结果: {preview}", flush=True)
        elif content is not None and content != "":
            print("\n  → 结果: (已返回)", flush=True)


async def run_turn(user_message: str) -> None:
    """发一条用户消息并流式打印回复。"""
    had_output = False
    async for message in run_agent(user_message=user_message):
        if message.get("type") == "text" and message.get("content"):
            had_output = True
        _print_message(message)
    if not had_output:
        print("(无回复，请检查 OpenCode 服务是否已启动)")
    print()  # 换行


async def main():
    print("零售贷款智能运营 Agent（输入 exit / quit / 空行 结束）\n")
    while True:
        try:
            line = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not line or line.lower() in ("exit", "quit", "q"):
            print("再见。")
            break
        print("Agent: ", end="", flush=True)
        await run_turn(line)


if __name__ == "__main__":
    anyio.run(main)
