"""
使用 opencode-sdk 驱动零售贷款智能运营，通过 opencode 服务器执行对话。
自定义工具通过 MCP 服务器提供，需要在 opencode 配置中注册。
支持流式回复：通过 event.list() 订阅 SSE，逐 delta 输出。
"""
import asyncio
import json
from typing import AsyncIterator, Any

from core.config import (
    OPENCODE_BASE_URL,
    OPENCODE_MODEL_PROVIDER,
    OPENCODE_MODEL_ID,
)
from agents.definitions import MAIN_SYSTEM_PROMPT


async def run_agent(user_message: str, stream: bool = True) -> AsyncIterator[Any]:
    """
    使用 opencode-sdk 通过 opencode 服务器执行对话，流式 yield 消息。
    stream=True 时通过 event 流逐字输出；stream=False 时等完整响应后一次性返回。
    """
    if not (user_message or "").strip():
        yield {"type": "text", "content": "请输入您的问题。"}
        return

    try:
        from opencode_ai import AsyncOpencode
    except ImportError as e:
        yield {"type": "error", "message": f"未安装 opencode-ai: {e}"}
        return

    client = None
    try:
        client = AsyncOpencode(base_url=OPENCODE_BASE_URL)

        sessions = await client.session.list()
        if sessions and len(sessions) > 0:
            session_id = sessions[0].id
        else:
            new_session = await client.session.create(body={"title": "零售贷款智能运营"})
            session_id = new_session.id

        if stream:
            # 流式：先订阅 event 流，再发 chat，从 message.part.updated 里取 delta 输出
            event_stream = await client.event.list()
            chat_done = False
            chat_task = None

            async def send_chat():
                nonlocal chat_done
                try:
                    await client.session.chat(
                        id=session_id,
                        provider_id=OPENCODE_MODEL_PROVIDER,
                        model_id=OPENCODE_MODEL_ID,
                        parts=[{"type": "text", "text": user_message}],
                        system=MAIN_SYSTEM_PROMPT,
                    )
                finally:
                    chat_done = True

            chat_task = asyncio.create_task(send_chat())
            last_text_by_part: dict[str, str] = {}
            assistant_message_id = None
            stream_yielded_any = False
            try:
                async for event in event_stream:
                    if chat_done and chat_task and chat_task.done():
                        break
                    ev_type = getattr(event, "type", None)
                    props = getattr(event, "properties", None)
                    if not props:
                        continue
                    if ev_type == "message.part.updated":
                        part = getattr(props, "part", None)
                        if not part:
                            continue
                        part_sid = getattr(part, "session_id", None) or getattr(part, "sessionID", None)
                        if part_sid != session_id:
                            continue
                        msg_id = getattr(part, "message_id", None) or getattr(part, "messageID", None)
                        if assistant_message_id is not None and msg_id != assistant_message_id:
                            continue
                        p_type = getattr(part, "type", None)
                        if p_type == "step-finish":
                            break
                        if p_type == "tool_use" and assistant_message_id is not None and msg_id == assistant_message_id:
                            name = getattr(part, "name", "") or ""
                            inp = getattr(part, "input", None) or {}
                            if isinstance(inp, str):
                                try:
                                    inp = json.loads(inp) if inp else {}
                                except Exception:
                                    inp = {}
                            yield {"type": "tool_use", "name": name, "input": inp if isinstance(inp, dict) else {}}
                        if p_type == "text" and assistant_message_id is not None and msg_id == assistant_message_id:
                            delta = getattr(props, "delta", None)
                            if delta:
                                yield {"type": "text", "content": delta}
                                stream_yielded_any = True
                            else:
                                text = getattr(part, "text", None) or ""
                                pid = getattr(part, "id", "") or ""
                                prev = last_text_by_part.get(pid, "")
                                if len(text) > len(prev) and text.startswith(prev):
                                    yield {"type": "text", "content": text[len(prev):]}
                                    stream_yielded_any = True
                                last_text_by_part[pid] = text
                    elif ev_type == "message.updated":
                        info = getattr(props, "info", None)
                        if info and getattr(info, "role", None) == "assistant":
                            sid = getattr(info, "session_id", None) or getattr(info, "sessionID", None)
                            if sid == session_id:
                                assistant_message_id = getattr(info, "id", None)
            except asyncio.CancelledError:
                pass
            finally:
                # 等待 chat 完成（含子 Agent Task），避免取消导致无完整回复
                if chat_task and not chat_task.done():
                    try:
                        await asyncio.wait_for(chat_task, timeout=120.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass

            # 从 session.messages 拉取「本轮」对话的 tool 调用，便于 OpenCode 侧打印「委派子Agent」「MCP 工具」日志
            # 取最近几条 assistant 消息（主 Agent 的 Task 与子 Agent 的 MCP 可能在不同消息里；最后一条可能只有最终文本），最多 3 条，再按时间正序输出
            # OpenCode API 返回 List[SessionMessagesResponseItem]，每项为 { info: Message, parts: List[Part] }；Part 类型为 "tool"，工具名为 .tool，输入/输出在 .state 中
            try:
                await asyncio.sleep(0.3)
                messages = await client.session.messages(id=session_id)
                if messages and len(messages) > 0:
                    assistant_count = 0
                    max_assistant_messages = 3
                    collected = []
                    for item in reversed(messages):
                        info = getattr(item, "info", None)
                        role = getattr(info, "role", None) if info is not None else getattr(item, "role", None)
                        if role != "assistant":
                            continue
                        assistant_count += 1
                        if assistant_count > max_assistant_messages:
                            break
                        parts = getattr(item, "parts", None) or []
                        for part in parts or []:
                            p_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                            if p_type == "tool":
                                name = part.get("tool", "") if isinstance(part, dict) else getattr(part, "tool", "")
                                state = part.get("state") if isinstance(part, dict) else getattr(part, "state", None)
                                inp = {}
                                if state is not None:
                                    inp = state.get("input", {}) if isinstance(state, dict) else getattr(state, "input", None) or {}
                                if isinstance(inp, str):
                                    try:
                                        inp = json.loads(inp) if inp else {}
                                    except Exception:
                                        inp = {}
                                collected.append({"type": "tool_use", "name": name, "input": inp if isinstance(inp, dict) else {}})
                                if state is not None:
                                    out = state.get("output", "") if isinstance(state, dict) else getattr(state, "output", None) or ""
                                    if out is not None and str(out).strip():
                                        collected.append({"type": "tool_result", "content": str(out)})
                    for msg in reversed(collected):
                        yield msg
            except Exception:
                pass

            # 流式未产出任何文本时（如主 Agent 只委派 Task、无 text delta），从 session.messages 取最后一条助理消息全文
            if not stream_yielded_any:
                try:
                    await asyncio.sleep(0.8)
                    messages = await client.session.messages(id=session_id)
                    if messages and len(messages) > 0:
                        for msg in reversed(messages):
                            parts = getattr(msg, "parts", None) or (msg if isinstance(msg, list) else [])
                            for part in parts or []:
                                p_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                                if p_type == "text":
                                    text = part.get("text", "") if isinstance(part, dict) else (getattr(part, "text", None) or "")
                                    if text.strip():
                                        yield {"type": "text", "content": text}
                                        stream_yielded_any = True
                                        break
                            if stream_yielded_any:
                                break
                except Exception:
                    pass
            if not stream_yielded_any:
                yield {"type": "text", "content": "未收到 Agent 回复，请确认 OpenCode 服务已启动（opencode serve），或稍后重试。"}
            return

        # 非流式：原有逻辑，等 chat 完成后解析整段回复
        response = await client.session.chat(
            id=session_id,
            provider_id=OPENCODE_MODEL_PROVIDER,
            model_id=OPENCODE_MODEL_ID,
            parts=[{"type": "text", "text": user_message}],
            system=MAIN_SYSTEM_PROMPT,
        )

        parts = getattr(response, "parts", None)
        if not parts and hasattr(response, "model_dump"):
            raw = response.model_dump()
            parts = raw.get("parts") or []

        yielded_text = False
        for part in parts or []:
            p_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            if p_type == "text":
                text = part.get("text", "") if isinstance(part, dict) else (getattr(part, "text", None) or getattr(part, "content", "") or "")
                if text:
                    yield {"type": "text", "content": text}
                    yielded_text = True
            elif p_type == "tool_use":
                name = part.get("name", "") if isinstance(part, dict) else getattr(part, "name", "")
                inp = part.get("input", {}) if isinstance(part, dict) else getattr(part, "input", {})
                yield {"type": "tool_use", "name": name, "input": inp}
            elif p_type == "tool_result":
                content = part.get("content", "") if isinstance(part, dict) else getattr(part, "content", "")
                yield {"type": "tool_result", "content": content}
            elif not yielded_text:
                text = part.get("text", "") if isinstance(part, dict) else getattr(part, "text", None)
                if text:
                    yield {"type": "text", "content": text}
                    yielded_text = True

        if not yielded_text:
            try:
                messages = await client.session.messages(id=session_id)
                if messages and len(messages) > 0:
                    last = messages[-1]
                    last_parts = getattr(last, "parts", None) or (last if isinstance(last, list) else [])
                    for part in last_parts or []:
                        p_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                        text = part.get("text", "") if isinstance(part, dict) else getattr(part, "text", None)
                        if (p_type == "text" or text) and text:
                            yield {"type": "text", "content": text}
                            yielded_text = True
                            break
            except Exception:
                pass
        if not yielded_text:
            yield {"type": "text", "content": "未收到 Agent 回复，请确认 OpenCode 服务已启动（opencode serve），或稍后重试。"}

    except Exception as e:
        yield {"type": "error", "message": f"执行失败: {e}"}
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass
