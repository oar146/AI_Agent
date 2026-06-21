"""调试脚本：查看 Agent 图流式输出的事件类型"""
import asyncio
import sys

sys.path.insert(0, ".")

from agent_config import get_agent
from langchain_core.messages import HumanMessage


async def main():
    agent = get_agent()
    
    state = {
        "messages": [HumanMessage(content="你好")],
        "user_input": "你好",
        "username": "test",
        "session_id": "test",
        "memory_context": "",
        "plan": "",
        "critique": "",
        "revision_count": 0,
    }
    
    print("=" * 60)
    print("前 20 个事件类型：")
    print("=" * 60)
    count = 0
    async for event in agent.astream_events(state, version="v1"):
        kind = event["event"]
        name = event.get("name", "")
        node = event.get("metadata", {}).get("langgraph_node", "N/A")
        
        # 只看关键事件
        if kind in ("on_chat_model_stream", "on_chat_model_end", "on_chain_end", "on_tool_start", "on_tool_end"):
            print(f"  {kind:35s} name={name:20s} node={node}")
            count += 1
            if count >= 20:
                break
    
    print()
    print("=" * 60)
    print("on_chat_model_stream 的 content 样例：")
    print("=" * 60)
    count = 0
    async for event in agent.astream_events(state, version="v1"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                print(f"  [{chunk.content[:80]}]")
                count += 1
                if count >= 5:
                    break
    
    if count == 0:
        print("  ❌ 没有抓到任何 on_chat_model_stream 事件！")


if __name__ == "__main__":
    asyncio.run(main())