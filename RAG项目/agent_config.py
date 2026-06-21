"""Agent 核心配置：完整工作流（记忆检索 → 规划 → ReAct执行 → 反思 → 记忆存储）"""
import json
import re
from typing import Sequence, TypedDict, Literal, Optional

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

import config_data as config
from tools import knowledge_base_search, file_content_analyzer, calculator
from memory_store import memory_store


# ======== 工具注册 ========
AGENT_TOOLS = [
    knowledge_base_search,
    file_content_analyzer,
    calculator,
]


# ======== Agent 状态定义 ========
class AgentState(TypedDict):
    """完整的 Agent 运行时状态"""
    messages: Sequence[BaseMessage]    # 对话消息流（含ReAct的工具调用记录）
    user_input: str                     # 当前用户输入
    username: str                       # 用户名（记忆隔离用）
    session_id: str                     # 会话ID
    memory_context: str                 # 长期记忆上下文文本
    plan: str                           # 执行计划
    critique: str                       # 反思评价
    revision_count: int                 # 已修订次数


# ======== LLM 工厂 ========
def _get_llm():
    return ChatTongyi(model=config.chat_model_name)

def _get_llm_with_tools():
    return ChatTongyi(model=config.chat_model_name).bind_tools(AGENT_TOOLS)


# ====================================================================
#  图节点（Graph Nodes）
# ====================================================================

# --- 1. 记忆检索节点 ------------------------------------------------
def node_load_memory(state: AgentState) -> dict:
    """从 Chroma 长期记忆中检索与当前问题相关的历史对话"""
    username = state.get("username", "unknown")
    query = state.get("user_input", "")
    memories = memory_store.query_relevant(query, username)

    if memories:
        context = "以下是与你相关的历史对话摘要（有助于理解当前问题）：\n" + \
                  "\n---\n".join(f"• {m}" for m in memories)
    else:
        context = ""

    return {"memory_context": context}


# --- 2. 规划节点 ----------------------------------------------------
def node_planner(state: AgentState) -> dict:
    """分析用户问题，生成结构化的执行计划"""
    llm = _get_llm()
    user_input = state.get("user_input", "")
    memory_context = state.get("memory_context", "")

    prompt = f"""你是AI助手的规划器。请分析用户问题，制定清晰的执行计划。

可用工具：
- knowledge_base_search：搜索知识库获取信息
- calculator：执行数学计算
- file_content_analyzer：分析已上传文件

用户问题：{user_input}
{f"相关记忆：\n{memory_context}" if memory_context else ""}

要求：
1. 如果问题需要查询知识或数据，规划 knowledge_base_search 步骤
2. 如果需要计算，规划 calculator 步骤
3. 如果涉及已上传文件，规划 file_content_analyzer 步骤
4. 如果不需要任何工具，输出"无需工具，直接回答"

输出格式：每行一个步骤，用数字编号。"""

    response = llm.invoke([HumanMessage(content=prompt)])
    plan = response.content.strip()
    return {"plan": plan}


# --- 3. ReAct 模型调用节点（核心！）---------------------------------
def node_call_model(state: AgentState) -> dict:
    """带工具绑定的 LLM 调用，支持 ReAct 模式"""
    llm = _get_llm_with_tools()
    messages = list(state.get("messages", []))

    # 构建系统提示
    system_parts = [
        "你是一个智能AI助手，可以自主使用工具来回答用户问题。\n",
        "## 可用工具",
        f"- knowledge_base_search：从知识库中检索信息",
        f"- calculator：执行数学计算",
        f"- file_content_analyzer：分析已上传文件内容",
    ]

    # 注入长期记忆
    memory = state.get("memory_context", "")
    if memory:
        system_parts.append(f"\n## 历史相关记忆\n{memory}")

    # 注入执行计划
    plan = state.get("plan", "")
    if plan and "无需工具" not in plan:
        system_parts.append(f"\n## 执行计划\n请参考以下计划执行，完成所有步骤后再回答：\n{plan}")

    # 注入反思反馈（修订时）
    critique = state.get("critique", "")
    if critique:
        system_parts.append(f"\n## 反思反馈\n上次回答需要改进：{critique}")

    system_parts.append(
        "\n## 行为准则\n"
        "- 按计划逐步执行，每一步调用一个工具\n"
        "- 获取足够信息后再综合回答\n"
        "- 工具返回'没有找到相关信息'时，不要重复调用同一工具，直接告诉用户\n"
        "- 最多调用 3 次工具，超出后必须直接回答\n"
        "- 知识库无相关内容时如实告知，不要编造\n"
        "- 回答简洁专业"
    )

    system_msg = SystemMessage(content="\n".join(system_parts))
    response = llm.invoke([system_msg] + messages)
    # 手动累积消息（不依赖 LangGraph 的 add_messages reducer）
    return {"messages": messages + [response]}


# --- 4. 工具执行节点（手动累积消息）---------------------------------
_tool_node = ToolNode(AGENT_TOOLS)

def node_call_tool(state: AgentState) -> dict:
    """执行工具调用，手动累积 ToolMessage"""
    messages = list(state.get("messages", []))
    result = _tool_node.invoke(state)
    tool_messages = list(result.get("messages", []))
    return {"messages": messages + tool_messages}


# --- 5. 反思节点 ----------------------------------------------------
def node_reflector(state: AgentState) -> dict:
    """自我评估回答质量，决定是否需要修订"""
    llm = _get_llm()
    messages = state.get("messages", [])

    # 找到最后一条 AI 消息
    last_ai_msg = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            last_ai_msg = msg
            break

    if not last_ai_msg:
        return {"critique": "", "revision_count": state.get("revision_count", 0)}

    prompt = f"""你是质量审查员。评估以下回答的质量。

用户问题：{state.get("user_input", "")}
执行计划：{state.get("plan", "")}
AI回答：{last_ai_msg.content}

评估标准：
1. 是否完整回答了用户问题？
2. 是否充分利用了工具检索到的信息？
3. 是否有遗漏或错误？

请输出 JSON（仅 JSON，不要其他文字）：
{{"satisfied": true/false, "critique": "不满意的原因或空字符串"}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            satisfied = result.get("satisfied", True)
            critique = result.get("critique", "")
        else:
            satisfied, critique = True, ""
    except Exception:
        satisfied, critique = True, ""

    revision_count = state.get("revision_count", 0)
    return {
        "critique": "" if satisfied else critique,
        "revision_count": revision_count + (0 if satisfied else 1),
    }


# --- 6. 记忆存储节点 ------------------------------------------------
def node_save_memory(state: AgentState) -> dict:
    """将本次对话摘要存入 Chroma 长期记忆"""
    username = state.get("username", "unknown")
    session_id = state.get("session_id", "default")
    user_input = state.get("user_input", "")

    # 找到最后一条 AI 回答
    messages = state.get("messages", [])
    last_response = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            last_response = msg.content
            break

    if not last_response:
        return {}

    # 生成摘要
    try:
        llm = _get_llm()
        summary = llm.invoke([
            HumanMessage(
                content=f"用一句话总结以下对话的核心信息：\n用户：{user_input}\n回答：{last_response}"
            )
        ]).content.strip()

        memory_store.add_memory(username, session_id, summary, user_input, last_response)
    except Exception as e:
        print(f"[Agent] 保存长期记忆失败: {e}")

    return {}


# ====================================================================
#  条件边函数
# ====================================================================

def edge_after_call_model(state: AgentState) -> Literal["call_tool", "reflector"]:
    """ReAct 循环决策：有工具调用就去执行工具，否则去反思"""
    messages = state.get("messages", [])
    if not messages:
        return "reflector"

    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        # 防死循环：统计本轮 AI 发起工具调用的次数，超过 3 次强制结束
        tool_call_count = sum(
            1 for msg in messages
            if isinstance(msg, AIMessage)
            and hasattr(msg, "tool_calls") and msg.tool_calls
        )
        if tool_call_count > 3:
            return "reflector"
        return "call_tool"
    return "reflector"


def edge_after_reflector(state: AgentState) -> Literal["call_model", "save_memory"]:
    """反思决策：不满意且未达上限则修订，否则存记忆结束"""
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    if critique and revision_count < 2:
        return "call_model"      # 需要修订，回到模型重新回答
    return "save_memory"         # 通过或已达上限，结束


# ====================================================================
#  构建 Agent 图
# ====================================================================

def build_agent() -> StateGraph:
    """构建完整的 Agent 工作流图"""
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("load_memory", node_load_memory)
    graph.add_node("planner", node_planner)
    graph.add_node("call_model", node_call_model)
    graph.add_node("call_tool", node_call_tool)
    graph.add_node("reflector", node_reflector)
    graph.add_node("save_memory", node_save_memory)

    # 边：起始 → 记忆检索
    graph.set_entry_point("load_memory")

    # 记忆检索 → 规划 → 执行
    graph.add_edge("load_memory", "planner")
    graph.add_edge("planner", "call_model")

    # ReAct 循环：模型 <→ 工具
    graph.add_conditional_edges(
        "call_model",
        edge_after_call_model,
        {"call_tool": "call_tool", "reflector": "reflector"},
    )
    graph.add_edge("call_tool", "call_model")

    # 反思循环：反思 → 修订 / 结束
    graph.add_conditional_edges(
        "reflector",
        edge_after_reflector,
        {"call_model": "call_model", "save_memory": "save_memory"},
    )

    # 记忆存储 → 结束
    graph.add_edge("save_memory", END)

    return graph.compile()


# ====================================================================
#  全局单例
# ====================================================================
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        print("[Agent] 初始化 Agent 工作流图...")
        _agent = build_agent()
        print("[Agent] Agent 工作流图初始化完成")
    return _agent