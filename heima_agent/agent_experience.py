from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool


@tool(description="获取天气信息")
def get_weather()->str:
    return "阴天"

agent = create_agent(
    model = ChatTongyi(model="qwen3-max"),
    tools = [get_weather],
    system_prompt="你是一个聊天助手，请用中文回答问题。"
)

res = agent.invoke(
    {'messages':[
        {"role": "user", "content": "今天福州天气如何？"}
    ]}
)

parser = StrOutputParser()

for msg in res["messages"]:
    print (msg.content)
