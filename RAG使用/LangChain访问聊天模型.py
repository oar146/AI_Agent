from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatTongyi(model="qwen3-max")

messages = [SystemMessage(content="你是明朝皇帝朱元璋"),
    HumanMessage(content="给我展示最像你会说的一段话")]

for chunk in model.stream(input=messages):
    print(chunk.content,end="",flush=True)