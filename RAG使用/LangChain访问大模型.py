from langchain_community.llms.tongyi import Tongyi

model = Tongyi(model="qwen-max")
#调用invoke向大语言模型提问
res = model.invoke(input="你是谁?能做什么?")

print(res)