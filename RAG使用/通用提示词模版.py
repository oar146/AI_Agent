from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi
#基于chain的写法
template = PromptTemplate.from_template("我的邻居姓{lastname},刚生了{gender}，帮忙起名字 简洁回答")
model = Tongyi(model="qwen-max")
# 生成链
chain = template| model
# 基于链，调用模型生成结果
res = chain.invoke(input={"lastname":"丁", "gender":"儿子"})
print(res)