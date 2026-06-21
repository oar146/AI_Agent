from openai import OpenAI
#1.获取client对象
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
#2.调用模型
response = client.chat.completions.create(
    model="qwen3-max-2026-01-23",
    messages=[
        {"role": "system", "content": "你是我的助手，名叫4-，并且话多"},
        {"role": "assistant", "content": "我是4-，并且话多，你需要什么帮助？"},
        {"role": "user", "content": "你的名字是什么"},
    ],
    stream = True,  #开启流式输出
)
#3处理结果
#print(response.choices[0].message.content)
for delta in response:
    if delta.choices[0].delta.content:
        print(delta.choices[0].delta.content, end="", flush=True)#flush是用来刷新缓冲区 让缓冲区的数据立刻输出