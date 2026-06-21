from langchain_community.embeddings import DashScopeEmbeddings
embed = DashScopeEmbeddings()
# 测试
print(embed.embed_query("你好"))
print(embed.embed_documents(["你好", "世界"]))