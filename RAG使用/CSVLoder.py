from langchain_community.document_loaders import CSVLoader
loader = CSVLoader(file_path=r"E:\AI_RAG\RAG使用\stu.csv")
#批量加载
docs = loader.load()

print(docs)