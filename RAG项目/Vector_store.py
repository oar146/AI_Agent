from langchain_chroma import Chroma
import config_data as config

class VectorStoreService(object):
    def __init__(self, embedding):
        '''embedding 是嵌入模型的传入'''
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory   # 存储在这里 这个配置了数据库的路径 或者说是名字
        )

    def get_retriever(self):
        '''返回向量检索器,方便加入chain'''
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})
if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    embedding = DashScopeEmbeddings(model="text-embedding-v4")
    vector_store = VectorStoreService(embedding)
    retriever = vector_store.get_retriever()
    res=retriever.invoke("我的体重180斤，给我尺码推荐")
    print(res)