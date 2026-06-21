"""知识库检索工具：供 Agent 调用的知识库搜索功能"""
from langchain_core.tools import tool
from Vector_store import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config


# 全局复用 VectorStoreService（与 rag.py 共用同一个 Chroma 实例）
_vector_service = None


def _get_vector_service():
    global _vector_service
    if _vector_service is None:
        embedding = DashScopeEmbeddings(model=config.embedding_model_name)
        _vector_service = VectorStoreService(embedding=embedding)
    return _vector_service


@tool
def knowledge_base_search(query: str) -> str:
    """从知识库中检索与 query 相关的信息。
    当用户的问题可能涉及知识库中的文档内容时，调用此工具来获取参考资料。"""
    vector_service = _get_vector_service()
    retriever = vector_service.get_retriever()
    docs = retriever.invoke(query)

    if not docs:
        return "知识库中没有找到相关信息。"

    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"[{i}] {doc.page_content}")
    return "\n\n".join(results)