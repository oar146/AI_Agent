"""长期记忆系统：使用 Chroma 存储和检索历史对话摘要"""
import os
from datetime import datetime

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config

MEMORY_PERSIST_DIR = "./chroma_db_memory"
MEMORY_COLLECTION = "agent_memory"


class MemoryStore:
    """基于 Chroma 的长期记忆存储"""
    
    def __init__(self):
        os.makedirs(MEMORY_PERSIST_DIR, exist_ok=True)
        self.embedding = DashScopeEmbeddings(model=config.embedding_model_name)
        self.vector_store = Chroma(
            collection_name=MEMORY_COLLECTION,
            embedding_function=self.embedding,
            persist_directory=MEMORY_PERSIST_DIR,
        )

    def add_memory(self, username: str, session_id: str, summary: str, 
                   user_input: str, response: str) -> None:
        """存储一条对话记忆"""
        metadata = {
            "username": username,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "user_input_preview": user_input[:200],
        }
        try:
            self.vector_store.add_texts(
                texts=[summary],
                metadatas=[metadata],
            )
        except Exception as e:
            print(f"[MemoryStore] 保存记忆失败: {e}")

    def query_relevant(self, query: str, username: str, top_k: int = 3) -> list[str]:
        """检索与当前问题相关的历史记忆（按用户隔离）"""
        try:
            docs = self.vector_store.similarity_search(query, k=top_k * 2)
            # 只返回当前用户的记忆
            user_docs = [
                d for d in docs 
                if d.metadata.get("username") == username
            ]
            return [d.page_content for d in user_docs[:top_k]]
        except Exception as e:
            print(f"[MemoryStore] 检索记忆失败: {e}")
            return []


# 全局单例
memory_store = MemoryStore()