
"""
RRF
Reciprocal Rank Fusion
混合检索重排序算法
结合全文检索（BM25/关键词匹配）和向量检索（语义相似度）
"""

import numpy as np
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from collections import Counter
import math


class HybridRetriever:
    """混合检索器：结合全文检索和向量检索"""

    def __init__(self, vector_retriever, alpha=0.7):
        """
        初始化混合检索器

        Args:
            vector_retriever: 向量检索器
            alpha: 向量检索权重 (0-1)，全文检索权重为 (1-alpha)
                  alpha=0.7 表示向量检索占70%，全文检索占30%
        """
        self.vector_retriever = vector_retriever
        self.alpha = alpha
        self.documents = []

    def set_documents(self, documents: List[Document]):
        """设置文档库用于全文检索"""
        self.documents = documents

    def full_text_search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        全文检索：基于TF-IDF的简单实现

        Args:
            query: 查询文本
            top_k: 返回前k个结果

        Returns:
            List of (Document, score) tuples
        """
        # 简单的分词（中文需要更复杂的分词，这里用字符级别）
        query_terms = self._tokenize(query)

        # 计算每个文档的TF-IDF分数
        doc_scores = []
        for doc in self.documents:
            doc_terms = self._tokenize(doc.page_content)
            score = self._calculate_tfidf_score(query_terms, doc_terms)
            if score > 0:
                doc_scores.append((doc, score))

        # 按分数排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        """
        简单分词（实际项目中建议使用jieba等中文分词工具）

        Args:
            text: 输入文本

        Returns:
            分词列表
        """
        # 简单实现：按空格和标点分割
        import re
        tokens = re.findall(r'\w+|[^\s\w]', text.lower())
        return [t for t in tokens if t.strip()]

    def _calculate_tfidf_score(self, query_terms: List[str], doc_terms: List[str]) -> float:
        """
        计算简化的TF-IDF分数

        Args:
            query_terms: 查询词列表
            doc_terms: 文档词列表

        Returns:
            TF-IDF分数
        """
        if not query_terms or not doc_terms:
            return 0.0

        # 计算词频
        doc_term_freq = Counter(doc_terms)

        # 计算分数：查询词在文档中的出现频率
        score = 0.0
        for term in query_terms:
            if term in doc_term_freq:
                # TF: 词频 / 文档总词数
                tf = doc_term_freq[term] / len(doc_terms)
                # 简化的IDF（假设所有文档都包含该词）
                idf = 1.0
                score += tf * idf

        return score

    def vector_search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        向量检索

        Args:
            query: 查询文本
            top_k: 返回前k个结果

        Returns:
            List of (Document, score) tuples
        """
        # 调用向量检索器
        docs = self.vector_retriever.invoke(query)

        # 这里假设文档已经有相似度分数，实际中可能需要调整
        # 如果没有分数，给一个默认值
        results = []
        for i, doc in enumerate(docs[:top_k]):
            # LangChain的文档可能没有score属性，这里用排名作为分数
            score = 1.0 / (i + 1)  # 排名越靠前分数越高
            results.append((doc, score))

        return results

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Min-Max归一化分数到[0, 1]范围

        Args:
            scores: 原始分数列表

        Returns:
            归一化后的分数列表
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [0.5] * len(scores)

        normalized = [(s - min_score) / (max_score - min_score) for s in scores]
        return normalized

    def hybrid_search(self, query: str, top_k: int = 10,
                      ft_top_k: int = 20, vec_top_k: int = 20) -> List[Tuple[Document, float]]:
        """
        混合检索：结合全文检索和向量检索并重排序

        Args:
            query: 查询文本
            top_k: 最终返回的结果数量
            ft_top_k: 全文检索候选数量
            vec_top_k: 向量检索候选数量

        Returns:
            重排序后的 (Document, score) 列表
        """
        # 1. 执行全文检索
        ft_results = self.full_text_search(query, top_k=ft_top_k)

        # 2. 执行向量检索
        vec_results = self.vector_search(query, top_k=vec_top_k)

        # 3. 合并结果（去重）
        doc_dict = {}  # {doc_id: (doc, ft_score, vec_score)}

        # 添加全文检索结果
        for doc, score in ft_results:
            doc_id = id(doc)
            doc_dict[doc_id] = (doc, score, 0.0)

        # 添加向量检索结果
        for doc, score in vec_results:
            doc_id = id(doc)
            if doc_id in doc_dict:
                # 已存在，更新向量分数
                old_doc, ft_score, _ = doc_dict[doc_id]
                doc_dict[doc_id] = (old_doc, ft_score, score)
            else:
                doc_dict[doc_id] = (doc, 0.0, score)

        # 4. 提取分数并归一化
        all_docs = list(doc_dict.values())
        ft_scores = [item[1] for item in all_docs]
        vec_scores = [item[2] for item in all_docs]

        normalized_ft = self.normalize_scores(ft_scores)
        normalized_vec = self.normalize_scores(vec_scores)

        # 5. 计算混合分数
        hybrid_results = []
        for i, (doc, ft_score, vec_score) in enumerate(all_docs):
            # 加权融合
            final_score = (
                    self.alpha * normalized_vec[i] +
                    (1 - self.alpha) * normalized_ft[i]
            )
            hybrid_results.append((doc, final_score))

        # 6. 按混合分数排序
        hybrid_results.sort(key=lambda x: x[1], reverse=True)

        return hybrid_results[:top_k]

    def reciprocal_rank_fusion(self, query: str, top_k: int = 10,
                               ft_top_k: int = 20, vec_top_k: int = 20,
                               k_param: float = 60.0) -> List[Tuple[Document, float]]:
        """
        使用倒数排名融合(RRF)算法进行混合检索

        Args:
            query: 查询文本
            top_k: 最终返回的结果数量
            ft_top_k: 全文检索候选数量
            vec_top_k: 向量检索候选数量
            k_param: RRF平滑参数，通常取60

        Returns:
            RRF重排序后的 (Document, score) 列表
        """
        # 1. 执行两种检索
        ft_results = self.full_text_search(query, top_k=ft_top_k)
        vec_results = self.vector_search(query, top_k=vec_top_k)

        # 2. 计算RRF分数
        doc_rrf_scores = {}  # {doc_id: (doc, rrf_score)}

        # 全文检索的RRF分数
        for rank, (doc, _) in enumerate(ft_results, 1):
            doc_id = id(doc)
            if doc_id not in doc_rrf_scores:
                doc_rrf_scores[doc_id] = (doc, 0.0)
            old_doc, old_score = doc_rrf_scores[doc_id]
            doc_rrf_scores[doc_id] = (old_doc, old_score + 1.0 / (k_param + rank))

        # 向量检索的RRF分数
        for rank, (doc, _) in enumerate(vec_results, 1):
            doc_id = id(doc)
            if doc_id not in doc_rrf_scores:
                doc_rrf_scores[doc_id] = (doc, 0.0)
            old_doc, old_score = doc_rrf_scores[doc_id]
            doc_rrf_scores[doc_id] = (old_doc, old_score + 1.0 / (k_param + rank))

        # 3. 转换为列表并排序
        rrf_results = list(doc_rrf_scores.values())
        rrf_results.sort(key=lambda x: x[1], reverse=True)

        return rrf_results[:top_k]


# 使用示例
if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    from Vector_store import VectorStoreService
    import config_data as config

    # 1. 初始化向量检索器
    embedding = DashScopeEmbeddings(model=config.embedding_model_name)
    vector_service = VectorStoreService(embedding=embedding)
    vector_retriever = vector_service.get_retriever()

    # 2. 创建混合检索器
    hybrid_retriever = HybridRetriever(vector_retriever, alpha=0.7)

    # 3. 设置文档库（从向量存储中获取所有文档）
    # 注意：实际使用时需要从数据库加载所有文档
    # 这里仅作示例
    all_docs = vector_service.vector_store.get()
    if all_docs['documents']:
        documents = [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(all_docs['documents'], all_docs['metadatas'])
        ]
        hybrid_retriever.set_documents(documents)

    # 4. 执行混合检索
    query = "我的体重180斤，给我尺码推荐"

    print("=" * 50)
    print("方法1: 加权融")