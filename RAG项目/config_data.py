md5_path = "./md5.text"


# Chroma
collection_name="rag"
persist_directory="./chroma_db"

# Splitter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", " ", "", "。", "！","?", "？", "；", "，"]
max_spliter_char_num=1000 #文本分割的阈值



#
similarity_threshold = 3    #检索返回匹配的文档数量

embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"

# ========== MySQL 配置 ==========
MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "rag_chat",
    "charset": "utf8mb4",
}