'''
知识库     防止文件重新上传的函数写在这里
'''
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
def chek_md5(md5_str:str):
    """检查传入的md5字符串是否已经被处理过了，用来文件去重
    return False表示该md5没有被处理过
    """
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_path, 'r', encoding='utf-8').readlines():
            line = line.strip()   #处理了字符串前后的空格和回车
            if line == md5_str:
                return True   # 已经处理过了
        return False

def save_md5(md5_str:str):
    """将传入的md5字符串，记录到文件内保存"""
    with open(config.md5_path, 'a', encoding='utf-8') as f:
        f.write(md5_str + '\n')

def get_string_md5(input_str:str,encoding='utf-8'):
    """传入一个字符串，返回md5
    字符串——》bytes——》md5 要走这个流程
    """
    #将字符串转换成bytes字节数据
    str_bytes = input_str.encode(encoding=encoding)

    # 创建md5对象
    md5_obj = hashlib.md5()#得到md5对象
    md5_obj.update(str_bytes)  # 更新md5对象(传入即将要转换的字节数据)
    md5_hex = md5_obj.hexdigest()  # 得到md5对象转换的16进制字符串
    return md5_hex

class KnowledgeBaseService(object):
    def __init__(self):
        # 如果文件夹不存在则创建 如果存在则跳过
        os.makedirs(config.persist_directory, exist_ok=True)
        self.chroma= Chroma(
            collection_name=config.collection_name,    #数据库的表名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),   #嵌入的模型是哪个
            persist_directory=config.persist_directory   #数据库本地存储文件夹路径
        )  #向量存储的实例 Chroma向量库对象  上面就是创建了一个数据库吧 只是在这里我们叫向量库


        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size = config.chunk_size,   # 分割后的文本段最大长度
            chunk_overlap= config.chunk_overlap,    #连续文本段之间的字符重叠数量
            separators= config.separators,      #  自然段划分的符号 就是那种空格 回车
            length_function=len     #使用python自带的len函数做长度统计的依据
        ) # 文本分割器的对象

    def uploder_by_str(self,data:str,filename):
        """传入一个字符串，进行向量化再存到向量数据库"""
        # 先得到字符串的md5值
        md5_hex = get_string_md5(data)
        if chek_md5(md5_hex):
            return "该文件已经处理过了"
        if len(data)>=config.chunk_size:
            knowledge_chunks:list [str]=self.spliter.split_text(data)
        else:
            knowledge_chunks=[data]
        metadata={"source":filename,"create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"operator":"曼巴"}
        self.chroma.add_texts(knowledge_chunks,metadatas=[metadata for _ in knowledge_chunks])  # 执行完这个内容就添加到向量库中了
        save_md5(md5_hex)
        return "上传成功"


if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.uploder_by_str("黑曼巴out","manba1.txt")
    print(r)
