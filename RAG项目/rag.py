from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from Vector_store import VectorStoreService
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, format_document,MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi


class RagService(object):
    def __init__(self):
        self.Vector_service = VectorStoreService(
            embedding= DashScopeEmbeddings(model=config.embedding_model_name))

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system","以我提供的已知参考资料为主，""简洁和专业的回答用户问题。参考资料:{context}."),
            ("system","并且我提供用户的对话历史记录"),
            MessagesPlaceholder("history"),
            ("user","请回答用户提问:{input}")
        ])

        self.chat_model = ChatTongyi(model=config.chat_model_name)

        self.chain = self.__get_chain()

    def __get_chain(self):
        "获取最终的执行链"
        retriever = self.Vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"
            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段:\n{doc.page_content}\n文档元数据:\n{doc.metadata}\n\n"
            return formatted_str
        def temp1(value:dict)->str:
            return value['input']
        def temp2(value):
            new_value={}
            new_value['input']=value['input']['input']
            new_value['context']=value["context"]
            new_value['history']=value["input"]['history']
            return new_value
        chain = ({
            "input": RunnablePassthrough(),
            "context":RunnableLambda(temp1) | retriever | format_document
        } | RunnableLambda(temp2) | self.prompt_template | self.chat_model | StrOutputParser()

        )

        conversion_chain = RunnableWithMessageHistory(
             chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",

        )
        return conversion_chain
if __name__ == '__main__':
    # session_id配置
    session_config={
        "configurable":{
            "session_id": "user_001",
        }
    }
    rag_service = RagService()
    res = rag_service.chain.invoke({"input":"我的体重180斤，给我尺码推荐"},session_config)
    print(res)