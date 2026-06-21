"""文件分析工具：供 Agent 调用的文件内容分析功能"""
from langchain_core.tools import tool


# 临时文件存储，由上传接口写入，Agent 读取
_file_storage: dict[str, str] = {}


def store_file_content(file_id: str, content: str) -> None:
    """存储解析后的文件内容，供 Agent 后续调用"""
    _file_storage[file_id] = content


@tool
def file_content_analyzer(file_id: str) -> str:
    """分析已上传文件的内容。
    当你需要了解用户上传的文件内容时，使用 file_id 来获取并分析文件内容。"""
    content = _file_storage.get(file_id)
    if content is None:
        return f"未找到文件（file_id: {file_id}），文件可能已过期或尚未上传。"
    return content