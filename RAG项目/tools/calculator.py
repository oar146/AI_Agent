"""计算器工具：供 Agent 进行数学计算"""
import math
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """执行数学计算并返回结果。
    支持四则运算、括号、数学函数（如 math.sqrt, math.sin 等）。
    输入应为合法的 Python 数学表达式，例如："(180 - 80) * 0.7" 或 "math.sqrt(144)"。"""
    try:
        # 替换常见数学函数
        expr = expression.replace("^", "**")
        # 在受限的命名空间中执行
        namespace = {"math": math}
        result = eval(expr, {"__builtins__": {}}, namespace)
        return str(result)
    except Exception as e:
        return f"计算失败: {str(e)}"