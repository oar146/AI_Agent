import time

import streamlit as st
from knowledge_base import KnowledgeBaseService

# 自定义 CSS 样式
st.markdown("""
<style>
    /* 标题粉色渐变 */
    .main h1 {
        background: linear-gradient(90deg, #ff6b9d, #c44dff, #ff6b9d);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient 3s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% center; }
        50% { background-position: 100% center; }
        100% { background-position: 0% center; }
    }

    /* Upload 按钮渐变 */
    .stFileUploader [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #ff6b9d 0%, #c44dff 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }

    .stFileUploader [data-testid="stFileUploaderDropzone"] button:hover {
        background: linear-gradient(135deg, #c44dff 0%, #ff6b9d 100%) !important;
        transform: scale(1.05) !important;
        box-shadow: 0 4px 15px rgba(255, 107, 157, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)



# 添加网页标题
st.title('文件上传')
# 文件上传file_uploader
file = st.file_uploader('请上传文件',
                        type=['txt'],
                        accept_multiple_files=False,# 是否允许上传多个文件 这里是false是不允许上传多个文件
                        )
#  session_state就是一个字典
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if file is not None:
    file_name = file.name
    file_type = file.type
    file_size = file.size / 1024   # 获取KB单位

    st.subheader(f"文件名:{file_name}")   #也是标题 只是这个是子标题
    st.write(f"文件类型:{file_type} | 文件大小:{file_size:.2f}KB")
# 使用get_value 来获取文件的内容 但获取到的是字节数据 要通过编码转换成字符串
    text = file.getvalue().decode('utf-8')
    with st.spinner("文件处理中..."):   # 在spinner内的代码执行过程中，会有一个转圈动画
         time.sleep(3)
         result = st.session_state["service"].uploder_by_str(text,file_name)
         st.write(result)