import os
import json
import uuid
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel

from rag import RagService
from auth import init_database, register_user, login_user, logout_user, validate_token
from file_parser import parse_file

app = FastAPI(title="智能对话助手")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 RAG 服务
rag_service = RagService()

# 初始化数据库（启动时自动创建 user 表）
try:
    init_database()
except Exception as e:
    print(f"[警告] 数据库初始化失败: {e}，请检查 MySQL 连接配置")

# 历史记录存储路径
CHAT_HISTORY_DIR = os.path.join(os.path.dirname(__file__), "chat_history")
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)


def _get_current_user(authorization: str = "") -> str:
    """从 Authorization header 中提取并验证当前用户名"""
    token = authorization.replace("Bearer ", "") if authorization else ""
    username = validate_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    return username


# ========== 数据模型 ==========

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class AuthRequest(BaseModel):
    username: str
    password: str


# ========== 认证 API 路由 ==========

@app.post("/api/auth/register")
async def api_register(req: AuthRequest):
    """用户注册"""
    if len(req.username) < 2 or len(req.username) > 32:
        return JSONResponse(status_code=400, content={"message": "用户名长度需在 2-32 个字符之间"})
    if len(req.password) < 6:
        return JSONResponse(status_code=400, content={"message": "密码长度不能少于 6 位"})
    success, msg = register_user(req.username, req.password)
    if not success:
        return JSONResponse(status_code=400, content={"message": msg})
    return {"message": msg}


@app.post("/api/auth/login")
async def api_login(req: AuthRequest):
    """用户登录"""
    success, msg, token = login_user(req.username, req.password)
    if not success:
        return JSONResponse(status_code=401, content={"message": msg})
    return {"message": msg, "token": token, "username": req.username}


@app.post("/api/auth/logout")
async def api_logout(authorization: str = Header("")):
    """用户退出"""
    token = authorization.replace("Bearer ", "") if authorization else ""
    logout_user(token)
    return {"message": "已退出"}


@app.get("/api/auth/verify")
async def api_verify(authorization: str = Header("")):
    """验证令牌"""
    token = authorization.replace("Bearer ", "") if authorization else ""
    username = validate_token(token)
    if not username:
        return JSONResponse(status_code=401, content={"message": "未登录或令牌已过期"})
    return {"username": username}


# ========== API 路由 ==========

@app.get("/api/sessions")
async def list_sessions(authorization: str = Header("")):
    """获取当前用户的所有会话列表"""
    username = _get_current_user(authorization)
    user_dir = os.path.join(CHAT_HISTORY_DIR, username)
    if not os.path.exists(user_dir):
        return {"sessions": []}
    sessions = []
    for fname in os.listdir(user_dir):
        filepath = os.path.join(user_dir, fname)
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            preview = ""
            for msg in data:
                if msg.get("type") == "human":
                    content = msg.get("data", {}).get("content", "")
                    preview = content[:60] if content else ""
                    break
            sessions.append({
                "id": fname,
                "preview": preview or "空对话",
                "message_count": len(data) if isinstance(data, list) else 0,
            })
        except Exception:
            sessions.append({"id": fname, "preview": "空对话", "message_count": 0})
    # 按文件名倒序（最新的在前）
    sessions.sort(key=lambda x: x["id"], reverse=True)
    return {"sessions": sessions}


@app.post("/api/sessions")
async def create_session(authorization: str = Header("")):
    """创建新会话"""
    username = _get_current_user(authorization)
    session_id = str(uuid.uuid4())
    user_dir = os.path.join(CHAT_HISTORY_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    filepath = os.path.join(user_dir, session_id)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump([], f)
    return {"session_id": session_id}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, authorization: str = Header("")):
    """删除会话"""
    username = _get_current_user(authorization)
    filepath = os.path.join(CHAT_HISTORY_DIR, username, session_id)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="会话不存在")
    os.remove(filepath)
    return {"message": "删除成功"}


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, authorization: str = Header("")):
    """获取会话消息历史"""
    username = _get_current_user(authorization)
    filepath = os.path.join(CHAT_HISTORY_DIR, username, session_id)
    if not os.path.exists(filepath):
        return {"messages": []}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        messages = []
        for msg in data:
            msg_type = msg.get("type", "")
            content = msg.get("data", {}).get("content", "")
            role = "user" if msg_type == "human" else "assistant"
            messages.append({"role": role, "content": content})
        return {"messages": messages}
    except Exception:
        return {"messages": []}


@app.post("/api/chat")
async def chat(request: ChatRequest, authorization: str = Header("")):
    """聊天接口（SSE 流式响应）"""
    username = _get_current_user(authorization)
    # 使用 "用户名/会话ID" 作为 session_id，实现历史记录按用户隔离
    namespaced_session_id = f"{username}/{request.session_id}"
    session_config = {
        "configurable": {
            "session_id": namespaced_session_id,
        }
    }

    async def generate():
        try:
            async for chunk in rag_service.chain.astream(
                {"input": request.message},
                session_config,
            ):
                if chunk:
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/with-file")
async def chat_with_file(
    message: str = Form(...),
    session_id: str = Form("default"),
    file: UploadFile = File(...),
    authorization: str = Header(""),
):
    """带文件上传的聊天接口（SSE 流式响应）"""
    username = _get_current_user(authorization)
    namespaced_session_id = f"{username}/{session_id}"
    session_config = {
        "configurable": {
            "session_id": namespaced_session_id,
        }
    }

    # 解析文件内容
    try:
        file_bytes = await file.read()
        file_content = parse_file(file_bytes, file.filename)
        # 将文件内容拼接到用户消息中
        full_message = (
            f"用户上传了文件《{file.filename}》，内容如下：\n"
            f"---文件内容开始---\n{file_content}\n---文件内容结束---\n\n"
            f"用户提问：{message}"
        )
    except Exception as e:
        full_message = message

    async def generate():
        try:
            async for chunk in rag_service.chain.astream(
                {"input": full_message},
                session_config,
            ):
                if chunk:
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# 前端静态文件
@app.get("/")
async def serve_frontend():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "static", "index.html")
    )


@app.get("/login")
async def serve_login():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "static", "login.html")
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)