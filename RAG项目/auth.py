import hashlib
import secrets
import pymysql
from config_data import MYSQL_CONFIG


def get_connection():
    """获取 MySQL 数据库连接"""
    return pymysql.connect(
        host=MYSQL_CONFIG["host"],
        port=MYSQL_CONFIG["port"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        charset=MYSQL_CONFIG["charset"],
    )


def init_database():
    """初始化数据库和 user 表"""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # 创建数据库
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{MYSQL_CONFIG['database']}` "
                f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cursor.execute(f"USE `{MYSQL_CONFIG['database']}`")
            # 创建 user 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS `user` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `username` VARCHAR(64) NOT NULL UNIQUE,
                    `password_hash` VARCHAR(128) NOT NULL,
                    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX `idx_username` (`username`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
        conn.commit()
    finally:
        conn.close()


def _hash_password(password: str) -> str:
    """对密码进行哈希处理"""
    salt = "RagChatSalt2024"
    return hashlib.sha256((password + salt).encode()).hexdigest()


def _generate_token(username: str) -> str:
    """生成会话令牌"""
    raw = f"{username}:{secrets.token_hex(32)}"
    return hashlib.sha256(raw.encode()).hexdigest()


# 简单的内存令牌存储 {token: username}
_token_store: dict[str, str] = {}


def register_user(username: str, password: str) -> tuple[bool, str]:
    """注册新用户，返回 (成功, 消息)"""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"USE `{MYSQL_CONFIG['database']}`")
            # 检查用户名是否已存在
            cursor.execute("SELECT id FROM `user` WHERE username = %s", (username,))
            if cursor.fetchone():
                return False, "用户名已存在"
            # 插入新用户
            password_hash = _hash_password(password)
            cursor.execute(
                "INSERT INTO `user` (username, password_hash) VALUES (%s, %s)",
                (username, password_hash),
            )
        conn.commit()
        return True, "注册成功"
    except pymysql.MySQLError as e:
        return False, f"注册失败: {str(e)}"
    finally:
        conn.close()


def login_user(username: str, password: str) -> tuple[bool, str, str]:
    """用户登录，返回 (成功, 消息, token)"""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"USE `{MYSQL_CONFIG['database']}`")
            cursor.execute(
                "SELECT password_hash FROM `user` WHERE username = %s",
                (username,),
            )
            row = cursor.fetchone()
            if not row:
                return False, "登录失败：用户名或密码错误", ""
            stored_hash = row[0]
            if stored_hash != _hash_password(password):
                return False, "登录失败：用户名或密码错误", ""
            # 生成并存储令牌
            token = _generate_token(username)
            _token_store[token] = username
            return True, "登录成功", token
    except pymysql.MySQLError as e:
        return False, f"登录失败: {str(e)}", ""
    finally:
        conn.close()


def validate_token(token: str) -> str | None:
    """验证令牌，返回用户名或 None"""
    return _token_store.get(token)


def logout_user(token: str) -> bool:
    """退出登录，移除令牌"""
    return bool(_token_store.pop(token, None))