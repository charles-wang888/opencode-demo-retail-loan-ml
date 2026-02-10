"""
配置：模型后端、工作目录、opencode 服务器地址等。
使用 opencode-sdk 调用 opencode 服务器。
"""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# 项目根目录（core 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据与状态目录
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------- OpenCode 服务器配置 ----------
# opencode 服务器地址（默认本地）
OPENCODE_BASE_URL = os.getenv("OPENCODE_BASE_URL", "http://localhost:4096")

# 模型配置（与 opencode.json 中 model 对应；若为 "deepseek-chat" 则 provider=deepseek, model_id=deepseek-chat）
OPENCODE_MODEL_PROVIDER = os.getenv("OPENCODE_MODEL_PROVIDER", "deepseek")
OPENCODE_MODEL_ID = os.getenv("OPENCODE_MODEL_ID", "deepseek-chat")

# Web UI：默认监听所有网卡，可通过 http://localhost:7860 访问
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))
