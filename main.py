"""
零售贷款智能运营 — 入口。
启动 Gradio Web UI，通过 opencode-sdk 调用 opencode 服务器。
"""
import os
os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

import argparse
import gradio as gr

from core.config import HOST, PORT
from ui.app import create_app


def parse_args():
    parser = argparse.ArgumentParser(description="零售贷款智能运营（Agent）")
    parser.add_argument("--host", default=HOST, help="监听地址")
    parser.add_argument("--port", type=int, default=PORT, help="端口")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    return parser.parse_args()


def main():
    args = parse_args()
    app = create_app()
    print(f"启动后请在浏览器打开: http://localhost:{args.port}")
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=".block { max-width: 1200px !important; }",
    )


if __name__ == "__main__":
    main()
