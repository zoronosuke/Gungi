#!/usr/bin/env python
"""
軍儀 開発サーバ起動スクリプト
"""

import sys
import os

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# appを直接インポート
from src.api.main import app
import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("軍儀 (Gungi) 開発サーバを起動します")
    print("=" * 60)
    print("APIサーバ: http://localhost:8001")
    print("API ドキュメント: http://localhost:8001/docs")
    print("=" * 60)
    print()
    
    # appオブジェクトを直接渡す
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,  # Windowsでは問題が出やすいのでオフ
        log_level="info"
    )
