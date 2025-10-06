"""
最小限のテストサーバー
"""

import sys
sys.path.insert(0, '.')

from fastapi import FastAPI

app = FastAPI(title="軍儀 API テスト")

@app.get("/")
def root():
    return {"message": "軍儀 API is running!"}

@app.get("/test")
def test():
    # エンジンのインポートをテスト
    try:
        from src.engine import Board, Player
        board = Board()
        return {
            "status": "OK",
            "message": "Engine imported successfully",
            "player_black": Player.BLACK.name
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("軍儀テストサーバーを起動します")
    print("http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000)
