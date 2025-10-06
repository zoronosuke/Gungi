@echo off
REM 軍儀 FastAPI サーバ起動スクリプト (Windows)

cd /d "%~dp0"
set PYTHONPATH=%CD%

echo ============================================================
echo 軍儀 (Gungi) FastAPI サーバを起動します
echo ============================================================
echo.
echo APIサーバ: http://localhost:8000
echo APIドキュメント: http://localhost:8000/docs
echo ============================================================
echo.

REM 仮想環境をチェック
if exist venv\Scripts\python.exe (
    echo 仮想環境を使用します
    venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
) else (
    echo システムのPythonを使用します
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
)
