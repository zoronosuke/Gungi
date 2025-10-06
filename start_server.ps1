# 軍儀 FastAPI サーバ起動スクリプト (PowerShell)

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $SCRIPT_DIR
$env:PYTHONPATH = $SCRIPT_DIR

Write-Host "============================================================"
Write-Host "軍儀 (Gungi) FastAPI サーバを起動します"
Write-Host "============================================================"
Write-Host ""
Write-Host "APIサーバ: http://localhost:8000"
Write-Host "APIドキュメント: http://localhost:8000/docs"
Write-Host "============================================================"
Write-Host ""

# 仮想環境をチェック
if (Test-Path "venv\Scripts\python.exe") {
    Write-Host "仮想環境を使用します"
    & "venv\Scripts\python.exe" -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
} else {
    Write-Host "システムのPythonを使用します"
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
}
