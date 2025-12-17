@echo off
REM クイックテスト用（30分〜1時間で効果確認）

echo ============================================================
echo Gungi AI - Quick Test Training (30min-1hour)
echo ============================================================
echo 改善の効果をすぐに確認するためのテスト設定
echo ============================================================
echo.

call .venv\Scripts\activate.bat

REM テスト設定
python scripts/train.py ^
    --optimized ^
    --mcts-sims 50 ^
    --games 10 ^
    --iterations 20 ^
    --parallel-games 16 ^
    --batch-size 128 ^
    --epochs 5 ^
    --lr 0.001 ^
    --resume

pause
