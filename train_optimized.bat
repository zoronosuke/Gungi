@echo off
echo ============================================
echo Gungi AI Training - Optimized Settings
echo ============================================
echo.
echo 学習設定:
echo   - 並行ゲーム数: 32
echo   - ゲーム/イテレーション: 20
echo   - 総イテレーション: 100
echo   - MCTS: 50
echo   - 総ゲーム数: 2,000
echo   - 予想時間: 約4-5時間
echo.
echo Ctrl+C で中断可能（チェックポイント保存されます）
echo --resume で再開可能
echo ============================================
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts/train.py --test --parallel-games 32 --games 20 --iterations 100

pause
