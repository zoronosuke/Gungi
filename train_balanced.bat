@echo off
echo ============================================
echo Gungi AI Training - BALANCED MODE (推奨)
echo ============================================
echo.
echo バランス型設定:
echo   - 並行ゲーム数: 48
echo   - ゲーム/イテレーション: 30
echo   - 総イテレーション: 150
echo   - MCTS: 50 (バランス)
echo   - 総ゲーム数: 4,500
echo   - 予想時間: 約4-5時間
echo.
echo 品質と量のバランスを取った設定です
echo.
echo Ctrl+C で中断可能
echo ============================================
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts/train.py --test --parallel-games 48 --games 30 --iterations 150 --mcts-sims 50

pause
