@echo off
echo ============================================
echo Gungi AI Training - HIGH GPU UTILIZATION
echo ============================================
echo.
echo GPU使用率を最大化する設定:
echo   - 並行ゲーム数: 128 (GPUにもっと仕事を)
echo   - ゲーム/イテレーション: 30
echo   - 総イテレーション: 100
echo   - MCTS: 50
echo   - 総ゲーム数: 3,000
echo.
echo Ctrl+C で中断可能（チェックポイント保存されます）
echo ============================================
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts/train.py --test --parallel-games 128 --games 30 --iterations 100

pause
