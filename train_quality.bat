@echo off
echo ============================================
echo Gungi AI Training - QUALITY MODE (長期学習用)
echo ============================================
echo.
echo 品質重視設定（長期学習に最適）:
echo   - 並行ゲーム数: 64
echo   - ゲーム/イテレーション: 20
echo   - 総イテレーション: 500
echo   - MCTS: 100 (高品質)
echo   - 総ゲーム数: 10,000
echo   - 予想時間: 約10-12時間
echo.
echo 長時間学習で強いAIを目指す設定です
echo.
echo Ctrl+C で中断可能（--resume で再開）
echo ============================================
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts/train.py --test --parallel-games 64 --games 20 --iterations 500 --mcts-sims 100

pause
