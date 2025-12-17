@echo off
echo ============================================
echo Gungi AI Training - 1 WEEK MARATHON
echo ============================================
echo.
echo 1週間本格学習設定:
echo   - 並行ゲーム数: 64
echo   - ゲーム/イテレーション: 25
echo   - 総イテレーション: 2000
echo   - MCTS: 100 (高品質)
echo   - 総ゲーム数: 50,000
echo   - 予想時間: 約5-7日
echo.
echo 強いAIを目指す本格設定です！
echo.
echo Ctrl+C で中断可能（--resume で再開）
echo ============================================
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts/train.py --test --parallel-games 64 --games 25 --iterations 2000 --mcts-sims 100

pause
