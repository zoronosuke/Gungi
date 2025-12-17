@echo off
echo ============================================
echo Gungi AI Training - FAST MODE
echo ============================================
echo.
echo 高速学習設定（ゲーム数重視）:
echo   - 並行ゲーム数: 64
echo   - ゲーム/イテレーション: 50
echo   - 総イテレーション: 100
echo   - MCTS: 30 (軽量化)
echo   - 総ゲーム数: 5,000
echo.
echo GPU使用率は30-50%%程度ですが、
echo これはCPUでのゲームロジックがボトルネックのため
echo （AlphaZeroの典型的な特性）
echo.
echo Ctrl+C で中断可能
echo ============================================
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts/train.py --test --parallel-games 64 --games 50 --iterations 100 --mcts-sims 30

pause
