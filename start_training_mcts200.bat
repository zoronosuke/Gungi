@echo off
cd /d C:\Users\iamzo\Documents\GUNGI\Gungi
call .venv\Scripts\activate.bat
echo.
echo ===================================================
echo   Gungi AI - 深層強化学習トレーニング
echo   修正版: 引き分けバグ修正済み
echo ===================================================
echo   MCTS シミュレーション: 200回
echo   ゲーム数/イテレーション: 10
echo   総イテレーション数: 20
echo   修正内容:
echo     1. 引き分け報酬バグ修正
echo     2. MCTS内の引き分け評価を-0.1に変更
echo     3. MAX_MOVES統一（200手）
echo ===================================================
echo.
python scripts\train.py --mcts-sims 200 --games 10 --iterations 20 --batch-size 256
echo.
echo 学習が完了しました！
pause
