@echo off
REM AlphaZero/将棋AIスタイルの推奨学習設定
REM RTX 3060 Ti 8GB向け - 長時間学習用

echo ============================================================
echo Gungi AI - AlphaZero-style Recommended Training
echo ============================================================
echo.
echo この設定は将棋AIやAlphaZeroを参考にした推奨設定です
echo - MCTSシミュレーション: 100回（精度と速度のバランス）
echo - 1イテレーションあたり25ゲーム
echo - 温度スケジューリング: 序盤30手は探索重視
echo - Dirichletノイズ: 探索の多様性を確保
echo - 引き分けペナルティ: -0.1（千日手を避ける学習）
echo.
echo 推定時間: 約1週間（2000イテレーション）
echo ============================================================
echo.

call .venv\Scripts\activate.bat

REM 推奨設定
REM - mcts-sims=100: 探索精度（将棋AIは800だが、計算コスト削減）
REM - games=25: 1イテレーションの多様性
REM - iterations=2000: 十分な学習回数
REM - parallel-games=32: メモリ効率と速度のバランス
REM - batch-size=256: 安定した学習
REM - epochs=10: 1イテレーションの学習密度
REM - lr=0.001: 標準的な学習率

python scripts/train.py ^
    --optimized ^
    --mcts-sims 100 ^
    --games 25 ^
    --iterations 2000 ^
    --parallel-games 32 ^
    --batch-size 256 ^
    --epochs 10 ^
    --lr 0.001 ^
    --resume

pause
