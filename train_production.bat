@echo off
echo ============================================================
echo Gungi AI - Production Training
echo ============================================================
echo.
echo Settings:
echo   - Model: 8 res_blocks (full)
echo   - MCTS simulations: 200
echo   - Games per iteration: 50
echo   - Parallel games: 64
echo   - Batch size: 512
echo   - FP16 (half precision): enabled
echo   - Iterations: 100
echo.
echo Penalty Settings:
echo   - Repetition (senjite) penalty: -0.9
echo   - Max moves penalty: -0.1
echo   - Repetition threshold: 3
echo   - Max moves: 200
echo.
echo ============================================================

cd /d %~dp0
call .venv\Scripts\activate.bat

python scripts/train.py --full --optimized --iterations 100 --games 50 --mcts-sims 200 --batch-size 512 --use-fp16

echo.
echo Training completed!
pause
