@echo off
echo ============================================
echo Gungi AI Training - Resume
echo ============================================
echo.
echo 前回の学習から再開します
echo ============================================
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts/train.py --test --parallel-games 32 --games 20 --iterations 100 --resume

pause
