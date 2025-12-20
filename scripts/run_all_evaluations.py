#!/usr/bin/env python
"""
全ての評価を実行してレポートを生成するスクリプト

使用例:
    python scripts/run_all_evaluations.py
    python scripts/run_all_evaluations.py --quick  # 簡易評価
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def run_command(cmd: list, description: str):
    """コマンドを実行"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run all evaluations')
    parser.add_argument('--quick', action='store_true',
                        help='Quick evaluation with fewer games')
    parser.add_argument('--skip-elo', action='store_true',
                        help='Skip Elo calculation (slow)')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"# Gungi AI - Complete Evaluation Suite")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    python_cmd = sys.executable
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. モデル評価（vs Random, Greedy, SimpleMCTS）
    eval_cmd = [python_cmd, os.path.join(scripts_dir, 'evaluate_model.py')]
    if args.quick:
        eval_cmd.append('--quick')
    
    success = run_command(eval_cmd, "Model Evaluation (vs baseline AIs)")
    if not success:
        print("Warning: Model evaluation failed or incomplete")
    
    # 2. Elo計算（時間がかかるのでオプション）
    if not args.skip_elo:
        elo_cmd = [python_cmd, os.path.join(scripts_dir, 'calculate_elo.py')]
        if args.quick:
            elo_cmd.extend(['--games', '4', '--mcts-sims', '30'])
        
        success = run_command(elo_cmd, "Elo Rating Calculation")
        if not success:
            print("Warning: Elo calculation failed or incomplete")
    else:
        print("\nSkipping Elo calculation (--skip-elo)")
    
    # 3. 可視化
    plot_cmd = [python_cmd, os.path.join(scripts_dir, 'plot_training.py'), '--all']
    success = run_command(plot_cmd, "Training Visualization")
    if not success:
        print("Warning: Visualization failed or incomplete")
    
    # 完了メッセージ
    print(f"\n{'#'*60}")
    print(f"# Evaluation Complete!")
    print(f"# Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    print(f"\nGenerated files:")
    print(f"  - evaluation_results.json  (対戦結果)")
    print(f"  - training_report.png      (学習曲線)")
    print(f"  - training_curves.png      (詳細Loss曲線)")
    if not args.skip_elo:
        print(f"  - elo_results.json         (Elo推移)")
    if os.path.exists('./evaluation_chart.png'):
        print(f"  - evaluation_chart.png     (評価チャート)")


if __name__ == "__main__":
    main()
