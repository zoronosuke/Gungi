"""
訓練結果のグラフを生成するスクリプト
失敗分析やレポート作成用
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 日本語フォントの設定（Windows環境向け）
plt.rcParams['font.family'] = 'MS Gothic'

def generate_graphs(json_path, output_dir=None):
    """訓練結果のグラフを生成"""
    if not os.path.exists(json_path):
        print(f"Error: {json_path} が見つかりません。")
        return

    # 出力ディレクトリの設定
    if output_dir is None:
        output_dir = os.path.join(project_root, 'reports')
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    iterations = []
    draw_rates = []
    win_rates = []
    loss_rates = []
    avg_values = []
    avg_entropies = []
    avg_lengths = []
    policy_losses = []
    value_losses = []

    for it in data['iterations']:
        iterations.append(it['iteration'])
        total = it['total_games']
        
        # 勝敗比率
        draw_rates.append(it['draws'] / total * 100)
        # 黒勝ち + 白勝ち = 勝負がついた数
        win_loss_total = (it['black_wins'] + it['white_wins']) / total * 100
        win_rates.append(win_loss_total)

        # 損失の取得
        policy_losses.append(it.get('policy_loss', 0))
        value_losses.append(it.get('value_loss', 0))

        # 平均値の取得
        if 'game_stats' in it and it['game_stats']:
            game_stats = it['game_stats']
            avg_val = np.mean([g.get('avg_value_prediction', 0) for g in game_stats])
            avg_ent = np.mean([g.get('avg_policy_entropy', 0) for g in game_stats])
            avg_len = np.mean([g.get('total_moves', 0) for g in game_stats])
        else:
            # game_statsがない場合は代替データを使用
            avg_val = it.get('value_near_zero_ratio', 0) * 2 - 1  # 0-1を-1-1に変換
            avg_ent = it.get('avg_policy_entropy', 0)
            avg_len = it.get('avg_game_length', 0)

        avg_values.append(avg_val)
        avg_entropies.append(avg_ent)
        avg_lengths.append(avg_len)

    print(f"=== 訓練データ分析 ===")
    print(f"総イテレーション数: {len(iterations)}")
    print(f"最終引き分け率: {draw_rates[-1]:.1f}%")
    print(f"出力先: {output_dir}")
    print()

    # 1. 勝敗内訳の推移 (Stack Plot)
    plt.figure(figsize=(10, 6))
    plt.stackplot(iterations, win_rates, draw_rates, 
                  labels=['Decisive (Win/Loss)', 'Draw (Repetition)'], 
                  colors=['#66b3ff', '#ff9999'])
    plt.title('Game Outcome Distribution per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Percentage (%)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    output_path = os.path.join(output_dir, 'report_outcome_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graph 1 Saved: {output_path}")

    # 2. 評価値の崩壊 (Value Collapse)
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, avg_values, marker='o', color='purple', linewidth=2, markersize=4)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=-1, color='red', linestyle='--', label='Sure Loss (-1.0)')
    plt.axhline(y=1, color='blue', linestyle='--', label='Sure Win (1.0)')
    plt.title('Average Value Prediction (AI Optimism)')
    plt.xlabel('Iteration')
    plt.ylabel('Value (-1: Loss, 1: Win)')
    plt.ylim(-1.2, 1.2)
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(output_dir, 'report_value_collapse.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graph 2 Saved: {output_path}")

    # 3. エントロピー（迷い）の推移
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, avg_entropies, marker='x', color='green', linewidth=2, markersize=4)
    plt.title('Policy Entropy (Uncertainty)')
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')
    plt.grid(True)
    output_path = os.path.join(output_dir, 'report_entropy.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graph 3 Saved: {output_path}")

    # 4. 平均ゲーム長の推移
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, avg_lengths, marker='s', color='orange', linewidth=2, markersize=4)
    plt.title('Average Game Length per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Moves')
    plt.grid(True)
    output_path = os.path.join(output_dir, 'report_game_length.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graph 4 Saved: {output_path}")

    # 5. Policy Loss と Value Loss の推移
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Policy Loss', color='tab:blue')
    ax1.plot(iterations, policy_losses, marker='o', color='tab:blue', linewidth=2, markersize=4, label='Policy Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Value Loss', color='tab:red')
    ax2.plot(iterations, value_losses, marker='x', color='tab:red', linewidth=2, markersize=4, label='Value Loss')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Training Loss per Iteration')
    fig.tight_layout()
    output_path = os.path.join(output_dir, 'report_training_loss.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graph 5 Saved: {output_path}")

    # 6. 引き分け率の単独グラフ（千日手問題の可視化）
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, draw_rates, marker='o', color='red', linewidth=2, markersize=4)
    plt.axhline(y=30, color='orange', linestyle='--', label='Warning Threshold (30%)')
    plt.axhline(y=50, color='red', linestyle='--', label='Critical Threshold (50%)')
    plt.fill_between(iterations, draw_rates, alpha=0.3, color='red')
    plt.title('Draw Rate per Iteration (Repetition Detection)')
    plt.xlabel('Iteration')
    plt.ylabel('Draw Rate (%)')
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(output_dir, 'report_draw_rate.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graph 6 Saved: {output_path}")

    print()
    print("=" * 50)
    print("すべてのグラフ作成が完了しました！")
    print(f"出力先: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate training report graphs')
    parser.add_argument('--json', type=str, default='checkpoints/training_stats.json',
                        help='Path to training_stats.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for graphs')
    args = parser.parse_args()
    
    json_path = os.path.join(project_root, args.json) if not os.path.isabs(args.json) else args.json
    generate_graphs(json_path, args.output)
