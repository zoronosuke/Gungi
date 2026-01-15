"""
訓練統計の可視化スクリプト
training_stats.jsonから詳細なグラフを生成
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']


def load_stats(filepath: str) -> dict:
    """統計ファイルを読み込み"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_training_overview(data: dict, output_path: str):
    """訓練の概要グラフを生成"""
    iterations = data['iterations']
    if not iterations:
        print("データがありません")
        return
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Gungi AI 訓練統計 - 詳細分析', fontsize=16)
    
    # データ抽出
    iter_nums = [it['iteration'] for it in iterations]
    policy_losses = [it['policy_loss'] for it in iterations]
    value_losses = [it['value_loss'] for it in iterations]
    black_wins = [it['black_wins'] for it in iterations]
    white_wins = [it['white_wins'] for it in iterations]
    draws = [it['draws'] for it in iterations]
    avg_game_lengths = [it['avg_game_length'] for it in iterations]
    buffer_draw_ratios = [it['buffer_draw_ratio'] for it in iterations]
    avg_entropies = [it['avg_policy_entropy'] for it in iterations]
    value_near_zero = [it['value_near_zero_ratio'] for it in iterations]
    selfplay_times = [it['selfplay_duration_seconds'] for it in iterations]
    training_times = [it['training_duration_seconds'] for it in iterations]
    
    # 1. Policy Loss
    ax = axes[0, 0]
    ax.plot(iter_nums, policy_losses, 'b-', marker='o', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Policy Loss')
    ax.set_title('Policy Loss')
    ax.grid(True, alpha=0.3)
    
    # 2. Value Loss
    ax = axes[0, 1]
    ax.plot(iter_nums, value_losses, 'r-', marker='o', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value Loss')
    ax.set_title('Value Loss')
    ax.grid(True, alpha=0.3)
    
    # 3. 勝敗分布
    ax = axes[0, 2]
    ax.stackplot(iter_nums, black_wins, white_wins, draws, 
                 labels=['Black', 'White', 'Draw'],
                 colors=['#333333', '#AAAAAA', '#FF6B6B'],
                 alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Games')
    ax.set_title('勝敗分布')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 4. 引き分け率
    ax = axes[0, 3]
    total_games = [b + w + d for b, w, d in zip(black_wins, white_wins, draws)]
    draw_rates = [d / t * 100 if t > 0 else 0 for d, t in zip(draws, total_games)]
    ax.plot(iter_nums, draw_rates, 'r-', marker='o', markersize=4)
    ax.axhline(y=50, color='orange', linestyle='--', label='50% 警告ライン')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Draw Rate (%)')
    ax.set_title('引き分け率')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 終了理由の内訳
    ax = axes[1, 0]
    rep_counts = []
    max_moves_counts = []
    checkmate_counts = []
    for it in iterations:
        reasons = it.get('termination_reasons', {})
        rep_counts.append(reasons.get('REPETITION', 0))
        max_moves_counts.append(reasons.get('MAX_MOVES', 0))
        checkmate_counts.append(reasons.get('CHECKMATE', 0))
    
    ax.stackplot(iter_nums, checkmate_counts, rep_counts, max_moves_counts,
                 labels=['Checkmate', 'Repetition', 'Max Moves'],
                 colors=['#4CAF50', '#FF6B6B', '#FFA726'],
                 alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Games')
    ax.set_title('終了理由の内訳')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 6. 平均ゲーム長
    ax = axes[1, 1]
    ax.plot(iter_nums, avg_game_lengths, 'g-', marker='o', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Moves')
    ax.set_title('平均ゲーム長')
    ax.grid(True, alpha=0.3)
    
    # 7. バッファ内の引き分け比率
    ax = axes[1, 2]
    ax.plot(iter_nums, [r * 100 for r in buffer_draw_ratios], 'purple', marker='o', markersize=4)
    ax.axhline(y=30, color='orange', linestyle='--', label='30% 制限')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Draw Ratio (%)')
    ax.set_title('バッファ内 引き分け比率')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. Policy エントロピー
    ax = axes[1, 3]
    ax.plot(iter_nums, avg_entropies, 'orange', marker='o', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Entropy')
    ax.set_title('平均 Policy エントロピー')
    ax.grid(True, alpha=0.3)
    
    # 9. Value 0近傍率
    ax = axes[2, 0]
    ax.plot(iter_nums, [r * 100 for r in value_near_zero], 'red', marker='o', markersize=4)
    ax.axhline(y=50, color='orange', linestyle='--', label='50% 警告')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Near Zero Ratio (%)')
    ax.set_title('Value予測 0近傍率 (|v|<0.1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 10. Value分布
    ax = axes[2, 1]
    if iterations:
        last_it = iterations[-1]
        value_dist = last_it.get('value_prediction_distribution', {})
        if value_dist:
            labels = list(value_dist.keys())
            values = list(value_dist.values())
            colors = ['#4CAF50', '#8BC34A', '#FFA726', '#FF6B6B', '#E53935']
            ax.bar(range(len(labels)), values, color=colors)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Value Range')
    ax.set_ylabel('Count')
    ax.set_title('最新イテレーション Value分布')
    ax.grid(True, alpha=0.3)
    
    # 11. 所要時間
    ax = axes[2, 2]
    ax.bar(iter_nums, selfplay_times, label='Self-play', alpha=0.7)
    ax.bar(iter_nums, training_times, bottom=selfplay_times, label='Training', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Time (s)')
    ax.set_title('イテレーション所要時間')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 12. Policy Loss vs Draw Rate (相関分析)
    ax = axes[2, 3]
    ax.scatter(policy_losses, draw_rates, c=iter_nums, cmap='viridis', s=50)
    ax.set_xlabel('Policy Loss')
    ax.set_ylabel('Draw Rate (%)')
    ax.set_title('Policy Loss vs Draw Rate')
    ax.grid(True, alpha=0.3)
    
    # 相関係数を計算
    if len(policy_losses) > 1:
        corr = np.corrcoef(policy_losses, draw_rates)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"グラフを保存しました: {output_path}")


def plot_game_details(data: dict, output_path: str):
    """ゲームごとの詳細グラフを生成"""
    iterations = data['iterations']
    if not iterations:
        print("データがありません")
        return
    
    # 最新のイテレーションを取得
    last_it = iterations[-1]
    game_stats = last_it.get('game_stats', [])
    
    if not game_stats:
        print("ゲーム統計がありません")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'イテレーション {last_it["iteration"]} - ゲーム詳細分析', fontsize=16)
    
    # 1. ゲームごとの手数分布
    ax = axes[0, 0]
    game_ids = [g['game_id'] for g in game_stats]
    total_moves = [g['total_moves'] for g in game_stats]
    colors = ['green' if g['winner'] != 'DRAW' else 'red' for g in game_stats]
    ax.bar(game_ids, total_moves, color=colors, alpha=0.7)
    ax.set_xlabel('Game ID')
    ax.set_ylabel('Total Moves')
    ax.set_title('ゲームごとの手数')
    ax.grid(True, alpha=0.3)
    
    # 2. Policy エントロピー分布
    ax = axes[0, 1]
    entropies = [g['avg_policy_entropy'] for g in game_stats]
    ax.hist(entropies, bins=20, color='orange', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Average Policy Entropy')
    ax.set_ylabel('Frequency')
    ax.set_title('ゲームごとの平均Policy Entropy分布')
    ax.grid(True, alpha=0.3)
    
    # 3. Value予測分布
    ax = axes[0, 2]
    avg_values = [g['avg_value_prediction'] for g in game_stats]
    ax.hist(avg_values, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', label='Zero')
    ax.set_xlabel('Average Value Prediction')
    ax.set_ylabel('Frequency')
    ax.set_title('ゲームごとの平均Value予測分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 終了理由の円グラフ
    ax = axes[1, 0]
    termination_counts = {}
    for g in game_stats:
        reason = g['termination_reason']
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
    
    if termination_counts:
        labels = list(termination_counts.keys())
        sizes = list(termination_counts.values())
        colors = {'CHECKMATE': '#4CAF50', 'REPETITION': '#FF6B6B', 'MAX_MOVES': '#FFA726'}
        pie_colors = [colors.get(l, '#888888') for l in labels]
        ax.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('終了理由の内訳')
    
    # 5. 往復運動回数
    ax = axes[1, 1]
    back_forth = [g.get('back_and_forth_count', 0) for g in game_stats]
    ax.bar(game_ids, back_forth, color='purple', alpha=0.7)
    ax.set_xlabel('Game ID')
    ax.set_ylabel('Count')
    ax.set_title('ゲームごとの往復運動回数')
    ax.grid(True, alpha=0.3)
    
    # 6. Value予測の変動（要約版）
    ax = axes[1, 2]
    for g in game_stats[:5]:  # 最初の5ゲームのみ
        summary = g.get('move_stats_summary', {})
        values = summary.get('first_10_values', []) + summary.get('last_10_values', [])
        if values:
            x = list(range(len(values)))
            ax.plot(x, values, alpha=0.5, label=f"Game {g['game_id']}")
    ax.set_xlabel('Sample Points')
    ax.set_ylabel('Value Prediction')
    ax.set_title('Value予測の推移サンプル')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ゲーム詳細グラフを保存しました: {output_path}")


def print_summary(data: dict):
    """統計サマリーを表示"""
    global_stats = data.get('global_stats', {})
    iterations = data.get('iterations', [])
    
    print("\n" + "=" * 60)
    print("訓練統計サマリー")
    print("=" * 60)
    print(f"総ゲーム数: {global_stats.get('total_games', 0)}")
    print(f"総手数: {global_stats.get('total_moves', 0)}")
    print(f"総訓練時間: {global_stats.get('total_training_time', 0):.1f}秒")
    print(f"開始時刻: {global_stats.get('start_time', 'N/A')}")
    
    if iterations:
        latest = iterations[-1]
        print(f"\n--- 最新イテレーション {latest['iteration']} ---")
        print(f"勝敗: 黒{latest['black_wins']} 白{latest['white_wins']} 引分{latest['draws']}")
        print(f"終了理由: {latest.get('termination_reasons', {})}")
        print(f"平均ゲーム長: {latest['avg_game_length']:.1f}手")
        print(f"Policy Loss: {latest['policy_loss']:.4f}")
        print(f"Value Loss: {latest['value_loss']:.4f}")
        print(f"平均Policy Entropy: {latest['avg_policy_entropy']:.4f}")
        print(f"Value 0近傍率: {latest['value_near_zero_ratio']*100:.1f}%")
        print(f"バッファDraw率: {latest['buffer_draw_ratio']*100:.1f}%")
        
        # 問題診断
        print("\n--- 問題診断 ---")
        if latest['draws'] == latest['black_wins'] + latest['white_wins'] + latest['draws']:
            print("⚠️ 警告: 全ゲームが引き分け!")
        
        rep_count = latest.get('termination_reasons', {}).get('REPETITION', 0)
        if rep_count > 0:
            print(f"⚠️ 千日手: {rep_count}ゲーム")
        
        if latest['value_near_zero_ratio'] > 0.5:
            print(f"⚠️ Value予測が0に収束中 ({latest['value_near_zero_ratio']*100:.1f}%)")
        
        if latest['avg_policy_entropy'] < 1.0:
            print(f"⚠️ Policy Entropyが低い（探索の多様性不足）")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='訓練統計の可視化')
    parser.add_argument('--input', '-i', default='checkpoints/training_stats.json',
                        help='統計ファイルのパス')
    parser.add_argument('--output', '-o', default='training_analysis.png',
                        help='出力画像のパス')
    parser.add_argument('--details', '-d', action='store_true',
                        help='ゲーム詳細グラフも生成')
    
    args = parser.parse_args()
    
    # データ読み込み
    try:
        data = load_stats(args.input)
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {args.input}")
        return
    except json.JSONDecodeError:
        print(f"JSONファイルの解析に失敗: {args.input}")
        return
    
    # サマリー表示
    print_summary(data)
    
    # グラフ生成
    plot_training_overview(data, args.output)
    
    if args.details:
        details_path = args.output.replace('.png', '_details.png')
        plot_game_details(data, details_path)


if __name__ == '__main__':
    main()
