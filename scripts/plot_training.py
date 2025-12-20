#!/usr/bin/env python
"""
学習曲線の可視化スクリプト

使用例:
    # 基本的な使用
    python scripts/plot_training.py
    
    # 出力ファイル指定
    python scripts/plot_training.py --output training_curves.png
    
    # 評価結果も含めて表示
    python scripts/plot_training.py --eval evaluation_results.json
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI不要モード

# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_training_state(checkpoint_dir: str = './checkpoints') -> dict:
    """training_state.jsonを読み込む"""
    state_path = os.path.join(checkpoint_dir, 'training_state.json')
    
    if not os.path.exists(state_path):
        print(f"Warning: {state_path} not found")
        return None
    
    with open(state_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_evaluation_results(eval_path: str) -> dict:
    """evaluation_results.jsonを読み込む"""
    if not os.path.exists(eval_path):
        return None
    
    with open(eval_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_training_curves(training_state: dict, output_path: str = 'training_curves.png',
                        eval_results: dict = None):
    """学習曲線をプロット"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gungi AI Training Progress', fontsize=16, fontweight='bold')
    
    iterations = list(range(1, len(training_state['policy_loss_history']) + 1))
    
    # 1. Policy Loss
    ax1 = axes[0, 0]
    ax1.plot(iterations, training_state['policy_loss_history'], 
             'b-o', linewidth=2, markersize=4, label='Policy Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Policy Loss (Cross Entropy)')
    ax1.set_title('Policy Network Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 改善率を表示
    if len(training_state['policy_loss_history']) >= 2:
        initial = training_state['policy_loss_history'][0]
        final = training_state['policy_loss_history'][-1]
        improvement = (initial - final) / initial * 100
        ax1.annotate(f'Improvement: {improvement:.1f}%', 
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 2. Value Loss
    ax2 = axes[0, 1]
    ax2.plot(iterations, training_state['value_loss_history'], 
             'r-o', linewidth=2, markersize=4, label='Value Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Value Loss (MSE)')
    ax2.set_title('Value Network Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 改善率を表示
    if len(training_state['value_loss_history']) >= 2:
        initial = training_state['value_loss_history'][0]
        final = training_state['value_loss_history'][-1]
        improvement = (initial - final) / initial * 100
        ax2.annotate(f'Improvement: {improvement:.1f}%', 
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # 3. Combined Loss (Log scale)
    ax3 = axes[1, 0]
    total_loss = [p + v for p, v in zip(training_state['policy_loss_history'], 
                                         training_state['value_loss_history'])]
    ax3.semilogy(iterations, total_loss, 'g-o', linewidth=2, markersize=4, label='Total Loss')
    ax3.semilogy(iterations, training_state['policy_loss_history'], 
                 'b--', linewidth=1, alpha=0.7, label='Policy')
    ax3.semilogy(iterations, training_state['value_loss_history'], 
                 'r--', linewidth=1, alpha=0.7, label='Value')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss (log scale)')
    ax3.set_title('Combined Loss Progress')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Training Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 統計情報テーブル
    stats_text = f"""
    Training Statistics
    ══════════════════════════════════════
    
    Total Iterations:    {training_state['iteration']}
    Total Games:         {training_state['total_games']:,}
    Total Examples:      {training_state['total_examples']:,}
    
    Policy Loss:
      Initial:  {training_state['policy_loss_history'][0]:.4f}
      Final:    {training_state['policy_loss_history'][-1]:.4f}
      Best:     {min(training_state['policy_loss_history']):.4f}
    
    Value Loss:
      Initial:  {training_state['value_loss_history'][0]:.4f}
      Final:    {training_state['value_loss_history'][-1]:.4f}
      Best:     {min(training_state['value_loss_history']):.4f}
    
    Start Time: {training_state['start_time'][:19]}
    Last Update: {training_state['last_update_time'][:19]}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {output_path}")
    
    return fig


def plot_evaluation_results(eval_results: dict, output_path: str = 'evaluation_results.png'):
    """評価結果をプロット"""
    
    if eval_results is None:
        print("No evaluation results to plot")
        return None
    
    opponents = list(eval_results['opponents'].keys())
    win_rates = [eval_results['opponents'][o]['win_rate'] * 100 for o in opponents]
    draw_rates = [eval_results['opponents'][o]['draw_rate'] * 100 for o in opponents]
    loss_rates = [100 - w - d for w, d in zip(win_rates, draw_rates)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Evaluation Results', fontsize=14, fontweight='bold')
    
    # 1. 勝率棒グラフ
    ax1 = axes[0]
    x = np.arange(len(opponents))
    width = 0.25
    
    bars1 = ax1.bar(x - width, win_rates, width, label='Win', color='green', alpha=0.8)
    bars2 = ax1.bar(x, draw_rates, width, label='Draw', color='gray', alpha=0.8)
    bars3 = ax1.bar(x + width, loss_rates, width, label='Loss', color='red', alpha=0.8)
    
    ax1.set_ylabel('Rate (%)')
    ax1.set_title('Win/Draw/Loss Rate by Opponent')
    ax1.set_xticks(x)
    ax1.set_xticklabels(opponents)
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 数値を棒の上に表示
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 5:
                ax1.annotate(f'{height:.0f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
    
    # 2. スタック棒グラフ
    ax2 = axes[1]
    
    ax2.barh(opponents, win_rates, label='Win', color='green', alpha=0.8)
    ax2.barh(opponents, draw_rates, left=win_rates, label='Draw', color='gray', alpha=0.8)
    ax2.barh(opponents, loss_rates, left=[w+d for w,d in zip(win_rates, draw_rates)], 
             label='Loss', color='red', alpha=0.8)
    
    ax2.set_xlabel('Rate (%)')
    ax2.set_title('Result Distribution')
    ax2.legend(loc='lower right')
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Evaluation results saved to: {output_path}")
    
    return fig


def plot_comprehensive_report(training_state: dict, eval_results: dict = None,
                             output_path: str = 'comprehensive_report.png'):
    """包括的なレポートを生成"""
    
    if eval_results:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Gungi AI - Comprehensive Training Report', fontsize=18, fontweight='bold')
    
    iterations = list(range(1, len(training_state['policy_loss_history']) + 1))
    
    # 1. Policy Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iterations, training_state['policy_loss_history'], 
             'b-o', linewidth=2, markersize=5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Policy Loss (Cross Entropy)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Value Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iterations, training_state['value_loss_history'], 
             'r-o', linewidth=2, markersize=5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Value Loss (MSE)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(iterations, training_state['policy_loss_history'], 
                 'b-', linewidth=2, label='Policy')
    ax3.semilogy(iterations, training_state['value_loss_history'], 
                 'r-', linewidth=2, label='Value')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss (log scale)')
    ax3.set_title('Loss Comparison (Log Scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    policy_improvement = (training_state['policy_loss_history'][0] - training_state['policy_loss_history'][-1]) / training_state['policy_loss_history'][0] * 100
    value_improvement = (training_state['value_loss_history'][0] - training_state['value_loss_history'][-1]) / training_state['value_loss_history'][0] * 100
    
    stats_text = f"""
    ╔══════════════════════════════════════╗
    ║       TRAINING STATISTICS            ║
    ╠══════════════════════════════════════╣
    ║                                      ║
    ║  Iterations:     {training_state['iteration']:>6}             ║
    ║  Total Games:    {training_state['total_games']:>6,}             ║
    ║  Total Examples: {training_state['total_examples']:>6,}             ║
    ║                                      ║
    ║  Policy Loss Improvement: {policy_improvement:>5.1f}%     ║
    ║  Value Loss Improvement:  {value_improvement:>5.1f}%     ║
    ║                                      ║
    ║  Final Policy Loss: {training_state['policy_loss_history'][-1]:>7.4f}        ║
    ║  Final Value Loss:  {training_state['value_loss_history'][-1]:>7.4f}        ║
    ║                                      ║
    ╚══════════════════════════════════════╝
    """
    
    ax4.text(0.5, 0.5, stats_text, transform=ax4.transAxes,
             fontsize=11, ha='center', va='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # 評価結果がある場合
    if eval_results:
        # 5. Win Rate Bar Chart
        ax5 = fig.add_subplot(gs[2, 0])
        opponents = list(eval_results['opponents'].keys())
        win_rates = [eval_results['opponents'][o]['win_rate'] * 100 for o in opponents]
        
        colors = ['green' if w >= 50 else 'orange' if w >= 30 else 'red' for w in win_rates]
        bars = ax5.bar(opponents, win_rates, color=colors, alpha=0.8, edgecolor='black')
        ax5.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% baseline')
        ax5.set_ylabel('Win Rate (%)')
        ax5.set_title('Win Rate vs Different Opponents')
        ax5.set_ylim(0, 100)
        ax5.grid(True, alpha=0.3, axis='y')
        
        for bar, rate in zip(bars, win_rates):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 6. Evaluation Summary
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        eval_text = "╔══════════════════════════════════════╗\n"
        eval_text += "║       EVALUATION RESULTS             ║\n"
        eval_text += "╠══════════════════════════════════════╣\n"
        
        for opp in opponents:
            data = eval_results['opponents'][opp]
            eval_text += f"║  vs {opp:<12}: {data['win_rate']*100:>5.1f}% wins    ║\n"
        
        eval_text += "║                                      ║\n"
        avg_win_rate = np.mean(win_rates)
        eval_text += f"║  Average Win Rate: {avg_win_rate:>5.1f}%           ║\n"
        eval_text += "╚══════════════════════════════════════╝"
        
        ax6.text(0.5, 0.5, eval_text, transform=ax6.transAxes,
                fontsize=11, ha='center', va='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comprehensive report saved to: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize Gungi AI training progress')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory containing training_state.json')
    parser.add_argument('--eval', type=str, default='./evaluation_results.json',
                        help='Path to evaluation results JSON')
    parser.add_argument('--output', type=str, default='./training_report.png',
                        help='Output path for the main report')
    parser.add_argument('--all', action='store_true',
                        help='Generate all plots separately')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Gungi AI Training Visualization")
    print(f"{'='*60}")
    
    # 学習状態を読み込み
    training_state = load_training_state(args.checkpoint_dir)
    if training_state is None:
        print("Error: Could not load training state")
        return
    
    print(f"Loaded training state: {training_state['iteration']} iterations")
    
    # 評価結果を読み込み（あれば）
    eval_results = load_evaluation_results(args.eval)
    if eval_results:
        print(f"Loaded evaluation results")
    else:
        print("No evaluation results found (run evaluate_model.py first)")
    
    # プロット生成
    if args.all:
        # 個別プロット
        plot_training_curves(training_state, 'training_curves.png')
        if eval_results:
            plot_evaluation_results(eval_results, 'evaluation_chart.png')
    
    # 包括的レポート
    plot_comprehensive_report(training_state, eval_results, args.output)
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
