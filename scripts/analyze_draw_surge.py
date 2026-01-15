#!/usr/bin/env python
"""
引き分け急増問題の総合分析スクリプト

AlphaZero型AIの学習で「Policy Lossが下がるのに引き分けが増える」という
矛盾を分析し、根本原因を特定するためのデータ可視化・分析ツール。

使用例:
    python scripts/analyze_draw_surge.py
    python scripts/analyze_draw_surge.py --output report.png --log training_log.txt
"""

import argparse
import os
import sys
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI不要モード
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from scipy import stats

# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# ===== データパース =====

def parse_training_log(log_text: str) -> List[Dict]:
    """学習ログテキストをパースしてイテレーションごとのデータを抽出"""
    iterations = []
    
    # 正規表現パターン
    iter_pattern = r'--- Iteration (\d+)/\d+ ---'
    results_pattern = r'Results: BLACK=(\d+), WHITE=(\d+), DRAW=(\d+)'
    draw_reason_pattern = r'Draw reasons: REPETITION=(\d+), MAX_MOVES=(\d+)'
    loss_pattern = r'Policy Loss: ([\d.]+), Value Loss: ([\d.]+)'
    
    lines = log_text.strip().split('\n')
    current_iter = {}
    
    for line in lines:
        iter_match = re.search(iter_pattern, line)
        if iter_match:
            if current_iter:
                iterations.append(current_iter)
            current_iter = {'iteration': int(iter_match.group(1))}
        
        results_match = re.search(results_pattern, line)
        if results_match:
            current_iter['black_wins'] = int(results_match.group(1))
            current_iter['white_wins'] = int(results_match.group(2))
            current_iter['draws'] = int(results_match.group(3))
            total = current_iter['black_wins'] + current_iter['white_wins'] + current_iter['draws']
            current_iter['total_games'] = total
            current_iter['draw_rate'] = current_iter['draws'] / total if total > 0 else 0
        
        draw_match = re.search(draw_reason_pattern, line)
        if draw_match:
            current_iter['repetition_draws'] = int(draw_match.group(1))
            current_iter['max_moves_draws'] = int(draw_match.group(2))
        
        loss_match = re.search(loss_pattern, line)
        if loss_match:
            current_iter['policy_loss'] = float(loss_match.group(1))
            current_iter['value_loss'] = float(loss_match.group(2))
    
    if current_iter:
        iterations.append(current_iter)
    
    return iterations


def load_training_state(checkpoint_dir: str = './checkpoints') -> dict:
    """training_state.jsonを読み込む"""
    state_path = os.path.join(checkpoint_dir, 'training_state.json')
    if not os.path.exists(state_path):
        return None
    with open(state_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_replay_buffer(checkpoint_dir: str = './checkpoints') -> dict:
    """replay_buffer.npzを読み込んでValue分布を分析"""
    buffer_path = os.path.join(checkpoint_dir, 'replay_buffer.npz')
    if not os.path.exists(buffer_path):
        return None
    
    data = np.load(buffer_path, allow_pickle=True)
    return {
        'states': data['states'] if 'states' in data else None,
        'policies': data['policies'] if 'policies' in data else None,
        'values': data['values'] if 'values' in data else None
    }


# ===== 分析関数 =====

def analyze_value_distribution(values: np.ndarray) -> Dict:
    """Value分布を分析"""
    if values is None or len(values) == 0:
        return {}
    
    # 値の分布
    win_count = np.sum(values > 0.5)
    loss_count = np.sum(values < -0.5)
    draw_count = np.sum(np.abs(values) <= 0.5)
    total = len(values)
    
    return {
        'total_samples': total,
        'win_ratio': win_count / total,
        'loss_ratio': loss_count / total,
        'draw_ratio': draw_count / total,
        'mean_value': float(np.mean(values)),
        'std_value': float(np.std(values)),
        'median_value': float(np.median(values)),
        'values': values  # 元データも保持
    }


def calculate_correlations(iterations: List[Dict]) -> Dict:
    """各指標間の相関を計算"""
    if len(iterations) < 3:
        return {}
    
    draw_rates = [it.get('draw_rate', 0) for it in iterations]
    policy_losses = [it.get('policy_loss', 0) for it in iterations]
    value_losses = [it.get('value_loss', 0) for it in iterations]
    
    results = {}
    
    # Policy Loss vs Draw Rate
    if len(policy_losses) >= 3 and len(draw_rates) >= 3:
        corr, p_val = stats.pearsonr(policy_losses, draw_rates)
        results['policy_loss_vs_draw_rate'] = {
            'correlation': corr,
            'p_value': p_val,
            'interpretation': '負の相関（Loss↓ = Draw↑）' if corr < -0.5 else 
                             '正の相関（Loss↓ = Draw↓）' if corr > 0.5 else '弱い相関'
        }
    
    # Value Loss vs Draw Rate
    if len(value_losses) >= 3 and len(draw_rates) >= 3:
        corr, p_val = stats.pearsonr(value_losses, draw_rates)
        results['value_loss_vs_draw_rate'] = {
            'correlation': corr,
            'p_value': p_val,
            'interpretation': '負の相関（Loss↓ = Draw↑）' if corr < -0.5 else 
                             '正の相関（Loss↓ = Draw↓）' if corr > 0.5 else '弱い相関'
        }
    
    return results


def identify_turning_point(iterations: List[Dict]) -> Dict:
    """引き分け急増の「転換点」を特定"""
    if len(iterations) < 2:
        return {}
    
    draw_rates = [it.get('draw_rate', 0) for it in iterations]
    
    # 最大の増加ポイントを探す
    max_increase = 0
    turning_point = 0
    for i in range(1, len(draw_rates)):
        increase = draw_rates[i] - draw_rates[i-1]
        if increase > max_increase:
            max_increase = increase
            turning_point = i
    
    # 100%到達ポイント
    full_draw_iter = None
    for i, rate in enumerate(draw_rates):
        if rate >= 0.9:
            full_draw_iter = i + 1
            break
    
    return {
        'turning_point_iteration': turning_point + 1,
        'turning_point_increase': max_increase,
        'full_draw_iteration': full_draw_iter,
        'draw_rate_at_turning': draw_rates[turning_point] if turning_point < len(draw_rates) else None
    }


# ===== 可視化関数 =====

def create_comprehensive_report(iterations: List[Dict], 
                                 training_state: dict,
                                 value_analysis: Dict,
                                 correlations: Dict,
                                 turning_point: Dict,
                                 output_path: str = 'draw_surge_analysis.png'):
    """総合分析レポートを生成"""
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('引き分け急増問題 - 総合分析レポート', fontsize=18, fontweight='bold')
    
    # GridSpecでレイアウト
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # データ準備
    iters = [it['iteration'] for it in iterations]
    draw_rates = [it.get('draw_rate', 0) * 100 for it in iterations]
    policy_losses = [it.get('policy_loss', 0) for it in iterations]
    value_losses = [it.get('value_loss', 0) for it in iterations]
    
    # 引き分け理由の分離
    rep_draws = [it.get('repetition_draws', 0) for it in iterations]
    max_draws = [it.get('max_moves_draws', 0) for it in iterations]
    
    # 勝敗データ
    black_wins = [it.get('black_wins', 0) for it in iterations]
    white_wins = [it.get('white_wins', 0) for it in iterations]
    draws = [it.get('draws', 0) for it in iterations]
    
    # ===== 1. 引き分け率の推移 =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iters, draw_rates, 'r-o', linewidth=2, markersize=8, label='引き分け率')
    ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='警戒ライン(50%)')
    ax1.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='危険ライン(90%)')
    
    # 転換点マーカー
    if turning_point.get('turning_point_iteration'):
        tp = turning_point['turning_point_iteration']
        if tp <= len(draw_rates):
            ax1.axvline(x=tp, color='purple', linestyle=':', linewidth=2, label=f'転換点(Iter {tp})')
    
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('引き分け率 (%)', fontsize=11)
    ax1.set_title('【A1】引き分け率の推移', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)
    
    # ===== 2. 千日手 vs MAX_MOVES =====
    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.35
    x = np.arange(len(iters))
    
    # データがある場合のみプロット
    has_draw_details = any(r > 0 or m > 0 for r, m in zip(rep_draws, max_draws))
    if has_draw_details:
        bars1 = ax2.bar(x - width/2, rep_draws, width, label='千日手', color='red', alpha=0.8)
        bars2 = ax2.bar(x + width/2, max_draws, width, label='MAX_MOVES', color='orange', alpha=0.8)
        ax2.set_ylabel('件数', fontsize=11)
    else:
        ax2.text(0.5, 0.5, 'Draw理由の詳細データなし\n（ログに記録がありません）', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_title('【A2】引き分け理由の内訳', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(iters)
    if has_draw_details:
        ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ===== 3. Loss曲線 =====
    ax3 = fig.add_subplot(gs[0, 2])
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(iters, policy_losses, 'b-o', linewidth=2, markersize=6, label='Policy Loss')
    line2 = ax3_twin.plot(iters, value_losses, 'g-s', linewidth=2, markersize=6, label='Value Loss')
    
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Policy Loss', color='blue', fontsize=11)
    ax3_twin.set_ylabel('Value Loss', color='green', fontsize=11)
    ax3.set_title('【A3】Loss曲線', fontsize=12, fontweight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ===== 4. 勝敗分布のスタック =====
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.stackplot(iters, black_wins, white_wins, draws, 
                  labels=['BLACK勝ち', 'WHITE勝ち', '引き分け'],
                  colors=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.8)
    ax4.set_xlabel('Iteration', fontsize=11)
    ax4.set_ylabel('ゲーム数', fontsize=11)
    ax4.set_title('【A4】結果分布の推移', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # ===== 5. Policy Loss vs Draw Rate 散布図 =====
    ax5 = fig.add_subplot(gs[1, 1])
    scatter = ax5.scatter(policy_losses, draw_rates, c=iters, cmap='viridis', 
                          s=100, edgecolors='black', linewidth=1)
    
    # 回帰線
    if len(policy_losses) >= 2:
        z = np.polyfit(policy_losses, draw_rates, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(policy_losses), max(policy_losses), 100)
        ax5.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label='回帰線')
    
    plt.colorbar(scatter, ax=ax5, label='Iteration')
    ax5.set_xlabel('Policy Loss', fontsize=11)
    ax5.set_ylabel('引き分け率 (%)', fontsize=11)
    ax5.set_title('【B5】Policy Loss vs 引き分け率', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 相関係数を表示
    if correlations.get('policy_loss_vs_draw_rate'):
        corr_info = correlations['policy_loss_vs_draw_rate']
        ax5.text(0.05, 0.95, f"相関: r={corr_info['correlation']:.3f}\np={corr_info['p_value']:.3f}",
                transform=ax5.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ===== 6. Value Loss vs Draw Rate 散布図 =====
    ax6 = fig.add_subplot(gs[1, 2])
    scatter2 = ax6.scatter(value_losses, draw_rates, c=iters, cmap='plasma', 
                           s=100, edgecolors='black', linewidth=1)
    
    if len(value_losses) >= 2:
        z = np.polyfit(value_losses, draw_rates, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(value_losses), max(value_losses), 100)
        ax6.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label='回帰線')
    
    plt.colorbar(scatter2, ax=ax6, label='Iteration')
    ax6.set_xlabel('Value Loss', fontsize=11)
    ax6.set_ylabel('引き分け率 (%)', fontsize=11)
    ax6.set_title('【B6】Value Loss vs 引き分け率', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    if correlations.get('value_loss_vs_draw_rate'):
        corr_info = correlations['value_loss_vs_draw_rate']
        ax6.text(0.05, 0.95, f"相関: r={corr_info['correlation']:.3f}\np={corr_info['p_value']:.3f}",
                transform=ax6.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ===== 7. Value分布ヒストグラム =====
    ax7 = fig.add_subplot(gs[2, 0])
    if value_analysis and 'values' in value_analysis:
        values = value_analysis['values']
        ax7.hist(values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax7.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Value=0 (引き分け)')
        ax7.axvline(x=value_analysis['mean_value'], color='green', linestyle='-', 
                   linewidth=2, label=f'平均={value_analysis["mean_value"]:.3f}')
        ax7.set_xlabel('Value', fontsize=11)
        ax7.set_ylabel('頻度', fontsize=11)
        ax7.legend(fontsize=9)
    else:
        ax7.text(0.5, 0.5, 'replay_bufferデータなし', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=12)
    ax7.set_title('【C7】Value分布（学習データ）', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # ===== 8. Value分布の内訳 =====
    ax8 = fig.add_subplot(gs[2, 1])
    if value_analysis and value_analysis.get('total_samples'):
        labels = ['勝ち(v>0.5)', '負け(v<-0.5)', '引き分け(-0.5≤v≤0.5)']
        sizes = [value_analysis['win_ratio'], value_analysis['loss_ratio'], 
                value_analysis['draw_ratio']]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        explode = (0, 0, 0.1)  # 引き分けを強調
        
        wedges, texts, autotexts = ax8.pie(sizes, explode=explode, labels=labels, 
                                           colors=colors, autopct='%1.1f%%',
                                           shadow=True, startangle=90)
        ax8.axis('equal')
    else:
        ax8.text(0.5, 0.5, 'データなし', ha='center', va='center', 
                transform=ax8.transAxes, fontsize=12)
    ax8.set_title('【C8】学習データ内訳', fontsize=12, fontweight='bold')
    
    # ===== 9. Loss減少率 vs Draw増加率 =====
    ax9 = fig.add_subplot(gs[2, 2])
    
    if len(iterations) >= 2:
        # 各イテレーション間の変化率
        policy_changes = []
        draw_changes = []
        iter_labels = []
        
        for i in range(1, len(iterations)):
            p_change = (policy_losses[i-1] - policy_losses[i]) / policy_losses[i-1] * 100 if policy_losses[i-1] > 0 else 0
            d_change = draw_rates[i] - draw_rates[i-1]
            policy_changes.append(p_change)
            draw_changes.append(d_change)
            iter_labels.append(f'{i}→{i+1}')
        
        x = np.arange(len(iter_labels))
        width = 0.35
        
        bars1 = ax9.bar(x - width/2, policy_changes, width, label='Policy Loss減少率(%)', 
                       color='blue', alpha=0.7)
        bars2 = ax9.bar(x + width/2, draw_changes, width, label='Draw率増加(pp)', 
                       color='red', alpha=0.7)
        
        ax9.set_xlabel('Iteration変化', fontsize=11)
        ax9.set_ylabel('変化量', fontsize=11)
        ax9.set_xticks(x)
        ax9.set_xticklabels(iter_labels)
        ax9.legend(fontsize=9)
        ax9.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax9.set_title('【分析】Loss減少 vs Draw増加', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='y')
    
    # ===== 10-12. 分析結果テキスト =====
    ax_text = fig.add_subplot(gs[3, :])
    ax_text.axis('off')
    
    # 分析結果のテキスト生成
    analysis_text = generate_analysis_text(iterations, training_state, value_analysis, 
                                           correlations, turning_point)
    
    ax_text.text(0.02, 0.95, analysis_text, transform=ax_text.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nレポートを保存しました: {output_path}")
    
    return fig


def generate_analysis_text(iterations: List[Dict], 
                           training_state: dict,
                           value_analysis: Dict,
                           correlations: Dict,
                           turning_point: Dict) -> str:
    """分析結果のテキストを生成"""
    
    text = """
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    【根本原因の仮説と改善提案】                                                   ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

■ 転換点の特定
"""
    
    if turning_point.get('turning_point_iteration'):
        text += f"""  • 引き分け急増の転換点: Iteration {turning_point['turning_point_iteration']}
  • この時点での増加幅: +{turning_point['turning_point_increase']*100:.1f}%ポイント
"""
    
    if turning_point.get('full_draw_iteration'):
        text += f"""  • 引き分け90%超到達: Iteration {turning_point['full_draw_iteration']}
"""
    
    text += """
■ 相関分析の結果
"""
    
    if correlations.get('policy_loss_vs_draw_rate'):
        corr = correlations['policy_loss_vs_draw_rate']
        text += f"""  • Policy Loss vs 引き分け率: r = {corr['correlation']:.3f} ({corr['interpretation']})
"""
    
    if correlations.get('value_loss_vs_draw_rate'):
        corr = correlations['value_loss_vs_draw_rate']
        text += f"""  • Value Loss vs 引き分け率: r = {corr['correlation']:.3f} ({corr['interpretation']})
"""
    
    text += """
■ 仮説: 「Policy Lossが下がるのに引き分けが増える」理由

  【仮説1】Value予測の0収束問題
    - Value Networkが「引き分けが最も安全」と学習
    - Value Loss低下 = 「すべて0と予測」が正解に近づいている可能性
    - 確認方法: Value分布が0に集中しているか
"""
    
    if value_analysis and value_analysis.get('draw_ratio'):
        draw_ratio = value_analysis['draw_ratio'] * 100
        mean_val = value_analysis.get('mean_value', 0)
        text += f"""    → 現状: 引き分け相当データ比率 = {draw_ratio:.1f}%, 平均Value = {mean_val:.3f}
"""
        if draw_ratio > 30 or abs(mean_val) < 0.2:
            text += f"""    → ⚠️ Value分布が0付近に集中している可能性が高い
"""
    
    text += """
  【仮説2】Policy Entropy崩壊
    - Policy Networkが特定の手のみを選択するように収束
    - 「安全な手」= 引き分けに導く手を過学習
    - Policy Lossは下がるが、探索多様性が失われる

  【仮説3】引き分けデータの自己強化ループ
    - 引き分けゲーム → バッファに引き分けデータ蓄積 → Value=0学習 → さらに引き分けが増加
    - 「引き分けが引き分けを生む」悪循環

■ 改善提案

  1. 【即効性】引き分けペナルティの調整
     現在: 千日手=-0.9, MAX_MOVES=-0.2
     提案: 千日手=-0.95, MAX_MOVES=-0.5  (より強いペナルティ)

  2. 【中期】探索多様性の強化
     - MCTSのDirichlet noiseを増加 (α: 0.3→0.5)
     - 温度パラメータを上げる (τ: 1.0→1.5)
     - 探索回数を増やす (200→400 sims)

  3. 【長期】学習アルゴリズムの改善
     - Value Lossに正則化追加（極端な予測を抑制）
     - 引き分けデータのサンプリング比率を制限
     - Policyエントロピーをボーナスとして追加

  4. 【モニタリング】追加すべき指標
     - イテレーションごとのPolicyエントロピー平均
     - Value予測の分散（0に収束していないか）
     - 引き分け理由の内訳（千日手 vs MAX_MOVES）

══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""
    
    return text


# ===== メイン =====

def main():
    parser = argparse.ArgumentParser(description='引き分け急増問題の総合分析')
    parser.add_argument('--log', type=str, default=None, help='学習ログファイルパス')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints', help='チェックポイントディレクトリ')
    parser.add_argument('--output', type=str, default='draw_surge_analysis.png', help='出力ファイル名')
    args = parser.parse_args()
    
    print("="*70)
    print("引き分け急増問題 - 総合分析")
    print("="*70)
    
    # 1. 学習ログのパース（ユーザー提供データを使用）
    print("\n【1】学習ログを解析中...")
    
    # デフォルトの学習ログ（ユーザー提供）
    default_log = """
--- Iteration 1/20 ---
Results: BLACK=3, WHITE=7, DRAW=0
Policy Loss: 2.6627, Value Loss: 0.0566

--- Iteration 2/20 ---
Results: BLACK=4, WHITE=1, DRAW=5
Draw reasons: REPETITION=0, MAX_MOVES=5
Policy Loss: 1.2388, Value Loss: 0.0346

--- Iteration 3/20 ---
Results: BLACK=5, WHITE=0, DRAW=5
Draw reasons: REPETITION=0, MAX_MOVES=5
Policy Loss: 0.8606, Value Loss: 0.0192

--- Iteration 4/20 ---
Results: BLACK=2, WHITE=0, DRAW=8
Draw reasons: REPETITION=1, MAX_MOVES=7
Policy Loss: 0.7764, Value Loss: 0.0137

--- Iteration 5/20 ---
Results: BLACK=4, WHITE=1, DRAW=5
Policy Loss: 0.7334, Value Loss: 0.0168

--- Iteration 6/20 ---
Results: BLACK=0, WHITE=0, DRAW=10
Draw reasons: REPETITION=3, MAX_MOVES=7
Policy Loss: 0.6719, Value Loss: 0.0131
"""
    
    if args.log and os.path.exists(args.log):
        with open(args.log, 'r', encoding='utf-8') as f:
            log_text = f.read()
    else:
        log_text = default_log
        print("  (ユーザー提供の学習ログを使用)")
    
    iterations = parse_training_log(log_text)
    print(f"  解析完了: {len(iterations)} イテレーション")
    
    # データ表示
    print("\n  --- パースされたデータ ---")
    for it in iterations:
        draw_rate = it.get('draw_rate', 0) * 100
        print(f"  Iter {it['iteration']}: Draw={draw_rate:.0f}%, "
              f"PolicyLoss={it.get('policy_loss', 0):.4f}, "
              f"ValueLoss={it.get('value_loss', 0):.4f}")
    
    # 2. training_state.jsonの読み込み
    print("\n【2】training_state.jsonを読み込み中...")
    training_state = load_training_state(args.checkpoint)
    if training_state:
        print(f"  総イテレーション: {training_state.get('iteration', 'N/A')}")
        print(f"  総ゲーム数: {training_state.get('total_games', 'N/A')}")
        print(f"  総サンプル数: {training_state.get('total_examples', 'N/A')}")
    else:
        print("  (training_state.json が見つかりません)")
    
    # 3. replay_bufferの分析
    print("\n【3】replay_buffer.npzを分析中...")
    buffer_data = load_replay_buffer(args.checkpoint)
    value_analysis = {}
    if buffer_data and buffer_data.get('values') is not None:
        value_analysis = analyze_value_distribution(buffer_data['values'])
        print(f"  総サンプル数: {value_analysis['total_samples']:,}")
        print(f"  勝ちデータ(v>0.5): {value_analysis['win_ratio']*100:.1f}%")
        print(f"  負けデータ(v<-0.5): {value_analysis['loss_ratio']*100:.1f}%")
        print(f"  引き分けデータ: {value_analysis['draw_ratio']*100:.1f}%")
        print(f"  平均Value: {value_analysis['mean_value']:.4f}")
        print(f"  標準偏差: {value_analysis['std_value']:.4f}")
    else:
        print("  (replay_buffer.npz が見つかりません)")
    
    # 4. 相関分析
    print("\n【4】相関分析を実行中...")
    correlations = calculate_correlations(iterations)
    if correlations.get('policy_loss_vs_draw_rate'):
        corr = correlations['policy_loss_vs_draw_rate']
        print(f"  Policy Loss vs Draw Rate: r={corr['correlation']:.3f} ({corr['interpretation']})")
    if correlations.get('value_loss_vs_draw_rate'):
        corr = correlations['value_loss_vs_draw_rate']
        print(f"  Value Loss vs Draw Rate: r={corr['correlation']:.3f} ({corr['interpretation']})")
    
    # 5. 転換点の特定
    print("\n【5】転換点を特定中...")
    turning_point = identify_turning_point(iterations)
    if turning_point.get('turning_point_iteration'):
        print(f"  転換点: Iteration {turning_point['turning_point_iteration']}")
        print(f"  増加幅: +{turning_point['turning_point_increase']*100:.1f}%ポイント")
    
    # 6. レポート生成
    print("\n【6】レポートを生成中...")
    create_comprehensive_report(
        iterations, 
        training_state, 
        value_analysis, 
        correlations, 
        turning_point,
        output_path=args.output
    )
    
    print("\n" + "="*70)
    print("分析完了！")
    print("="*70)
    
    return iterations, training_state, value_analysis, correlations, turning_point


if __name__ == '__main__':
    main()
