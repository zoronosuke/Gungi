"""
訓練のリアルタイム監視スクリプト
training_stats.jsonを監視して更新を表示
"""

import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime

# ANSIカラーコード
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'


def clear_screen():
    """画面をクリア"""
    os.system('cls' if os.name == 'nt' else 'clear')


def load_stats(filepath: str) -> dict:
    """統計ファイルを読み込み"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def format_time(seconds: float) -> str:
    """秒を読みやすい形式に変換"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分"
    else:
        return f"{seconds/3600:.1f}時間"


def get_progress_bar(value: float, max_value: float, width: int = 30) -> str:
    """プログレスバーを生成"""
    if max_value <= 0:
        return "[" + " " * width + "]"
    ratio = min(value / max_value, 1.0)
    filled = int(width * ratio)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def display_dashboard(data: dict, target_iterations: int = 20):
    """ダッシュボードを表示"""
    clear_screen()
    
    global_stats = data.get('global_stats', {})
    iterations = data.get('iterations', [])
    
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║              🎮 GUNGI AI 訓練リアルタイムモニター 🎮                 ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")
    
    # グローバル統計
    current_iter = len(iterations)
    progress = current_iter / target_iterations if target_iterations > 0 else 0
    
    print(f"{Colors.WHITE}📊 進捗状況{Colors.RESET}")
    print(f"   イテレーション: {Colors.GREEN}{current_iter}{Colors.RESET} / {target_iterations}")
    print(f"   {get_progress_bar(current_iter, target_iterations)} {progress*100:.0f}%")
    print(f"   総ゲーム数: {Colors.YELLOW}{global_stats.get('total_games', 0)}{Colors.RESET}")
    print(f"   総手数: {global_stats.get('total_moves', 0):,}")
    print(f"   総訓練時間: {format_time(global_stats.get('total_training_time', 0))}")
    print()
    
    if not iterations:
        print(f"{Colors.YELLOW}⏳ 訓練データを待機中...{Colors.RESET}")
        return
    
    # 最新イテレーションの詳細
    latest = iterations[-1]
    
    print(f"{Colors.WHITE}🎯 最新イテレーション {latest['iteration']}{Colors.RESET}")
    print("─" * 70)
    
    # 勝敗統計
    total_games = latest['black_wins'] + latest['white_wins'] + latest['draws']
    black_pct = latest['black_wins'] / total_games * 100 if total_games > 0 else 0
    white_pct = latest['white_wins'] / total_games * 100 if total_games > 0 else 0
    draw_pct = latest['draws'] / total_games * 100 if total_games > 0 else 0
    
    print(f"   勝敗: ", end="")
    print(f"{Colors.WHITE}⬛黒 {latest['black_wins']}({black_pct:.0f}%){Colors.RESET} | ", end="")
    print(f"{Colors.WHITE}⬜白 {latest['white_wins']}({white_pct:.0f}%){Colors.RESET} | ", end="")
    
    # 引き分けは赤で警告
    draw_color = Colors.RED if draw_pct > 50 else Colors.YELLOW if draw_pct > 30 else Colors.GREEN
    print(f"{draw_color}🤝引分 {latest['draws']}({draw_pct:.0f}%){Colors.RESET}")
    
    # 終了理由
    reasons = latest.get('termination_reasons', {})
    checkmate = reasons.get('CHECKMATE', 0)
    repetition = reasons.get('REPETITION', 0)
    max_moves = reasons.get('MAX_MOVES', 0)
    
    rep_color = Colors.RED if repetition > 5 else Colors.YELLOW if repetition > 0 else Colors.GREEN
    print(f"   終了理由: ✓詰み {Colors.GREEN}{checkmate}{Colors.RESET} | ", end="")
    print(f"🔄千日手 {rep_color}{repetition}{Colors.RESET} | ", end="")
    print(f"⏰最大手数 {max_moves}")
    print()
    
    # 損失
    print(f"   📉 Policy Loss: {Colors.CYAN}{latest['policy_loss']:.4f}{Colors.RESET}")
    print(f"   📉 Value Loss:  {Colors.CYAN}{latest['value_loss']:.4f}{Colors.RESET}")
    print()
    
    # 重要な指標
    print(f"   📈 重要指標:")
    
    # Policy Entropy
    entropy = latest['avg_policy_entropy']
    entropy_color = Colors.GREEN if entropy > 2.0 else Colors.YELLOW if entropy > 1.0 else Colors.RED
    print(f"      Policy Entropy: {entropy_color}{entropy:.4f}{Colors.RESET} ", end="")
    if entropy < 1.0:
        print(f"{Colors.RED}⚠️ 探索多様性が低い{Colors.RESET}")
    elif entropy < 2.0:
        print(f"{Colors.YELLOW}📉 やや低い{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}✓ 良好{Colors.RESET}")
    
    # Value 0近傍率
    v0_ratio = latest['value_near_zero_ratio'] * 100
    v0_color = Colors.RED if v0_ratio > 50 else Colors.YELLOW if v0_ratio > 30 else Colors.GREEN
    print(f"      Value 0近傍率: {v0_color}{v0_ratio:.1f}%{Colors.RESET} ", end="")
    if v0_ratio > 50:
        print(f"{Colors.RED}⚠️ 0収束問題{Colors.RESET}")
    elif v0_ratio > 30:
        print(f"{Colors.YELLOW}📉 注意{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}✓ 良好{Colors.RESET}")
    
    # バッファDraw率
    buf_draw = latest['buffer_draw_ratio'] * 100
    buf_color = Colors.RED if buf_draw > 30 else Colors.YELLOW if buf_draw > 20 else Colors.GREEN
    print(f"      バッファDraw率: {buf_color}{buf_draw:.1f}%{Colors.RESET} ", end="")
    if buf_draw > 30:
        print(f"{Colors.RED}⚠️ 制限値到達{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}✓ 良好{Colors.RESET}")
    
    # 平均ゲーム長
    avg_len = latest['avg_game_length']
    print(f"      平均ゲーム長: {Colors.CYAN}{avg_len:.1f}手{Colors.RESET}")
    print()
    
    # 推移グラフ（簡易版）
    if len(iterations) > 1:
        print(f"{Colors.WHITE}📈 推移 (最新{min(10, len(iterations))}イテレーション){Colors.RESET}")
        print("─" * 70)
        
        recent = iterations[-10:]
        
        # Draw率の推移
        print("   Draw率: ", end="")
        for it in recent:
            total = it['black_wins'] + it['white_wins'] + it['draws']
            rate = it['draws'] / total * 100 if total > 0 else 0
            if rate >= 80:
                print(f"{Colors.RED}█{Colors.RESET}", end="")
            elif rate >= 50:
                print(f"{Colors.YELLOW}▓{Colors.RESET}", end="")
            elif rate >= 20:
                print(f"{Colors.CYAN}▒{Colors.RESET}", end="")
            else:
                print(f"{Colors.GREEN}░{Colors.RESET}", end="")
        print(f" (█:80%+ ▓:50%+ ▒:20%+ ░:<20%)")
        
        # 千日手の推移
        print("   千日手: ", end="")
        for it in recent:
            rep = it.get('termination_reasons', {}).get('REPETITION', 0)
            total = it['black_wins'] + it['white_wins'] + it['draws']
            if total > 0:
                rate = rep / total
                if rate >= 0.8:
                    print(f"{Colors.RED}█{Colors.RESET}", end="")
                elif rate >= 0.5:
                    print(f"{Colors.YELLOW}▓{Colors.RESET}", end="")
                elif rate > 0:
                    print(f"{Colors.CYAN}▒{Colors.RESET}", end="")
                else:
                    print(f"{Colors.GREEN}░{Colors.RESET}", end="")
            else:
                print("?", end="")
        print()
        
        # Policy Lossの推移
        print("   P.Loss: ", end="")
        p_losses = [it['policy_loss'] for it in recent]
        max_pl = max(p_losses) if p_losses else 1
        min_pl = min(p_losses) if p_losses else 0
        for pl in p_losses:
            normalized = (pl - min_pl) / (max_pl - min_pl + 0.001)
            if normalized > 0.8:
                print("█", end="")
            elif normalized > 0.6:
                print("▓", end="")
            elif normalized > 0.4:
                print("▒", end="")
            elif normalized > 0.2:
                print("░", end="")
            else:
                print("_", end="")
        print(f" ({min_pl:.3f} ~ {max_pl:.3f})")
    
    print()
    print(f"{Colors.CYAN}最終更新: {datetime.now().strftime('%H:%M:%S')} | Ctrl+C で終了{Colors.RESET}")
    
    # 問題診断
    problems = []
    if draw_pct >= 100:
        problems.append("🚨 全ゲームが引き分け")
    if repetition == total_games and total_games > 0:
        problems.append("🚨 全て千日手で終了")
    if v0_ratio > 70:
        problems.append("🚨 Value予測が0に強く収束中")
    if entropy < 0.5:
        problems.append("🚨 Policy Entropyが非常に低い")
    
    if problems:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️ 問題検出:{Colors.RESET}")
        for p in problems:
            print(f"   {Colors.RED}{p}{Colors.RESET}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='訓練リアルタイムモニター')
    parser.add_argument('--input', '-i', default='checkpoints/training_stats.json',
                        help='統計ファイルのパス')
    parser.add_argument('--interval', '-t', type=float, default=5.0,
                        help='更新間隔（秒）')
    parser.add_argument('--iterations', '-n', type=int, default=20,
                        help='目標イテレーション数')
    
    args = parser.parse_args()
    
    stats_path = Path(args.input)
    last_mtime = 0
    
    print(f"{Colors.CYAN}訓練モニターを開始します...{Colors.RESET}")
    print(f"監視ファイル: {stats_path}")
    print(f"更新間隔: {args.interval}秒")
    print()
    
    try:
        while True:
            # ファイルの更新をチェック
            if stats_path.exists():
                current_mtime = stats_path.stat().st_mtime
                
                # ファイルが更新されたか、初回の場合
                if current_mtime != last_mtime:
                    data = load_stats(str(stats_path))
                    if data:
                        display_dashboard(data, args.iterations)
                        last_mtime = current_mtime
                        
                        # 完了チェック
                        if len(data.get('iterations', [])) >= args.iterations:
                            print(f"\n{Colors.GREEN}{Colors.BOLD}✅ 訓練完了!{Colors.RESET}")
                            break
            else:
                clear_screen()
                print(f"{Colors.YELLOW}⏳ 統計ファイルを待機中: {stats_path}{Colors.RESET}")
                print(f"   訓練が開始されるとデータが表示されます...")
            
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print(f"\n{Colors.CYAN}モニターを終了しました{Colors.RESET}")


if __name__ == '__main__':
    main()
