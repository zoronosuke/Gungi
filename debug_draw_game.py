"""
引き分けの原因を調査するデバッグスクリプト
1ゲームだけ詳細ログを出力
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from src.engine.board import Board
from src.engine.piece import Player, PieceType
from src.engine.rules import Rules
from src.engine.initial_setup import load_initial_board, get_initial_hand_pieces
from src.model.encoder import StateEncoder, ActionEncoder
from src.model.network import create_model

MAX_MOVES = 300

def run_debug_game():
    print("=" * 60)
    print("引き分け原因調査 - 1ゲーム詳細ログ")
    print("=" * 60)
    
    # モデル読み込み
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    network = create_model()
    
    # チェックポイント読み込み
    try:
        checkpoint = torch.load('checkpoints/latest.pt', map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            network.load_state_dict(checkpoint['model_state_dict'])
        else:
            network.load_state_dict(checkpoint)
        print("Model loaded from checkpoint")
    except Exception as e:
        print(f"No checkpoint loaded: {e}")
    
    network.to(device)
    network.eval()
    
    state_encoder = StateEncoder()
    action_encoder = ActionEncoder()
    
    # ゲーム初期化
    board = load_initial_board()
    current_player = Player.BLACK
    hands = {
        Player.BLACK: get_initial_hand_pieces(Player.BLACK),
        Player.WHITE: get_initial_hand_pieces(Player.WHITE)
    }
    
    move_count = 0
    position_history = {}  # 局面の繰り返しを検出
    last_10_moves = []  # 最後の10手を記録
    
    print(f"\n初期局面の駒数:")
    board_piece_count = sum(board.get_stack_height((r, c)) for r in range(9) for c in range(9))
    print(f"  盤上の駒: {board_piece_count}")
    print(f"  黒の持ち駒: {sum(hands[Player.BLACK].values())}")
    print(f"  白の持ち駒: {sum(hands[Player.WHITE].values())}")
    
    print("\n" + "=" * 60)
    print("ゲーム開始")
    print("=" * 60)
    
    while move_count < MAX_MOVES:
        # 合法手を取得
        legal_moves = Rules.get_legal_moves(board, current_player, hands[current_player])
        
        if not legal_moves:
            print(f"\n手番: {move_count + 1}, プレイヤー: {current_player.name}")
            print(f"合法手なし！ {current_player.opponent.name}の勝利")
            return current_player.opponent, move_count, "NO_LEGAL_MOVES", last_10_moves
        
        # 局面をハッシュ化（千日手チェック）
        board_str = str(board)  # 簡易的なハッシュ
        position_key = (board_str, current_player)
        position_history[position_key] = position_history.get(position_key, 0) + 1
        
        if position_history[position_key] >= 4:
            print(f"\n手番: {move_count + 1}")
            print(f"千日手発生！同じ局面が4回")
            return None, move_count, "REPETITION", last_10_moves
        
        # NN推論
        my_hand = hands[current_player]
        opponent_hand = hands[current_player.opponent]
        state = state_encoder.encode(board, current_player, my_hand, opponent_hand)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            log_policy, value = network(state_tensor)
            policy = torch.exp(log_policy).cpu().numpy()[0]
        
        # 合法手でマスク
        action_probs = {}
        for move in legal_moves:
            action_idx = action_encoder.encode_move(move)
            if action_idx is not None:
                action_probs[action_idx] = policy[action_idx]
        
        if not action_probs:
            print(f"\n手番: {move_count + 1}")
            print(f"エンコード可能な合法手なし！")
            return None, move_count, "NO_ENCODABLE_MOVES", last_10_moves
        
        # 最善手を選択
        best_action = max(action_probs.keys(), key=lambda a: action_probs[a])
        move = action_encoder.decode_action(best_action, current_player, board)
        
        # 手を適用
        success, captured = Rules.apply_move(board, move, my_hand)
        
        if not success:
            print(f"\n手番: {move_count + 1}")
            print(f"手の適用に失敗！")
            return current_player.opponent, move_count, "MOVE_FAILED", last_10_moves
        
        # 最後の10手を記録
        move_info = {
            'move_num': move_count + 1,
            'player': current_player.name,
            'move_type': move.move_type.name,
            'piece': move.piece_type.name if move.piece_type else 'N/A',
            'from': move.from_pos,
            'to': move.to_pos,
            'captured': captured.piece_type.name if captured else None,
            'legal_moves_count': len(legal_moves),
            'value': value.item()
        }
        last_10_moves.append(move_info)
        if len(last_10_moves) > 20:
            last_10_moves.pop(0)
        
        # 10手ごとに状況報告
        if (move_count + 1) % 50 == 0:
            print(f"\n--- {move_count + 1}手目完了 ---")
            print(f"  プレイヤー: {current_player.name}")
            print(f"  合法手数: {len(legal_moves)}")
            print(f"  評価値: {value.item():.4f}")
            # 簡易的に盤上の駒数をカウント
            total_pieces = sum(board.get_stack_height((r, c)) for r in range(9) for c in range(9))
            black_pieces = total_pieces // 2  # 概算
            white_pieces = total_pieces - black_pieces
            print(f"  盤上の駒: 黒={black_pieces}, 白={white_pieces}")
        
        # 勝敗判定
        is_over, winner = Rules.is_game_over(board)
        if is_over:
            print(f"\n手番: {move_count + 1}")
            print(f"ゲーム終了！勝者: {winner.name if winner else '引き分け'}")
            return winner, move_count, "GAME_OVER", last_10_moves
        
        # 手番交代
        current_player = current_player.opponent
        move_count += 1
    
    # 最大手数到達
    print(f"\n最大手数 {MAX_MOVES} に到達！引き分け")
    return None, move_count, "MAX_MOVES", last_10_moves


def main():
    winner, move_count, reason, last_moves = run_debug_game()
    
    print("\n" + "=" * 60)
    print("結果サマリー")
    print("=" * 60)
    print(f"勝者: {winner.name if winner else '引き分け'}")
    print(f"総手数: {move_count}")
    print(f"終了理由: {reason}")
    
    print("\n" + "=" * 60)
    print("最後の20手:")
    print("=" * 60)
    for m in last_moves:
        captured_str = f" (捕獲: {m['captured']})" if m['captured'] else ""
        print(f"  {m['move_num']:3d}. {m['player']:5s} {m['move_type']:6s} "
              f"{m['piece']:10s} {m['from']} -> {m['to']}{captured_str} "
              f"(合法手数={m['legal_moves_count']}, 評価値={m['value']:.3f})")
    
    # 引き分け分析
    if winner is None:
        print("\n" + "=" * 60)
        print("引き分け分析:")
        print("=" * 60)
        if reason == "MAX_MOVES":
            print("  → 300手に到達したため引き分け")
            print("  → これはモデルが決着をつけられないことを示唆")
            print("  → 考えられる原因:")
            print("     1. 駒の動きがループしている")
            print("     2. モデルが攻めの手を学習していない")
            print("     3. 勝利条件の判定に問題がある可能性")
        elif reason == "REPETITION":
            print("  → 千日手による引き分け")
        elif reason == "NO_LEGAL_MOVES":
            print("  → 合法手がない（ステイルメイト）")
        elif reason == "NO_ENCODABLE_MOVES":
            print("  → 合法手はあるがエンコードできない")


if __name__ == "__main__":
    main()
