"""
自己対戦シミュレーション
ランダムな手を実行してルール違反やバグを検出する

使用方法:
    python scripts/self_play_simulation.py [--games 1000] [--max-moves 200] [--verbose]
"""

import sys
import os
import argparse
import random
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.engine import Board, Player, PieceType, Piece, Rules, Move, MoveType
from src.engine.initial_setup import load_initial_board, get_initial_hand_pieces
from src.engine.board import BOARD_SIZE


@dataclass
class GameResult:
    """ゲーム結果"""
    game_id: int
    winner: Optional[Player]
    total_moves: int
    termination_reason: str
    error: Optional[str] = None
    error_move: Optional[int] = None


@dataclass
class SimulationStats:
    """シミュレーション統計"""
    total_games: int = 0
    completed_games: int = 0
    error_games: int = 0
    black_wins: int = 0
    white_wins: int = 0
    draws: int = 0
    max_moves_reached: int = 0
    total_moves: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class SelfPlaySimulator:
    """自己対戦シミュレータ"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.stats = SimulationStats()
    
    def run_game(self, game_id: int, max_moves: int = 200) -> GameResult:
        """1ゲームを実行"""
        board = load_initial_board()
        hand_pieces = {
            Player.BLACK: get_initial_hand_pieces(Player.BLACK),
            Player.WHITE: get_initial_hand_pieces(Player.WHITE),
        }
        
        current_player = Player.BLACK
        move_count = 0
        
        try:
            while move_count < max_moves:
                # ゲーム終了チェック
                is_over, winner = Rules.is_game_over(board)
                if is_over:
                    return GameResult(
                        game_id=game_id,
                        winner=winner,
                        total_moves=move_count,
                        termination_reason="game_over"
                    )
                
                # 不変条件のチェック
                self._check_invariants(board, move_count)
                
                # 合法手を取得
                legal_moves = Rules.get_legal_moves(
                    board, current_player, hand_pieces[current_player]
                )
                
                if not legal_moves:
                    # 合法手がない（ステイルメイト）
                    return GameResult(
                        game_id=game_id,
                        winner=None,
                        total_moves=move_count,
                        termination_reason="stalemate"
                    )
                
                # ランダムに手を選択
                move = random.choice(legal_moves)
                
                # 手を適用
                success, captured = Rules.apply_move(
                    board, move, hand_pieces[current_player]
                )
                
                if not success:
                    # 合法手が失敗した（バグ！）
                    error_msg = f"合法手が失敗: {move} at move {move_count}"
                    return GameResult(
                        game_id=game_id,
                        winner=None,
                        total_moves=move_count,
                        termination_reason="error",
                        error=error_msg,
                        error_move=move_count
                    )
                
                # 手番交代
                current_player = current_player.opponent
                move_count += 1
            
            # 最大手数に到達
            return GameResult(
                game_id=game_id,
                winner=None,
                total_moves=move_count,
                termination_reason="max_moves"
            )
            
        except Exception as e:
            error_msg = f"例外発生: {type(e).__name__}: {e} at move {move_count}"
            return GameResult(
                game_id=game_id,
                winner=None,
                total_moves=move_count,
                termination_reason="exception",
                error=error_msg,
                error_move=move_count
            )
    
    def _check_invariants(self, board: Board, move_count: int):
        """不変条件をチェック"""
        # スタック高さのチェック
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                height = board.get_stack_height((row, col))
                if height > 3:
                    raise AssertionError(
                        f"スタック高さ違反: ({row}, {col}) = {height} at move {move_count}"
                    )
        
        # 帥の数のチェック
        for player in [Player.BLACK, Player.WHITE]:
            sui_count = 0
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    stack = board.get_stack((row, col))
                    for level in range(stack.get_height()):
                        piece = stack.get_piece_at_level(level)
                        if piece and piece.owner == player and piece.piece_type == PieceType.SUI:
                            sui_count += 1
            
            if sui_count > 1:
                raise AssertionError(
                    f"{player}の帥が{sui_count}個 at move {move_count}"
                )
    
    def run_simulation(self, num_games: int, max_moves: int = 200) -> SimulationStats:
        """複数ゲームのシミュレーションを実行"""
        self.stats = SimulationStats()
        
        print(f"🎮 自己対戦シミュレーション開始: {num_games}ゲーム")
        print(f"   最大手数: {max_moves}")
        print("=" * 50)
        
        for i in range(num_games):
            result = self.run_game(i + 1, max_moves)
            self.stats.total_games += 1
            self.stats.total_moves += result.total_moves
            
            if result.termination_reason == "game_over":
                self.stats.completed_games += 1
                if result.winner == Player.BLACK:
                    self.stats.black_wins += 1
                elif result.winner == Player.WHITE:
                    self.stats.white_wins += 1
            elif result.termination_reason == "stalemate":
                self.stats.draws += 1
            elif result.termination_reason == "max_moves":
                self.stats.max_moves_reached += 1
            elif result.termination_reason in ["error", "exception"]:
                self.stats.error_games += 1
                self.stats.errors.append(result.error)
            
            if self.verbose or result.error:
                self._print_result(result)
            
            # 進捗表示
            if (i + 1) % 100 == 0:
                print(f"  進捗: {i + 1}/{num_games} ゲーム完了")
        
        self._print_summary()
        return self.stats
    
    def _print_result(self, result: GameResult):
        """ゲーム結果を表示"""
        status = "✅" if result.termination_reason == "game_over" else "⚠️"
        if result.error:
            status = "❌"
        
        winner_str = result.winner.name if result.winner else "なし"
        print(f"{status} Game {result.game_id}: "
              f"手数={result.total_moves}, "
              f"勝者={winner_str}, "
              f"終了理由={result.termination_reason}")
        
        if result.error:
            print(f"   エラー: {result.error}")
    
    def _print_summary(self):
        """統計サマリーを表示"""
        print("\n" + "=" * 50)
        print("📊 シミュレーション結果")
        print("=" * 50)
        print(f"総ゲーム数:     {self.stats.total_games}")
        print(f"完了ゲーム:     {self.stats.completed_games}")
        print(f"エラーゲーム:   {self.stats.error_games}")
        print(f"黒の勝利:       {self.stats.black_wins}")
        print(f"白の勝利:       {self.stats.white_wins}")
        print(f"引き分け:       {self.stats.draws}")
        print(f"最大手数到達:   {self.stats.max_moves_reached}")
        
        if self.stats.total_games > 0:
            avg_moves = self.stats.total_moves / self.stats.total_games
            print(f"平均手数:       {avg_moves:.1f}")
        
        if self.stats.errors:
            print("\n❌ 発見されたエラー:")
            for i, error in enumerate(self.stats.errors[:10], 1):
                print(f"   {i}. {error}")
            if len(self.stats.errors) > 10:
                print(f"   ... 他 {len(self.stats.errors) - 10} 件")
        else:
            print("\n✅ エラーは発見されませんでした！")


def main():
    parser = argparse.ArgumentParser(description="軍儀の自己対戦シミュレーション")
    parser.add_argument("--games", type=int, default=100, help="シミュレーションするゲーム数")
    parser.add_argument("--max-moves", type=int, default=200, help="ゲームあたりの最大手数")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細な出力を表示")
    
    args = parser.parse_args()
    
    simulator = SelfPlaySimulator(verbose=args.verbose)
    stats = simulator.run_simulation(args.games, args.max_moves)
    
    # エラーがあった場合は終了コード1を返す
    if stats.error_games > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
