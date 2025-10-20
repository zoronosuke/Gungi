"""
統合テスト: ゲームフロー全体のテスト
初期配置から複数手のゲーム進行を確認
"""

import pytest
from src.engine import Board, Player, Rules
from src.engine.initial_setup import load_initial_board


class TestGameFlow:
    """ゲーム全体の流れをテストするクラス"""
    
    def test_initial_setup_loads_correctly(self, initial_board):
        """初期配置が正しく読み込まれることを確認"""
        # 帥の位置が設定されているか確認
        black_sui = initial_board.get_sui_position(Player.BLACK)
        white_sui = initial_board.get_sui_position(Player.WHITE)
        
        assert black_sui is not None, "黒の帥の位置が設定されていません"
        assert white_sui is not None, "白の帥の位置が設定されていません"
        assert black_sui != white_sui, "帥の位置が重複しています"
    
    def test_first_turn_has_legal_moves(self, initial_board):
        """初手に合法手が存在することを確認"""
        legal_moves = Rules.get_legal_moves(initial_board, Player.BLACK)
        
        assert len(legal_moves) > 0, "初手に合法手が存在しません"
        print(f"初手の合法手数: {len(legal_moves)}")
    
    def test_move_application_succeeds(self, initial_board):
        """手の適用が成功することを確認"""
        legal_moves = Rules.get_legal_moves(initial_board, Player.BLACK)
        
        assert len(legal_moves) > 0, "合法手が存在しません"
        
        # 最初の合法手を適用
        move = legal_moves[0]
        success = Rules.apply_move(initial_board, move)
        
        assert success, f"手の適用に失敗しました: {move}"
    
    def test_turn_alternation(self, initial_board):
        """手番が正しく交代することを確認"""
        current_player = Player.BLACK
        
        for turn in range(10):
            legal_moves = Rules.get_legal_moves(initial_board, current_player)
            
            if not legal_moves:
                break
            
            # 最初の合法手を適用
            move = legal_moves[0]
            success = Rules.apply_move(initial_board, move)
            
            assert success, f"{turn + 1}手目の適用に失敗しました"
            
            # ゲーム終了判定
            is_over, winner = Rules.is_game_over(initial_board)
            if is_over:
                print(f"{turn + 1}手でゲーム終了。勝者: {winner.name}")
                break
            
            # 手番交代
            current_player = current_player.opponent
        
        print(f"10手の進行が完了しました")
    
    def test_game_does_not_crash_after_many_moves(self, initial_board):
        """多数の手を進めてもクラッシュしないことを確認"""
        current_player = Player.BLACK
        max_moves = 100
        
        for turn in range(max_moves):
            legal_moves = Rules.get_legal_moves(initial_board, current_player)
            
            if not legal_moves:
                print(f"{turn + 1}手目で合法手がなくなりました")
                break
            
            # 最初の合法手を適用
            move = legal_moves[0]
            success = Rules.apply_move(initial_board, move)
            
            assert success, f"{turn + 1}手目の適用に失敗しました"
            
            # ゲーム終了判定
            is_over, winner = Rules.is_game_over(initial_board)
            if is_over:
                print(f"{turn + 1}手でゲーム終了。勝者: {winner.name}")
                return
            
            # 手番交代
            current_player = current_player.opponent
        
        print(f"{max_moves}手まで正常に進行しました")
    
    def test_empty_board_creation(self):
        """空の盤面を作成できることを確認"""
        board = Board()
        
        # 全てのマスが空であることを確認
        for row in range(9):
            for col in range(9):
                stack = board.get_stack((row, col))
                assert stack.is_empty(), f"({row}, {col})が空ではありません"
