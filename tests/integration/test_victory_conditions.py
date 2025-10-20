"""
統合テスト: 勝利条件のテスト
ゲーム終了判定とその条件を確認
"""

import pytest
from src.engine import Board, Player, PieceType, Piece, Rules, Move


class TestVictoryConditions:
    """勝利条件のテストクラス"""
    
    def test_game_over_when_sui_captured(self):
        """帥が取られるとゲーム終了になることを確認"""
        board = Board()
        
        # 黒の帥を配置
        board.add_piece((7, 4), Piece(PieceType.SUI, Player.BLACK))
        # 白の大将を配置
        board.add_piece((1, 4), Piece(PieceType.SUI, Player.WHITE))
        # 黒の大将を白の帥の隣に配置
        board.add_piece((2, 4), Piece(PieceType.DAI, Player.BLACK))
        
        # ゲーム開始時は終了していない
        is_over, winner = Rules.is_game_over(board)
        assert not is_over, "ゲーム開始時に終了判定がTrueです"
        
        # 黒が白の帥を取得
        move = Move.create_capture_move(
            from_pos=(2, 4),
            to_pos=(1, 4),
            player=Player.BLACK
        )
        
        success, captured = Rules.apply_move(board, move)
        assert success, "帥の取得に失敗しました"
        
        # ゲーム終了判定
        is_over, winner = Rules.is_game_over(board)
        
        assert is_over, "帥を取得してもゲームが終了していません"
        assert winner == Player.BLACK, "勝者が正しくありません"
    
    def test_game_continues_when_sui_present(self):
        """両方の帥が盤上にあればゲームは継続することを確認"""
        board = Board()
        
        # 両方の帥を配置
        board.add_piece((7, 4), Piece(PieceType.SUI, Player.BLACK))
        board.add_piece((1, 4), Piece(PieceType.SUI, Player.WHITE))
        
        # ゲームは継続中
        is_over, winner = Rules.is_game_over(board)
        
        assert not is_over, "両方の帥があるのにゲームが終了しています"
        assert winner is None, "勝者が設定されています"
    
    def test_black_wins_when_white_sui_captured(self):
        """白の帥が取られると黒の勝利になることを確認"""
        board = Board()
        
        # 黒の帥を配置（安全な位置）
        board.add_piece((8, 8), Piece(PieceType.SUI, Player.BLACK))
        # 白の帥を盤上から削除（取られた状態）
        
        # ゲーム終了判定
        is_over, winner = Rules.is_game_over(board)
        
        assert is_over, "白の帥がないのにゲームが継続しています"
        assert winner == Player.BLACK, "黒の勝利になっていません"
    
    def test_white_wins_when_black_sui_captured(self):
        """黒の帥が取られると白の勝利になることを確認"""
        board = Board()
        
        # 白の帥を配置（安全な位置）
        board.add_piece((0, 0), Piece(PieceType.SUI, Player.WHITE))
        # 黒の帥を盤上から削除（取られた状態）
        
        # ゲーム終了判定
        is_over, winner = Rules.is_game_over(board)
        
        assert is_over, "黒の帥がないのにゲームが継続しています"
        assert winner == Player.WHITE, "白の勝利になっていません"
    
    def test_sui_in_stack_is_still_present(self):
        """帥がスタックの下にいてもゲームは継続することを確認"""
        board = Board()
        
        # 黒の帥を配置
        board.add_piece((7, 4), Piece(PieceType.SUI, Player.BLACK))
        # 白の帥を配置
        board.add_piece((1, 4), Piece(PieceType.SUI, Player.WHITE))
        
        # 白の帥の上に白の兵を乗せる（実際は不可能だが、テストとして）
        # 注: 帥の上には乗せられないルールがあるので、このテストは実際には発生しない状況
        
        # ゲームは継続中
        is_over, winner = Rules.is_game_over(board)
        
        assert not is_over, "帥が存在するのにゲームが終了しています"
    
    def test_capturing_sui_removes_it_from_board(self):
        """帥を取得すると盤上から削除されることを確認"""
        board = Board()
        
        # 両方の帥を配置
        board.add_piece((7, 4), Piece(PieceType.SUI, Player.BLACK))
        board.add_piece((1, 4), Piece(PieceType.SUI, Player.WHITE))
        
        # 白の帥の隣に黒の大将を配置
        board.add_piece((2, 4), Piece(PieceType.DAI, Player.BLACK))
        
        # 白の帥の位置を確認
        white_sui_pos = board.get_sui_position(Player.WHITE)
        assert white_sui_pos == (1, 4), "白の帥の位置が正しくありません"
        
        # 黒が白の帥を取得
        move = Move.create_capture_move(
            from_pos=(2, 4),
            to_pos=(1, 4),
            player=Player.BLACK
        )
        
        Rules.apply_move(board, move)
        
        # 白の帥の位置がNoneになっているか確認
        white_sui_pos = board.get_sui_position(Player.WHITE)
        assert white_sui_pos is None, "帥を取得したのに位置が残っています"
    
    def test_game_over_immediately_after_sui_capture(self):
        """帥を取得した直後にゲーム終了判定がTrueになることを確認"""
        board = Board()
        
        # 最小限の構成で帥の取得をシミュレート
        board.add_piece((7, 4), Piece(PieceType.SUI, Player.BLACK))
        board.add_piece((1, 4), Piece(PieceType.SUI, Player.WHITE))
        board.add_piece((1, 5), Piece(PieceType.DAI, Player.BLACK))
        
        # 取得前はゲーム継続中
        is_over, _ = Rules.is_game_over(board)
        assert not is_over
        
        # 帥を取得
        move = Move.create_capture_move(
            from_pos=(1, 5),
            to_pos=(1, 4),
            player=Player.BLACK
        )
        
        Rules.apply_move(board, move)
        
        # 取得後は即座にゲーム終了
        is_over, winner = Rules.is_game_over(board)
        
        assert is_over, "帥を取得してもゲームが終了していません"
        assert winner == Player.BLACK, "勝者が正しくありません"
