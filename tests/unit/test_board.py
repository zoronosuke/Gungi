"""
単体テスト: 盤面管理のテスト
Board クラスの基本機能を確認
"""

import pytest
from src.engine import Board, Player, PieceType, Piece


class TestBoard:
    """盤面管理のテストクラス"""
    
    def test_board_initialization(self):
        """盤面が正しく初期化されることを確認"""
        board = Board()
        
        # 9x9の盤面が作成されていることを確認
        for row in range(9):
            for col in range(9):
                stack = board.get_stack((row, col))
                assert stack is not None, f"({row}, {col})のスタックが存在しません"
                assert stack.is_empty(), f"({row}, {col})が空ではありません"
    
    def test_add_piece_to_empty_square(self, empty_board):
        """空のマスに駒を追加できることを確認"""
        piece = Piece(PieceType.HYO, Player.BLACK)
        position = (4, 4)
        
        success = empty_board.add_piece(position, piece)
        
        assert success, "駒の追加に失敗しました"
        
        stack = empty_board.get_stack(position)
        assert not stack.is_empty(), "スタックが空のままです"
        assert stack.get_height() == 1, "スタックの高さが1ではありません"
        
        top_piece = stack.get_top_piece()
        assert top_piece.piece_type == PieceType.HYO, "駒の種類が正しくありません"
        assert top_piece.owner == Player.BLACK, "駒の所有者が正しくありません"
    
    def test_stack_height_increases(self, empty_board):
        """駒を積むとスタック高さが増えることを確認"""
        position = (4, 4)
        
        # 1個目
        piece1 = Piece(PieceType.HYO, Player.BLACK)
        empty_board.add_piece(position, piece1)
        assert empty_board.get_stack_height(position) == 1
        
        # 2個目
        piece2 = Piece(PieceType.YARI, Player.BLACK)
        empty_board.add_piece(position, piece2)
        assert empty_board.get_stack_height(position) == 2
        
        # 3個目
        piece3 = Piece(PieceType.UMA, Player.BLACK)
        empty_board.add_piece(position, piece3)
        assert empty_board.get_stack_height(position) == 3
    
    def test_cannot_stack_more_than_3_pieces(self, empty_board):
        """4段目以上は積めないことを確認"""
        position = (4, 4)
        
        # 3段まで積む
        for i, piece_type in enumerate([PieceType.HYO, PieceType.YARI, PieceType.UMA]):
            piece = Piece(piece_type, Player.BLACK)
            success = empty_board.add_piece(position, piece)
            assert success, f"{i + 1}個目の駒の追加に失敗しました"
        
        # 4個目は失敗するべき
        piece4 = Piece(PieceType.SAMURAI, Player.BLACK)
        success = empty_board.add_piece(position, piece4)
        
        assert not success, "4段目の駒が追加できてしまいました"
        assert empty_board.get_stack_height(position) == 3, "スタックの高さが3を超えています"
    
    def test_cannot_stack_on_sui(self, empty_board):
        """帥の上に駒を乗せられないことを確認"""
        position = (4, 4)
        
        # 帥を配置
        sui = Piece(PieceType.SUI, Player.BLACK)
        success = empty_board.add_piece(position, sui)
        assert success, "帥の配置に失敗しました"
        
        # 帥の上に兵を乗せようとする
        hyo = Piece(PieceType.HYO, Player.BLACK)
        success = empty_board.add_piece(position, hyo)
        
        assert not success, "帥の上に駒を乗せられてしまいました"
        assert empty_board.get_stack_height(position) == 1, "帥の上に駒が乗っています"
    
    def test_sui_position_tracking(self, empty_board):
        """帥の位置が正しく記録されることを確認"""
        black_sui_pos = (7, 4)
        white_sui_pos = (1, 4)
        
        # 黒の帥を配置
        black_sui = Piece(PieceType.SUI, Player.BLACK)
        empty_board.add_piece(black_sui_pos, black_sui)
        
        # 白の帥を配置
        white_sui = Piece(PieceType.SUI, Player.WHITE)
        empty_board.add_piece(white_sui_pos, white_sui)
        
        # 位置が正しく記録されているか確認
        assert empty_board.get_sui_position(Player.BLACK) == black_sui_pos
        assert empty_board.get_sui_position(Player.WHITE) == white_sui_pos
    
    def test_remove_top_piece(self, empty_board):
        """最上段の駒を取り除けることを確認"""
        position = (4, 4)
        
        # 2段積む
        piece1 = Piece(PieceType.HYO, Player.BLACK)
        piece2 = Piece(PieceType.YARI, Player.BLACK)
        empty_board.add_piece(position, piece1)
        empty_board.add_piece(position, piece2)
        
        assert empty_board.get_stack_height(position) == 2
        
        # 最上段を取り除く
        stack = empty_board.get_stack(position)
        removed_piece = stack.remove_top_piece()
        
        assert removed_piece is not None, "駒を取り除けませんでした"
        assert removed_piece.piece_type == PieceType.YARI, "取り除いた駒が正しくありません"
        assert empty_board.get_stack_height(position) == 1, "スタックの高さが正しくありません"
    
    def test_boundary_check(self, empty_board):
        """盤面の境界が正しく機能することを確認"""
        # 有効な位置
        valid_positions = [(0, 0), (0, 8), (8, 0), (8, 8), (4, 4)]
        for pos in valid_positions:
            stack = empty_board.get_stack(pos)
            assert stack is not None, f"有効な位置{pos}でスタックを取得できません"
        
        # 無効な位置（範囲外）は ValueError を raise するべき
        invalid_positions = [(-1, 0), (0, -1), (9, 0), (0, 9), (10, 10)]
        for pos in invalid_positions:
            with pytest.raises(ValueError):
                stack = empty_board.get_stack(pos)
    
    def test_to_dict_serialization(self, empty_board):
        """盤面を辞書形式にシリアライズできることを確認"""
        # いくつか駒を配置
        empty_board.add_piece((0, 0), Piece(PieceType.HYO, Player.BLACK))
        empty_board.add_piece((8, 8), Piece(PieceType.SUI, Player.WHITE))
        
        # 辞書に変換
        board_dict = empty_board.to_dict()
        
        assert board_dict is not None, "to_dict()がNoneを返しました"
        assert 'board' in board_dict, "board_dictに'board'キーがありません"
        assert 'sui_positions' in board_dict, "board_dictに'sui_positions'キーがありません"
        
        # board は9x9の配列
        board_data = board_dict['board']
        assert len(board_data) == 9, "盤面の行数が9ではありません"
        assert len(board_data[0]) == 9, "盤面の列数が9ではありません"
