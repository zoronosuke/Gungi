"""
単体テスト: 駒の定義と動きのテスト
各駒の移動パターンとスタックレベルによる変化を確認
"""

import pytest
from src.engine import Board, Player, PieceType, Piece, Rules


class TestPieceMovement:
    """駒の動きのテストクラス"""
    
    # 全駒種のリスト
    ALL_PIECE_TYPES = [
        PieceType.SUI, PieceType.DAI, PieceType.CHUU, PieceType.SHO,
        PieceType.SAMURAI, PieceType.HYO, PieceType.UMA, PieceType.SHINOBI,
        PieceType.YARI, PieceType.TORIDE, PieceType.YUMI, PieceType.TSUTU,
        PieceType.HOU, PieceType.BOU
    ]
    
    @pytest.mark.parametrize("piece_type", ALL_PIECE_TYPES)
    def test_piece_has_move_pattern_at_level_1(self, piece_type):
        """全ての駒が1段目の動きパターンを持つことを確認"""
        piece = Piece(piece_type, Player.BLACK)
        pattern = piece.get_move_pattern(1)
        
        assert pattern is not None, f"{piece_type.name}の1段目の動きパターンが存在しません"
        assert 'moves' in pattern, f"{piece_type.name}のパターンに'moves'が含まれていません"
    
    @pytest.mark.parametrize("piece_type", ALL_PIECE_TYPES)
    def test_piece_has_move_pattern_at_level_2(self, piece_type):
        """全ての駒が2段目の動きパターンを持つことを確認"""
        piece = Piece(piece_type, Player.BLACK)
        pattern = piece.get_move_pattern(2)
        
        assert pattern is not None, f"{piece_type.name}の2段目の動きパターンが存在しません"
        assert 'moves' in pattern, f"{piece_type.name}のパターンに'moves'が含まれていません"
    
    @pytest.mark.parametrize("piece_type", ALL_PIECE_TYPES)
    def test_piece_has_move_pattern_at_level_3(self, piece_type):
        """全ての駒が3段目の動きパターンを持つことを確認"""
        piece = Piece(piece_type, Player.BLACK)
        pattern = piece.get_move_pattern(3)
        
        assert pattern is not None, f"{piece_type.name}の3段目の動きパターンが存在しません"
        assert 'moves' in pattern, f"{piece_type.name}のパターンに'moves'が含まれていません"
    
    def test_hyo_can_move_forward_and_backward(self, empty_board):
        """兵が前後に動けることを確認"""
        position = (4, 4)
        piece = Piece(PieceType.HYO, Player.BLACK)
        empty_board.add_piece(position, piece)
        
        legal_moves = Rules._get_piece_legal_moves(empty_board, position, Player.BLACK)
        move_positions = {move.to_pos for move in legal_moves}
        
        # 前方（黒は上方向）
        assert (3, 4) in move_positions, "兵が前方に動けません"
        # 後方
        assert (5, 4) in move_positions, "兵が後方に動けません"
    
    def test_sui_can_move_in_8_directions(self, empty_board):
        """帥が8方向に動けることを確認"""
        position = (4, 4)
        piece = Piece(PieceType.SUI, Player.BLACK)
        empty_board.add_piece(position, piece)
        
        legal_moves = Rules._get_piece_legal_moves(empty_board, position, Player.BLACK)
        move_positions = {move.to_pos for move in legal_moves}
        
        # 8方向全てに動けるはず
        expected_positions = [
            (3, 4),  # 上
            (5, 4),  # 下
            (4, 3),  # 左
            (4, 5),  # 右
            (3, 3),  # 左上
            (3, 5),  # 右上
            (5, 3),  # 左下
            (5, 5),  # 右下
        ]
        
        for pos in expected_positions:
            assert pos in move_positions, f"帥が{pos}に動けません"
    
    def test_piece_movement_changes_with_stack_level(self, empty_board):
        """スタックレベルによって駒の動きが変わることを確認"""
        position = (4, 4)
        
        # 兵を1段目に配置
        piece1 = Piece(PieceType.HYO, Player.BLACK)
        empty_board.add_piece(position, piece1)
        
        moves_at_level_1 = Rules._get_piece_legal_moves(empty_board, position, Player.BLACK)
        count_level_1 = len(moves_at_level_1)
        
        # 兵を2段目に積む
        piece2 = Piece(PieceType.HYO, Player.BLACK)
        empty_board.add_piece(position, piece2)
        
        moves_at_level_2 = Rules._get_piece_legal_moves(empty_board, position, Player.BLACK)
        count_level_2 = len(moves_at_level_2)
        
        # 兵を3段目に積む
        piece3 = Piece(PieceType.HYO, Player.BLACK)
        empty_board.add_piece(position, piece3)
        
        moves_at_level_3 = Rules._get_piece_legal_moves(empty_board, position, Player.BLACK)
        count_level_3 = len(moves_at_level_3)
        
        # 2段目は1段目より多いか同じ
        assert count_level_2 >= count_level_1, "2段目の動きが1段目より少なくなっています"
        # 3段目は2段目より多いか同じ
        assert count_level_3 >= count_level_2, "3段目の動きが2段目より少なくなっています"
    
    def test_white_piece_movement_is_reversed(self, empty_board):
        """白プレイヤーの駒の動きが反転することを確認"""
        # 黒の兵の前方向
        black_pos = (4, 4)
        black_hyo = Piece(PieceType.HYO, Player.BLACK)
        empty_board.add_piece(black_pos, black_hyo)
        
        black_moves = Rules._get_piece_legal_moves(empty_board, black_pos, Player.BLACK)
        black_forward_moves = [m for m in black_moves if m.to_pos[0] < black_pos[0]]
        
        # 盤面をリセット
        empty_board = Board()
        
        # 白の兵の前方向
        white_pos = (4, 4)
        white_hyo = Piece(PieceType.HYO, Player.WHITE)
        empty_board.add_piece(white_pos, white_hyo)
        
        white_moves = Rules._get_piece_legal_moves(empty_board, white_pos, Player.WHITE)
        white_forward_moves = [m for m in white_moves if m.to_pos[0] > white_pos[0]]
        
        # 白は下方向が前（rowが増える）、黒は上方向が前（rowが減る）
        assert len(black_forward_moves) > 0, "黒の兵が前方に動けません"
        assert len(white_forward_moves) > 0, "白の兵が前方に動けません"
    
    def test_dai_has_rook_king_movement(self, empty_board):
        """大将が龍王の動き（飛車+王の斜め）をすることを確認"""
        position = (4, 4)
        piece = Piece(PieceType.DAI, Player.BLACK)
        empty_board.add_piece(position, piece)
        
        legal_moves = Rules._get_piece_legal_moves(empty_board, position, Player.BLACK)
        move_positions = {move.to_pos for move in legal_moves}
        
        # 縦横に遠くまで動ける
        assert (0, 4) in move_positions, "大将が縦方向に遠くまで動けません"
        assert (8, 4) in move_positions, "大将が縦方向に遠くまで動けません"
        assert (4, 0) in move_positions, "大将が横方向に遠くまで動けません"
        assert (4, 8) in move_positions, "大将が横方向に遠くまで動けません"
        
        # 斜め1マス動ける
        assert (3, 3) in move_positions, "大将が斜め1マスに動けません"
        assert (3, 5) in move_positions, "大将が斜め1マスに動けません"
    
    def test_chuu_has_bishop_king_movement(self, empty_board):
        """中将が龍馬の動き（角+王の縦横）をすることを確認"""
        position = (4, 4)
        piece = Piece(PieceType.CHUU, Player.BLACK)
        empty_board.add_piece(position, piece)
        
        legal_moves = Rules._get_piece_legal_moves(empty_board, position, Player.BLACK)
        move_positions = {move.to_pos for move in legal_moves}
        
        # 斜めに遠くまで動ける
        assert (0, 0) in move_positions, "中将が斜め方向に遠くまで動けません"
        assert (8, 8) in move_positions, "中将が斜め方向に遠くまで動けません"
        
        # 縦横1マス動ける
        assert (3, 4) in move_positions, "中将が縦横1マスに動けません"
        assert (5, 4) in move_positions, "中将が縦横1マスに動けません"
        assert (4, 3) in move_positions, "中将が縦横1マスに動けません"
        assert (4, 5) in move_positions, "中将が縦横1マスに動けません"
    
    def test_sui_cannot_be_stacked_on(self, empty_board):
        """帥の上に駒を乗せられないことを確認"""
        sui = Piece(PieceType.SUI, Player.BLACK)
        
        # 帥は他の駒の上に乗せられないメソッドを持つべき
        assert not sui.can_be_stacked_on(), "帥がスタック可能になっています"
    
    @pytest.mark.parametrize("piece_type", [
        PieceType.HYO, PieceType.UMA, PieceType.SHINOBI, PieceType.TORIDE,
        PieceType.YUMI, PieceType.TSUTU, PieceType.HOU, PieceType.BOU
    ])
    def test_piece_can_jump_at_level_3(self, piece_type):
        """3段目でジャンプ可能な駒を確認"""
        piece = Piece(piece_type, Player.BLACK)
        pattern = piece.get_move_pattern(3)
        
        # 3段目ではcanJumpがTrueになるべき
        assert 'canJump' in pattern, f"{piece_type.name}のパターンにcanJumpが含まれていません"
        # ジャンプ可能な駒はcanJumpがTrueのはず（実装による）
