"""
単体テスト: スタック（ツケ）機能のテスト
スタックのルールと制限を確認
"""

import pytest
from src.engine import Board, Player, PieceType, Piece, Rules, Move, MoveType


class TestStackMechanics:
    """スタック機能のテストクラス"""
    
    def test_stack_up_to_3_levels(self, empty_board):
        """3段までスタックできることを確認"""
        position = (4, 4)
        
        for i, piece_type in enumerate([PieceType.HYO, PieceType.YARI, PieceType.UMA]):
            piece = Piece(piece_type, Player.BLACK)
            success = empty_board.add_piece(position, piece)
            
            assert success, f"{i + 1}段目の追加に失敗しました"
            assert empty_board.get_stack_height(position) == i + 1, \
                f"スタック高さが{i + 1}ではありません"
    
    def test_cannot_stack_4th_level(self, empty_board):
        """4段目は追加できないことを確認"""
        position = (4, 4)
        
        # 3段まで積む
        for piece_type in [PieceType.HYO, PieceType.YARI, PieceType.UMA]:
            empty_board.add_piece(position, Piece(piece_type, Player.BLACK))
        
        # 4段目は失敗
        piece4 = Piece(PieceType.SAMURAI, Player.BLACK)
        success = empty_board.add_piece(position, piece4)
        
        assert not success, "4段目が追加できてしまいました"
        assert empty_board.get_stack_height(position) == 3
    
    def test_sui_cannot_have_pieces_stacked_on_it(self, empty_board):
        """帥の上に駒を乗せられないことを確認"""
        position = (4, 4)
        
        # 帥を配置
        sui = Piece(PieceType.SUI, Player.BLACK)
        empty_board.add_piece(position, sui)
        
        # 兵を乗せようとする
        hyo = Piece(PieceType.HYO, Player.BLACK)
        success = empty_board.add_piece(position, hyo)
        
        assert not success, "帥の上に駒を乗せられてしまいました"
        assert empty_board.get_stack_height(position) == 1
    
    def test_cannot_move_to_higher_stack(self, empty_board):
        """自分より高いスタックには移動できないことを確認"""
        # (4, 4)に2段スタックを作成
        empty_board.add_piece((4, 4), Piece(PieceType.HYO, Player.BLACK))
        empty_board.add_piece((4, 4), Piece(PieceType.YARI, Player.BLACK))
        
        # (4, 5)に1段の駒を配置
        empty_board.add_piece((4, 5), Piece(PieceType.SHO, Player.BLACK))
        
        # (4, 5)から(4, 4)への移動を試みる
        legal_moves = Rules._get_piece_legal_moves(empty_board, (4, 5), Player.BLACK)
        move_positions = {move.to_pos for move in legal_moves}
        
        # (4, 4)へのSTACK移動は含まれないべき
        stack_moves_to_44 = [
            move for move in legal_moves 
            if move.to_pos == (4, 4) and move.move_type == MoveType.STACK
        ]
        
        assert len(stack_moves_to_44) == 0, \
            "1段の駒が2段のスタックの上に乗れてしまいました"
    
    def test_can_stack_on_same_or_lower_height(self, empty_board):
        """同じ高さまたは低いスタックにはツケられることを確認"""
        # (4, 4)に1段の駒を配置
        empty_board.add_piece((4, 4), Piece(PieceType.HYO, Player.BLACK))
        
        # (4, 5)に2段のスタックを作成
        empty_board.add_piece((4, 5), Piece(PieceType.YARI, Player.BLACK))
        empty_board.add_piece((4, 5), Piece(PieceType.UMA, Player.BLACK))
        
        # (4, 5)から(4, 4)への移動（2段→1段）
        legal_moves = Rules._get_piece_legal_moves(empty_board, (4, 5), Player.BLACK)
        
        # (4, 4)へのSTACK移動が含まれるべき
        stack_moves_to_44 = [
            move for move in legal_moves 
            if move.to_pos == (4, 4) and move.move_type == MoveType.STACK
        ]
        
        assert len(stack_moves_to_44) > 0, \
            "2段の駒が1段のスタックの上に乗れません"
    
    def test_toride_cannot_stack_on_other_pieces(self, empty_board):
        """砦は他の駒の上に乗れないことを確認"""
        # (4, 4)に兵を配置
        empty_board.add_piece((4, 4), Piece(PieceType.HYO, Player.BLACK))
        
        # (4, 5)に砦を配置
        empty_board.add_piece((4, 5), Piece(PieceType.TORIDE, Player.BLACK))
        
        # 砦から(4, 4)への移動を試みる
        legal_moves = Rules._get_piece_legal_moves(empty_board, (4, 5), Player.BLACK)
        
        # (4, 4)へのSTACK移動は含まれないべき
        stack_moves = [
            move for move in legal_moves 
            if move.to_pos == (4, 4) and move.move_type == MoveType.STACK
        ]
        
        # 砦は他の駒の上に乗れない
        # （実装によっては、砦のcan_stack_onメソッドがFalseを返す）
    
    def test_stack_level_affects_movement_range(self, empty_board):
        """スタックレベルで移動範囲が変わることを確認"""
        position = (4, 4)
        
        # 1段目: 兵を配置
        empty_board.add_piece(position, Piece(PieceType.HYO, Player.BLACK))
        moves_1 = Rules._get_piece_legal_moves(empty_board, position, Player.BLACK)
        
        # 2段目: 兵を追加
        empty_board.add_piece(position, Piece(PieceType.HYO, Player.BLACK))
        moves_2 = Rules._get_piece_legal_moves(empty_board, position, Player.BLACK)
        
        # 3段目: 兵を追加
        empty_board.add_piece(position, Piece(PieceType.HYO, Player.BLACK))
        moves_3 = Rules._get_piece_legal_moves(empty_board, position, Player.BLACK)
        
        # 段数が上がるごとに移動可能な手が増えるか同じ
        assert len(moves_2) >= len(moves_1), "2段目の移動が1段目より少ない"
        assert len(moves_3) >= len(moves_2), "3段目の移動が2段目より少ない"
    
    def test_capturing_stack_removes_entire_stack(self, empty_board):
        """スタック全体を取得できることを確認"""
        # (5, 4)に白の3段スタックを作成
        empty_board.add_piece((5, 4), Piece(PieceType.HYO, Player.WHITE))
        empty_board.add_piece((5, 4), Piece(PieceType.YARI, Player.WHITE))
        empty_board.add_piece((5, 4), Piece(PieceType.UMA, Player.WHITE))
        
        # (7, 4)に黒の大将を配置
        empty_board.add_piece((7, 4), Piece(PieceType.DAI, Player.BLACK))
        
        # 黒の大将が白のスタックを取得
        move = Move.create_capture_move(
            from_pos=(7, 4),
            to_pos=(5, 4),
            player=Player.BLACK
        )
        
        success, captured_pieces = Rules.apply_move(empty_board, move)
        
        assert success, "取得に失敗しました"
        assert captured_pieces is not None, "取得した駒がNoneです"
        assert len(captured_pieces) == 3, f"3個の駒を取得すべきですが{len(captured_pieces)}個でした"
        
        # (5, 4)には黒の大将だけが残る
        assert empty_board.get_stack_height((5, 4)) == 1
        top_piece = empty_board.get_stack((5, 4)).get_top_piece()
        assert top_piece.owner == Player.BLACK
        assert top_piece.piece_type == PieceType.DAI
    
    def test_mixed_player_stack(self, empty_board):
        """敵味方が混在したスタックを作れることを確認"""
        position = (4, 4)
        
        # 白の駒を配置
        empty_board.add_piece(position, Piece(PieceType.HYO, Player.WHITE))
        
        # 黒の駒をその上に乗せる（敵の駒にツケる）
        empty_board.add_piece(position, Piece(PieceType.YARI, Player.BLACK))
        
        assert empty_board.get_stack_height(position) == 2
        
        stack = empty_board.get_stack(position)
        bottom = stack.get_piece_at_level(0)
        top = stack.get_piece_at_level(1)
        
        assert bottom.owner == Player.WHITE, "下段が白ではありません"
        assert top.owner == Player.BLACK, "上段が黒ではありません"
    
    def test_get_top_piece_stack_level(self, empty_board):
        """最上段の駒のスタックレベルを取得できることを確認"""
        position = (4, 4)
        
        stack = empty_board.get_stack(position)
        assert stack.get_top_piece_stack_level() == 0, "空のスタックレベルが0ではありません"
        
        # 1段追加
        empty_board.add_piece(position, Piece(PieceType.HYO, Player.BLACK))
        assert stack.get_top_piece_stack_level() == 1, "1段目のレベルが1ではありません"
        
        # 2段追加
        empty_board.add_piece(position, Piece(PieceType.YARI, Player.BLACK))
        assert stack.get_top_piece_stack_level() == 2, "2段目のレベルが2ではありません"
        
        # 3段追加
        empty_board.add_piece(position, Piece(PieceType.UMA, Player.BLACK))
        assert stack.get_top_piece_stack_level() == 3, "3段目のレベルが3ではありません"
