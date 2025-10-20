"""
単体テスト: ルール判定のテスト
合法手生成、手の適用、特殊ルールを確認
"""

import pytest
from src.engine import Board, Player, PieceType, Piece, Rules, Move, MoveType


class TestRules:
    """ルール判定のテストクラス"""
    
    def test_get_legal_moves_returns_list(self, initial_board):
        """合法手生成がリストを返すことを確認"""
        legal_moves = Rules.get_legal_moves(initial_board, Player.BLACK)
        
        assert isinstance(legal_moves, list), "合法手がリストではありません"
        assert len(legal_moves) > 0, "初期配置で合法手が0です"
    
    def test_cannot_move_to_own_piece(self, empty_board):
        """自分の駒を取得できないことを確認"""
        # 黒の駒を2つ配置
        empty_board.add_piece((4, 4), Piece(PieceType.SHO, Player.BLACK))
        empty_board.add_piece((4, 5), Piece(PieceType.HYO, Player.BLACK))
        
        legal_moves = Rules._get_piece_legal_moves(empty_board, (4, 4), Player.BLACK)
        
        # (4, 5)へのCAPTURE移動は含まれないべき
        capture_moves = [
            move for move in legal_moves 
            if move.to_pos == (4, 5) and move.move_type == MoveType.CAPTURE
        ]
        
        assert len(capture_moves) == 0, "自分の駒を取得できてしまいます"
    
    def test_can_capture_enemy_piece(self, empty_board):
        """敵の駒を取得できることを確認"""
        # 黒の駒を配置
        empty_board.add_piece((4, 4), Piece(PieceType.DAI, Player.BLACK))
        # 白の駒を配置
        empty_board.add_piece((4, 5), Piece(PieceType.HYO, Player.WHITE))
        
        legal_moves = Rules._get_piece_legal_moves(empty_board, (4, 4), Player.BLACK)
        
        # (4, 5)へのCAPTURE移動が含まれるべき
        capture_moves = [
            move for move in legal_moves 
            if move.to_pos == (4, 5) and move.move_type == MoveType.CAPTURE
        ]
        
        assert len(capture_moves) > 0, "敵の駒を取得できません"
    
    def test_can_stack_on_friendly_piece(self, empty_board):
        """味方の駒にツケられることを確認"""
        # 黒の駒を2つ配置
        empty_board.add_piece((4, 4), Piece(PieceType.HYO, Player.BLACK))
        empty_board.add_piece((4, 5), Piece(PieceType.SHO, Player.BLACK))
        
        legal_moves = Rules._get_piece_legal_moves(empty_board, (4, 5), Player.BLACK)
        
        # (4, 4)へのSTACK移動が含まれるべき
        stack_moves = [
            move for move in legal_moves 
            if move.to_pos == (4, 4) and move.move_type == MoveType.STACK
        ]
        
        assert len(stack_moves) > 0, "味方の駒にツケられません"
    
    def test_apply_normal_move(self, empty_board):
        """通常の移動が正しく適用されることを確認"""
        # 駒を配置
        empty_board.add_piece((4, 4), Piece(PieceType.HYO, Player.BLACK))
        
        # 移動を作成
        move = Move.create_normal_move(
            from_pos=(4, 4),
            to_pos=(3, 4),
            player=Player.BLACK
        )
        
        # 移動を適用
        success = Rules.apply_move(empty_board, move)
        
        assert success, "移動の適用に失敗しました"
        assert empty_board.get_stack((4, 4)).is_empty(), "元の位置に駒が残っています"
        assert not empty_board.get_stack((3, 4)).is_empty(), "移動先に駒がありません"
    
    def test_apply_capture_move(self, empty_board):
        """取得の手が正しく適用されることを確認"""
        # 黒の駒を配置
        empty_board.add_piece((4, 4), Piece(PieceType.DAI, Player.BLACK))
        # 白の駒を配置
        empty_board.add_piece((4, 5), Piece(PieceType.HYO, Player.WHITE))
        
        # 取得の手を作成
        move = Move.create_capture_move(
            from_pos=(4, 4),
            to_pos=(4, 5),
            player=Player.BLACK
        )
        
        # 手を適用
        success, captured = Rules.apply_move(empty_board, move)
        
        assert success, "取得の適用に失敗しました"
        assert captured is not None, "取得した駒がNoneです"
        assert len(captured) > 0, "取得した駒が空です"
        assert empty_board.get_stack((4, 4)).is_empty(), "元の位置に駒が残っています"
        
        # 移動先には黒の駒だけ
        top = empty_board.get_stack((4, 5)).get_top_piece()
        assert top.owner == Player.BLACK, "移動先の駒が黒ではありません"
    
    def test_apply_stack_move(self, empty_board):
        """ツケの手が正しく適用されることを確認"""
        # 黒の駒を2つ配置
        empty_board.add_piece((4, 4), Piece(PieceType.HYO, Player.BLACK))
        empty_board.add_piece((4, 5), Piece(PieceType.YARI, Player.BLACK))
        
        # ツケの手を作成
        move = Move.create_stack_move(
            from_pos=(4, 5),
            to_pos=(4, 4),
            player=Player.BLACK
        )
        
        # 手を適用
        success = Rules.apply_move(empty_board, move)
        
        assert success, "ツケの適用に失敗しました"
        assert empty_board.get_stack((4, 5)).is_empty(), "元の位置に駒が残っています"
        assert empty_board.get_stack_height((4, 4)) == 2, "スタック高さが2ではありません"
    
    def test_drop_piece_to_empty_square(self, empty_board):
        """持ち駒を空マスに打てることを確認"""
        # 持ち駒を設定
        hand_pieces = {PieceType.HYO: 1}
        
        # 新の手を作成
        move = Move.create_drop_move(
            to_pos=(4, 4),
            piece_type=PieceType.HYO,
            player=Player.BLACK
        )
        
        # 手を適用
        success = Rules.apply_move(empty_board, move, hand_pieces)
        
        assert success, "持ち駒を打つのに失敗しました"
        assert not empty_board.get_stack((4, 4)).is_empty(), "駒が配置されていません"
        
        # 持ち駒が減っているか確認
        assert hand_pieces[PieceType.HYO] == 0, "持ち駒が減っていません"
    
    def test_cannot_drop_on_sui(self, empty_board):
        """帥の上に持ち駒を打てないことを確認"""
        # 帥を配置
        empty_board.add_piece((4, 4), Piece(PieceType.SUI, Player.BLACK))
        
        # 持ち駒を設定
        hand_pieces = {PieceType.HYO: 1}
        
        # 帥の上に打とうとする
        move = Move.create_drop_move(
            to_pos=(4, 4),
            piece_type=PieceType.HYO,
            player=Player.BLACK
        )
        
        # 手を適用
        success, captured = Rules.apply_move(empty_board, move, hand_pieces)
        
        assert not success, "帥の上に駒を打ててしまいました"
        assert hand_pieces[PieceType.HYO] == 1, "持ち駒が減っています"
    
    def test_cannot_drop_on_3_height_stack(self, empty_board):
        """高さ3のスタックには打てないことを確認"""
        # 3段のスタックを作成
        position = (4, 4)
        empty_board.add_piece(position, Piece(PieceType.HYO, Player.BLACK))
        empty_board.add_piece(position, Piece(PieceType.YARI, Player.BLACK))
        empty_board.add_piece(position, Piece(PieceType.UMA, Player.BLACK))
        
        # 持ち駒を設定
        hand_pieces = {PieceType.SAMURAI: 1}
        
        # 3段のスタックの上に打とうとする
        move = Move.create_drop_move(
            to_pos=position,
            piece_type=PieceType.SAMURAI,
            player=Player.BLACK
        )
        
        # 手を適用
        success, captured = Rules.apply_move(empty_board, move, hand_pieces)
        
        assert not success, "高さ3のスタックに駒を打ててしまいました"
        assert hand_pieces[PieceType.SAMURAI] == 1, "持ち駒が減っています"
    
    def test_is_game_over_when_sui_captured(self, empty_board):
        """帥が取られるとゲーム終了になることを確認"""
        # 両方の帥を配置（ゲームに必要）
        empty_board.add_piece((7, 4), Piece(PieceType.SUI, Player.BLACK))
        empty_board.add_piece((4, 5), Piece(PieceType.SUI, Player.WHITE))
        # 黒の大将を配置
        empty_board.add_piece((4, 4), Piece(PieceType.DAI, Player.BLACK))
        
        # 白の帥を取得
        move = Move.create_capture_move(
            from_pos=(4, 4),
            to_pos=(4, 5),
            player=Player.BLACK
        )
        
        Rules.apply_move(empty_board, move)
        
        # ゲーム終了判定
        is_over, winner = Rules.is_game_over(empty_board)
        
        assert is_over, "ゲームが終了していません"
        assert winner == Player.BLACK, "勝者が黒ではありません"
    
    def test_cannot_move_out_of_board(self, empty_board):
        """盤外には移動できないことを確認"""
        # 端に駒を配置
        empty_board.add_piece((0, 0), Piece(PieceType.DAI, Player.BLACK))
        
        legal_moves = Rules._get_piece_legal_moves(empty_board, (0, 0), Player.BLACK)
        
        # 盤外の位置(-1, 0)や(0, -1)への移動は含まれないべき
        for move in legal_moves:
            row, col = move.to_pos
            assert 0 <= row < 9, f"盤外の行{row}への移動が含まれています"
            assert 0 <= col < 9, f"盤外の列{col}への移動が含まれています"
    
    def test_piece_blocked_by_obstacle(self, empty_board):
        """障害物があると移動できないことを確認（ジャンプなし）"""
        # 黒の大将を配置
        empty_board.add_piece((4, 4), Piece(PieceType.DAI, Player.BLACK))
        # 障害物となる駒を配置
        empty_board.add_piece((4, 5), Piece(PieceType.HYO, Player.BLACK))
        
        legal_moves = Rules._get_piece_legal_moves(empty_board, (4, 4), Player.BLACK)
        
        # (4, 6)への移動は含まれないべき（障害物があるため）
        moves_to_46 = [move for move in legal_moves if move.to_pos == (4, 6)]
        
        # 大将は障害物を飛び越えられない（ジャンプなし）
        # ただし、(4, 5)にはSTACKできるはず
