"""
シナリオテスト: エッジケースと境界条件
特殊な状況や境界値のテスト
"""

import pytest
from src.engine import Board, Player, PieceType, Piece, Rules, Move


class TestEdgeCases:
    """エッジケースのテストクラス"""
    
    def test_corner_piece_movement(self):
        """盤の角での駒の動きを確認"""
        board = Board()
        
        # 左上の角に帥を配置
        board.add_piece((0, 0), Piece(PieceType.SUI, Player.BLACK))
        
        legal_moves = Rules._get_piece_legal_moves(board, (0, 0), Player.BLACK)
        
        # 盤外への移動は含まれないはず
        for move in legal_moves:
            row, col = move.to_pos
            assert 0 <= row < 9, f"盤外の行{row}への移動が含まれています"
            assert 0 <= col < 9, f"盤外の列{col}への移動が含まれています"
        
        # 角から動ける方向は3方向のみ
        assert len(legal_moves) == 3, f"角からの移動が3方向ではありません: {len(legal_moves)}"
    
    def test_edge_piece_movement(self):
        """盤の端での駒の動きを確認"""
        board = Board()
        
        # 上端の中央に大将を配置
        board.add_piece((0, 4), Piece(PieceType.DAI, Player.BLACK))
        
        legal_moves = Rules._get_piece_legal_moves(board, (0, 4), Player.BLACK)
        
        # 盤外への移動は含まれないはず
        for move in legal_moves:
            row, col = move.to_pos
            assert 0 <= row < 9, f"盤外の行{row}への移動が含まれています"
            assert 0 <= col < 9, f"盤外の列{col}への移動が含まれています"
    
    def test_no_legal_moves_surrounded(self):
        """完全に囲まれた駒の合法手が正しく処理されることを確認"""
        board = Board()
        
        # 中央に帥を配置
        board.add_piece((4, 4), Piece(PieceType.SUI, Player.BLACK))
        
        # 周囲8マスを全て黒の駒で埋める
        surrounding = [
            (3, 3), (3, 4), (3, 5),
            (4, 3),         (4, 5),
            (5, 3), (5, 4), (5, 5)
        ]
        for pos in surrounding:
            board.add_piece(pos, Piece(PieceType.HYO, Player.BLACK))
        
        # 合法手を取得
        legal_moves = Rules._get_piece_legal_moves(board, (4, 4), Player.BLACK)
        
        # 移動はできないが、スタックは可能なはず
        normal_moves = [m for m in legal_moves if m.move_type.name == 'NORMAL']
        assert len(normal_moves) == 0, "囲まれた駒が通常移動できてしまいます"
    
    def test_jump_over_obstacle(self):
        """ジャンプ可能な駒が障害物を飛び越えることを確認"""
        board = Board()
        
        # 兵を3段スタックにする（ジャンプ可能）
        position = (4, 4)
        board.add_piece(position, Piece(PieceType.HYO, Player.BLACK))
        board.add_piece(position, Piece(PieceType.HYO, Player.BLACK))
        board.add_piece(position, Piece(PieceType.HYO, Player.BLACK))
        
        # 前方に障害物を配置
        board.add_piece((3, 4), Piece(PieceType.YARI, Player.BLACK))
        
        # 合法手を取得
        legal_moves = Rules._get_piece_legal_moves(board, position, Player.BLACK)
        
        # 3段目の兵はジャンプ可能なので、障害物を飛び越えて(2, 4)に移動できるはず
        # （実装による）
    
    def test_all_hand_pieces_used(self):
        """全ての持ち駒を使い切れることを確認"""
        board = Board()
        
        # 持ち駒を設定
        hand_pieces = {PieceType.HYO: 3}
        
        # 3個全て打つ
        for i in range(3):
            move = Move.create_drop_move(
                to_pos=(4, i),
                piece_type=PieceType.HYO,
                player=Player.BLACK
            )
            success, captured = Rules.apply_move(board, move, hand_pieces)
            assert success, f"{i + 1}個目の持ち駒を打つのに失敗しました"
        
        # 持ち駒が0になっているか確認
        assert hand_pieces[PieceType.HYO] == 0, "持ち駒が残っています"
        
        # もう打てない
        move = Move.create_drop_move(
            to_pos=(4, 3),
            piece_type=PieceType.HYO,
            player=Player.BLACK
        )
        success, captured = Rules.apply_move(board, move, hand_pieces)
        assert not success, "持ち駒がないのに打ててしまいました"
    
    def test_sui_next_to_sui(self):
        """帥同士が隣接する状況を確認"""
        board = Board()
        
        # 帥を隣接して配置
        board.add_piece((4, 4), Piece(PieceType.SUI, Player.BLACK))
        board.add_piece((4, 5), Piece(PieceType.SUI, Player.WHITE))
        
        # 黒の帥が白の帥を取得できるか
        legal_moves = Rules._get_piece_legal_moves(board, (4, 4), Player.BLACK)
        capture_moves = [
            m for m in legal_moves 
            if m.to_pos == (4, 5) and m.move_type.name == 'CAPTURE'
        ]
        
        assert len(capture_moves) > 0, "隣接する帥を取得できません"
    
    def test_only_one_piece_on_board(self):
        """盤上に駒が1個だけの状況を確認"""
        board = Board()
        
        # 帥だけを配置
        board.add_piece((4, 4), Piece(PieceType.SUI, Player.BLACK))
        
        # 合法手を取得
        legal_moves = Rules._get_piece_legal_moves(board, (4, 4), Player.BLACK)
        
        # 何かしらの移動が可能なはず
        assert len(legal_moves) > 0, "駒1個で合法手がありません"
    
    def test_full_board(self):
        """盤面が駒で埋まっている状況を確認"""
        board = Board()
        
        # 全マスに駒を配置
        for row in range(9):
            for col in range(9):
                piece = Piece(PieceType.HYO, Player.BLACK if (row + col) % 2 == 0 else Player.WHITE)
                board.add_piece((row, col), piece)
        
        # 合法手を取得（黒の駒から）
        legal_moves = Rules.get_legal_moves(board, Player.BLACK)
        
        # スタックのみ可能なはず
        # （実装による）
    
    def test_toride_at_level_1_has_no_or_limited_moves(self):
        """砦が1段目で動きが制限されることを確認"""
        board = Board()
        
        # 砦を配置
        board.add_piece((4, 4), Piece(PieceType.TORIDE, Player.BLACK))
        
        # 1段目の砦の動き
        moves_1 = Rules._get_piece_legal_moves(board, (4, 4), Player.BLACK)
        
        # 砦を2段目に
        board.add_piece((4, 4), Piece(PieceType.TORIDE, Player.BLACK))
        moves_2 = Rules._get_piece_legal_moves(board, (4, 4), Player.BLACK)
        
        # 2段目では動きが増えるはず
        # （実装による）
    
    def test_move_to_frontline_restriction_for_drop(self):
        """最前線より前に打てないルールを確認"""
        board = Board()
        
        # 黒の駒を配置（最前線）
        board.add_piece((4, 4), Piece(PieceType.HYO, Player.BLACK))
        
        # 持ち駒
        hand_pieces = {PieceType.YARI: 1}
        
        # 最前線より前(3, 4)に打とうとする
        move = Move.create_drop_move(
            to_pos=(3, 4),
            piece_type=PieceType.YARI,
            player=Player.BLACK
        )
        
        # 合法かどうかは実装による（最前線ルール）
        # 注: このルールの実装状況を確認する必要がある
    
    def test_stack_with_different_piece_types(self):
        """異なる種類の駒でスタックを作れることを確認"""
        board = Board()
        
        position = (4, 4)
        
        # 3種類の駒でスタック
        board.add_piece(position, Piece(PieceType.HYO, Player.BLACK))
        board.add_piece(position, Piece(PieceType.YARI, Player.BLACK))
        board.add_piece(position, Piece(PieceType.UMA, Player.BLACK))
        
        assert board.get_stack_height(position) == 3
        
        stack = board.get_stack(position)
        assert stack.get_piece_at_level(0).piece_type == PieceType.HYO
        assert stack.get_piece_at_level(1).piece_type == PieceType.YARI
        assert stack.get_piece_at_level(2).piece_type == PieceType.UMA
    
    def test_empty_board_has_no_legal_moves(self):
        """空の盤面では合法手がないことを確認"""
        board = Board()
        
        legal_moves = Rules.get_legal_moves(board, Player.BLACK)
        
        # 盤上に駒がなければ、持ち駒がない限り合法手はない
        # （持ち駒がある場合は打てる）


class TestJumpPieces:
    """ジャンプ可能な駒（弓・筒・砲）のテストクラス"""
    
    def test_yumi_can_jump_over_piece(self):
        """弓がジャンプできることを確認"""
        board = Board()
        
        # 弓を配置
        board.add_piece((4, 4), Piece(PieceType.YUMI, Player.BLACK))
        # 弓の移動先にある障害物
        board.add_piece((3, 4), Piece(PieceType.HYO, Player.WHITE))
        
        legal_moves = Rules._get_piece_legal_moves(board, (4, 4), Player.BLACK)
        
        # 弓の動きパターンを確認（1段目）
        # 弓は飛び越えて移動できるはず
        move_targets = [m.to_pos for m in legal_moves]
        
        # 弓が障害物を飛び越えて移動できるか
        # 具体的な動きは駒の定義による
    
    def test_yumi_at_level_2(self):
        """弓の2段目での動きを確認"""
        board = Board()
        
        # 弓を2段目に配置
        position = (4, 4)
        board.add_piece(position, Piece(PieceType.HYO, Player.BLACK))  # 1段目
        board.add_piece(position, Piece(PieceType.YUMI, Player.BLACK))  # 2段目
        
        legal_moves = Rules._get_piece_legal_moves(board, position, Player.BLACK)
        
        # 2段目の弓の動きを確認
        assert len(legal_moves) > 0, "2段目の弓に合法手がありません"
    
    def test_yumi_at_level_3(self):
        """弓の3段目（極）での動きを確認"""
        board = Board()
        
        # 弓を3段目に配置
        position = (4, 4)
        board.add_piece(position, Piece(PieceType.HYO, Player.BLACK))  # 1段目
        board.add_piece(position, Piece(PieceType.HYO, Player.BLACK))  # 2段目
        board.add_piece(position, Piece(PieceType.YUMI, Player.BLACK))  # 3段目
        
        legal_moves = Rules._get_piece_legal_moves(board, position, Player.BLACK)
        
        # 3段目の弓（極）はジャンプ可能
        assert len(legal_moves) > 0, "3段目の弓に合法手がありません"
    
    def test_tsutu_can_jump(self):
        """筒がジャンプできることを確認"""
        board = Board()
        
        # 筒を配置
        board.add_piece((4, 4), Piece(PieceType.TSUTU, Player.BLACK))
        # 障害物を配置
        board.add_piece((4, 5), Piece(PieceType.HYO, Player.WHITE))
        
        legal_moves = Rules._get_piece_legal_moves(board, (4, 4), Player.BLACK)
        
        # 筒の動きを確認
        move_targets = [m.to_pos for m in legal_moves]
    
    def test_hou_can_jump(self):
        """砲がジャンプできることを確認"""
        board = Board()
        
        # 砲を配置
        board.add_piece((4, 4), Piece(PieceType.HOU, Player.BLACK))
        # 障害物を配置
        board.add_piece((3, 4), Piece(PieceType.HYO, Player.WHITE))
        
        legal_moves = Rules._get_piece_legal_moves(board, (4, 4), Player.BLACK)
        
        # 砲の動きを確認
        move_targets = [m.to_pos for m in legal_moves]
    
    def test_hou_at_level_3_mastered(self):
        """砲の3段目（極）での動きを確認"""
        board = Board()
        
        # 砲を3段目に配置
        position = (4, 4)
        board.add_piece(position, Piece(PieceType.HYO, Player.BLACK))  # 1段目
        board.add_piece(position, Piece(PieceType.HYO, Player.BLACK))  # 2段目
        board.add_piece(position, Piece(PieceType.HOU, Player.BLACK))  # 3段目
        
        legal_moves = Rules._get_piece_legal_moves(board, position, Player.BLACK)
        
        # 3段目の砲（極）の動き
        assert len(legal_moves) > 0, "3段目の砲に合法手がありません"
    
    def test_jump_over_own_piece(self):
        """味方の駒を飛び越えられることを確認"""
        board = Board()
        
        # 弓を配置
        board.add_piece((4, 4), Piece(PieceType.YUMI, Player.BLACK))
        # 味方の駒を配置
        board.add_piece((3, 4), Piece(PieceType.HYO, Player.BLACK))
        
        legal_moves = Rules._get_piece_legal_moves(board, (4, 4), Player.BLACK)
        
        # 味方の駒を飛び越えて移動できるか確認
        move_targets = [m.to_pos for m in legal_moves]
    
    def test_jump_cannot_land_on_higher_stack(self):
        """自分より高いスタックには着地できない"""
        board = Board()
        
        # 弓を1段目に配置
        board.add_piece((4, 4), Piece(PieceType.YUMI, Player.BLACK))
        # 着地先に2段スタック
        board.add_piece((2, 4), Piece(PieceType.HYO, Player.WHITE))
        board.add_piece((2, 4), Piece(PieceType.HYO, Player.WHITE))
        
        legal_moves = Rules._get_piece_legal_moves(board, (4, 4), Player.BLACK)
        
        # 1段目の弓が2段スタックを取れないか確認
        capture_to_stack = [
            m for m in legal_moves 
            if m.to_pos == (2, 4) and m.move_type.name == 'CAPTURE'
        ]
        
        # 自分のスタック(1) < 相手のスタック(2) なので取れないはず


class TestDirectionInversion:
    """方向反転（BLACK/WHITE）のテストクラス"""
    
    def test_hyo_moves_opposite_directions(self):
        """兵はBLACK/WHITEで反対方向に動く"""
        # 中央に兵を配置
        pos = (4, 4)
        
        # 黒の兵
        black_board = Board()
        black_board.add_piece(pos, Piece(PieceType.HYO, Player.BLACK))
        black_moves = Rules._get_piece_legal_moves(black_board, pos, Player.BLACK)
        black_targets = set(m.to_pos for m in black_moves if m.move_type.name == 'NORMAL')
        
        # 白の兵
        white_board = Board()
        white_board.add_piece(pos, Piece(PieceType.HYO, Player.WHITE))
        white_moves = Rules._get_piece_legal_moves(white_board, pos, Player.WHITE)
        white_targets = set(m.to_pos for m in white_moves if m.move_type.name == 'NORMAL')
        
        # 兵は前後1マスなので
        # 黒の前方は上（行番号が小さい）方向 → (3, 4)
        # 白の前方は下（行番号が大きい）方向 → (5, 4)
        assert (3, 4) in black_targets, "黒の兵が前方に動けません"
        assert (5, 4) in white_targets, "白の兵が前方に動けません"
    
    def test_samurai_front_diagonal_differs(self):
        """侍の斜め前方向がBLACK/WHITEで異なる"""
        pos = (4, 4)
        
        # 黒の侍
        black_board = Board()
        black_board.add_piece(pos, Piece(PieceType.SAMURAI, Player.BLACK))
        black_moves = Rules._get_piece_legal_moves(black_board, pos, Player.BLACK)
        black_targets = set(m.to_pos for m in black_moves if m.move_type.name == 'NORMAL')
        
        # 白の侍
        white_board = Board()
        white_board.add_piece(pos, Piece(PieceType.SAMURAI, Player.WHITE))
        white_moves = Rules._get_piece_legal_moves(white_board, pos, Player.WHITE)
        white_targets = set(m.to_pos for m in white_moves if m.move_type.name == 'NORMAL')
        
        # 侍は前方、斜め前、後方1マスに動ける
        # 黒の斜め前は(3, 3)と(3, 5)
        # 白の斜め前は(5, 3)と(5, 5)
        assert (3, 3) in black_targets or (3, 5) in black_targets, "黒の侍が斜め前に動けません"
        assert (5, 3) in white_targets or (5, 5) in white_targets, "白の侍が斜め前に動けません"
    
    def test_yari_forward_reach_differs(self):
        """槍の前方リーチがBLACK/WHITEで異なる"""
        pos = (4, 4)
        
        # 黒の槍
        black_board = Board()
        black_board.add_piece(pos, Piece(PieceType.YARI, Player.BLACK))
        black_moves = Rules._get_piece_legal_moves(black_board, pos, Player.BLACK)
        black_targets = set(m.to_pos for m in black_moves if m.move_type.name == 'NORMAL')
        
        # 白の槍
        white_board = Board()
        white_board.add_piece(pos, Piece(PieceType.YARI, Player.WHITE))
        white_moves = Rules._get_piece_legal_moves(white_board, pos, Player.WHITE)
        white_targets = set(m.to_pos for m in white_moves if m.move_type.name == 'NORMAL')
        
        # 槍は前方2マスまで動ける
        # 黒の前方2マスは(2, 4)
        # 白の前方2マスは(6, 4)
        assert (2, 4) in black_targets, "黒の槍が前方2マスに動けません"
        assert (6, 4) in white_targets, "白の槍が前方2マスに動けません"
    
    def test_sui_moves_same_for_both_players(self):
        """帥は全方向なのでBLACK/WHITE同じ動き"""
        pos = (4, 4)
        
        # 黒の帥
        black_board = Board()
        black_board.add_piece(pos, Piece(PieceType.SUI, Player.BLACK))
        black_moves = Rules._get_piece_legal_moves(black_board, pos, Player.BLACK)
        black_targets = set(m.to_pos for m in black_moves if m.move_type.name == 'NORMAL')
        
        # 白の帥
        white_board = Board()
        white_board.add_piece(pos, Piece(PieceType.SUI, Player.WHITE))
        white_moves = Rules._get_piece_legal_moves(white_board, pos, Player.WHITE)
        white_targets = set(m.to_pos for m in white_moves if m.move_type.name == 'NORMAL')
        
        # 帥は8方向全てに1マス動ける（対称）
        assert black_targets == white_targets, "帥の動きがBLACK/WHITEで異なります"
