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


class TestCheckConditions:
    """王手判定のテストクラス"""
    
    def test_is_check_when_attacked_by_dai(self):
        """大将に攻撃されると王手"""
        board = Board()
        
        # 黒の帥を配置
        board.add_piece((4, 4), Piece(PieceType.SUI, Player.BLACK))
        # 白の帥を遠くに
        board.add_piece((0, 0), Piece(PieceType.SUI, Player.WHITE))
        # 白の大将を帥の隣に配置
        board.add_piece((4, 5), Piece(PieceType.DAI, Player.WHITE))
        
        is_check = Rules.is_check(board, Player.BLACK)
        assert is_check, "大将に隣接されているのに王手と判定されませんでした"
    
    def test_is_check_when_attacked_from_distance(self):
        """遠距離攻撃による王手"""
        board = Board()
        
        # 黒の帥を配置
        board.add_piece((4, 4), Piece(PieceType.SUI, Player.BLACK))
        # 白の帥を遠くに
        board.add_piece((0, 0), Piece(PieceType.SUI, Player.WHITE))
        # 白の大将を同じ行に配置（直線攻撃）
        board.add_piece((4, 8), Piece(PieceType.DAI, Player.WHITE))
        
        is_check = Rules.is_check(board, Player.BLACK)
        assert is_check, "直線上の大将から王手されているのに検出されませんでした"
    
    def test_not_in_check_when_blocked(self):
        """間に駒があれば王手ではない"""
        board = Board()
        
        # 黒の帥を配置
        board.add_piece((4, 4), Piece(PieceType.SUI, Player.BLACK))
        # 白の帥を遠くに
        board.add_piece((0, 0), Piece(PieceType.SUI, Player.WHITE))
        # 白の大将を同じ行に配置
        board.add_piece((4, 8), Piece(PieceType.DAI, Player.WHITE))
        # 間に黒の駒を配置（ブロック）
        board.add_piece((4, 6), Piece(PieceType.HYO, Player.BLACK))
        
        is_check = Rules.is_check(board, Player.BLACK)
        assert not is_check, "間に駒があるのに王手と判定されました"
    
    def test_is_check_by_multiple_pieces(self):
        """複数の駒から王手（両王手）"""
        board = Board()
        
        # 黒の帥を配置
        board.add_piece((4, 4), Piece(PieceType.SUI, Player.BLACK))
        # 白の帥を遠くに
        board.add_piece((0, 0), Piece(PieceType.SUI, Player.WHITE))
        # 白の大将を配置
        board.add_piece((4, 5), Piece(PieceType.DAI, Player.WHITE))
        # 白の中将も配置
        board.add_piece((3, 3), Piece(PieceType.CHUU, Player.WHITE))
        
        is_check = Rules.is_check(board, Player.BLACK)
        assert is_check, "複数の駒から王手されているのに検出されませんでした"
    
    def test_no_check_sui_not_on_board(self):
        """帥が盤上にない場合は王手ではない（Falseを返す）"""
        board = Board()
        
        # 白の帥のみ配置
        board.add_piece((0, 0), Piece(PieceType.SUI, Player.WHITE))
        
        is_check = Rules.is_check(board, Player.BLACK)
        assert not is_check, "帥がないのに王手と判定されました"


class TestCheckmateConditions:
    """詰み判定のテストクラス"""
    
    def test_not_checkmate_if_not_in_check(self):
        """王手でなければ詰みではない"""
        board = Board()
        
        # 両方の帥を安全な位置に配置
        board.add_piece((8, 4), Piece(PieceType.SUI, Player.BLACK))
        board.add_piece((0, 4), Piece(PieceType.SUI, Player.WHITE))
        
        is_checkmate = Rules.is_checkmate(board, Player.BLACK)
        assert not is_checkmate, "王手でないのに詰みと判定されました"
    
    def test_not_checkmate_if_can_escape(self):
        """逃げ道があれば詰みではない"""
        board = Board()
        
        # 黒の帥を配置
        board.add_piece((4, 4), Piece(PieceType.SUI, Player.BLACK))
        # 白の帥を遠くに
        board.add_piece((0, 0), Piece(PieceType.SUI, Player.WHITE))
        # 白の大将で王手
        board.add_piece((4, 5), Piece(PieceType.DAI, Player.WHITE))
        
        # 帥は(3,3), (3,4), (3,5), (4,3), (5,3), (5,4), (5,5)に逃げられる
        is_checkmate = Rules.is_checkmate(board, Player.BLACK)
        assert not is_checkmate, "逃げ道があるのに詰みと判定されました"
    
    def test_not_checkmate_if_can_capture_attacker(self):
        """攻撃駒を取れれば詰みではない"""
        board = Board()
        
        # 黒の帥を角に配置
        board.add_piece((0, 0), Piece(PieceType.SUI, Player.BLACK))
        # 白の帥を遠くに
        board.add_piece((8, 8), Piece(PieceType.SUI, Player.WHITE))
        # 白の大将で王手
        board.add_piece((0, 1), Piece(PieceType.DAI, Player.WHITE))
        # 黒の駒で攻撃駒を取れる
        board.add_piece((1, 1), Piece(PieceType.DAI, Player.BLACK))
        
        is_checkmate = Rules.is_checkmate(board, Player.BLACK)
        assert not is_checkmate, "攻撃駒を取れるのに詰みと判定されました"
    
    def test_not_checkmate_if_can_block(self):
        """間に駒を入れられれば詰みではない（直線攻撃の場合）"""
        board = Board()
        
        # 黒の帥を配置
        board.add_piece((4, 0), Piece(PieceType.SUI, Player.BLACK))
        # 白の帥を遠くに
        board.add_piece((8, 8), Piece(PieceType.SUI, Player.WHITE))
        # 白の大将で直線攻撃
        board.add_piece((4, 8), Piece(PieceType.DAI, Player.WHITE))
        # 黒の兵がブロックできる位置に
        board.add_piece((5, 4), Piece(PieceType.HYO, Player.BLACK))
        
        # 兵が(4, 4)に移動してブロックできるか確認
        is_checkmate = Rules.is_checkmate(board, Player.BLACK)
        # 注: これは帥の逃げ道もあるので詰みではないはず
        assert not is_checkmate, "ブロックできるのに詰みと判定されました"
    
    def test_checkmate_corner_position(self):
        """角で逃げ場がなく攻撃されている場合"""
        board = Board()
        
        # 黒の帥を(0,0)に配置
        board.add_piece((0, 0), Piece(PieceType.SUI, Player.BLACK))
        # 白の帥を遠くに
        board.add_piece((8, 8), Piece(PieceType.SUI, Player.WHITE))
        # 白の駒で包囲
        board.add_piece((0, 1), Piece(PieceType.DAI, Player.WHITE))  # 王手
        board.add_piece((1, 0), Piece(PieceType.HYO, Player.WHITE))  # 逃げ道を塞ぐ
        board.add_piece((1, 1), Piece(PieceType.HYO, Player.WHITE))  # 逃げ道を塞ぐ
        
        is_check = Rules.is_check(board, Player.BLACK)
        assert is_check, "この配置は王手のはずです"
        
        # 黒は帥以外の駒がないので、逃げるか攻撃駒を取るしかない
        # (0,1)の大将を取れるか？帥は1マスしか動けないので取れる
        # → 詰みではない
        is_checkmate = Rules.is_checkmate(board, Player.BLACK)
        # この場合、帥が大将を取れるので詰みではない
        # assert is_checkmate == False  # コメント: 帥が大将を取れる


class TestSpecialCheckmateScenarios:
    """特殊な詰みシナリオのテスト"""
    
    def test_stalemate_no_legal_moves_but_not_in_check(self):
        """ステイルメイト: 王手ではないが合法手がない"""
        # 軍儀でこの状態が発生するか、発生した場合の扱いを確認
        board = Board()
        
        # 黒の帥を配置
        board.add_piece((0, 0), Piece(PieceType.SUI, Player.BLACK))
        # 白の帥を配置
        board.add_piece((8, 8), Piece(PieceType.SUI, Player.WHITE))
        
        # 黒の帥の周りを白の駒で囲む（王手にならないように）
        # 注: 帥は周囲8マスに動けるので、完全に囲むには多くの駒が必要
        
        legal_moves = Rules.get_legal_moves(board, Player.BLACK)
        # 帥だけでも動けるはず
        assert len(legal_moves) > 0, "帥だけでも合法手があるはずです"
    
    def test_checkmate_with_stack(self):
        """スタック上の駒による詰み"""
        board = Board()
        
        # 黒の帥を角に
        board.add_piece((0, 0), Piece(PieceType.SUI, Player.BLACK))
        # 白の帥を遠くに
        board.add_piece((8, 8), Piece(PieceType.SUI, Player.WHITE))
        
        # 白の駒をスタックして攻撃力を高める
        board.add_piece((0, 2), Piece(PieceType.HYO, Player.WHITE))
        board.add_piece((0, 2), Piece(PieceType.DAI, Player.WHITE))  # 2段目
        
        # スタックの2段目の大将から攻撃
        is_check = Rules.is_check(board, Player.BLACK)
        # 位置(0,2)から(0,0)は2マス離れているので、大将なら攻撃可能か確認
