"""
軍儀のルール判定を行うモジュール
"""

from typing import List, Tuple, Optional
from .board import Board, BOARD_SIZE
from .piece import Piece, Player, PieceType
from .move import Move, MoveType


class Rules:
    """軍儀のルールを管理するクラス"""
    
    @staticmethod
    def get_legal_moves(board: Board, player: Player, hand_pieces: dict = None) -> List[Move]:
        """
        指定プレイヤーの合法手をすべて取得
        hand_pieces: 持ち駒の辞書 {PieceType: count}
        """
        legal_moves = []
        
        # 盤上の駒の移動
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                pos = (row, col)
                top_piece = board.get_top_piece(pos)
                
                if top_piece and top_piece.owner == player:
                    # この駒の合法手を取得
                    piece_moves = Rules._get_piece_legal_moves(board, pos, player)
                    legal_moves.extend(piece_moves)
        
        # 持ち駒を打つ手（将来実装）
        if hand_pieces:
            drop_moves = Rules._get_drop_moves(board, player, hand_pieces)
            legal_moves.extend(drop_moves)
        
        return legal_moves

    @staticmethod
    def _get_piece_legal_moves(
        board: Board,
        from_pos: Tuple[int, int],
        player: Player
    ) -> List[Move]:
        """指定位置の駒の合法手を取得"""
        legal_moves = []
        piece = board.get_top_piece(from_pos)
        
        if not piece or piece.owner != player:
            return legal_moves
        
        # 一番上の駒のスタックレベルを取得（1, 2, 3）
        from_stack_level = board.get_top_piece_stack_level(from_pos)
        from_stack_height = board.get_stack_height(from_pos)
        
        # 駒の移動可能な位置を取得
        possible_positions = Rules._get_possible_move_positions(
            board, from_pos, piece, from_stack_level
        )
        
        for to_pos in possible_positions:
            # 移動先の状態を確認
            target_piece = board.get_top_piece(to_pos)
            to_stack_level = board.get_top_piece_stack_level(to_pos)
            to_stack_height = board.get_stack_height(to_pos)
            
            # 自分より高いスタックには移動できない（取ることもツケることもできない）
            if to_stack_level > from_stack_level:
                continue
            
            if target_piece is None:
                # 空マスへの移動
                move = Move.create_normal_move(from_pos, to_pos, player)
                legal_moves.append(move)
            
            elif target_piece.owner != player:
                # 敵の駒がいる場合
                
                # 敵の帥を取る（勝利条件）
                if target_piece.piece_type == PieceType.SUI:
                    move = Move.create_capture_move(from_pos, to_pos, player)
                    legal_moves.append(move)
                    continue
                
                # 敵の駒を取る
                move = Move.create_capture_move(from_pos, to_pos, player)
                legal_moves.append(move)
                
                # または敵の駒の上に重ねる（ツケ）
                # 最大スタック高さをチェック＆帥の上には乗せられない
                if to_stack_height < 3 and target_piece.can_be_stacked_on():
                    move = Move.create_stack_move(from_pos, to_pos, player)
                    legal_moves.append(move)
            
            else:
                # 味方の駒がいる場合
                
                # 味方の帥の上には乗せられない
                if target_piece.piece_type == PieceType.SUI:
                    continue
                
                # 味方の駒の上に重ねる（ツケ）
                # 最大スタック高さをチェック＆帥の上には乗せられない
                if to_stack_height < 3 and target_piece.can_be_stacked_on():
                    move = Move.create_stack_move(from_pos, to_pos, player)
                    legal_moves.append(move)
        
        return legal_moves

    @staticmethod
    def _get_possible_move_positions(
        board: Board,
        from_pos: Tuple[int, int],
        piece: Piece,
        stack_level: int
    ) -> List[Tuple[int, int]]:
        """駒が移動できる位置のリストを取得"""
        row, col = from_pos
        possible_positions = []
        
        # 駒の動きパターンを取得
        pattern = piece.get_move_pattern(stack_level)
        moves = pattern['moves']
        can_jump = pattern['canJump']
        
        # プレイヤーによって方向を調整
        # 駒の動きパターンでは正の値=前方向と定義
        # BLACK（下側・先手）: 前方向は上（負の方向）なので -1を掛ける
        # WHITE（上側・後手）: 前方向は下（正の方向）なので 1を掛ける  
        direction_multiplier = -1 if piece.owner == Player.BLACK else 1
        
        for move in moves:
            dr, dc = move
            # プレイヤーに応じて方向を反転
            dr = dr * direction_multiplier
            
            target_row = row + dr
            target_col = col + dc
            
            if not board.is_valid_position((target_row, target_col)):
                continue
            
            # ジャンプ可能な駒の場合は中間チェック不要
            if can_jump:
                possible_positions.append((target_row, target_col))
            else:
                # ジャンプできない駒は中間マスをチェック
                # 移動距離が2以上の場合、途中に駒があれば移動不可
                abs_dr = abs(dr)
                abs_dc = abs(dc)
                max_dist = max(abs_dr, abs_dc)
                
                if max_dist <= 1:
                    # 1マス移動は常に可能
                    possible_positions.append((target_row, target_col))
                else:
                    # 複数マス移動の場合、途中に駒がないか確認
                    path_clear = True
                    steps = max_dist
                    step_dr = dr // steps if dr != 0 else 0
                    step_dc = dc // steps if dc != 0 else 0
                    
                    # 途中のマスをチェック
                    for step in range(1, steps):
                        check_row = row + step_dr * step
                        check_col = col + step_dc * step
                        if board.is_occupied((check_row, check_col)):
                            path_clear = False
                            break
                    
                    if path_clear:
                        possible_positions.append((target_row, target_col))
        
        return possible_positions

    @staticmethod
    def _get_line_moves(
        board: Board,
        from_pos: Tuple[int, int],
        directions: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """直線方向の移動可能位置を取得"""
        row, col = from_pos
        positions = []
        
        for dr, dc in directions:
            r, c = row, col
            while True:
                r, c = r + dr, c + dc
                if not board.is_valid_position((r, c)):
                    break
                positions.append((r, c))
                # 駒がある場合はそこで止まる
                if board.is_occupied((r, c)):
                    break
        
        return positions

    @staticmethod
    def _get_jump_moves(
        board: Board,
        from_pos: Tuple[int, int],
        jumps: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """ジャンプ移動の可能位置を取得"""
        row, col = from_pos
        positions = []
        
        for dr, dc in jumps:
            # 中間地点
            mid_r, mid_c = row + dr // 2, col + dc // 2
            # 目標地点
            target_r, target_c = row + dr, col + dc
            
            if board.is_valid_position((target_r, target_c)):
                # 中間地点に駒がない場合のみ移動可能
                if not board.is_occupied((mid_r, mid_c)):
                    positions.append((target_r, target_c))
        
        return positions

    @staticmethod
    def _get_drop_moves(
        board: Board,
        player: Player,
        hand_pieces: dict
    ) -> List[Move]:
        """持ち駒を打つ手を取得（新）"""
        drop_moves = []
        
        # 最前線の行を取得
        frontline_row = Rules._get_frontline_row(board, player)
        
        for piece_type, count in hand_pieces.items():
            if count <= 0:
                continue
            
            # 各マスについて打てるか確認
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    pos = (row, col)
                    
                    # 最前線より前には打てない
                    if player == Player.BLACK:
                        if frontline_row is not None and row < frontline_row:
                            continue
                    else:  # Player.WHITE
                        if frontline_row is not None and row > frontline_row:
                            continue
                    
                    # 駒を打てる条件をチェック
                    if Rules._can_drop_piece_at(board, pos, piece_type, player):
                        move = Move(
                            move_type=MoveType.DROP,
                            from_pos=None,
                            to_pos=pos,
                            piece_type=piece_type,
                            player=player
                        )
                        drop_moves.append(move)
        
        return drop_moves
    
    @staticmethod
    def _get_frontline_row(board: Board, player: Player) -> Optional[int]:
        """指定プレイヤーの最前線の行を取得"""
        frontline = None
        
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                stack = board.get_stack((row, col))
                if not stack.is_empty():
                    # 一番下の駒（最初に置かれた駒）をチェック
                    bottom_piece = stack.get_piece_at_level(0)
                    if bottom_piece and bottom_piece.owner == player:
                        if player == Player.BLACK:
                            # 黒は小さい行番号が前線
                            if frontline is None or row < frontline:
                                frontline = row
                        else:  # Player.WHITE
                            # 白は大きい行番号が前線
                            if frontline is None or row > frontline:
                                frontline = row
        
        return frontline
    
    @staticmethod
    def _can_drop_piece_at(
        board: Board,
        pos: Tuple[int, int],
        piece_type: PieceType,
        player: Player
    ) -> bool:
        """指定位置に駒を打てるか確認"""
        target_piece = board.get_top_piece(pos)
        stack_height = board.get_stack_height(pos)
        
        # スタック高さが3の場合は打てない
        if stack_height >= 3:
            return False
        
        # 空マスには打てる
        if target_piece is None:
            return True
        
        # 味方の駒の上には打てる（帥を除く）
        if target_piece.owner == player:
            if target_piece.piece_type == PieceType.SUI:
                return False  # 帥の上には打てない
            return True
        
        # 敵の駒の上には打てない
        return False

    @staticmethod
    def is_check(board: Board, player: Player) -> bool:
        """
        指定プレイヤーの帥が王手されているか確認
        """
        sui_pos = board.get_sui_position(player)
        if sui_pos is None:
            return False  # 帥が既に取られている
        
        opponent = player.opponent
        
        # 相手の全ての駒から帥が攻撃されているか確認
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                pos = (row, col)
                piece = board.get_top_piece(pos)
                
                if piece and piece.owner == opponent:
                    stack_level = board.get_stack_height(pos)
                    possible_positions = Rules._get_possible_move_positions(
                        board, pos, piece, stack_level
                    )
                    
                    if sui_pos in possible_positions:
                        return True
        
        return False

    @staticmethod
    def is_checkmate(board: Board, player: Player) -> bool:
        """
        指定プレイヤーが詰んでいるか確認
        """
        if not Rules.is_check(board, player):
            return False
        
        # 全ての合法手を試して、王手を回避できるか確認
        legal_moves = Rules.get_legal_moves(board, player)
        
        for move in legal_moves:
            # 手を試す
            test_board = board.copy()
            Rules.apply_move(test_board, move)
            
            # この手で王手が回避できるか
            if not Rules.is_check(test_board, player):
                return False
        
        return True

    @staticmethod
    def apply_move(board: Board, move: Move) -> bool:
        """
        盤面に手を適用する
        返り値: 成功したらTrue
        """
        if move.move_type == MoveType.NORMAL:
            # 通常の移動
            return board.move_piece(move.from_pos, move.to_pos)
        
        elif move.move_type == MoveType.CAPTURE:
            # 駒を取る
            board.capture_piece(move.to_pos)
            return board.move_piece(move.from_pos, move.to_pos)
        
        elif move.move_type == MoveType.STACK:
            # 駒を重ねる
            return board.move_piece(move.from_pos, move.to_pos)
        
        elif move.move_type == MoveType.DROP:
            # 持ち駒を打つ
            piece = Piece(move.piece_type, move.player)
            return board.add_piece(move.to_pos, piece)
        
        elif move.move_type == MoveType.SETUP:
            # 初期配置
            piece = Piece(move.piece_type, move.player)
            return board.add_piece(move.to_pos, piece)
        
        return False

    @staticmethod
    def is_game_over(board: Board) -> Tuple[bool, Optional[Player]]:
        """
        ゲームが終了したか確認
        返り値: (終了フラグ, 勝者)
        """
        # 帥が取られているか確認
        for player in [Player.BLACK, Player.WHITE]:
            sui_pos = board.get_sui_position(player)
            if sui_pos is None:
                # 帥が取られた = 相手の勝ち
                return True, player.opponent
            
            # 詰んでいるか確認
            if Rules.is_checkmate(board, player):
                return True, player.opponent
        
        return False, None
