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
            
            if target_piece is None:
                # 空マスへの移動
                move = Move.create_normal_move(from_pos, to_pos, player)
                legal_moves.append(move)
            
            elif target_piece.owner != player:
                # 敵の駒がいる場合
                
                # 敵の帥は取るしかできない（勝利条件）
                # 自分のスタックレベル >= 相手のスタックレベルの場合のみ取れる
                if target_piece.piece_type == PieceType.SUI:
                    if from_stack_level >= to_stack_level:
                        move = Move.create_capture_move(from_pos, to_pos, player)
                        legal_moves.append(move)
                else:
                    # 敵の駒を取る（自分のスタックレベル >= 相手のスタックレベル）
                    if from_stack_level >= to_stack_level:
                        move = Move.create_capture_move(from_pos, to_pos, player)
                        legal_moves.append(move)
                    
                    # または敵の駒の上に重ねる（ツケ）
                    # 条件：
                    # 1. 最大スタック高さ（3）未満
                    # 2. 帥の上には乗せられない
                    # 3. 自分のスタック高さ >= 相手のスタック高さ
                    # 4. 砦は他の駒の上に乗れない
                    if (to_stack_height < 3 and 
                        target_piece.can_be_stacked_on() and
                        from_stack_height >= to_stack_height and
                        piece.can_stack_on_other()):
                        move = Move.create_stack_move(from_pos, to_pos, player)
                        legal_moves.append(move)
            
            else:
                # 味方の駒がいる場合
                
                # 味方の帥の上には乗せられない
                if target_piece.piece_type == PieceType.SUI:
                    continue
                
                # 味方の駒の上に重ねる（ツケ）
                # 条件：
                # 1. 最大スタック高さ（3）未満
                # 2. 帥の上には乗せられない
                # 3. 自分のスタック高さ >= 相手のスタック高さ
                # 4. 砦は他の駒の上に乗れない
                if (to_stack_height < 3 and 
                    target_piece.can_be_stacked_on() and
                    from_stack_height >= to_stack_height and
                    piece.can_stack_on_other()):
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
        max_steps = pattern.get('maxSteps', 1)
        
        # プレイヤーによって方向を調整
        # 駒の動きパターンでは正の値=前方向と定義
        # BLACK（下側・先手）: 前方向は上（負の方向）なので -1を掛ける
        # WHITE（上側・後手）: 前方向は下（正の方向）なので 1を掛ける  
        direction_multiplier = -1 if piece.owner == Player.BLACK else 1
        
        for move in moves:
            dr, dc = move
            # プレイヤーに応じて方向を反転
            dr = dr * direction_multiplier
            
            # 方向ベクトルを正規化（1マス分の移動量）
            abs_dr = abs(dr)
            abs_dc = abs(dc)
            steps_in_move = max(abs_dr, abs_dc)
            
            if steps_in_move == 0:
                continue
            
            # 方向の単位ベクトル
            step_dr = dr // steps_in_move if steps_in_move > 0 else 0
            step_dc = dc // steps_in_move if steps_in_move > 0 else 0
            
            # 直線的な動きかどうかを判定
            is_straight = (step_dr == 0 or step_dc == 0)
            is_diagonal = (abs(step_dr) == abs(step_dc) and step_dr != 0 and step_dc != 0)
            is_continuous = is_straight or is_diagonal
            
            if is_continuous and max_steps > 1:
                # 連続移動可能な方向（飛車・角・龍王・龍馬など）
                # maxSteps まで、または駒にぶつかるまで進む
                for step in range(1, max_steps + 1):
                    target_row = row + step_dr * step
                    target_col = col + step_dc * step
                    
                    if not board.is_valid_position((target_row, target_col)):
                        break
                    
                    # 駒があるか確認
                    if board.is_occupied((target_row, target_col)):
                        # 駒がある場合は、そこまでで止まる（取得またはツケは別途判定）
                        possible_positions.append((target_row, target_col))
                        break
                    else:
                        # 空マスなら追加して次へ
                        possible_positions.append((target_row, target_col))
            else:
                # 固定位置への移動（王将、金将、桂馬など）
                target_row = row + dr
                target_col = col + dc
                
                if not board.is_valid_position((target_row, target_col)):
                    continue
                
                # ジャンプ可能な駒の場合は中間チェック不要
                if can_jump:
                    possible_positions.append((target_row, target_col))
                elif not is_continuous:
                    # 特定マスへのジャンプ（桂馬の動きなど）- 途中チェック不要
                    possible_positions.append((target_row, target_col))
                else:
                    # 直線または斜めの移動 - 途中に駒があればブロック
                    if steps_in_move <= 1:
                        # 1マス移動は常に可能
                        possible_positions.append((target_row, target_col))
                    else:
                        # 複数マス移動の場合、途中に駒がないか確認
                        path_clear = True
                        for step in range(1, steps_in_move):
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
        """
        持ち駒を打つ手を取得（新）
        軍儀のルール: 
        - 自軍の最前線より前のマス（敵寄りのマス）には置けない
        - 空マスまたは味方の駒の上に置ける（帥の上を除く）
        - スタック高さ3未満
        """
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
                    
                    # 最前線より前（敵寄り）には打てない
                    if frontline_row is not None:
                        if player == Player.BLACK:
                            # 黒は上（小さい行番号）が前線なので、frontlineより小さい行には打てない
                            if row < frontline_row:
                                continue
                        else:  # Player.WHITE
                            # 白は下（大きい行番号）が前線なので、frontlineより大きい行には打てない
                            if row > frontline_row:
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
        """
        指定プレイヤーの最前線の行を取得
        
        軍儀のルール: 
        - 黒（先手）: 盤面下側から開始、上（小さい行番号）が敵側
        - 白（後手）: 盤面上側から開始、下（大きい行番号）が敵側
        - 最前線 = 最も敵側に近い自分の駒がある行
        
        返り値: 最前線の行番号、駒がない場合はNone
        """
        frontline = None
        
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                stack = board.get_stack((row, col))
                if not stack.is_empty():
                    # スタック内のいずれかの駒が自分の駒ならその行を考慮
                    has_own_piece = False
                    for level in range(stack.get_height()):
                        piece = stack.get_piece_at_level(level)
                        if piece and piece.owner == player:
                            has_own_piece = True
                            break
                    
                    if has_own_piece:
                        if player == Player.BLACK:
                            # 黒は小さい行番号が前線（敵側）
                            if frontline is None or row < frontline:
                                frontline = row
                        else:  # Player.WHITE
                            # 白は大きい行番号が前線（敵側）
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
        """
        指定位置に駒を打てるか確認
        
        軍儀のルール:
        - 空マスには打てる
        - 味方の駒の上には打てる（帥の上を除く、スタック高さ3未満）
        - 敵の駒の上には打てない
        - 砦は他の駒の上に乗れない
        """
        target_piece = board.get_top_piece(pos)
        stack_height = board.get_stack_height(pos)
        
        # スタック高さが3の場合は打てない
        if stack_height >= 3:
            return False
        
        # 砦は他の駒の上に乗れない
        if piece_type == PieceType.TORIDE and stack_height > 0:
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
            success, _ = Rules.apply_move(test_board, move)
            
            # この手で王手が回避できるか
            if success and not Rules.is_check(test_board, player):
                return False
        
        return True

    @staticmethod
    def apply_move(board: Board, move: Move, hand_pieces: dict = None) -> Tuple[bool, Optional[List[Piece]]]:
        """
        盤面に手を適用する
        
        軍儀のルール:
        - 駒を取るときは、そのマスの駒全部を取る
        - 取った駒は自分の持ち駒として使えない（除外される）
        - 持ち駒を「新」で配置できる
        
        返り値: (成功したらTrue, 取った駒のリスト)
        """
        captured_pieces = None
        
        if move.move_type == MoveType.NORMAL:
            # 通常の移動
            success = board.move_piece(move.from_pos, move.to_pos)
            return success, None
        
        elif move.move_type == MoveType.CAPTURE:
            # 駒を取る（スタック全体）
            # 軍儀のルール: 駒を取るときは、そのマスの駒全部を取る
            captured_pieces = board.capture_piece(move.to_pos)
            success = board.move_piece(move.from_pos, move.to_pos)
            # 取った駒は自分の持ち駒として使えない（除外される）
            return success, captured_pieces
        
        elif move.move_type == MoveType.STACK:
            # 駒を重ねる
            success = board.move_piece(move.from_pos, move.to_pos)
            return success, None
        
        elif move.move_type == MoveType.DROP:
            # 持ち駒を打つ（新）
            
            # 持ち駒数のチェック
            if hand_pieces is None or move.piece_type not in hand_pieces:
                return False, None
            
            if hand_pieces[move.piece_type] <= 0:
                return False, None
            
            # 打つ先のチェック
            to_stack = board.get_stack(move.to_pos)
            to_height = to_stack.get_height()
            
            # 1. 高さ3のスタックには打てない
            if to_height >= 3:
                return False, None
            
            # 2. 帥の上には打てない
            if not to_stack.is_empty():
                top_piece = to_stack.get_top_piece()
                if top_piece and not top_piece.can_be_stacked_on():
                    return False, None
            
            # 駒を配置
            piece = Piece(move.piece_type, move.player)
            success = board.add_piece(move.to_pos, piece)
            
            # 持ち駒から減らす
            if success:
                hand_pieces[move.piece_type] -= 1
            
            return success, None
        
        elif move.move_type == MoveType.SETUP:
            # 初期配置
            piece = Piece(move.piece_type, move.player)
            success = board.add_piece(move.to_pos, piece)
            return success, None
        
        return False, None

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
