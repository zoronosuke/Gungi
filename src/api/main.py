"""
軍儀 FastAPI サーバ
ゲームの状態管理とAI推論のエンドポイントを提供
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid
import os
from pathlib import Path

from ..engine import (
    Board, Player, PieceType, Move, MoveType, Rules
)
from ..engine.initial_setup import load_initial_board, get_initial_hand_pieces

app = FastAPI(
    title="軍儀 API",
    description="軍儀ゲームのバックエンドAPI",
    version="1.0.0"
)

# 静的ファイルのディレクトリを設定
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# 静的ファイルをマウント（CSS, JS, HTML）
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# CORS設定（フロントエンドからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では制限すべき
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ゲームの状態を保持する辞書
games: Dict[str, 'GameState'] = {}


class GameState:
    """ゲームの状態を管理するクラス"""
    
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.board = load_initial_board()
        self.current_player = Player.BLACK
        self.move_history: List[Move] = []
        self.hand_pieces = {
            Player.BLACK: get_initial_hand_pieces(Player.BLACK),
            Player.WHITE: get_initial_hand_pieces(Player.WHITE),
        }
        self.game_over = False
        self.winner: Optional[Player] = None

    def switch_turn(self):
        """手番を交代"""
        self.current_player = self.current_player.opponent

    def to_dict(self) -> dict:
        """ゲーム状態を辞書形式に変換"""
        return {
            "game_id": self.game_id,
            "board": self.board.to_dict(),
            "current_player": self.current_player.name,
            "move_count": len(self.move_history),
            "hand_pieces": {
                "BLACK": {k.name: v for k, v in self.hand_pieces[Player.BLACK].items()},
                "WHITE": {k.name: v for k, v in self.hand_pieces[Player.WHITE].items()},
            },
            "game_over": self.game_over,
            "winner": self.winner.name if self.winner else None,
        }


# Pydanticモデル（リクエスト/レスポンス用）

class NewGameResponse(BaseModel):
    game_id: str
    message: str
    game_state: dict


class MoveRequest(BaseModel):
    from_row: Optional[int] = None
    from_col: Optional[int] = None
    to_row: int
    to_col: int
    move_type: str = "NORMAL"  # NORMAL, CAPTURE, STACK, DROP
    piece_type: Optional[str] = None  # DROPの場合に必要


class MoveResponse(BaseModel):
    success: bool
    message: str
    game_state: dict
    legal_moves: Optional[List[dict]] = None


class PredictRequest(BaseModel):
    game_id: str
    depth: int = 1  # MCTSの探索深さ（将来実装）


class PredictResponse(BaseModel):
    move: dict
    evaluation: float
    game_state: dict


# エンドポイント

@app.get("/api")
async def root():
    """APIルート"""
    return {
        "message": "軍儀 API へようこそ",
        "version": "1.0.0",
        "endpoints": [
            "/new_game",
            "/apply_move/{game_id}",
            "/predict/{game_id}",
            "/get_legal_moves/{game_id}",
            "/get_game/{game_id}",
            "/resign/{game_id}",
            "/delete_game/{game_id}",
        ]
    }


@app.post("/new_game", response_model=NewGameResponse)
async def new_game():
    """
    新しいゲームを開始する
    公式の初期盤面から始まる
    """
    game_id = str(uuid.uuid4())
    game_state = GameState(game_id)
    games[game_id] = game_state
    
    return NewGameResponse(
        game_id=game_id,
        message="新しいゲームを開始しました",
        game_state=game_state.to_dict()
    )


@app.get("/get_game/{game_id}")
async def get_game(game_id: str):
    """ゲームの状態を取得"""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="ゲームが見つかりません")
    
    return games[game_id].to_dict()


@app.post("/apply_move/{game_id}", response_model=MoveResponse)
async def apply_move(game_id: str, move_request: MoveRequest):
    """
    手を適用する
    """
    if game_id not in games:
        raise HTTPException(status_code=404, detail="ゲームが見つかりません")
    
    game_state = games[game_id]
    
    if game_state.game_over:
        raise HTTPException(status_code=400, detail="ゲームは既に終了しています")
    
    # 手を構築
    try:
        move_type = MoveType[move_request.move_type]
        
        if move_type == MoveType.DROP:
            if not move_request.piece_type:
                raise HTTPException(status_code=400, detail="DROPには piece_type が必要です")
            piece_type = PieceType[move_request.piece_type]
            move = Move.create_drop_move(
                (move_request.to_row, move_request.to_col),
                piece_type,
                game_state.current_player
            )
        else:
            if move_request.from_row is None or move_request.from_col is None:
                raise HTTPException(status_code=400, detail="移動元の座標が必要です")
            
            from_pos = (move_request.from_row, move_request.from_col)
            to_pos = (move_request.to_row, move_request.to_col)
            
            if move_type == MoveType.CAPTURE:
                move = Move.create_capture_move(from_pos, to_pos, game_state.current_player)
            elif move_type == MoveType.STACK:
                move = Move.create_stack_move(from_pos, to_pos, game_state.current_player)
            else:  # NORMAL
                move = Move.create_normal_move(from_pos, to_pos, game_state.current_player)
        
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"無効なパラメータ: {str(e)}")
    
    # 手を適用
    success, captured_piece = Rules.apply_move(
        game_state.board, 
        move, 
        game_state.hand_pieces[game_state.current_player]
    )
    
    if not success:
        return MoveResponse(
            success=False,
            message="無効な手です",
            game_state=game_state.to_dict()
        )
    
    # 履歴に追加
    game_state.move_history.append(move)
    
    # 駒を取った場合の処理
    sui_captured = False
    if captured_piece:
        # captured_pieceがリストの場合（スタックから取った場合）
        if isinstance(captured_piece, list):
            # スタック内の全ての駒をチェック
            for piece in captured_piece:
                # 帥が取られたかチェック
                if piece.piece_type == PieceType.SUI:
                    sui_captured = True
                    break
                # 帥以外の駒は持ち駒に追加
                piece_type = piece.piece_type
                if piece_type in game_state.hand_pieces[game_state.current_player]:
                    game_state.hand_pieces[game_state.current_player][piece_type] += 1
                else:
                    game_state.hand_pieces[game_state.current_player][piece_type] = 1
        else:
            # 単一の駒の場合
            if captured_piece.piece_type == PieceType.SUI:
                sui_captured = True
            else:
                piece_type = captured_piece.piece_type
                if piece_type in game_state.hand_pieces[game_state.current_player]:
                    game_state.hand_pieces[game_state.current_player][piece_type] += 1
                else:
                    game_state.hand_pieces[game_state.current_player][piece_type] = 1
    
    # ゲーム終了判定（帥が取られた場合は即座に終了）
    if sui_captured:
        game_state.game_over = True
        game_state.winner = game_state.current_player
        return MoveResponse(
            success=True,
            message=f"帥を取りました！{game_state.winner.name}の勝利です！",
            game_state=game_state.to_dict(),
            legal_moves=None
        )
    
    # 詰みの判定
    is_over, winner = Rules.is_game_over(game_state.board)
    if is_over:
        game_state.game_over = True
        game_state.winner = winner
    
    # 手番を交代
    if not game_state.game_over:
        game_state.switch_turn()
    
    # 次の合法手を取得
    legal_moves = None
    if not game_state.game_over:
        legal_moves_list = Rules.get_legal_moves(
            game_state.board,
            game_state.current_player,
            game_state.hand_pieces[game_state.current_player]
        )
        legal_moves = [move.to_dict() for move in legal_moves_list[:50]]  # 最大50手まで
    
    return MoveResponse(
        success=True,
        message="手を適用しました",
        game_state=game_state.to_dict(),
        legal_moves=legal_moves
    )


@app.get("/get_legal_moves/{game_id}")
async def get_legal_moves(game_id: str):
    """現在のプレイヤーの合法手を取得"""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="ゲームが見つかりません")
    
    game_state = games[game_id]
    
    if game_state.game_over:
        return {"legal_moves": [], "message": "ゲームは終了しています"}
    
    legal_moves = Rules.get_legal_moves(
        game_state.board,
        game_state.current_player,
        game_state.hand_pieces[game_state.current_player]
    )
    
    return {
        "legal_moves": [move.to_dict() for move in legal_moves],
        "count": len(legal_moves),
        "current_player": game_state.current_player.name
    }


@app.post("/predict/{game_id}", response_model=PredictResponse)
async def predict(game_id: str, request: PredictRequest):
    """
    AIが次の手を予測する
    現在は簡易的に評価関数付きランダム選択
    将来的にはニューラルネット+MCTSで実装
    """
    if game_id not in games:
        raise HTTPException(status_code=404, detail="ゲームが見つかりません")
    
    game_state = games[game_id]
    
    if game_state.game_over:
        raise HTTPException(status_code=400, detail="ゲームは終了しています")
    
    # 合法手を取得
    legal_moves = Rules.get_legal_moves(
        game_state.board,
        game_state.current_player,
        game_state.hand_pieces[game_state.current_player]
    )
    
    if not legal_moves:
        raise HTTPException(status_code=400, detail="合法手がありません")
    
    # 簡易評価関数で手を選択
    import random
    
    # 手の優先度を評価
    move_scores = []
    for move in legal_moves:
        score = 0
        
        # CAPTURE（駒を取る手）を優先
        if move.move_type == MoveType.CAPTURE:
            score += 100
            # 相手の帥を取る手は最優先
            target_stack = game_state.board.get_stack(move.to_pos)
            if target_stack and target_stack[-1].piece_type == PieceType.SUI:
                score += 1000
        
        # NORMAL（通常移動）
        elif move.move_type == MoveType.NORMAL:
            score += 10
            # 前進する手を少し優先
            if game_state.current_player == Player.BLACK:
                score += max(0, move.from_pos[0] - move.to_pos[0])  # 上方向への移動
            else:
                score += max(0, move.to_pos[0] - move.from_pos[0])  # 下方向への移動
        
        # STACK（ツケ）
        elif move.move_type == MoveType.STACK:
            score += 50
        
        # DROP（新）
        elif move.move_type == MoveType.DROP:
            score += 30
            # 前線に近い位置への配置を優先
            if game_state.current_player == Player.BLACK:
                score += (8 - move.to_pos[0]) * 2  # 上方向が前線
            else:
                score += move.to_pos[0] * 2  # 下方向が前線
        
        # ランダム要素を追加（同じスコアでもバリエーションを持たせる）
        score += random.random() * 5
        
        move_scores.append((move, score))
    
    # スコアでソートして上位から選択（確率的に）
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 上位30%からランダムに選択（完全に最善手だけでなく、多様性を持たせる）
    top_count = max(1, len(move_scores) // 3)
    best_move = random.choice(move_scores[:top_count])[0]  # タプルの最初の要素（Move）を取得
    
    return PredictResponse(
        move=best_move.to_dict(),
        evaluation=0.0,  # 将来: ニューラルネットの評価値
        game_state=game_state.to_dict()
    )


@app.post("/resign/{game_id}")
async def resign(game_id: str):
    """
    投了する
    現在のプレイヤーが投了し、相手の勝利となる
    """
    if game_id not in games:
        raise HTTPException(status_code=404, detail="ゲームが見つかりません")
    
    game_state = games[game_id]
    
    if game_state.game_over:
        raise HTTPException(status_code=400, detail="ゲームは既に終了しています")
    
    # 投了により相手の勝利
    game_state.game_over = True
    game_state.winner = game_state.current_player.opponent
    
    return {
        "message": f"{game_state.current_player.name}が投了しました",
        "winner": game_state.winner.name,
        "game_state": game_state.to_dict()
    }


@app.delete("/delete_game/{game_id}")
async def delete_game(game_id: str):
    """ゲームを削除"""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="ゲームが見つかりません")
    
    del games[game_id]
    return {"message": "ゲームを削除しました"}


# フロントエンド用の静的ファイル配信
@app.get("/")
async def serve_index():
    """ルートでindex.htmlを提供"""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "軍儀 API サーバが稼働中です。/docs でAPIドキュメントを確認できます。"}

@app.get("/{filename:path}")
async def serve_static_files(filename: str):
    """その他の静的ファイルを提供"""
    # APIエンドポイントと競合しないようにチェック
    if filename.startswith("api/") or filename in ["docs", "redoc", "openapi.json"]:
        raise HTTPException(status_code=404, detail="ファイルが見つかりません")
    
    file_path = FRONTEND_DIR / filename
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="ファイルが見つかりません")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
