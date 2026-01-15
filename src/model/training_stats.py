"""
訓練統計収集モジュール
Value予測0収束問題の分析のための詳細なデータ収集
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
import numpy as np


@dataclass
class MoveStats:
    """1手ごとの統計"""
    move_number: int
    policy_entropy: float  # ポリシーのエントロピー（探索の多様性）
    value_prediction: float  # その時点でのValue予測
    mcts_visits: int  # MCTSの総訪問回数
    top_move_prob: float  # 最善手の確率
    top3_move_prob: float  # 上位3手の合計確率
    temperature: float  # 使用された温度
    is_repetition_risk: bool  # 千日手リスクがあったか


@dataclass
class GameStats:
    """1ゲームの詳細統計"""
    game_id: int
    winner: str  # "BLACK", "WHITE", "DRAW"
    termination_reason: str  # "CHECKMATE", "REPETITION", "MAX_MOVES", etc.
    total_moves: int
    game_duration_seconds: float
    
    # 手ごとの統計
    move_stats: List[MoveStats] = field(default_factory=list)
    
    # 集計統計
    avg_policy_entropy: float = 0.0
    avg_value_prediction: float = 0.0
    value_prediction_std: float = 0.0
    avg_top_move_prob: float = 0.0
    
    # Value予測の推移
    value_predictions_black: List[float] = field(default_factory=list)
    value_predictions_white: List[float] = field(default_factory=list)
    
    # 千日手関連
    repetition_count: int = 0  # 同一局面の出現回数
    back_and_forth_count: int = 0  # 往復運動の回数
    
    def compute_aggregates(self):
        """集計統計を計算"""
        if not self.move_stats:
            return
        
        entropies = [m.policy_entropy for m in self.move_stats]
        values = [m.value_prediction for m in self.move_stats]
        top_probs = [m.top_move_prob for m in self.move_stats]
        
        self.avg_policy_entropy = np.mean(entropies) if entropies else 0.0
        self.avg_value_prediction = np.mean(values) if values else 0.0
        self.value_prediction_std = np.std(values) if values else 0.0
        self.avg_top_move_prob = np.mean(top_probs) if top_probs else 0.0


@dataclass
class IterationStats:
    """1イテレーションの詳細統計"""
    iteration: int
    timestamp: str
    
    # ゲーム結果
    total_games: int = 0
    black_wins: int = 0
    white_wins: int = 0
    draws: int = 0
    
    # 終了理由の内訳
    termination_reasons: Dict[str, int] = field(default_factory=dict)
    
    # ゲームの詳細統計
    game_stats: List[GameStats] = field(default_factory=list)
    
    # 集計統計
    avg_game_length: float = 0.0
    min_game_length: int = 0
    max_game_length: int = 0
    
    # Policy統計
    avg_policy_entropy: float = 0.0
    policy_entropy_trend: List[float] = field(default_factory=list)  # 手番ごとの平均
    
    # Value統計
    avg_value_prediction: float = 0.0
    value_near_zero_ratio: float = 0.0  # |v| < 0.1 の割合
    value_prediction_distribution: Dict[str, int] = field(default_factory=dict)
    
    # 学習統計
    policy_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    learning_rate: float = 0.0
    
    # ReplayBuffer統計
    buffer_size: int = 0
    buffer_draw_ratio: float = 0.0
    buffer_value_distribution: Dict[str, int] = field(default_factory=dict)
    
    # 時間統計
    selfplay_duration_seconds: float = 0.0
    training_duration_seconds: float = 0.0
    total_duration_seconds: float = 0.0
    
    def compute_aggregates(self):
        """集計統計を計算"""
        if not self.game_stats:
            return
        
        # ゲーム結果
        self.total_games = len(self.game_stats)
        self.black_wins = sum(1 for g in self.game_stats if g.winner == "BLACK")
        self.white_wins = sum(1 for g in self.game_stats if g.winner == "WHITE")
        self.draws = sum(1 for g in self.game_stats if g.winner == "DRAW")
        
        # 終了理由
        for g in self.game_stats:
            reason = g.termination_reason
            self.termination_reasons[reason] = self.termination_reasons.get(reason, 0) + 1
        
        # ゲーム長
        lengths = [g.total_moves for g in self.game_stats]
        self.avg_game_length = np.mean(lengths) if lengths else 0.0
        self.min_game_length = min(lengths) if lengths else 0
        self.max_game_length = max(lengths) if lengths else 0
        
        # Policy/Value統計
        all_entropies = []
        all_values = []
        for g in self.game_stats:
            for m in g.move_stats:
                all_entropies.append(m.policy_entropy)
                all_values.append(m.value_prediction)
        
        if all_entropies:
            self.avg_policy_entropy = np.mean(all_entropies)
        if all_values:
            self.avg_value_prediction = np.mean(all_values)
            self.value_near_zero_ratio = sum(1 for v in all_values if abs(v) < 0.1) / len(all_values)
            
            # Value分布
            bins = [(-1.0, -0.5), (-0.5, -0.1), (-0.1, 0.1), (0.1, 0.5), (0.5, 1.0)]
            for low, high in bins:
                key = f"{low:.1f}_to_{high:.1f}"
                self.value_prediction_distribution[key] = sum(1 for v in all_values if low <= v < high)


class TrainingStatsCollector:
    """訓練統計コレクター"""
    
    def __init__(self, save_dir: str = "checkpoints"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.iterations: List[IterationStats] = []
        self.current_iteration: Optional[IterationStats] = None
        self.current_game: Optional[GameStats] = None
        
        # グローバル統計
        self.global_stats = {
            "total_games": 0,
            "total_moves": 0,
            "total_training_time": 0.0,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    
    def start_iteration(self, iteration: int):
        """新しいイテレーションを開始"""
        self.current_iteration = IterationStats(
            iteration=iteration,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def start_game(self, game_id: int):
        """新しいゲームを開始"""
        self.current_game = GameStats(
            game_id=game_id,
            winner="",
            termination_reason="",
            total_moves=0,
            game_duration_seconds=0.0
        )
        self._game_start_time = time.time()
    
    def record_move(self, 
                    move_number: int,
                    policy_entropy: float,
                    value_prediction: float,
                    mcts_visits: int,
                    top_move_prob: float,
                    top3_move_prob: float,
                    temperature: float,
                    is_repetition_risk: bool = False,
                    current_player: str = "BLACK"):
        """1手の統計を記録"""
        if self.current_game is None:
            return
        
        move_stat = MoveStats(
            move_number=move_number,
            policy_entropy=policy_entropy,
            value_prediction=value_prediction,
            mcts_visits=mcts_visits,
            top_move_prob=top_move_prob,
            top3_move_prob=top3_move_prob,
            temperature=temperature,
            is_repetition_risk=is_repetition_risk
        )
        self.current_game.move_stats.append(move_stat)
        
        # プレイヤー別のValue予測を記録
        if current_player == "BLACK":
            self.current_game.value_predictions_black.append(value_prediction)
        else:
            self.current_game.value_predictions_white.append(value_prediction)
    
    def end_game(self, winner: str, termination_reason: str, 
                 repetition_count: int = 0, back_and_forth_count: int = 0):
        """ゲームを終了"""
        if self.current_game is None:
            return
        
        self.current_game.winner = winner
        self.current_game.termination_reason = termination_reason
        self.current_game.total_moves = len(self.current_game.move_stats)
        self.current_game.game_duration_seconds = time.time() - self._game_start_time
        self.current_game.repetition_count = repetition_count
        self.current_game.back_and_forth_count = back_and_forth_count
        
        # 集計統計を計算
        self.current_game.compute_aggregates()
        
        # イテレーションに追加
        if self.current_iteration:
            self.current_iteration.game_stats.append(self.current_game)
        
        # グローバル統計を更新
        self.global_stats["total_games"] += 1
        self.global_stats["total_moves"] += self.current_game.total_moves
        
        self.current_game = None
    
    def end_iteration(self, 
                      policy_loss: float = 0.0,
                      value_loss: float = 0.0,
                      total_loss: float = 0.0,
                      learning_rate: float = 0.0,
                      buffer_size: int = 0,
                      buffer_draw_ratio: float = 0.0,
                      buffer_value_distribution: Optional[Dict[str, int]] = None,
                      selfplay_duration: float = 0.0,
                      training_duration: float = 0.0):
        """イテレーションを終了"""
        if self.current_iteration is None:
            return
        
        # 学習統計を設定
        self.current_iteration.policy_loss = policy_loss
        self.current_iteration.value_loss = value_loss
        self.current_iteration.total_loss = total_loss
        self.current_iteration.learning_rate = learning_rate
        
        # バッファ統計を設定
        self.current_iteration.buffer_size = buffer_size
        self.current_iteration.buffer_draw_ratio = buffer_draw_ratio
        if buffer_value_distribution:
            self.current_iteration.buffer_value_distribution = buffer_value_distribution
        
        # 時間統計を設定
        self.current_iteration.selfplay_duration_seconds = selfplay_duration
        self.current_iteration.training_duration_seconds = training_duration
        self.current_iteration.total_duration_seconds = selfplay_duration + training_duration
        
        # 集計統計を計算
        self.current_iteration.compute_aggregates()
        
        # イテレーションリストに追加
        self.iterations.append(self.current_iteration)
        
        # グローバル統計を更新
        self.global_stats["total_training_time"] += self.current_iteration.total_duration_seconds
        
        self.current_iteration = None
    
    def save(self, filename: str = "training_stats.json"):
        """統計をJSONファイルに保存"""
        filepath = self.save_dir / filename
        
        data = {
            "global_stats": self.global_stats,
            "iterations": [self._iteration_to_dict(it) for it in self.iterations]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"統計を保存しました: {filepath}")
        return filepath
    
    def _iteration_to_dict(self, iteration: IterationStats) -> dict:
        """IterationStatsを辞書に変換"""
        d = asdict(iteration)
        # MoveStatsのリストは要約版に変換（サイズ削減）
        for game in d["game_stats"]:
            # 手ごとの詳細は削除し、要約のみ保持
            if len(game["move_stats"]) > 0:
                game["move_stats_summary"] = {
                    "total_moves": len(game["move_stats"]),
                    "first_10_entropies": [m["policy_entropy"] for m in game["move_stats"][:10]],
                    "last_10_entropies": [m["policy_entropy"] for m in game["move_stats"][-10:]],
                    "first_10_values": [m["value_prediction"] for m in game["move_stats"][:10]],
                    "last_10_values": [m["value_prediction"] for m in game["move_stats"][-10:]],
                }
            del game["move_stats"]
        return d
    
    def load(self, filename: str = "training_stats.json"):
        """統計をJSONファイルから読み込み"""
        filepath = self.save_dir / filename
        if not filepath.exists():
            print(f"統計ファイルが見つかりません: {filepath}")
            return False
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.global_stats = data.get("global_stats", self.global_stats)
        # 注: 完全な復元は複雑なので、読み込みは分析用に限定
        print(f"統計を読み込みました: {filepath}")
        return True
    
    def get_summary(self) -> str:
        """統計のサマリーを取得"""
        if not self.iterations:
            return "統計データがありません"
        
        latest = self.iterations[-1]
        lines = [
            f"=== 訓練統計サマリー ===",
            f"総イテレーション: {len(self.iterations)}",
            f"総ゲーム数: {self.global_stats['total_games']}",
            f"総手数: {self.global_stats['total_moves']}",
            f"総訓練時間: {self.global_stats['total_training_time']:.1f}秒",
            f"",
            f"=== 最新イテレーション {latest.iteration} ===",
            f"勝敗: 黒{latest.black_wins} 白{latest.white_wins} 引分{latest.draws}",
            f"終了理由: {latest.termination_reasons}",
            f"平均ゲーム長: {latest.avg_game_length:.1f}手",
            f"Policy Loss: {latest.policy_loss:.4f}",
            f"Value Loss: {latest.value_loss:.4f}",
            f"平均Policy Entropy: {latest.avg_policy_entropy:.4f}",
            f"Value 0近傍率: {latest.value_near_zero_ratio:.1%}",
            f"バッファDraw率: {latest.buffer_draw_ratio:.1%}",
        ]
        return "\n".join(lines)


def compute_policy_entropy(policy: np.ndarray) -> float:
    """ポリシー分布のエントロピーを計算"""
    # 0を除外し、正規化
    policy = np.array(policy).flatten()
    policy = policy[policy > 0]
    if len(policy) == 0:
        return 0.0
    policy = policy / policy.sum()
    return -np.sum(policy * np.log(policy + 1e-10))


def compute_top_k_prob(policy: np.ndarray, k: int = 1) -> float:
    """上位k手の確率の合計を計算"""
    policy = np.array(policy).flatten()
    if len(policy) == 0:
        return 0.0
    sorted_probs = np.sort(policy)[::-1]
    return float(np.sum(sorted_probs[:k]))
