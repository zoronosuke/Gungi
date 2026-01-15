"""
ニューラルネットワークの学習モジュール
AlphaZero方式の学習ループを実装
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
from datetime import datetime

from .network import GungiNetwork, create_model
from .encoder import StateEncoder, ActionEncoder
from .self_play import SelfPlay, TrainingExample, ReplayBuffer
from .gpu_self_play import GPUSelfPlay
from .max_efficiency_selfplay import MaxEfficiencySelfPlay
from .optimized_selfplay import OptimizedSelfPlay
from .training_stats import TrainingStatsCollector, IterationStats, GameStats


class GungiDataset(Dataset):
    """学習用データセット"""
    
    def __init__(self, examples: List[TrainingExample]):
        self.states = np.array([ex.state for ex in examples])
        self.policies = np.array([ex.policy for ex in examples])
        self.values = np.array([ex.value for ex in examples])
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = torch.from_numpy(self.states[idx]).float()
        policy = torch.from_numpy(self.policies[idx]).float()
        value = torch.tensor([self.values[idx]]).float()
        return state, policy, value


class Trainer:
    """ニューラルネットワークの学習"""
    
    def __init__(
        self,
        network: GungiNetwork,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = 'cuda'
    ):
        self.network = network
        self.device = device
        self.optimizer = optim.Adam(
            network.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=10, 
            gamma=0.9
        )
    
    def train(
        self,
        examples: List[TrainingExample],
        batch_size: int = 256,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        学習を実行
        
        Args:
            examples: 学習データ
            batch_size: バッチサイズ
            epochs: エポック数
            verbose: 詳細出力
        
        Returns:
            {'policy_loss': float, 'value_loss': float, 'total_loss': float}
        """
        self.network.train()
        
        dataset = GungiDataset(examples)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_batches = 0
        
        for epoch in range(epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_batches = 0
            
            iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else dataloader
            
            for states, target_policies, target_values in iterator:
                states = states.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(self.device)
                
                # 順伝播
                log_policies, values = self.network(states)
                
                # 損失計算
                policy_loss, value_loss = self._compute_loss(
                    log_policies, values, target_policies, target_values
                )
                total_loss = policy_loss + value_loss
                
                # 逆伝播
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_batches += 1
                
                if verbose and isinstance(iterator, tqdm):
                    iterator.set_postfix({
                        'p_loss': f"{policy_loss.item():.4f}",
                        'v_loss': f"{value_loss.item():.4f}"
                    })
            
            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss
            total_batches += epoch_batches
            
            # 学習率スケジューラを更新
            self.scheduler.step()
        
        avg_policy_loss = total_policy_loss / total_batches if total_batches > 0 else 0.0
        avg_value_loss = total_value_loss / total_batches if total_batches > 0 else 0.0
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'total_loss': avg_policy_loss + avg_value_loss
        }
    
    # Value予測の0収束問題対策パラメータ（千日手対策で強化）
    VALUE_REGULARIZATION_WEIGHT = 0.01  # Value正則化の重み
    ENTROPY_BONUS_WEIGHT = 0.05         # Policyエントロピーボーナスの重み（0.01→0.05に強化）
    
    def _compute_loss(
        self,
        log_policies: torch.Tensor,
        values: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        損失を計算
        
        Policy Loss: クロスエントロピー + エントロピーボーナス
            loss = -Σ target_policy * log(pred_policy) - β * entropy(pred_policy)
        
        Value Loss: MSE + 正則化
            loss = (target_value - pred_value)^2 + λ * value_regularization
        
        Value予測の0収束問題対策:
        1. Value正則化: 極端な予測（常に0など）を抑制
        2. Policyエントロピーボーナス: 探索多様性を維持
        """
        # Policy Loss: KLダイバージェンス（クロスエントロピー）
        # target_policies は既に確率分布、log_policies はlog確率
        policy_cross_entropy = -torch.sum(target_policies * log_policies, dim=1).mean()
        
        # Policyエントロピーボーナス（探索多様性を維持）
        # エントロピー = -Σ p * log(p) を最大化（負の値を足す）
        policies = torch.exp(log_policies)  # log確率から確率へ
        policy_entropy = -torch.sum(policies * log_policies, dim=1).mean()
        
        # エントロピーボーナスを引く（エントロピー最大化 = ボーナス）
        policy_loss = policy_cross_entropy - self.ENTROPY_BONUS_WEIGHT * policy_entropy
        
        # Value Loss: MSE
        value_mse = nn.functional.mse_loss(values, target_values)
        
        # Value正則化: 予測の分散を促進（0に収束することを防ぐ）
        # 予測値の分散が小さい（=すべて同じ値を予測）場合にペナルティ
        value_variance = torch.var(values)
        # 分散が小さいときにペナルティを大きくする（1/(var+ε)を使用）
        value_diversity_penalty = 1.0 / (value_variance + 0.1)
        
        value_loss = value_mse + self.VALUE_REGULARIZATION_WEIGHT * value_diversity_penalty
        
        return policy_loss, value_loss
    
    def set_learning_rate(self, lr: float):
        """学習率を設定"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


@dataclass
class TrainingState:
    """学習状態を保持するクラス（チェックポイント用）
    
    Value予測の0収束問題対策:
    - draw_rate_historyを追加して引き分け率の推移を監視
    - buffer_draw_ratio_historyでバッファ内の引き分け比率を追跡
    """
    iteration: int
    total_games: int
    total_examples: int
    policy_loss_history: List[float]
    value_loss_history: List[float]
    win_stats: Dict[str, int]
    start_time: str
    last_update_time: str
    # Value予測の0収束問題対策: 追加の統計情報
    draw_rate_history: List[float] = None  # イテレーションごとの引き分け率
    buffer_draw_ratio_history: List[float] = None  # バッファ内の引き分け比率
    
    def __post_init__(self):
        """リストの初期化"""
        if self.draw_rate_history is None:
            self.draw_rate_history = []
        if self.buffer_draw_ratio_history is None:
            self.buffer_draw_ratio_history = []


class AlphaZeroTrainer:
    """AlphaZero方式の学習ループ"""
    
    def __init__(
        self,
        network: GungiNetwork,
        state_encoder: StateEncoder = None,
        action_encoder: ActionEncoder = None,
        device: str = 'cuda',
        # MCTS設定
        mcts_simulations: int = 50,
        c_puct: float = 3.0,  # 探索幅を拡大（1.5→3.0、千日手対策）
        # 自己対戦設定
        games_per_iteration: int = 10,
        temperature_threshold: int = 30,  # 温度維持手数を延長（20→30）
        num_workers: int = 1,  # CPU並列版用
        use_gpu_selfplay: bool = True,  # GPU版自己対戦を使うか
        use_optimized: bool = False,  # 最適化版を使うか
        num_parallel_games: int = 16,  # 最大効率版の並行ゲーム数
        use_fp16: bool = True,  # 半精度を使うか
        # 学習設定
        batch_size: int = 256,
        epochs_per_iteration: int = 10,
        learning_rate: float = 0.001,
        # バッファ設定
        buffer_size: int = 50000,
        # 保存設定
        checkpoint_dir: str = './checkpoints'
    ):
        self.network = network
        self.state_encoder = state_encoder or StateEncoder()
        self.action_encoder = action_encoder or ActionEncoder()
        self.device = device
        
        # MCTS設定
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct
        
        # 自己対戦設定
        self.games_per_iteration = games_per_iteration
        self.temperature_threshold = temperature_threshold
        self.num_workers = num_workers  # 並列ワーカー数
        self.use_gpu_selfplay = use_gpu_selfplay  # GPU版を使うか
        self.use_optimized = use_optimized  # 最適化版を使うか
        self.num_parallel_games = num_parallel_games  # 並行ゲーム数
        self.use_fp16 = use_fp16  # 半精度
        
        # 学習設定
        self.batch_size = batch_size
        self.epochs_per_iteration = epochs_per_iteration
        self.learning_rate = learning_rate
        
        # リプレイバッファ
        self.buffer = ReplayBuffer(max_size=buffer_size)
        
        # 保存設定
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Trainer
        self.trainer = Trainer(
            network=network,
            lr=learning_rate,
            device=device
        )
        
        # 学習状態
        self.training_state = TrainingState(
            iteration=0,
            total_games=0,
            total_examples=0,
            policy_loss_history=[],
            value_loss_history=[],
            win_stats={'BLACK': 0, 'WHITE': 0, 'DRAW': 0},
            start_time=datetime.now().isoformat(),
            last_update_time=datetime.now().isoformat()
        )
        
        # 統計コレクター（Value予測0収束問題の診断用）
        self.stats_collector = TrainingStatsCollector(save_dir=checkpoint_dir)
    
    def run(self, num_iterations: int = 50, resume: bool = False):
        """
        学習ループを実行
        
        Args:
            num_iterations: 総イテレーション数
            resume: 前回の学習から再開するか
        """
        if resume:
            self.load_checkpoint()
            print(f"Resumed from iteration {self.training_state.iteration}")
        
        start_iter = self.training_state.iteration
        
        print(f"\n{'='*60}")
        print(f"AlphaZero Training for Gungi")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        if self.use_optimized:
            print(f"Self-play mode: OPTIMIZED (GPU max efficiency)")
        elif self.use_gpu_selfplay:
            print(f"Self-play mode: GPU-accelerated")
        else:
            print(f"Self-play mode: CPU parallel")
        print(f"MCTS simulations: {self.mcts_simulations}")
        print(f"Games per iteration: {self.games_per_iteration}")
        if self.use_gpu_selfplay or self.use_optimized:
            print(f"Parallel games: {self.num_parallel_games}")
            if self.use_optimized:
                print(f"FP16 (half precision): {self.use_fp16}")
        else:
            print(f"Parallel workers: {self.num_workers}")
        print(f"Total iterations: {num_iterations}")
        print(f"Starting from iteration: {start_iter}")
        print(f"{'='*60}\n")
        
        for iteration in range(start_iter, num_iterations):
            iter_start_time = time.time()
            selfplay_start_time = time.time()
            
            print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
            
            # 統計コレクターでイテレーション開始
            self.stats_collector.start_iteration(iteration + 1)
            
            # 1. 自己対戦でデータ生成
            print("Generating self-play data...")
            game_stats = []  # ゲーム統計
            
            if self.use_optimized:
                # 最適化版（GPU最大効率 + FP16 + Virtual Loss）
                optimized_self_play = OptimizedSelfPlay(
                    network=self.network,
                    state_encoder=self.state_encoder,
                    action_encoder=self.action_encoder,
                    mcts_simulations=self.mcts_simulations,
                    c_puct=self.c_puct,
                    device=self.device,
                    num_parallel_games=self.num_parallel_games,
                    use_fp16=self.use_fp16
                )
                
                examples = optimized_self_play.generate_data(
                    num_games=self.games_per_iteration,
                    temperature_threshold=self.temperature_threshold,
                    verbose=True
                )
            elif self.use_gpu_selfplay:
                # 最大効率版（GPU活用 + 並行ゲーム）- 統計収集対応
                max_eff_self_play = MaxEfficiencySelfPlay(
                    network=self.network,
                    state_encoder=self.state_encoder,
                    action_encoder=self.action_encoder,
                    mcts_simulations=self.mcts_simulations,
                    c_puct=self.c_puct,
                    device=self.device,
                    num_parallel_games=self.num_parallel_games
                )
                
                examples, game_stats = max_eff_self_play.generate_data(
                    num_games=self.games_per_iteration,
                    temperature_threshold=self.temperature_threshold,
                    verbose=True,
                    collect_stats=True  # 統計収集を有効化
                )
            else:
                # CPU並列版
                self_play = SelfPlay(
                    network=self.network,
                    state_encoder=self.state_encoder,
                    action_encoder=self.action_encoder,
                    mcts_simulations=self.mcts_simulations,
                    c_puct=self.c_puct,
                    device=self.device
                )
                
                examples = self_play.generate_data(
                    num_games=self.games_per_iteration,
                    temperature_threshold=self.temperature_threshold,
                    verbose=True,
                    num_workers=self.num_workers
                )
            
            selfplay_duration = time.time() - selfplay_start_time
            
            # 2. リプレイバッファに追加
            self.buffer.add(examples)
            print(f"Buffer size: {len(self.buffer)}")
            
            # Value予測の0収束問題対策: バッファ内のValue分布を監視
            value_dist = self.buffer.get_value_distribution()
            print(f"Buffer value distribution: Win={value_dist['win']}, Loss={value_dist['loss']}, Draw={value_dist['draw']} ({value_dist['draw_ratio']*100:.1f}%)")
            
            # 引き分け率の警告
            if value_dist['draw_ratio'] > 0.5:
                print(f"⚠️ WARNING: Draw ratio in buffer is {value_dist['draw_ratio']*100:.1f}% (>50%)")
            
            # 3. ネットワークを学習
            training_start_time = time.time()
            losses = {'policy_loss': 0.0, 'value_loss': 0.0, 'total_loss': 0.0}
            
            if len(self.buffer) >= self.batch_size:
                print("Training network...")
                train_examples = self.buffer.sample(
                    min(len(self.buffer), self.batch_size * 10)
                )
                
                losses = self.trainer.train(
                    examples=train_examples,
                    batch_size=self.batch_size,
                    epochs=self.epochs_per_iteration,
                    verbose=True
                )
                
                self.training_state.policy_loss_history.append(losses['policy_loss'])
                self.training_state.value_loss_history.append(losses['value_loss'])
                
                # Value予測の0収束問題対策: 引き分け率とバッファ比率を記録
                iter_draw_rate = sum(1 for ex in examples if -0.5 <= ex.value <= 0.5) / len(examples) if examples else 0
                self.training_state.draw_rate_history.append(iter_draw_rate)
                self.training_state.buffer_draw_ratio_history.append(value_dist['draw_ratio'])
                
                print(f"Policy Loss: {losses['policy_loss']:.4f}")
                print(f"Value Loss: {losses['value_loss']:.4f}")
                print(f"This iteration draw rate: {iter_draw_rate*100:.1f}%")
            
            training_duration = time.time() - training_start_time
            
            # 4. 統計コレクターにゲーム統計を追加
            if game_stats:
                for gs in game_stats:
                    self.stats_collector.current_iteration.game_stats.append(gs)
                    # グローバル統計も更新
                    self.stats_collector.global_stats["total_games"] += 1
                    self.stats_collector.global_stats["total_moves"] += gs.total_moves
            
            # イテレーション統計を終了
            self.stats_collector.end_iteration(
                policy_loss=losses['policy_loss'],
                value_loss=losses['value_loss'],
                total_loss=losses['total_loss'],
                learning_rate=self.trainer.optimizer.param_groups[0]['lr'],
                buffer_size=len(self.buffer),
                buffer_draw_ratio=value_dist['draw_ratio'],
                buffer_value_distribution={
                    'win': value_dist['win'],
                    'loss': value_dist['loss'],
                    'draw': value_dist['draw']
                },
                selfplay_duration=selfplay_duration,
                training_duration=training_duration
            )
            
            # 統計を保存
            self.stats_collector.save("training_stats.json")
            
            # 5. 状態を更新
            self.training_state.iteration = iteration + 1
            self.training_state.total_games += self.games_per_iteration
            self.training_state.total_examples += len(examples)
            self.training_state.last_update_time = datetime.now().isoformat()
            
            # 6. チェックポイント保存
            self.save_checkpoint(iteration + 1)
            
            iter_time = time.time() - iter_start_time
            print(f"Iteration time: {iter_time:.1f}s")
            
            # 統計サマリーを表示
            print(self.stats_collector.get_summary())
            
            # 学習率を調整（後半で下げる）
            if iteration == num_iterations // 2:
                new_lr = self.learning_rate * 0.1
                self.trainer.set_learning_rate(new_lr)
                print(f"Learning rate reduced to {new_lr}")
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Total games: {self.training_state.total_games}")
        print(f"Total examples: {self.training_state.total_examples}")
        print(f"{'='*60}")
    
    def save_checkpoint(self, iteration: int):
        """チェックポイントを保存"""
        # モデルを保存
        model_path = os.path.join(self.checkpoint_dir, f'model_iter_{iteration:04d}.pt')
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'iteration': iteration
        }, model_path)
        
        # 最新のモデルも保存（上書き）
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pt')
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'iteration': iteration
        }, latest_path)
        
        # 学習状態を保存
        state_path = os.path.join(self.checkpoint_dir, 'training_state.json')
        with open(state_path, 'w') as f:
            json.dump(asdict(self.training_state), f, indent=2)
        
        # リプレイバッファを保存
        buffer_path = os.path.join(self.checkpoint_dir, 'replay_buffer.npz')
        self.buffer.save(buffer_path)
        
        print(f"Checkpoint saved: iteration {iteration}")
    
    def load_checkpoint(self, path: str = None):
        """チェックポイントを読み込み"""
        if path is None:
            path = os.path.join(self.checkpoint_dir, 'latest.pt')
        
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return False
        
        # モデルを読み込み
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 学習状態を読み込み
        state_path = os.path.join(self.checkpoint_dir, 'training_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state_dict = json.load(f)
                self.training_state = TrainingState(**state_dict)
        
        # リプレイバッファを読み込み
        buffer_path = os.path.join(self.checkpoint_dir, 'replay_buffer.npz')
        if os.path.exists(buffer_path):
            self.buffer.load(buffer_path)
            print(f"Loaded {len(self.buffer)} examples from buffer")
        
        return True


if __name__ == "__main__":
    # テスト
    print("=== Trainer Test ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # モデルを作成
    network = create_model(device, test_mode=True)
    
    # ダミーデータを作成
    dummy_examples = [
        TrainingExample(
            state=np.random.randn(91, 9, 9).astype(np.float32),
            policy=np.random.dirichlet(np.ones(7695)).astype(np.float32),
            value=np.random.choice([-1.0, 0.0, 1.0])
        )
        for _ in range(100)
    ]
    
    # Trainerでテスト
    trainer = Trainer(network=network, device=device)
    losses = trainer.train(
        examples=dummy_examples,
        batch_size=32,
        epochs=2,
        verbose=True
    )
    
    print(f"\nFinal losses: {losses}")
