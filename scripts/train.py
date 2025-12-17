#!/usr/bin/env python
"""
軍儀AI 深層強化学習 トレーニングスクリプト

使用例:
    # 新規学習
    python train.py
    
    # 途中から再開
    python train.py --resume
    
    # 15時間テスト用（デフォルト）
    python train.py --test
    
    # 本番学習
    python train.py --full
"""

import argparse
import os
import sys
import torch
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.model.network import GungiNetwork, create_model
from src.model.encoder import StateEncoder, ActionEncoder
from src.model.trainer import AlphaZeroTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Gungi AI Training')
    
    # 学習モード
    parser.add_argument('--test', action='store_true', default=True,
                        help='15時間テスト用の軽量設定（デフォルト）')
    parser.add_argument('--full', action='store_true',
                        help='本番用の設定')
    parser.add_argument('--resume', action='store_true',
                        help='前回の学習から再開')
    
    # MCTS設定
    parser.add_argument('--mcts-sims', type=int, default=None,
                        help='MCTSシミュレーション数（デフォルト: test=50, full=200）')
    
    # 自己対戦設定
    parser.add_argument('--games', type=int, default=None,
                        help='イテレーションあたりのゲーム数（デフォルト: test=10, full=100）')
    parser.add_argument('--iterations', type=int, default=None,
                        help='総イテレーション数（デフォルト: test=50, full=100）')
    
    # 学習設定
    parser.add_argument('--batch-size', type=int, default=256,
                        help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=10,
                        help='イテレーションあたりのエポック数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学習率')
    
    # その他
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='チェックポイント保存先')
    parser.add_argument('--device', type=str, default=None,
                        help='使用デバイス（cuda/cpu）')
    parser.add_argument('--workers', type=int, default=None,
                        help='並列ワーカー数（デフォルト: CPUコア数の半分）')
    
    return parser.parse_args()


def get_config(args):
    """引数から設定を構築"""
    
    if args.full:
        # 本番設定
        config = {
            'mcts_simulations': args.mcts_sims or 200,
            'games_per_iteration': args.games or 100,
            'num_iterations': args.iterations or 100,
            'num_res_blocks': 8,
            'test_mode': False
        }
    else:
        # 15時間テスト設定
        config = {
            'mcts_simulations': args.mcts_sims or 50,
            'games_per_iteration': args.games or 10,
            'num_iterations': args.iterations or 50,
            'num_res_blocks': 4,
            'test_mode': True
        }
    
    config['batch_size'] = args.batch_size
    config['epochs_per_iteration'] = args.epochs
    config['learning_rate'] = args.lr
    config['checkpoint_dir'] = args.checkpoint_dir
    config['resume'] = args.resume
    
    # 並列ワーカー数
    if args.workers:
        config['num_workers'] = args.workers
    else:
        import os
        # CPUコア数の半分をデフォルトに（最低1、最大8）
        config['num_workers'] = max(1, min(8, os.cpu_count() // 2))
    
    # デバイス
    if args.device:
        config['device'] = args.device
    else:
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return config


def main():
    args = parse_args()
    config = get_config(args)
    
    print("=" * 60)
    print("Gungi AI - Deep Reinforcement Learning")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Full Training' if not config['test_mode'] else '15-Hour Test'}")
    print(f"Device: {config['device']}")
    print()
    print("Configuration:")
    print(f"  MCTS simulations: {config['mcts_simulations']}")
    print(f"  Games per iteration: {config['games_per_iteration']}")
    print(f"  Total iterations: {config['num_iterations']}")
    print(f"  Parallel workers: {config['num_workers']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs per iteration: {config['epochs_per_iteration']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Checkpoint dir: {config['checkpoint_dir']}")
    print(f"  Resume: {config['resume']}")
    print("=" * 60)
    
    # GPU情報
    if config['device'] == 'cuda':
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # モデル作成
    print("\nCreating model...")
    network = create_model(config['device'], test_mode=config['test_mode'])
    
    # パラメータ数を表示
    total_params = sum(p.numel() for p in network.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # エンコーダー作成
    state_encoder = StateEncoder()
    action_encoder = ActionEncoder()
    
    # トレーナー作成
    trainer = AlphaZeroTrainer(
        network=network,
        state_encoder=state_encoder,
        action_encoder=action_encoder,
        device=config['device'],
        mcts_simulations=config['mcts_simulations'],
        games_per_iteration=config['games_per_iteration'],
        num_workers=config['num_workers'],
        batch_size=config['batch_size'],
        epochs_per_iteration=config['epochs_per_iteration'],
        learning_rate=config['learning_rate'],
        checkpoint_dir=config['checkpoint_dir']
    )
    
    # 学習開始
    print("\nStarting training...")
    try:
        trainer.run(
            num_iterations=config['num_iterations'],
            resume=config['resume']
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer.save_checkpoint(trainer.training_state.iteration)
        print("Checkpoint saved. You can resume with --resume flag.")
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Saving emergency checkpoint...")
        trainer.save_checkpoint(trainer.training_state.iteration)
        raise
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
