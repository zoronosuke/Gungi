"""
学習状態をリセットするスクリプト
引き分けデータが多すぎる場合に使用
"""

import os
import json
import numpy as np
from datetime import datetime

def reset_training(checkpoint_dir='./checkpoints', reset_model=False, reset_buffer=True, reset_state=True):
    """
    学習状態をリセット
    
    Args:
        checkpoint_dir: チェックポイントディレクトリ
        reset_model: モデルの重みもリセットするか
        reset_buffer: リプレイバッファをリセットするか
        reset_state: 学習状態をリセットするか
    """
    
    if reset_buffer:
        buffer_path = os.path.join(checkpoint_dir, 'replay_buffer.npz')
        if os.path.exists(buffer_path):
            # バックアップ
            backup_path = os.path.join(checkpoint_dir, f'replay_buffer_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz')
            os.rename(buffer_path, backup_path)
            print(f"リプレイバッファをバックアップ: {backup_path}")
            
            # 空のバッファを作成
            np.savez(buffer_path, states=np.array([]), policies=np.array([]), values=np.array([]))
            print("リプレイバッファをリセットしました")
        else:
            print("リプレイバッファが見つかりません")
    
    if reset_state:
        state_path = os.path.join(checkpoint_dir, 'training_state.json')
        if os.path.exists(state_path):
            # バックアップ
            backup_path = os.path.join(checkpoint_dir, f'training_state_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            os.rename(state_path, backup_path)
            print(f"学習状態をバックアップ: {backup_path}")
            
            # 新しい状態を作成（イテレーションは継続）
            with open(backup_path, 'r') as f:
                old_state = json.load(f)
            
            new_state = {
                "iteration": old_state.get("iteration", 0),  # イテレーションは継続
                "total_games": 0,  # ゲーム数はリセット
                "total_examples": 0,  # サンプル数はリセット
                "policy_loss_history": [],
                "value_loss_history": [],
                "win_stats": {"BLACK": 0, "WHITE": 0, "DRAW": 0},
                "start_time": datetime.now().isoformat(),
                "last_update_time": datetime.now().isoformat()
            }
            
            with open(state_path, 'w') as f:
                json.dump(new_state, f, indent=2)
            print("学習状態をリセットしました（イテレーションは継続）")
        else:
            print("学習状態が見つかりません")
    
    if reset_model:
        model_path = os.path.join(checkpoint_dir, 'latest.pt')
        if os.path.exists(model_path):
            # バックアップ
            backup_path = os.path.join(checkpoint_dir, f'latest_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
            os.rename(model_path, backup_path)
            print(f"モデルをバックアップ: {backup_path}")
            print("注意: モデルがリセットされました。新しいモデルを作成する必要があります。")
        else:
            print("モデルが見つかりません")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='学習状態をリセット')
    parser.add_argument('--reset-model', action='store_true', help='モデルの重みもリセット')
    parser.add_argument('--keep-buffer', action='store_true', help='リプレイバッファを保持')
    parser.add_argument('--keep-state', action='store_true', help='学習状態を保持')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='チェックポイントディレクトリ')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("学習状態リセットツール")
    print("=" * 50)
    print(f"チェックポイント: {args.checkpoint_dir}")
    print(f"モデルリセット: {args.reset_model}")
    print(f"バッファリセット: {not args.keep_buffer}")
    print(f"状態リセット: {not args.keep_state}")
    print("=" * 50)
    
    confirm = input("続行しますか？ (y/N): ")
    if confirm.lower() == 'y':
        reset_training(
            checkpoint_dir=args.checkpoint_dir,
            reset_model=args.reset_model,
            reset_buffer=not args.keep_buffer,
            reset_state=not args.keep_state
        )
        print("\n完了しました")
    else:
        print("キャンセルしました")
