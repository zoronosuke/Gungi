"""
軍儀AI用のニューラルネットワークモデル
AlphaZero型の方策+価値ネットワーク
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResidualBlock(nn.Module):
    """
    ResNetの残差ブロック
    
    構造:
    入力 ─┬─→ Conv3×3 → BN → ReLU → Conv3×3 → BN ─┬─→ ReLU → 出力
          │                                        │
          └────────────────────────────────────────┘
    """
    
    def __init__(self, channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class GungiNetwork(nn.Module):
    """
    軍儀用のニューラルネットワーク（AlphaZero型）
    
    入力: 盤面の状態 (batch, 91, 9, 9)
    出力:
        - policy: 各手のlog確率分布 (batch, 7695)
        - value: 局面の評価値 (batch, 1), -1〜1
    
    構造:
        入力 → Conv → ResBlock×N → PolicyHead / ValueHead
    """
    
    def __init__(
        self,
        input_channels: int = 91,   # 状態エンコードのチャンネル数
        num_res_blocks: int = 4,    # 15時間テスト用は4ブロック（本番は8）
        num_filters: int = 128,     # フィルター数
        board_size: int = 9,
        num_actions: int = 7695     # 移動6561 + DROP1134
    ):
        super().__init__()
        
        self.board_size = board_size
        self.num_actions = num_actions
        self.input_channels = input_channels
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        
        # 入力層
        self.input_conv = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)
        
        # ResNetブロック
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy Head（方策）
        # Conv → BN → ReLU → Flatten → Linear → LogSoftmax
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, num_actions)
        
        # Value Head（価値）
        # Conv → BN → ReLU → Flatten → Linear → ReLU → Linear → Tanh
        self.value_conv = nn.Conv2d(num_filters, 4, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播
        
        Args:
            x: 盤面の状態テンソル (batch, 91, 9, 9)
        
        Returns:
            policy: 方策のlog確率分布 (batch, 7695)
            value: 価値の評価 (batch, 1)
        """
        # 入力層
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # ResNetブロック
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy Head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value Head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(
        self,
        state: torch.Tensor,
        legal_mask: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        推論用メソッド（合法手マスク適用済み）
        
        Args:
            state: (1, 91, 9, 9) 状態テンソル
            legal_mask: (7695,) 合法手マスク
        
        Returns:
            policy: (7695,) 合法手のみに確率を割り当て
            value: float 評価値
        """
        self.eval()
        with torch.no_grad():
            log_policy, value = self.forward(state)
            
            # log確率を確率に変換
            policy = torch.exp(log_policy).cpu().numpy()[0]
            
            # 合法手マスクを適用
            policy = policy * legal_mask
            
            # 正規化
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                # 合法手がない場合は均等分布（通常は起きない）
                num_legal = legal_mask.sum()
                if num_legal > 0:
                    policy = legal_mask / num_legal
            
            return policy, value.item()


def create_model(device: str = 'cpu', test_mode: bool = True) -> GungiNetwork:
    """
    モデルを作成して初期化
    
    Args:
        device: 'cpu' or 'cuda'
        test_mode: True=15時間テスト用（軽量）、False=本番用
    
    Returns:
        初期化されたGungiNetwork
    """
    if test_mode:
        # 15時間テスト用: 軽量版
        model = GungiNetwork(
            input_channels=91,
            num_res_blocks=4,
            num_filters=128,
            board_size=9,
            num_actions=7695
        )
    else:
        # 本番用
        model = GungiNetwork(
            input_channels=91,
            num_res_blocks=8,
            num_filters=128,
            board_size=9,
            num_actions=7695
        )
    
    model = model.to(device)
    return model


if __name__ == "__main__":
    # テスト
    print("=== GungiNetwork Test ===")
    model = create_model('cpu', test_mode=True)
    
    # ダミー入力
    dummy_input = torch.randn(1, 91, 9, 9)
    policy, value = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Value: {value.item():.4f}")
    
    # predictメソッドのテスト
    legal_mask = np.zeros(7695, dtype=np.float32)
    legal_mask[100:110] = 1.0  # ダミーの合法手
    
    policy_masked, value_scalar = model.predict(dummy_input, legal_mask)
    print(f"\nPredicted policy sum: {policy_masked.sum():.4f}")
    print(f"Predicted value: {value_scalar:.4f}")
    print(f"Non-zero policy entries: {np.sum(policy_masked > 0)}")
