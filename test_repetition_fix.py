"""
修正後のself-playをテスト（3ゲームだけ）
AlphaZero/将棋AIスタイルの改善を確認
"""
import sys
sys.path.insert(0, '.')

import torch
from src.model.network import create_model
from src.model.encoder import StateEncoder, ActionEncoder
from src.model.max_efficiency_selfplay import MaxEfficiencySelfPlay

print("Testing self-play with AlphaZero-style improvements...")
print(f"  - Dirichlet noise: alpha={MaxEfficiencySelfPlay.DIRICHLET_ALPHA}, eps={MaxEfficiencySelfPlay.DIRICHLET_EPSILON}")
print(f"  - Draw value: {MaxEfficiencySelfPlay.DRAW_VALUE}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

network = create_model()

# チェックポイント読み込み
try:
    checkpoint = torch.load('checkpoints/latest.pt', map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        network.load_state_dict(checkpoint['model_state_dict'])
    else:
        network.load_state_dict(checkpoint)
    print("Model loaded from checkpoint")
except Exception as e:
    print(f"No checkpoint loaded: {e}")

network.to(device)
network.eval()

# Self-playを実行（5ゲーム）
self_play = MaxEfficiencySelfPlay(
    network=network,
    mcts_simulations=50,  # テスト用に少なめ
    device=device,
    num_parallel_games=5,
    use_dirichlet_noise=True
)

print("\nRunning 5 test games...")
examples = self_play.generate_data(num_games=5, temperature_threshold=30, verbose=True)
print(f"\nTotal examples: {len(examples)}")
print(f"Average examples per game: {len(examples) / 5:.1f}")
