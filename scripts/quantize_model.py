"""
モデル量子化スクリプト
学習済みモデルをINT8量子化してサイズと推論速度を改善
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.network import create_model


def quantize_model(input_path: str, output_path: str, test_mode: bool = True):
    """
    モデルを動的量子化する
    
    Args:
        input_path: 入力チェックポイントのパス
        output_path: 出力先のパス
        test_mode: テストモード（小さいモデル）を使うか
    """
    print(f"Loading model from: {input_path}")
    
    # デバイスはCPU（量子化はCPUで行う）
    device = 'cpu'
    
    # モデル作成
    model = create_model(device, test_mode=test_mode)
    
    # チェックポイント読み込み
    checkpoint = torch.load(input_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        iteration = checkpoint.get('iteration', 'unknown')
        total_games = checkpoint.get('total_games', 'unknown')
    else:
        model.load_state_dict(checkpoint)
        iteration = 'unknown'
        total_games = 'unknown'
    
    model.eval()
    
    print(f"Model loaded (iteration: {iteration}, games: {total_games})")
    
    # 元のモデルサイズ
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Original model size: {original_size:.2f} MB")
    
    # 動的量子化（Linear層をINT8に）
    print("Applying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Linear層を量子化
        dtype=torch.qint8
    )
    
    # 量子化されたモデルを保存
    # 推論に必要な情報も含める
    quantized_checkpoint = {
        'model_state_dict': quantized_model.state_dict(),
        'iteration': iteration,
        'total_games': total_games,
        'quantized': True,
        'test_mode': test_mode,
    }
    
    torch.save(quantized_checkpoint, output_path)
    
    # 量子化後のサイズ
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    print(f"Saved to: {output_path}")
    
    return quantized_model


def verify_quantized_model(original_path: str, quantized_path: str, test_mode: bool = True):
    """量子化モデルの動作確認"""
    import numpy as np
    
    print("\n--- Verifying quantized model ---")
    
    device = 'cpu'
    
    # オリジナルモデル
    original_model = create_model(device, test_mode=test_mode)
    original_checkpoint = torch.load(original_path, map_location=device)
    if 'model_state_dict' in original_checkpoint:
        original_model.load_state_dict(original_checkpoint['model_state_dict'])
    else:
        original_model.load_state_dict(original_checkpoint)
    original_model.eval()
    
    # 量子化モデル
    quantized_model = create_model(device, test_mode=test_mode)
    quantized_model = torch.quantization.quantize_dynamic(
        quantized_model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    quantized_checkpoint = torch.load(quantized_path, map_location=device)
    quantized_model.load_state_dict(quantized_checkpoint['model_state_dict'])
    quantized_model.eval()
    
    # テスト入力
    batch_size = 1
    channels = 46  # StateEncoderのチャンネル数
    test_input = torch.randn(batch_size, channels, 9, 9)
    
    # 推論時間比較
    import time
    
    # ウォームアップ
    with torch.no_grad():
        _ = original_model(test_input)
        _ = quantized_model(test_input)
    
    # オリジナルモデルの推論時間
    n_runs = 100
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            orig_policy, orig_value = original_model(test_input)
    original_time = (time.time() - start) / n_runs * 1000
    
    # 量子化モデルの推論時間
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            quant_policy, quant_value = quantized_model(test_input)
    quantized_time = (time.time() - start) / n_runs * 1000
    
    print(f"Original inference time: {original_time:.2f} ms")
    print(f"Quantized inference time: {quantized_time:.2f} ms")
    print(f"Speedup: {original_time/quantized_time:.2f}x")
    
    # 出力の比較
    with torch.no_grad():
        orig_policy, orig_value = original_model(test_input)
        quant_policy, quant_value = quantized_model(test_input)
    
    policy_diff = torch.abs(orig_policy - quant_policy).mean().item()
    value_diff = torch.abs(orig_value - quant_value).mean().item()
    
    print(f"Policy difference (mean abs): {policy_diff:.6f}")
    print(f"Value difference (mean abs): {value_diff:.6f}")
    
    if policy_diff < 0.1 and value_diff < 0.1:
        print("✓ Quantization successful - outputs are similar")
    else:
        print("⚠ Warning: Large difference in outputs")


def main():
    parser = argparse.ArgumentParser(description='Quantize Gungi AI model')
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Input checkpoint path (default: checkpoints/latest.pt)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path (default: checkpoints/model_quantized.pt)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify quantized model'
    )
    parser.add_argument(
        '--no-test-mode',
        action='store_true',
        help='Use full model instead of test mode'
    )
    
    args = parser.parse_args()
    
    # デフォルトパス
    checkpoint_dir = project_root / 'checkpoints'
    
    if args.input is None:
        input_path = checkpoint_dir / 'latest.pt'
    else:
        input_path = Path(args.input)
    
    if args.output is None:
        output_path = checkpoint_dir / 'model_quantized.pt'
    else:
        output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    test_mode = not args.no_test_mode
    
    # 量子化実行
    quantize_model(str(input_path), str(output_path), test_mode=test_mode)
    
    # 検証
    if args.verify:
        verify_quantized_model(str(input_path), str(output_path), test_mode=test_mode)


if __name__ == '__main__':
    main()
