"""
「新」（持ち駒配置）のテストスクリプト
"""

import requests
import json

API_BASE_URL = 'http://localhost:8003'

def create_new_game():
    """新しいゲームを作成"""
    response = requests.post(f'{API_BASE_URL}/new_game')
    response.raise_for_status()
    return response.json()

def get_game_state(game_id):
    """ゲーム状態を取得"""
    response = requests.get(f'{API_BASE_URL}/get_game/{game_id}')
    response.raise_for_status()
    return response.json()

def apply_move(game_id, move_data):
    """手を適用"""
    response = requests.post(
        f'{API_BASE_URL}/apply_move/{game_id}',
        json=move_data
    )
    response.raise_for_status()
    return response.json()

def get_legal_moves(game_id):
    """合法手を取得"""
    response = requests.get(f'{API_BASE_URL}/get_legal_moves/{game_id}')
    response.raise_for_status()
    return response.json()

def print_hand_pieces(hand_pieces):
    """持ち駒を表示"""
    piece_names = {
        'HYO': '兵', 'YARI': '槍', 'UMA': '馬', 'SHINOBI': '忍',
        'SAMURAI': '侍', 'SHO': '小', 'TORIDE': '砦', 'YUMI': '弓',
        'TSUTU': '筒', 'HOU': '砲', 'BOU': '謀'
    }
    
    for piece_type, count in hand_pieces.items():
        if count > 0:
            name = piece_names.get(piece_type, piece_type)
            print(f"  {name}({piece_type}): {count}枚")

def test_drop_piece():
    """「新」のテスト"""
    print("=" * 60)
    print("「新」（持ち駒配置）のテスト")
    print("=" * 60)
    
    # 1. ゲーム作成
    print("\n1. 新しいゲームを作成...")
    game_data = create_new_game()
    game_id = game_data['game_id']
    print(f"   ゲームID: {game_id}")
    
    # 2. 初期状態の確認
    print("\n2. 初期状態を確認...")
    state = get_game_state(game_id)
    current_player = state['current_player']
    print(f"   現在のプレイヤー: {current_player}")
    print(f"   {current_player}の持ち駒:")
    print_hand_pieces(state['hand_pieces'][current_player])
    
    # 3. 合法手を取得
    print("\n3. 合法手を取得...")
    legal_moves_data = get_legal_moves(game_id)
    legal_moves = legal_moves_data['legal_moves']
    drop_moves = [m for m in legal_moves if m['type'] == 'DROP']
    print(f"   合法手の数: {len(legal_moves)}")
    print(f"   「新」の手の数: {len(drop_moves)}")
    
    if drop_moves:
        print(f"\n   最初の5つの「新」の手:")
        for i, move in enumerate(drop_moves[:5]):
            piece_type = move.get('piece_type')
            to_pos = move.get('to')
            print(f"   {i+1}. {piece_type} -> {to_pos}")
    
    # 4. 持ち駒がある場合、「新」を実行
    hand_pieces = state['hand_pieces'][current_player]
    available_pieces = [pt for pt, count in hand_pieces.items() if count > 0]
    
    if available_pieces and drop_moves:
        # 最初の持ち駒を使用
        piece_to_drop = available_pieces[0]
        drop_move = next((m for m in drop_moves if m.get('piece_type') == piece_to_drop), None)
        
        if drop_move:
            print(f"\n4. 「新」を実行: {piece_to_drop} を {drop_move['to']} に配置...")
            
            move_data = {
                'move_type': 'DROP',
                'piece_type': piece_to_drop,
                'to_row': drop_move['to'][0],
                'to_col': drop_move['to'][1],
                'from_row': None,
                'from_col': None
            }
            
            result = apply_move(game_id, move_data)
            
            if result['success']:
                print("   ✅ 成功！")
                print(f"   メッセージ: {result['message']}")
                
                # 更新後の持ち駒を表示
                new_state = result['game_state']
                # 手番が交代しているので、相手の持ち駒を表示
                prev_player = 'WHITE' if current_player == 'BLACK' else 'BLACK'
                print(f"\n   {prev_player}の残り持ち駒:")
                print_hand_pieces(new_state['hand_pieces'][prev_player])
            else:
                print(f"   ❌ 失敗: {result['message']}")
        else:
            print(f"\n4. 「新」の合法手が見つかりませんでした")
    else:
        print("\n4. 持ち駒がないため、「新」はスキップします")
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)

def test_drop_piece_manual():
    """手動で「新」をテスト"""
    print("=" * 60)
    print("手動で「新」をテスト")
    print("=" * 60)
    
    # ゲームを作成
    game_data = create_new_game()
    game_id = game_data['game_id']
    print(f"\nゲームID: {game_id}")
    
    # 初期状態を表示
    state = get_game_state(game_id)
    current_player = state['current_player']
    print(f"現在のプレイヤー: {current_player}")
    print(f"{current_player}の持ち駒:")
    print_hand_pieces(state['hand_pieces'][current_player])
    
    # 「新」を試す
    print("\n兵(HYO)を (7, 5) に配置してみます...")
    
    move_data = {
        'move_type': 'DROP',
        'piece_type': 'HYO',
        'to_row': 7,
        'to_col': 5,
        'from_row': None,
        'from_col': None
    }
    
    try:
        result = apply_move(game_id, move_data)
        
        if result['success']:
            print("✅ 成功！")
            print(f"メッセージ: {result['message']}")
        else:
            print(f"❌ 失敗: {result['message']}")
    except Exception as e:
        print(f"❌ エラー: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        print("\nサーバーに接続中...")
        response = requests.get(f'{API_BASE_URL}/')
        print("✅ サーバーが起動しています\n")
        
        # テストを実行
        test_drop_piece()
        
        print("\n" + "=" * 60)
        print("追加テスト: 手動で「新」を試す")
        print("=" * 60)
        test_drop_piece_manual()
        
    except requests.exceptions.ConnectionError:
        print("❌ エラー: サーバーに接続できません")
        print("   python run_server.py でサーバーを起動してください")
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
