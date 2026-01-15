"""
訓練リアルタイム可視化ダッシュボード
FlaskベースのWebサーバーでtraining_stats.jsonを可視化
"""

import os
import json
import time
from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path

app = Flask(__name__, 
            static_folder='../frontend',
            template_folder='../frontend')
CORS(app)

# チェックポイントディレクトリ
CHECKPOINT_DIR = Path(__file__).parent.parent / 'checkpoints'
STATS_FILE = CHECKPOINT_DIR / 'training_stats.json'

@app.route('/')
def index():
    """ダッシュボードページ"""
    return send_from_directory('../frontend', 'training_dashboard.html')

@app.route('/api/stats')
def get_stats():
    """訓練統計をJSON形式で返す"""
    try:
        if not STATS_FILE.exists():
            return jsonify({
                'status': 'waiting',
                'message': '訓練データがまだありません...',
                'data': None
            })
        
        with open(STATS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # データを整形
        iterations = data.get('iterations', [])
        global_stats = data.get('global_stats', {})
        
        # グラフ用データを整理
        chart_data = {
            'iterations': [],
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'black_wins': [],
            'white_wins': [],
            'draws': [],
            'avg_game_length': [],
            'policy_entropy': [],
            'value_near_zero': [],
            'draw_rate': [],
            'termination_reasons': []
        }
        
        for it in iterations:
            chart_data['iterations'].append(it.get('iteration', 0))
            chart_data['policy_loss'].append(it.get('policy_loss', 0))
            chart_data['value_loss'].append(it.get('value_loss', 0))
            chart_data['total_loss'].append(it.get('policy_loss', 0) + it.get('value_loss', 0))
            chart_data['black_wins'].append(it.get('black_wins', 0))
            chart_data['white_wins'].append(it.get('white_wins', 0))
            chart_data['draws'].append(it.get('draws', 0))
            chart_data['avg_game_length'].append(it.get('avg_game_length', 0))
            chart_data['policy_entropy'].append(it.get('avg_policy_entropy', 0))
            chart_data['value_near_zero'].append(it.get('value_near_zero_ratio', 0) * 100)
            
            # 引き分け率を計算
            total_games = it.get('black_wins', 0) + it.get('white_wins', 0) + it.get('draws', 0)
            draw_rate = (it.get('draws', 0) / total_games * 100) if total_games > 0 else 0
            chart_data['draw_rate'].append(draw_rate)
            
            chart_data['termination_reasons'].append(it.get('termination_reasons', {}))
        
        return jsonify({
            'status': 'ok',
            'global_stats': global_stats,
            'chart_data': chart_data,
            'latest_iteration': iterations[-1] if iterations else None,
            'total_iterations': len(iterations)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'data': None
        })

@app.route('/api/checkpoints')
def get_checkpoints():
    """保存されたチェックポイント一覧"""
    try:
        checkpoints = []
        for f in CHECKPOINT_DIR.glob('model_iter_*.pt'):
            checkpoints.append({
                'name': f.name,
                'size': f.stat().st_size,
                'modified': f.stat().st_mtime
            })
        checkpoints.sort(key=lambda x: x['modified'], reverse=True)
        return jsonify({'checkpoints': checkpoints})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("=" * 50)
    print("🎮 Gungi AI 訓練ダッシュボード")
    print("=" * 50)
    print(f"📊 統計ファイル: {STATS_FILE}")
    print(f"🌐 ブラウザで http://localhost:5000 を開いてください")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
