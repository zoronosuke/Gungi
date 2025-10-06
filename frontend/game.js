/**
 * 軍儀 フロントエンド JavaScript
 */

const API_BASE_URL = 'http://localhost:8001';

// ゲーム状態
let gameState = {
    gameId: null,
    board: null,
    currentPlayer: 'BLACK',
    moveCount: 0,
    selectedPiece: null,
    legalMoves: []
};

// 駒の表示名
const PIECE_NAMES = {
    SUI: '帥',
    DAI: '大',
    CHUU: '中',
    SHO: '小',
    SAMURAI: '侍',
    HYO: '兵',
    UMA: '馬',
    SHINOBI: '忍',
    YARI: '槍',
    TORIDE: '砦',
    YUMI: '弓',
    TSUTU: '筒',
    HOU: '砲',
    BOU: '謀'
};

/**
 * 初期化
 */
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    startNewGame();
});

/**
 * イベントリスナーの設定
 */
function setupEventListeners() {
    document.getElementById('new-game-btn').addEventListener('click', startNewGame);
    document.getElementById('ai-move-btn').addEventListener('click', requestAIMove);
}

/**
 * 新しいゲームを開始
 */
async function startNewGame() {
    try {
        showMessage('新しいゲームを開始しています...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/new_game`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error('ゲームの開始に失敗しました');
        }
        
        const data = await response.json();
        gameState.gameId = data.game_id;
        updateGameState(data.game_state);
        showMessage('ゲームを開始しました！', 'success');
        
    } catch (error) {
        console.error('Error starting new game:', error);
        showMessage('エラー: ' + error.message, 'error');
    }
}

/**
 * ゲーム状態を更新
 */
function updateGameState(state) {
    gameState.board = state.board;
    gameState.currentPlayer = state.current_player;
    gameState.moveCount = state.move_count;
    
    renderBoard(state.board);
    renderHandPieces(state.hand_pieces);
    updateGameInfo();
    
    if (state.game_over) {
        const winner = state.winner === 'BLACK' ? '先手（黒）' : '後手（白）';
        showMessage(`ゲーム終了！${winner}の勝利です！`, 'success');
    }
}

/**
 * 盤面を描画
 */
function renderBoard(boardData) {
    const boardElement = document.getElementById('board');
    boardElement.innerHTML = '';
    
    // 列ラベル（右から左に1-9）
    const colLabels = document.createElement('div');
    colLabels.className = 'col-labels';
    colLabels.innerHTML = '<span></span>' + Array.from({length: 9}, (_, i) => 
        `<span>${9 - i}</span>`
    ).join('');
    boardElement.appendChild(colLabels);
    
    // 各行
    for (let row = 0; row < 9; row++) {
        const rowElement = document.createElement('div');
        rowElement.className = 'board-row';
        
        // 行ラベル
        const rowLabel = document.createElement('span');
        rowLabel.className = 'row-label';
        rowLabel.textContent = String.fromCharCode(65 + row); // A-I
        rowElement.appendChild(rowLabel);
        
        // 各マス（右から左に表示するため逆順）
        for (let col = 8; col >= 0; col--) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.row = row;
            cell.dataset.col = col;
            
            // スタック（駒の重なり）を表示（下から上へ）
            const stack = boardData.board[row][col];
            if (stack && stack.length > 0) {
                // スタックを逆順にして、一番上の駒（最後に追加された駒）が上に表示される
                stack.slice().reverse().forEach((piece, displayIndex) => {
                    const level = stack.length - 1 - displayIndex;
                    const pieceElement = document.createElement('div');
                    pieceElement.className = `piece ${piece.owner.toLowerCase()} level-${displayIndex}`;
                    pieceElement.textContent = PIECE_NAMES[piece.type] || piece.type;
                    pieceElement.dataset.owner = piece.owner;
                    cell.appendChild(pieceElement);
                });
                
                // スタックレベルのインジケーターを追加
                if (stack.length > 1) {
                    const stackIndicator = document.createElement('div');
                    stackIndicator.className = 'stack-indicator';
                    stackIndicator.textContent = stack.length;
                    cell.appendChild(stackIndicator);
                }
            }
            
            // クリックイベント
            cell.addEventListener('click', () => handleCellClick(row, col));
            
            rowElement.appendChild(cell);
        }
        
        boardElement.appendChild(rowElement);
    }
}

/**
 * 持ち駒を描画
 */
function renderHandPieces(handPieces) {
    const blackPieces = handPieces.BLACK || {};
    const whitePieces = handPieces.WHITE || {};
    
    const blackElement = document.getElementById('hand-black-pieces');
    const whiteElement = document.getElementById('hand-white-pieces');
    
    blackElement.innerHTML = formatHandPieces(blackPieces);
    whiteElement.innerHTML = formatHandPieces(whitePieces);
}

/**
 * 持ち駒を整形
 */
function formatHandPieces(pieces) {
    const entries = Object.entries(pieces).filter(([_, count]) => count > 0);
    
    if (entries.length === 0) {
        return '<span class="no-pieces">なし</span>';
    }
    
    return entries.map(([type, count]) => {
        const name = PIECE_NAMES[type] || type;
        return `<span class="hand-piece">${name} × ${count}</span>`;
    }).join('');
}

/**
 * ゲーム情報を更新
 */
function updateGameInfo() {
    const playerText = gameState.currentPlayer === 'BLACK' ? '先手（黒）' : '後手（白）';
    document.getElementById('current-player').textContent = `${playerText}の番`;
    document.getElementById('move-count').textContent = `手数: ${gameState.moveCount}`;
}

/**
 * マスのクリック処理
 */
async function handleCellClick(row, col) {
    if (!gameState.gameId) {
        showMessage('先にゲームを開始してください', 'warning');
        return;
    }
    
    const topPiece = getTopPiece(row, col);
    
    if (gameState.selectedPiece) {
        // 駒が選択されている場合 → 合法手かチェックしてから移動
        const clickedCell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
        if (clickedCell && clickedCell.classList.contains('legal-move')) {
            attemptMove(gameState.selectedPiece.row, gameState.selectedPiece.col, row, col);
        } else {
            showMessage('そこには移動できません', 'warning');
        }
        clearSelection();
    } else if (topPiece && topPiece.owner === gameState.currentPlayer) {
        // 自分の駒を選択
        await selectPiece(row, col);
    }
}

/**
 * 指定位置の一番上の駒を取得
 */
function getTopPiece(row, col) {
    if (!gameState.board) return null;
    const stack = gameState.board.board[row][col];
    if (!stack || stack.length === 0) return null;
    return stack[stack.length - 1];
}

/**
 * 駒を選択
 */
async function selectPiece(row, col) {
    gameState.selectedPiece = { row, col };
    
    // 選択状態を視覚化
    document.querySelectorAll('.cell').forEach(cell => {
        cell.classList.remove('selected', 'legal-move');
    });
    
    const cell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
    if (cell) {
        cell.classList.add('selected');
    }
    
    // 合法手を取得して表示
    await fetchAndDisplayLegalMoves(row, col);
}

/**
 * 選択をクリア
 */
function clearSelection() {
    gameState.selectedPiece = null;
    document.querySelectorAll('.cell').forEach(cell => {
        cell.classList.remove('selected', 'legal-move');
    });
}

/**
 * 合法手を取得して表示
 */
async function fetchAndDisplayLegalMoves(fromRow, fromCol) {
    try {
        const response = await fetch(`${API_BASE_URL}/get_legal_moves/${gameState.gameId}`);
        const data = await response.json();
        
        if (data.legal_moves) {
            // この駒から移動できるマスをフィルタリング
            // APIレスポンスは "from" と "to" というキーを使用
            const movesFromThisPiece = data.legal_moves.filter(move => {
                return move.from && 
                       move.from[0] === fromRow && 
                       move.from[1] === fromCol;
            });
            
            console.log(`移動可能な手: ${movesFromThisPiece.length}個`);
            
            // 移動可能なマスをハイライト
            movesFromThisPiece.forEach(move => {
                if (move.to) {
                    const [toRow, toCol] = move.to;
                    const targetCell = document.querySelector(`[data-row="${toRow}"][data-col="${toCol}"]`);
                    if (targetCell) {
                        targetCell.classList.add('legal-move');
                        console.log(`ハイライト追加: (${toRow}, ${toCol})`);
                    }
                }
            });
        }
    } catch (error) {
        console.error('合法手の取得に失敗:', error);
    }
}

/**
 * 手を試みる
 */
async function attemptMove(fromRow, fromCol, toRow, toCol) {
    try {
        const moveData = {
            from_row: fromRow,
            from_col: fromCol,
            to_row: toRow,
            to_col: toCol,
            move_type: 'NORMAL'
        };
        
        const response = await fetch(`${API_BASE_URL}/apply_move/${gameState.gameId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(moveData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            updateGameState(data.game_state);
            showMessage('手を適用しました', 'success');
        } else {
            showMessage('無効な手です: ' + data.message, 'warning');
        }
        
    } catch (error) {
        console.error('Error applying move:', error);
        showMessage('エラー: ' + error.message, 'error');
    }
}

/**
 * AIに手を打たせる
 */
async function requestAIMove() {
    if (!gameState.gameId) {
        showMessage('先にゲームを開始してください', 'warning');
        return;
    }
    
    try {
        showMessage('AIが考えています...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/predict/${gameState.gameId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ game_id: gameState.gameId, depth: 1 })
        });
        
        if (!response.ok) {
            throw new Error('AI応答の取得に失敗しました');
        }
        
        const data = await response.json();
        const move = data.move;
        
        // AIの手を適用
        if (move.from) {
            await attemptMove(move.from[0], move.from[1], move.to[0], move.to[1]);
        }
        
    } catch (error) {
        console.error('Error requesting AI move:', error);
        showMessage('エラー: ' + error.message, 'error');
    }
}

/**
 * メッセージを表示
 */
function showMessage(message, type = 'info') {
    const messageElement = document.getElementById('status-message');
    messageElement.textContent = message;
    messageElement.className = `status-message ${type}`;
    messageElement.style.display = 'block';
    
    setTimeout(() => {
        messageElement.style.display = 'none';
    }, 3000);
}
