/**
 * 軍儀 フロントエンド JavaScript
 */

// API URLを環境に応じて設定
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8003'
    : window.location.origin;  // 本番環境では同じオリジンを使用

// ゲーム状態
let gameState = {
    gameId: null,
    board: null,
    currentPlayer: 'BLACK',
    moveCount: 0,
    selectedPiece: null,
    selectedHandPiece: null,  // 選択された持ち駒
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
    console.log('updateGameState called with:', state);
    gameState.board = state;  // 全体の状態を保存
    gameState.currentPlayer = state.current_player;
    gameState.moveCount = state.move_count;
    
    // state.boardは盤面オブジェクトで、state.board.boardが実際の盤面配列
    renderBoard(state.board);
    renderHandPieces(state.hand_pieces);
    updateGameInfo();
    
    // 選択をクリア
    clearSelection();
    
    if (state.game_over) {
        const winner = state.winner === 'BLACK' ? '先手（黒）' : '後手（白）';
        showMessage(`ゲーム終了！${winner}の勝利です！`, 'success');
    }
}

/**
 * 盤面を描画
 */
function renderBoard(boardData) {
    console.log('renderBoard called with:', boardData);
    const boardElement = document.getElementById('board');
    boardElement.innerHTML = '';
    
    // 列ラベル（9-1、右から左）
    const colLabels = document.createElement('div');
    colLabels.className = 'col-labels';
    colLabels.innerHTML = '<span></span>' + Array.from({length: 9}, (_, i) => 
        `<span>${9 - i}</span>`
    ).join('') + '<span></span>'; // 右側にも空白を追加
    boardElement.appendChild(colLabels);
    
    // boardDataの構造: {board: Array(9), sui_positions: {...}}
    // board配列を取得
    const actualBoard = boardData.board || boardData;
    console.log('actualBoard:', actualBoard);
    
    // 各行
    for (let row = 0; row < 9; row++) {
        const rowElement = document.createElement('div');
        rowElement.className = 'board-row';
        
        // 各マス（左から右に表示）
        for (let col = 0; col < 9; col++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.row = row;
            cell.dataset.col = col;
            
            // スタック（駒の重なり）を表示（下から上へ）
            const stack = actualBoard[row][col];
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
        
        // 行ラベル（右側）
        const rowLabelRight = document.createElement('span');
        rowLabelRight.className = 'row-label row-label-right';
        rowLabelRight.textContent = String.fromCharCode(65 + row); // A-I
        rowElement.appendChild(rowLabelRight);
        
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
    
    blackElement.innerHTML = formatHandPieces(blackPieces, 'BLACK');
    whiteElement.innerHTML = formatHandPieces(whitePieces, 'WHITE');
}

/**
 * 持ち駒を整形
 */
function formatHandPieces(pieces, player) {
    const entries = Object.entries(pieces).filter(([_, count]) => count > 0);
    
    if (entries.length === 0) {
        return '<span class="no-pieces">なし</span>';
    }
    
    return entries.map(([type, count]) => {
        const name = PIECE_NAMES[type] || type;
        const isClickable = player === gameState.currentPlayer && gameState.selectedHandPiece !== type;
        const isSelected = gameState.selectedHandPiece === type;
        const className = `hand-piece ${isClickable ? 'clickable' : ''} ${isSelected ? 'selected' : ''}`;
        return `<span class="${className}" data-piece-type="${type}" data-player="${player}" onclick="handleHandPieceClick('${type}', '${player}')">${name} × ${count}</span>`;
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
 * 持ち駒のクリック処理（「新」を実行）
 */
async function handleHandPieceClick(pieceType, player) {
    console.log(`持ち駒クリック: ${pieceType} (${player})`);
    
    if (!gameState.gameId) {
        showMessage('先にゲームを開始してください', 'warning');
        return;
    }
    
    if (player !== gameState.currentPlayer) {
        showMessage('相手の持ち駒は使えません', 'warning');
        return;
    }
    
    // 既に同じ駒が選択されている場合は選択解除
    if (gameState.selectedHandPiece === pieceType) {
        gameState.selectedHandPiece = null;
        clearSelection();
        renderHandPieces(gameState.board ? {
            BLACK: gameState.board.hand_pieces?.BLACK || {},
            WHITE: gameState.board.hand_pieces?.WHITE || {}
        } : {BLACK: {}, WHITE: {}});
        showMessage('持ち駒の選択を解除しました', 'info');
        return;
    }
    
    // 盤上の駒の選択を解除
    clearSelection();
    
    // 持ち駒を選択
    gameState.selectedHandPiece = pieceType;
    
    // 合法手を取得して配置可能な場所を表示
    await fetchAndDisplayDropMoves(pieceType);
    
    // 持ち駒の表示を更新
    renderHandPieces(gameState.board ? {
        BLACK: gameState.board.hand_pieces?.BLACK || {},
        WHITE: gameState.board.hand_pieces?.WHITE || {}
    } : {BLACK: {}, WHITE: {}});
    
    showMessage(`${PIECE_NAMES[pieceType]}を選択しました。配置する場所をクリックしてください`, 'info');
}

/**
 * 持ち駒の配置可能な場所を取得して表示
 */
async function fetchAndDisplayDropMoves(pieceType) {
    try {
        const response = await fetch(`${API_BASE_URL}/get_legal_moves/${gameState.gameId}`);
        const data = await response.json();
        
        if (!data.legal_moves) {
            console.error('合法手が取得できませんでした');
            return;
        }
        
        // DROPタイプで指定された駒種類の手のみをフィルタ
        const dropMoves = data.legal_moves.filter(move => 
            move.type === 'DROP' && move.piece_type === pieceType
        );
        
        console.log(`${pieceType}の配置可能な場所: ${dropMoves.length}箇所`);
        
        // 配置可能な場所をハイライト
        dropMoves.forEach(move => {
            const [row, col] = move.to;
            const cell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
            if (cell) {
                cell.classList.add('legal-move');
            }
        });
        
    } catch (error) {
        console.error('Error fetching drop moves:', error);
        showMessage('エラー: 配置可能な場所の取得に失敗しました', 'error');
    }
}

/**
 * マスのクリック処理
 */
async function handleCellClick(row, col) {
    console.log(`クリック: (${row}, ${col})`);
    console.log('gameState:', gameState);
    
    if (!gameState.gameId) {
        showMessage('先にゲームを開始してください', 'warning');
        return;
    }
    
    // 持ち駒が選択されている場合は「新」を実行
    if (gameState.selectedHandPiece) {
        const clickedCell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
        if (clickedCell && clickedCell.classList.contains('legal-move')) {
            await attemptDrop(gameState.selectedHandPiece, row, col);
            gameState.selectedHandPiece = null;
            clearSelection();
        } else {
            showMessage('そこには配置できません', 'warning');
        }
        return;
    }
    
    const topPiece = getTopPiece(row, col);
    
    if (gameState.selectedPiece) {
        // 駒が選択されている場合 → 合法手かチェックしてから移動
        const clickedCell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
        if (clickedCell && clickedCell.classList.contains('legal-move')) {
            // 目的地の駒と可能な手の種類をチェック
            const possibleMoves = getPossibleMovesForPosition(gameState.selectedPiece.row, gameState.selectedPiece.col, row, col);
            
            console.log(`可能な手の数: ${possibleMoves.length}`, possibleMoves);
            
            if (possibleMoves.length === 1) {
                // 1つだけの選択肢の場合、そのまま実行
                const fromRow = gameState.selectedPiece.row;
                const fromCol = gameState.selectedPiece.col;
                const moveType = possibleMoves[0].type;
                clearSelection();
                await attemptMove(fromRow, fromCol, row, col, moveType);
            } else if (possibleMoves.length > 1) {
                // 複数の選択肢がある場合、ユーザーに選択させる
                showMoveTypeSelection(gameState.selectedPiece.row, gameState.selectedPiece.col, row, col, possibleMoves);
                return; // clearSelectionは選択後に呼ばれる
            } else {
                showMessage('そこには移動できません', 'warning');
                clearSelection();
            }
        } else {
            // 選択解除または別の駒を選択
            if (topPiece && topPiece.owner === gameState.currentPlayer) {
                // 別の自分の駒を選択
                await selectPiece(row, col);
            } else {
                clearSelection();
            }
        }
    } else if (topPiece && topPiece.owner === gameState.currentPlayer) {
        // 自分の駒を選択
        await selectPiece(row, col);
    }
}

/**
 * 指定位置の一番上の駒を取得
 */
function getTopPiece(row, col) {
    console.log('getTopPiece called:', row, col);
    if (!gameState.board) {
        console.log('gameState.board is null');
        return null;
    }
    
    // gameState.boardは全体の状態オブジェクト
    // gameState.board.board.boardが実際の盤面配列
    let boardArray;
    if (gameState.board.board && gameState.board.board.board) {
        boardArray = gameState.board.board.board;
    } else if (gameState.board.board && Array.isArray(gameState.board.board)) {
        boardArray = gameState.board.board;
    } else if (Array.isArray(gameState.board)) {
        boardArray = gameState.board;
    } else {
        console.error('Invalid board structure:', gameState.board);
        return null;
    }
    
    const stack = boardArray[row][col];
    console.log('stack at', row, col, ':', stack);
    if (!stack || stack.length === 0) {
        console.log('No piece at this position');
        return null;
    }
    const topPiece = stack[stack.length - 1];
    console.log('topPiece:', topPiece);
    return topPiece;
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
    gameState.selectedHandPiece = null;
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
            
            // 合法手を保存（後で使用）
            gameState.legalMoves = movesFromThisPiece;
            
            console.log(`移動可能な手: ${movesFromThisPiece.length}個`);
            
            // 移動可能なマスをハイライト
            movesFromThisPiece.forEach(move => {
                if (move.to) {
                    const [toRow, toCol] = move.to;
                    const targetCell = document.querySelector(`[data-row="${toRow}"][data-col="${toCol}"]`);
                    if (targetCell) {
                        targetCell.classList.add('legal-move');
                        console.log(`ハイライト追加: (${toRow}, ${toCol}), タイプ: ${move.type}`);
                    }
                }
            });
        }
    } catch (error) {
        console.error('合法手の取得に失敗:', error);
    }
}

/**
 * 指定位置への可能な手の種類を取得
 */
function getPossibleMovesForPosition(fromRow, fromCol, toRow, toCol) {
    const moves = gameState.legalMoves.filter(move => {
        return move.from && 
               move.from[0] === fromRow && 
               move.from[1] === fromCol &&
               move.to &&
               move.to[0] === toRow && 
               move.to[1] === toCol;
    });
    return moves;
}

/**
 * 手の種類選択UIを表示
 */
function showMoveTypeSelection(fromRow, fromCol, toRow, toCol, possibleMoves) {
    // モーダルを作成
    const modal = document.createElement('div');
    modal.className = 'move-type-modal';
    modal.innerHTML = `
        <div class="move-type-content">
            <h3>手の種類を選択してください</h3>
            <div class="move-type-buttons">
                ${possibleMoves.map(move => {
                    let label = '';
                    if (move.type === 'CAPTURE') label = '駒を取る';
                    else if (move.type === 'STACK') label = '駒を重ねる（ツケ）';
                    else if (move.type === 'NORMAL') label = '通常移動';
                    
                    return `<button class="move-type-btn" data-type="${move.type}">${label}</button>`;
                }).join('')}
                <button class="move-type-btn cancel-btn">キャンセル</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // ボタンのイベントリスナー
    modal.querySelectorAll('.move-type-btn:not(.cancel-btn)').forEach(btn => {
        btn.addEventListener('click', async () => {
            const moveType = btn.dataset.type;
            document.body.removeChild(modal);
            clearSelection();
            await attemptMove(fromRow, fromCol, toRow, toCol, moveType);
        });
    });
    
    modal.querySelector('.cancel-btn').addEventListener('click', () => {
        document.body.removeChild(modal);
        clearSelection();
    });
}

/**
 * 手を試みる
 */
async function attemptMove(fromRow, fromCol, toRow, toCol, moveType = 'NORMAL') {
    try {
        console.log(`手を試みる: (${fromRow}, ${fromCol}) -> (${toRow}, ${toCol}), タイプ: ${moveType}`);
        
        const moveData = {
            from_row: fromRow,
            from_col: fromCol,
            to_row: toRow,
            to_col: toCol,
            move_type: moveType
        };
        
        console.log('送信データ:', moveData);
        
        const response = await fetch(`${API_BASE_URL}/apply_move/${gameState.gameId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(moveData)
        });
        
        console.log('レスポンスステータス:', response.status);
        
        const data = await response.json();
        console.log('レスポンスデータ:', data);
        
        if (data.success) {
            updateGameState(data.game_state);
            showMessage('手を適用しました', 'success');
        } else {
            showMessage('無効な手です: ' + (data.message || ''), 'warning');
        }
        
    } catch (error) {
        console.error('Error applying move:', error);
        showMessage('エラー: ' + error.message, 'error');
    }
}

/**
 * 持ち駒を配置する（「新」を実行）
 */
async function attemptDrop(pieceType, toRow, toCol) {
    try {
        showMessage(`${PIECE_NAMES[pieceType]}を(${toRow}, ${toCol})に配置中...`, 'info');
        
        const moveData = {
            move_type: 'DROP',
            piece_type: pieceType,
            to_row: toRow,
            to_col: toCol,
            from_row: null,
            from_col: null
        };
        
        console.log('「新」送信データ:', moveData);
        
        const response = await fetch(`${API_BASE_URL}/apply_move/${gameState.gameId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(moveData)
        });
        
        console.log('レスポンスステータス:', response.status);
        
        const data = await response.json();
        console.log('レスポンスデータ:', data);
        
        if (data.success) {
            updateGameState(data.game_state);
            showMessage(`${PIECE_NAMES[pieceType]}を配置しました`, 'success');
        } else {
            showMessage('無効な配置です: ' + (data.message || ''), 'warning');
        }
        
    } catch (error) {
        console.error('Error dropping piece:', error);
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
