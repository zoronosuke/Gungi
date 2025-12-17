/**
 * 軍儀 フロントエンド JavaScript
 */

// API URLを環境に応じて設定
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000'  // ポートを8000に固定
    : window.location.origin;  // 本番環境では同じオリジンを使用

// ゲーム状態
let gameState = {
    gameId: null,
    board: null,
    currentPlayer: 'BLACK',
    moveCount: 0,
    selectedPiece: null,
    selectedHandPiece: null,  // 選択された持ち駒
    legalMoves: [],
    gameMode: null,  // 'ai' or 'pvp'
    playerColor: 'black',  // プレイヤーの色（AI対戦時）
    aiDifficulty: 'medium',  // AI難易度
    isAIThinking: false  // AI思考中フラグ
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
    showMessage('新しいゲームを開始してください', 'info');
});

/**
 * イベントリスナーの設定
 */
function setupEventListeners() {
    document.getElementById('new-game-btn').addEventListener('click', showNewGameModal);
    
    // モーダル関連
    document.getElementById('close-new-game-modal').addEventListener('click', closeNewGameModal);
    
    // ゲームモード選択
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            selectGameMode(this.dataset.mode);
        });
    });
    
    // ゲーム開始ボタン
    document.getElementById('start-ai-game-btn').addEventListener('click', startAIGame);
    document.getElementById('start-pvp-game-btn').addEventListener('click', startPvPGame);
    
    // ルールボタン
    document.getElementById('rules-btn').addEventListener('click', showRulesModal);
    document.getElementById('close-rules-modal').addEventListener('click', closeRulesModal);
    
    // タブ切り替え
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            switchTab(this.dataset.tab);
        });
    });
    
    // モーダル外クリックで閉じる
    document.getElementById('new-game-modal').addEventListener('click', function(e) {
        if (e.target === this) {
            closeNewGameModal();
        }
    });
    
    document.getElementById('rules-modal').addEventListener('click', function(e) {
        if (e.target === this) {
            closeRulesModal();
        }
    });
    
    document.getElementById('piece-detail-modal').addEventListener('click', function(e) {
        if (e.target === this) {
            closePieceDetailModal();
        }
    });
    
    document.getElementById('practice-modal').addEventListener('click', function(e) {
        if (e.target === this) {
            closePracticeModal();
        }
    });
    
    // 駒詳細モーダル
    document.getElementById('close-piece-detail').addEventListener('click', closePieceDetailModal);
    
    // 練習モーダル
    document.getElementById('close-practice-modal').addEventListener('click', closePracticeModal);
    
    // 練習モードのレベル変更
    document.getElementById('practice-level').addEventListener('change', function() {
        updatePracticeBoard(currentPracticePiece, parseInt(this.value));
    });
}

/**
 * 新しいゲームモーダルを表示
 */
function showNewGameModal() {
    const modal = document.getElementById('new-game-modal');
    modal.style.display = 'flex';
    modal.classList.add('show');
    
    // 設定をリセット
    document.getElementById('ai-settings').style.display = 'none';
    document.getElementById('pvp-settings').style.display = 'none';
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('selected');
    });
}

/**
 * モーダルを閉じる
 */
function closeNewGameModal() {
    const modal = document.getElementById('new-game-modal');
    modal.style.display = 'none';
    modal.classList.remove('show');
}

/**
 * ゲームモードを選択
 */
function selectGameMode(mode) {
    // ボタンの選択状態を更新
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('selected');
    });
    event.target.closest('.mode-btn').classList.add('selected');
    
    // 設定画面を表示
    document.getElementById('ai-settings').style.display = mode === 'ai' ? 'block' : 'none';
    document.getElementById('pvp-settings').style.display = mode === 'pvp' ? 'block' : 'none';
}

/**
 * AI対戦を開始
 */
async function startAIGame() {
    const playerColor = document.querySelector('input[name="player-color"]:checked').value;
    const aiDifficulty = document.querySelector('input[name="ai-difficulty"]:checked').value;
    
    gameState.gameMode = 'ai';
    gameState.playerColor = playerColor;
    gameState.aiDifficulty = aiDifficulty;
    
    closeNewGameModal();
    await startNewGame();
    
    // ゲームモード表示を更新
    const difficultyText = { easy: '初級', medium: '中級', hard: '上級', expert: '🔥最強' }[aiDifficulty];
    document.getElementById('game-mode').textContent = `🤖 AI対戦（${difficultyText}）`;
    
    // プレイヤーが後手の場合、AIに先手を打たせる
    if (playerColor === 'white') {
        setTimeout(() => requestAIMove(), 1000);
    }
}

/**
 * 対人対戦を開始
 */
async function startPvPGame() {
    gameState.gameMode = 'pvp';
    
    closeNewGameModal();
    await startNewGame();
    
    // ゲームモード表示を更新
    document.getElementById('game-mode').textContent = '👥 対人対戦';
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
    
    // AI対戦モードで、プレイヤーのターンでない場合は操作不可
    if (gameState.gameMode === 'ai') {
        const playerIsBlack = gameState.playerColor === 'black';
        const currentPlayerIsBlack = gameState.currentPlayer === 'BLACK';
        const isPlayerTurn = (playerIsBlack && currentPlayerIsBlack) || (!playerIsBlack && !currentPlayerIsBlack);
        
        if (!isPlayerTurn || gameState.isAIThinking) {
            showMessage('AIのターンです', 'warning');
            return;
        }
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
    
    // AI対戦モードで、プレイヤーのターンでない場合は操作不可
    if (gameState.gameMode === 'ai') {
        const playerIsBlack = gameState.playerColor === 'black';
        const currentPlayerIsBlack = gameState.currentPlayer === 'BLACK';
        const isPlayerTurn = (playerIsBlack && currentPlayerIsBlack) || (!playerIsBlack && !currentPlayerIsBlack);
        
        if (!isPlayerTurn || gameState.isAIThinking) {
            showMessage('AIのターンです', 'warning');
            return;
        }
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
            // サーバーから返されたメッセージを表示（勝利メッセージを含む）
            if (data.message) {
                showMessage(data.message, 'success');
            } else {
                showMessage('手を適用しました', 'success');
            }
            
            // AI対戦モードで、かつゲームが終了していない場合、AIに自動で手を打たせる
            if (gameState.gameMode === 'ai' && !data.game_state.game_over) {
                await checkAndTriggerAIMove();
            }
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
            // サーバーから返されたメッセージを表示（勝利メッセージを含む）
            if (data.message) {
                showMessage(data.message, 'success');
            } else {
                showMessage(`${PIECE_NAMES[pieceType]}を配置しました`, 'success');
            }
            
            // AI対戦モードで、かつゲームが終了していない場合、AIに自動で手を打たせる
            if (gameState.gameMode === 'ai' && !data.game_state.game_over) {
                await checkAndTriggerAIMove();
            }
        } else {
            showMessage('無効な配置です: ' + (data.message || ''), 'warning');
        }
        
    } catch (error) {
        console.error('Error dropping piece:', error);
        showMessage('エラー: ' + error.message, 'error');
    }
}

/**
 * AIのターンかチェックして、必要ならAIに手を打たせる
 */
async function checkAndTriggerAIMove() {
    if (gameState.isAIThinking) {
        return; // 既にAIが考え中
    }
    
    // プレイヤーの色とAIの色を判定
    const playerIsBlack = gameState.playerColor === 'black';
    const currentPlayerIsBlack = gameState.currentPlayer === 'BLACK';
    
    // AIのターンかチェック
    const isAITurn = (playerIsBlack && !currentPlayerIsBlack) || (!playerIsBlack && currentPlayerIsBlack);
    
    if (isAITurn) {
        // 少し待ってからAIに手を打たせる（自然な感じにするため）
        setTimeout(() => requestAIMove(), 800);
    }
}

/**
 * AIに手を打たせる
 */
async function requestAIMove() {
    if (!gameState.gameId) {
        return;
    }
    
    if (gameState.isAIThinking) {
        return; // 既にAIが考え中
    }
    
    try {
        gameState.isAIThinking = true;
        showMessage('AIが考えています...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/predict/${gameState.gameId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                game_id: gameState.gameId, 
                difficulty: gameState.aiDifficulty || 'medium'
            })
        });
        
        if (!response.ok) {
            throw new Error('AI応答の取得に失敗しました');
        }
        
        const data = await response.json();
        const move = data.move;
        
        console.log('AIの手:', move);
        
        // AIの手を適用
        // move_typeの正規化（NORMAL, CAPTURE, STACK, DROPのいずれか）
        let moveType = move.move_type || move.type || 'NORMAL';
        
        const moveData = {
            to_row: move.to[0],
            to_col: move.to[1],
            move_type: moveType
        };
        
        // 移動元がある場合（DROP以外）
        if (move.from) {
            moveData.from_row = move.from[0];
            moveData.from_col = move.from[1];
        }
        
        // 駒の種類がある場合（DROP）
        if (move.piece_type) {
            moveData.piece_type = move.piece_type;
        }
        
        console.log('適用する手:', moveData);
        
        const applyResponse = await fetch(`${API_BASE_URL}/apply_move/${gameState.gameId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(moveData)
        });
        
        if (!applyResponse.ok) {
            const errorData = await applyResponse.json();
            throw new Error(`手の適用に失敗: ${errorData.detail || applyResponse.statusText}`);
        }
        
        const applyData = await applyResponse.json();
        
        if (applyData.success) {
            updateGameState(applyData.game_state);
            showMessage('AIが手を打ちました', 'success');
        } else {
            console.error('AIの手の適用失敗:', applyData);
            showMessage(`AIの手の適用に失敗: ${applyData.message || '不明なエラー'}`, 'error');
            
            // 適用失敗の場合、もう一度AIに手を要求
            console.log('再度AIに手を要求します...');
            gameState.isAIThinking = false;
            setTimeout(() => requestAIMove(), 1000);
            return;
        }
        
    } catch (error) {
        console.error('Error requesting AI move:', error);
        showMessage('エラー: ' + error.message, 'error');
        
        // エラーの場合もリトライを試みる（最大3回まで）
        if (!gameState.aiRetryCount) {
            gameState.aiRetryCount = 0;
        }
        
        if (gameState.aiRetryCount < 3) {
            gameState.aiRetryCount++;
            console.log(`AIの手の取得を再試行します (${gameState.aiRetryCount}/3)...`);
            gameState.isAIThinking = false;
            setTimeout(() => requestAIMove(), 1000);
            return;
        } else {
            console.error('AIの手の取得に3回失敗しました');
            gameState.aiRetryCount = 0;
        }
    } finally {
        gameState.isAIThinking = false;
        gameState.aiRetryCount = 0;  // 成功したらリセット
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

// 駒のデータ（重要度順）
const PIECES_DATA = {
    SUI: { kanji: '帥', name: 'スイ', fullName: '帥（スイ）- 王将', count: 1, desc: '王将と同じ動き' },
    HYO: { kanji: '兵', name: 'ヒョウ', fullName: '兵（ヒョウ）- 歩兵', count: 4, desc: '前後に1マス' },
    SAMURAI: { kanji: '侍', name: 'サムライ', fullName: '侍（サムライ）', count: 2, desc: '前方に強い' },
    SHO: { kanji: '小', name: 'ショウ', fullName: '小（ショウショウ）- 小将', count: 2, desc: '金将と同じ' },
    UMA: { kanji: '馬', name: 'キバ', fullName: '馬（キバ）- 騎馬', count: 2, desc: '縦横に2マス' },
    SHINOBI: { kanji: '忍', name: 'シノビ', fullName: '忍（シノビ）- 忍者', count: 2, desc: '斜めに2マス' },
    YARI: { kanji: '槍', name: 'ヤリ', fullName: '槍（ヤリ）', count: 3, desc: '前方に2マス' },
    BOU: { kanji: '謀', name: 'ボウ', fullName: '謀（ボウショウ）- 謀将', count: 1, desc: '斜め専用' },
    DAI: { kanji: '大', name: 'タイショウ', fullName: '大（タイショウ）- 大将', count: 1, desc: '龍王と同じ' },
    CHUU: { kanji: '中', name: 'チュウジョウ', fullName: '中（チュウジョウ）- 中将', count: 1, desc: '龍馬と同じ' },
    TORIDE: { kanji: '砦', name: 'トリデ', fullName: '砦（トリデ）', count: 2, desc: '防御駒' },
    YUMI: { kanji: '弓', name: 'ユミ', fullName: '弓（ユミ）', count: 2, desc: '飛び越え攻撃' },
    TSUTU: { kanji: '筒', name: 'ツツ', fullName: '筒（ツツ）', count: 1, desc: 'ジャンプ移動' },
    HOU: { kanji: '砲', name: 'オオヅツ', fullName: '砲（オオヅツ）', count: 1, desc: '前方ジャンプ' }
};

let currentPracticePiece = null;

/**
 * ルールモーダルを表示
 */
function showRulesModal() {
    const modal = document.getElementById('rules-modal');
    modal.style.display = 'flex';
    modal.classList.add('show');
    
    // 駒グリッドを生成（初回のみ）
    const piecesGrid = document.getElementById('pieces-grid');
    if (piecesGrid.children.length === 0) {
        generatePiecesGrid();
    }
}

/**
 * ルールモーダルを閉じる
 */
function closeRulesModal() {
    const modal = document.getElementById('rules-modal');
    modal.style.display = 'none';
    modal.classList.remove('show');
}

/**
 * タブを切り替え
 */
function switchTab(tabName) {
    // タブボタンの状態を更新
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    
    // タブコンテンツの表示を更新
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`tab-${tabName}`).classList.add('active');
}

/**
 * 駒グリッドを生成
 */
function generatePiecesGrid() {
    const piecesGrid = document.getElementById('pieces-grid');
    
    Object.keys(PIECES_DATA).forEach(pieceType => {
        const data = PIECES_DATA[pieceType];
        const card = document.createElement('div');
        card.className = 'piece-card';
        card.onclick = () => showPieceDetail(pieceType);
        
        card.innerHTML = `
            <div class="piece-card-kanji">${data.kanji}</div>
            <div class="piece-card-name">${data.name}</div>
            <div class="piece-card-count">${data.count}枚</div>
        `;
        
        piecesGrid.appendChild(card);
    });
}

/**
 * 駒の詳細を表示
 */
function showPieceDetail(pieceType) {
    const data = PIECES_DATA[pieceType];
    const modal = document.getElementById('piece-detail-modal');
    const title = document.getElementById('piece-detail-title');
    const body = document.getElementById('piece-detail-body');
    
    title.textContent = data.fullName;
    
    body.innerHTML = `
        <div class="piece-info">
            <div class="piece-info-item">
                <span class="piece-info-label">枚数</span>
                <span>${data.count}枚</span>
            </div>
            <div class="piece-info-item">
                <span class="piece-info-label">説明</span>
                <span>${data.desc}</span>
            </div>
        </div>
        
        <div class="piece-movements">
            <div class="movement-level">
                <h4>📐 1段目（基本）</h4>
                ${generateMovementGrid(pieceType, 1)}
            </div>
            
            <div class="movement-level">
                <h4>📐 2段目（強化）</h4>
                ${generateMovementGrid(pieceType, 2)}
                <p style="color: #666; margin-top: 10px;">動ける範囲が広がります</p>
            </div>
            
            <div class="movement-level">
                <h4>📐 3段目（極）</h4>
                ${generateMovementGrid(pieceType, 3)}
                <p style="color: #666; margin-top: 10px;">最大範囲で動けます</p>
            </div>
        </div>
        
        ${getSpecialRules(pieceType)}
        
        <button class="btn btn-primary" onclick="showPracticeMode('${pieceType}')" style="width: 100%; margin-top: 20px;">
            🎮 盤面で試す
        </button>
    `;
    
    modal.style.display = 'flex';
    modal.classList.add('show');
}

/**
 * 駒詳細モーダルを閉じる
 */
function closePieceDetailModal() {
    const modal = document.getElementById('piece-detail-modal');
    modal.style.display = 'none';
    modal.classList.remove('show');
}

/**
 * 移動グリッドを生成
 */
function generateMovementGrid(pieceType, level) {
    const size = 7;
    const center = 3;
    const moves = getMovesForPiece(pieceType, level);
    
    let html = '<div class="movement-grid">';
    for (let row = 0; row < size; row++) {
        html += '<div class="movement-row">';
        for (let col = 0; col < size; col++) {
            // 座標系を修正: 表示上は上が敵陣（負の値）、下が自陣（正の値）
            // drを反転させて、画面上部が前方（敵陣方向）になるようにする
            const dr = center - row;  // 反転: 上が負、下が正
            const dc = col - center;
            let className = 'movement-cell';
            
            if (row === center && col === center) {
                className += ' piece';
                html += `<div class="${className}">${PIECES_DATA[pieceType].kanji}</div>`;
            } else if (moves.some(m => m[0] === dr && m[1] === dc)) {
                className += ' can-move';
                html += `<div class="${className}">○</div>`;
            } else {
                html += `<div class="${className}"></div>`;
            }
        }
        html += '</div>';
    }
    html += '</div>';
    
    return html;
}

/**
 * 駒の動きを取得（piece.pyのPIECE_MOVE_PATTERNSと同期）
 */
function getMovesForPiece(pieceType, level) {
    // piece.pyのデータ構造と完全に同期
    const PIECE_MOVE_PATTERNS = {
        'SUI': {  // 帥 - 8方向
            1: [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]],
            2: [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1],
                [2, 2], [2, 0], [2, -2], [0, 2], [0, -2], [-2, 2], [-2, 0], [-2, -2]],
            3: [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1],
                [2, 2], [2, 0], [2, -2], [0, 2], [0, -2], [-2, 2], [-2, 0], [-2, -2],
                [3, 3], [3, 0], [3, -3], [0, 3], [0, -3], [-3, 3], [-3, 0], [-3, -3]]
        },
        'DAI': {  // 大 - 斜め1マス＋長距離直線（龍王型）
            1: [[1, 1], [1, -1], [-1, 1], [-1, -1]],
            2: [[1, 1], [1, -1], [-1, 1], [-1, -1], [2, 2], [2, -2], [-2, 2], [-2, -2]],
            3: [[1, 1], [1, -1], [-1, 1], [-1, -1], [2, 2], [2, -2], [-2, 2], [-2, -2],
                [3, 3], [3, -3], [-3, 3], [-3, -3]]
        },
        'CHUU': {  // 中 - 直線1マス＋長距離斜め（龍馬型）
            1: [[-1, 0], [0, 1], [0, -1], [1, 0]],
            2: [[-1, 0], [0, 1], [0, -1], [1, 0], [2, 0], [0, 2], [0, -2], [-2, 0]],
            3: [[-1, 0], [0, 1], [0, -1], [1, 0], [2, 0], [0, 2], [0, -2], [-2, 0],
                [3, 0], [0, 3], [0, -3], [-3, 0]]
        },
        'SHO': {  // 小 - 金将
            1: [[-1, 0], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]],
            2: [[-1, 0], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1],
                [-2, 0], [0, -2], [0, 2], [2, -2], [2, 0], [2, 2]],
            3: [[-1, 0], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1],
                [-2, 0], [0, -2], [0, 2], [2, -2], [2, 0], [2, 2],
                [-3, 0], [0, -3], [0, 3], [3, -3], [3, 0], [3, 3]]
        },
        'SAMURAI': {  // 侍 - 前方と斜め前、後方
            1: [[1, 1], [1, 0], [1, -1], [-1, 0]],
            2: [[1, 1], [1, 0], [1, -1], [-1, 0], [2, 2], [2, 0], [2, -2], [-2, 0]],
            3: [[1, 1], [1, 0], [1, -1], [-1, 0], [2, 2], [2, 0], [2, -2], [-2, 0],
                [3, 3], [3, 0], [3, -3], [-3, 0]]
        },
        'HYO': {  // 兵 - 前後
            1: [[1, 0], [-1, 0]],
            2: [[1, 0], [-1, 0], [2, 0], [-2, 0]],
            3: [[1, 0], [-1, 0], [2, 0], [-2, 0], [3, 0], [-3, 0]]
        },
        'UMA': {  // 馬 - 縦2マス＋横1マス
            1: [[2, 0], [1, 0], [0, 1], [0, -1], [-1, 0], [-2, 0]],
            2: [[2, 0], [1, 0], [0, 1], [0, -1], [-1, 0], [-2, 0],
                [3, 0], [0, 2], [0, -2], [-3, 0]],
            3: [[2, 0], [1, 0], [0, 1], [0, -1], [-1, 0], [-2, 0],
                [3, 0], [0, 2], [0, -2], [-3, 0], [4, 0], [0, 3], [0, -3], [-4, 0]]
        },
        'SHINOBI': {  // 忍 - 斜め1-2マス
            1: [[2, 2], [2, -2], [1, 1], [1, -1], [-1, 1], [-1, -1], [-2, 2], [-2, -2]],
            2: [[2, 2], [2, -2], [1, 1], [1, -1], [-1, 1], [-1, -1], [-2, 2], [-2, -2],
                [3, 3], [3, -3], [-3, 3], [-3, -3]],
            3: [[2, 2], [2, -2], [1, 1], [1, -1], [-1, 1], [-1, -1], [-2, 2], [-2, -2],
                [3, 3], [3, -3], [-3, 3], [-3, -3], [4, 4], [4, -4], [-4, 4], [-4, -4]]
        },
        'YARI': {  // 槍 - 前方2マス＋侍
            1: [[2, 0], [1, 1], [1, 0], [1, -1], [-1, 0]],
            2: [[2, 0], [1, 1], [1, 0], [1, -1], [-1, 0],
                [3, 0], [2, 2], [2, -2], [-2, 0]],
            3: [[2, 0], [1, 1], [1, 0], [1, -1], [-1, 0],
                [3, 0], [2, 2], [2, -2], [-2, 0], [4, 0], [3, 3], [3, -3], [-3, 0]]
        },
        'TORIDE': {  // 砦 - 前方＋横＋後方斜め
            1: [[1, 0], [0, 1], [0, -1], [-1, 1], [-1, -1]],
            2: [[1, 0], [0, 1], [0, -1], [-1, 1], [-1, -1],
                [2, 0], [0, 2], [0, -2], [-2, 2], [-2, -2]],
            3: [[1, 0], [0, 1], [0, -1], [-1, 1], [-1, -1],
                [2, 0], [0, 2], [0, -2], [-2, 2], [-2, -2],
                [3, 0], [0, 3], [0, -3], [-3, 3], [-3, -3]]
        },
        'YUMI': {  // 弓 - 前方2マス＋後方1マス
            1: [[2, -1], [2, 0], [2, 1], [-1, 0]],
            2: [[2, -1], [2, 0], [2, 1], [-1, 0], [3, -2], [3, 0], [3, 2], [-2, 0]],
            3: [[2, -1], [2, 0], [2, 1], [-1, 0], [3, -2], [3, 0], [3, 2], [-2, 0],
                [4, -3], [4, 0], [4, 3], [-3, 0]]
        },
        'TSUTU': {  // 筒 - 前方2マス＋後方斜め
            1: [[2, 0], [-1, 1], [-1, -1]],
            2: [[2, 0], [-1, 1], [-1, -1], [3, 0], [-2, 2], [-2, -2]],
            3: [[2, 0], [-1, 1], [-1, -1], [3, 0], [-2, 2], [-2, -2],
                [4, 0], [-3, 3], [-3, -3]]
        },
        'HOU': {  // 砲 - 前方3マス＋横＋後方
            1: [[3, 0], [0, 1], [0, -1], [-1, 0]],
            2: [[3, 0], [0, 1], [0, -1], [-1, 0], [4, 0], [0, 2], [0, -2], [-2, 0]],
            3: [[3, 0], [0, 1], [0, -1], [-1, 0], [4, 0], [0, 2], [0, -2], [-2, 0],
                [5, 0], [0, 3], [0, -3], [-3, 0]]
        },
        'BOU': {  // 謀 - 前方斜め＋後方
            1: [[1, 1], [1, -1], [-1, 0]],
            2: [[1, 1], [1, -1], [-1, 0], [2, 2], [2, -2], [-2, 0]],
            3: [[1, 1], [1, -1], [-1, 0], [2, 2], [2, -2], [-2, 0],
                [3, 3], [3, -3], [-3, 0]]
        }
    };
    
    const pattern = PIECE_MOVE_PATTERNS[pieceType];
    if (!pattern) {
        return [];
    }
    
    return pattern[level] || pattern[1];
}

/**
 * 特殊ルールを取得
 */
function getSpecialRules(pieceType) {
    let rules = '';
    
    if (pieceType === 'SUI') {
        rules = `
            <div class="special-rules">
                <h4>⚠️ 特殊ルール</h4>
                <ul>
                    <li>帥の上に他の駒は乗せられない</li>
                </ul>
            </div>
        `;
    } else if (pieceType === 'TORIDE') {
        rules = `
            <div class="special-rules">
                <h4>⚠️ 特殊ルール</h4>
                <ul>
                    <li>砦は他の駒の上に乗れない</li>
                    <li>他の駒を砦の上に乗せることは可能</li>
                </ul>
            </div>
        `;
    }
    
    return rules;
}

/**
 * 練習モードを表示
 */
function showPracticeMode(pieceType) {
    currentPracticePiece = pieceType;
    const modal = document.getElementById('practice-modal');
    
    // レベルを1にリセット
    document.getElementById('practice-level').value = '1';
    
    updatePracticeBoard(pieceType, 1);
    
    modal.style.display = 'flex';
    modal.classList.add('show');
}

/**
 * 練習盤面を更新
 */
function updatePracticeBoard(pieceType, level) {
    const board = document.getElementById('practice-board');
    const size = 7;
    const center = 3;
    const moves = getMovesForPiece(pieceType, level);
    
    let html = '';
    for (let row = 0; row < size; row++) {
        html += '<div class="practice-row">';
        for (let col = 0; col < size; col++) {
            const dr = row - center;
            const dc = col - center;
            let className = 'practice-cell';
            
            if (row === center && col === center) {
                className += ' piece';
                html += `<div class="${className}">${PIECES_DATA[pieceType].kanji}</div>`;
            } else if (moves.some(m => m[0] === dr && m[1] === dc)) {
                className += ' can-move';
                html += `<div class="${className}"></div>`;
            } else {
                html += `<div class="${className}"></div>`;
            }
        }
        html += '</div>';
    }
    
    board.innerHTML = html;
}

/**
 * 練習モーダルを閉じる
 */
function closePracticeModal() {
    const modal = document.getElementById('practice-modal');
    modal.style.display = 'none';
    modal.classList.remove('show');
}
