# Gungi Server Restart Script
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Gungi Server Restart" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Stop existing Python processes on port 8000
Write-Host ""
Write-Host "Stopping existing server processes..." -ForegroundColor Yellow

try {
    # Find and stop processes using port 8000
    $port = 8000
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    
    if ($connections) {
        foreach ($conn in $connections) {
            $process = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
            if ($process) {
                Write-Host "Stopping process $($process.ProcessName) (PID: $($process.Id))..." -ForegroundColor Yellow
                Stop-Process -Id $process.Id -Force
            }
        }
        Write-Host "Existing processes stopped" -ForegroundColor Green
    } else {
        Write-Host "No process using port 8000" -ForegroundColor Gray
    }
} catch {
    Write-Host "Error stopping processes: $_" -ForegroundColor Red
}

# Wait a moment
Start-Sleep -Seconds 2

# Start new server
Write-Host ""
Write-Host "Starting new server..." -ForegroundColor Green
python run_server.py
