param(
    [int]$Width = 20,
    [int]$Height = 20,
    [int]$Seed = 7
)

$ErrorActionPreference = 'Stop'
$dir = 'C:\Users\USER\Desktop\agent'

$venv = Join-Path $dir '.venv\Scripts\python.exe'
if (Test-Path $venv) {
    $python = $venv
} else {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { Write-Error 'python not found'; exit 1 }
    $python = $cmd.Source
}

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

Write-Host "Starting Snake ${Width}x${Height} (window height must be >= $($Height + 6))"
Write-Host "Controls: WASD/Arrows = move, p = pause, r = restart, q = quit"

Start-Process -FilePath $python `
    -ArgumentList @('-m', 'snake_game.main', '--seed', $Seed,
                   '--width', $Width, '--height', $Height) `
    -WorkingDirectory $dir `
    -WindowStyle Normal

Write-Host 'Launched in a separate console window.'
