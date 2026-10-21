@echo off
setlocal
title ATLAS — Structural Design Studio

set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

:: 1. Native Windows virtual environment (preferred on Windows: uses Windows CUDA/CPU + native Edge WebView2,
::    and still automatically bridges into WSL2 FEniCS when available!)
if exist "%REPO_DIR%\.venv\Scripts\python.exe" (
    "%REPO_DIR%\.venv\Scripts\python.exe" "%REPO_DIR%\webapp\desktop.py" %*
    exit /b %ERRORLEVEL%
)

:: 2. WSL2 environment (when running from \\wsl.localhost\... or when WSL2 holds the Atlas conda environment)
where powershell.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
      "$repo = $env:REPO_DIR; $extra = $args; " ^
      "if (Get-Command wsl.exe -ErrorAction SilentlyContinue) { " ^
      "  if ($repo -match '^\\\\wsl(?:\.localhost|\$)\\([^\\]+)\\(.*)$') { " ^
      "    $distro = $Matches[1]; $linuxPath = '/' + ($Matches[2] -replace '\\','/'); " ^
      "    Write-Host ('Starting ATLAS Desktop Studio in WSL2 (' + $distro + ')...'); " ^
      "    & wsl.exe -d $distro bash -lc (\"cd '\" + $linuxPath + \"' && chmod +x launch_atlas.sh && ./launch_atlas.sh \" + ($extra -join ' ')); " ^
      "    exit $LASTEXITCODE; " ^
      "  } " ^
      "  $raw = (& wsl.exe -l -q 2>$null) -join \"`n\"; " ^
      "  $distros = @($raw -split \"`r?`n\" | ForEach-Object { $_.Trim() -replace \"`0\",'' } | Where-Object { $_ -and $_ -notmatch '^docker-desktop' }); " ^
      "  foreach ($d in $distros) { " ^
      "    $lp = (& wsl.exe -d $d wslpath -a $repo 2>$null); " ^
      "    if ($LASTEXITCODE -eq 0 -and $lp) { " ^
      "      $ok = (& wsl.exe -d $d bash -lc '[ -x \"$HOME/miniconda3/envs/Atlas/bin/python\" ] || python3 -c \"import torch, fastapi, uvicorn\" >/dev/null 2>&1; echo $?' 2>$null); " ^
      "      if ($ok -match '^0') { " ^
      "        Write-Host ('Starting ATLAS Desktop Studio via WSL2 (' + $d + ')...'); " ^
      "        & wsl.exe -d $d bash -lc (\"cd '\" + $lp.Trim() + \"' && ./launch_atlas.sh \" + ($extra -join ' ')); " ^
      "        exit $LASTEXITCODE; " ^
      "      } " ^
      "    } " ^
      "  } " ^
      "}"
    if %ERRORLEVEL% EQU 0 exit /b 0
)

:: 3. System Python on Windows if dependencies are already installed
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python -c "import torch, scipy, fastapi, uvicorn" >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        python "%REPO_DIR%\webapp\desktop.py" %*
        exit /b %ERRORLEVEL%
    )
)

:: 4. First-time launch: automatically open the Guided Setup Wizard if present
if exist "%REPO_DIR%\Setup-ATLAS.bat" (
    echo ATLAS runtime environment is not configured yet. Opening Guided Setup Wizard...
    call "%REPO_DIR%\Setup-ATLAS.bat"
    exit /b %ERRORLEVEL%
)

echo Could not find a configured ATLAS Python environment or WSL2 Atlas installation.
echo Run Setup-ATLAS.bat first to install everything automatically.
pause
exit /b 1
