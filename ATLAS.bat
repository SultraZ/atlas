@echo off
setlocal
title ATLAS — Structural Design Console

set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

where powershell.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
      "$repo = $env:REPO_DIR; $extra = $args; " ^
      "if (Get-Command wsl.exe -ErrorAction SilentlyContinue) { " ^
      "  if ($repo -match '^\\\\wsl(?:\.localhost|\$)\\([^\\]+)\\(.*)$') { " ^
      "    $distro = $Matches[1]; $linuxPath = '/' + ($Matches[2] -replace '\\','/'); " ^
      "    Write-Host ('Starting ATLAS Desktop Application in WSL2 (' + $distro + ' — CUDA + FEniCS)...'); " ^
      "    & wsl.exe -d $distro bash -lc (\"cd '\" + $linuxPath + \"' && chmod +x launch_atlas.sh && ./launch_atlas.sh \" + ($extra -join ' ')); " ^
      "    exit $LASTEXITCODE; " ^
      "  } " ^
      "  $raw = (& wsl.exe -l -q 2>$null) -join \"`n\"; " ^
      "  $distros = @($raw -split \"`r?`n\" | ForEach-Object { $_.Trim() -replace \"`0\",'' } | Where-Object { $_ -and $_ -notmatch '^docker-desktop' }); " ^
      "  foreach ($d in $distros) { " ^
      "    $lp = (& wsl.exe -d $d wslpath -a $repo 2>$null); " ^
      "    if ($LASTEXITCODE -eq 0 -and $lp) { " ^
      "      $ok = (& wsl.exe -d $d bash -lc '[ -x \"$HOME/miniconda3/envs/Atlas/bin/python\" ] || command -v python3 >/dev/null 2>&1; echo $?' 2>$null); " ^
      "      if ($ok -match '^0') { " ^
      "        Write-Host ('Starting ATLAS Desktop Application via WSL2 (' + $d + ' — CUDA + FEniCS)...'); " ^
      "        & wsl.exe -d $d bash -lc (\"cd '\" + $lp.Trim() + \"' && ./launch_atlas.sh \" + ($extra -join ' ')); " ^
      "        exit $LASTEXITCODE; " ^
      "      } " ^
      "    } " ^
      "  } " ^
      "}"
    if %ERRORLEVEL% EQU 0 exit /b 0
)

if exist "%REPO_DIR%\.venv\Scripts\python.exe" (
    "%REPO_DIR%\.venv\Scripts\python.exe" "%REPO_DIR%\webapp\desktop.py" %*
    exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python "%REPO_DIR%\webapp\desktop.py" %*
    exit /b %ERRORLEVEL%
)

echo Could not find WSL2 ^(Atlas conda environment^) or a local Python installation.
pause
exit /b 1
