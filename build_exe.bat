@echo off
setlocal
title Build ATLAS.exe for Windows

set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"
set "ICON_PATH=%REPO_DIR%\webapp\frontend\icon.ico"

if /I "%~1"=="--standalone" goto :build_standalone

echo ====================================================================
echo   Building native Windows launcher: ATLAS.exe
echo   ^(Uses built-in Windows .NET compiler — zero extra downloads^)
echo ====================================================================

set "CSC=%SystemRoot%\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if not exist "%CSC%" set "CSC=%SystemRoot%\Microsoft.NET\Framework\v4.0.30319\csc.exe"
if not exist "%CSC%" (
    echo Error: Could not find built-in Windows C# compiler ^(csc.exe^).
    exit /b 1
)

set "CS_SRC=%TEMP%\atlas_launcher_%RANDOM%.cs"
> "%CS_SRC%" (
echo using System;
echo using System.Diagnostics;
echo using System.IO;
echo using System.Windows.Forms;
echo.
echo static class AtlasLauncher {
echo     [STAThread]
echo     static int Main^(string[] args^) {
echo         try {
echo             string exeDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd^('\\', '/'^);
echo             string batPath = Path.Combine^(exeDir, "ATLAS.bat"^);
echo             if ^(!File.Exists^(batPath^)^) {
echo                 MessageBox.Show^("Could not find ATLAS.bat next to ATLAS.exe:\n" + batPath,
echo                     "ATLAS — Launch Error", MessageBoxButtons.OK, MessageBoxIcon.Error^);
echo                 return 1;
echo             }
echo             string joinedArgs = "";
echo             foreach ^(string a in args^) {
echo                 joinedArgs += " \"" + a.Replace^("\"", "\\\""^) + "\"";
echo             }
echo             ProcessStartInfo psi = new ProcessStartInfo^();
echo             psi.FileName = "cmd.exe";
echo             psi.Arguments = "/c \"\"" + batPath + "\"" + joinedArgs + "\"";
echo             psi.WorkingDirectory = Environment.SystemDirectory;
echo             psi.UseShellExecute = false;
echo             psi.CreateNoWindow = true;
echo             psi.WindowStyle = ProcessWindowStyle.Hidden;
echo             psi.EnvironmentVariables["REPO_DIR"] = exeDir;
echo             Process proc = Process.Start^(psi^);
echo             if ^(proc != null^) proc.WaitForExit^();
echo             return proc != null ? proc.ExitCode : 0;
echo         } catch ^(Exception ex^) {
echo             MessageBox.Show^(ex.Message, "ATLAS — Launch Error",
echo                 MessageBoxButtons.OK, MessageBoxIcon.Error^);
echo             return 1;
echo         }
echo     }
echo }
)

"%CSC%" /nologo /optimize+ /target:winexe /out:"%REPO_DIR%\ATLAS.exe" /win32icon:"%ICON_PATH%" /r:System.Windows.Forms.dll "%CS_SRC%"
set "RC=%ERRORLEVEL%"
del /q "%CS_SRC%" >nul 2>nul

if %RC% NEQ 0 (
    echo Build failed with exit code %RC%.
    pause
    exit /b %RC%
)

echo.
echo Done! Created native Windows executable:
echo   %REPO_DIR%\ATLAS.exe
echo You can double-click ATLAS.exe or pin it to your Windows Taskbar / Start Menu.
echo ^(Runs WSL2 CUDA + genuine FEniCS automatically with no black console window.^)
exit /b 0

:build_standalone
echo ====================================================================
echo   Building standalone bundled distribution: dist\ATLAS\ATLAS.exe
echo   ^(Packages Python + PyTorch + WebView + results + FEniCS refs^)
echo ====================================================================

set "PY_BIN=python"
if exist "%REPO_DIR%\.venv\Scripts\python.exe" set "PY_BIN=%REPO_DIR%\.venv\Scripts\python.exe"

"%PY_BIN%" -m PyInstaller --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing PyInstaller into %PY_BIN% ...
    "%PY_BIN%" -m pip install pyinstaller
    if %ERRORLEVEL% NEQ 0 exit /b 1
)

pushd "%REPO_DIR%"
"%PY_BIN%" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --windowed ^
    --name ATLAS ^
    --icon "%ICON_PATH%" ^
    --add-data "atlas;atlas" ^
    --add-data "webapp;webapp" ^
    --add-data "results;results" ^
    --add-data "fem;fem" ^
    --hidden-import uvicorn.logging ^
    --hidden-import uvicorn.loops.auto ^
    --hidden-import uvicorn.protocols.http.auto ^
    --hidden-import uvicorn.lifespan.on ^
    --collect-submodules atlas ^
    webapp\desktop.py
set "RC=%ERRORLEVEL%"
popd

if %RC% EQU 0 (
    echo.
    echo Done! Standalone Windows app bundle created at:
    echo   %REPO_DIR%\dist\ATLAS\ATLAS.exe
)
exit /b %RC%
