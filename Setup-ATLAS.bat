@@echo off
@@setlocal
@@title ATLAS Desktop Studio - Guided Setup Wizard
@@set "ATLAS_BAT_FILE=%~f0"
@@set "ATLAS_BAT_DIR=%~dp0"
@@powershell.exe -NoProfile -Sta -ExecutionPolicy Bypass -Command "$src = (Get-Content -LiteralPath $env:ATLAS_BAT_FILE -Raw) -replace '(?m)^@@.*$',''; & ([ScriptBlock]::Create($src)) @args" %*
@@exit /b %ERRORLEVEL%

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$REPO_SLUG   = 'SultraZ/atlas'
$APP_VERSION = 'v1.0.0'
$ZIP_URL_PRI = "https://github.com/$REPO_SLUG/releases/latest/download/ATLAS-Desktop-Studio.zip"
$ZIP_URL_ALT = "https://github.com/$REPO_SLUG/releases/download/$APP_VERSION/ATLAS-Desktop-Studio-$APP_VERSION.zip"

$ScriptDir = ($env:ATLAS_BAT_DIR).TrimEnd('\', '/')
$LocalCheckout = Test-Path (Join-Path $ScriptDir 'webapp\desktop.py')
$DefaultInstallDir = if ($LocalCheckout) { $ScriptDir } else { Join-Path $env:LOCALAPPDATA 'Programs\ATLAS' }

$AutoMode    = $false
$CliMode     = $false
$WithFenics  = $false
$NoLaunch    = $false
$DevicePref  = 'auto'
$TargetDir   = $DefaultInstallDir

for ($i = 0; $i -lt $args.Count; $i++) {
    switch -Regex ($args[$i]) {
        '^--auto$|^--unattended$|^-y$' { $AutoMode = $true; $CliMode = $true }
        '^--cli$'                      { $CliMode = $true }
        '^--with-fenics$'              { $WithFenics = $true }
        '^--no-launch$'                { $NoLaunch = $true }
        '^--device$'                   { if ($i + 1 -lt $args.Count) { $DevicePref = $args[++$i] } }
        '^--dir$'                      { if ($i + 1 -lt $args.Count) { $TargetDir = $args[++$i] } }
    }
}

function Get-HardwareSummary {
    $gpuName = $null
    try {
        $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
        foreach ($g in $gpus) {
            if ($g.Name -match 'NVIDIA') { $gpuName = $g.Name; break }
        }
    } catch {}
    if (-not $gpuName -and (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
        try {
            $out = (& nvidia-smi.exe --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
            if ($out) { $gpuName = $out.Trim() }
        } catch {}
    }

    $pyCmd = Find-SystemPython
    $hasWsl = $false
    try {
        if (Get-Command wsl.exe -ErrorAction SilentlyContinue) {
            $raw = (& wsl.exe -l -q 2>$null) -join "`n"
            $distros = @($raw -split "`r?`n" | ForEach-Object { $_.Trim() -replace "`0",'' } | Where-Object { $_ -and $_ -notmatch '^docker-desktop' })
            if ($distros.Count -gt 0) { $hasWsl = $true }
        }
    } catch {}

    return [PSCustomObject]@{
        GpuName = $gpuName
        HasNvidia = [bool]$gpuName
        PythonExe = $pyCmd
        HasWslDistro = $hasWsl
    }
}

function Find-SystemPython {
    $candidates = @()
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        foreach ($ver in @('-3.11', '-3.12', '-3.10')) {
            try {
                $p = (& py.exe $ver -c "import sys; print(sys.executable)" 2>$null | Select-Object -First 1)
                if ($LASTEXITCODE -eq 0 -and $p -and (Test-Path $p.Trim())) { return $p.Trim() }
            } catch {}
        }
    }
    foreach ($cmd in @('python.exe', 'python3.exe')) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found -and $found.Source -notmatch 'WindowsApps\\python') {
            try {
                $ok = (& $found.Source -c "import sys; print(sys.executable if sys.version_info >= (3,10) else '')" 2>$null | Select-Object -First 1)
                if ($LASTEXITCODE -eq 0 -and $ok -and (Test-Path $ok.Trim())) { return $ok.Trim() }
            } catch {}
        }
    }
    $localPy = Get-ChildItem -Path "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
    if ($localPy) { return $localPy.FullName }
    return $null
}

function Invoke-AtlasInstall {
    param(
        [string]$InstallDir,
        [bool]$InstallFenicsWsl,
        [string]$ComputeChoice,
        [bool]$CreateDesktopShortcut,
        [bool]$CreateStartMenuShortcut,
        [scriptblock]$Log,
        [scriptblock]$Progress
    )

    & $Progress 5 'Preparing installation directory...'
    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }

    if (-not (Test-Path (Join-Path $InstallDir 'webapp\desktop.py'))) {
        & $Progress 12 'Downloading ATLAS Desktop Studio package from GitHub Releases...'
        $tmpZip = Join-Path $env:TEMP ("atlas_release_" + [Guid]::NewGuid().ToString('N') + ".zip")
        $tmpExtract = Join-Path $env:TEMP ("atlas_extract_" + [Guid]::NewGuid().ToString('N'))
        $downloaded = $false
        foreach ($u in @($ZIP_URL_PRI, $ZIP_URL_ALT)) {
            try {
                & $Log "Downloading $u ..."
                Invoke-WebRequest -Uri $u -OutFile $tmpZip -UseBasicParsing
                $downloaded = $true
                break
            } catch {
                & $Log "Mirror check: $($_.Exception.Message)"
            }
        }
        if (-not $downloaded) {
            throw "Could not download ATLAS release archive from GitHub Releases. Please check your internet connection or download ATLAS-Desktop-Studio-$APP_VERSION.zip directly."
        }
        & $Progress 22 'Extracting application files...'
        Expand-Archive -Path $tmpZip -DestinationPath $tmpExtract -Force
        $innerRoot = Get-ChildItem -Path $tmpExtract -Directory | Select-Object -First 1
        $copySrc = if ($innerRoot -and (Test-Path (Join-Path $innerRoot.FullName 'webapp\desktop.py'))) { $innerRoot.FullName } else { $tmpExtract }
        Copy-Item -Path (Join-Path $copySrc '*') -Destination $InstallDir -Recurse -Force
        Remove-Item -Path $tmpZip -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $tmpExtract -Recurse -Force -ErrorAction SilentlyContinue
    } else {
        & $Log "Using existing ATLAS files in: $InstallDir"
    }

    try {
        Get-ChildItem -Path $InstallDir -Recurse -File -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue
    } catch {}

    & $Progress 28 'Checking Python 3.10+ runtime...'
    $pyExe = Find-SystemPython
    if (-not $pyExe) {
        if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
            throw 'Python 3.10+ was not found and winget is unavailable. Please install Python 3.11 from https://www.python.org/downloads/ (check Add Python to PATH) and re-run Setup-ATLAS.bat.'
        }
        & $Log 'Python not found on system. Installing Python 3.11 automatically via Windows Package Manager (winget)...'
        $wg = Start-Process -FilePath 'winget.exe' -ArgumentList 'install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements' -Wait -PassThru -NoNewWindow
        $env:Path = [System.Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path', 'User')
        $pyExe = Find-SystemPython
        if (-not $pyExe) {
            throw 'Python installation finished, but python.exe could not be located yet. Please close and re-run Setup-ATLAS.bat.'
        }
    }
    & $Log "Found Python runtime: $pyExe"

    $venvDir = Join-Path $InstallDir '.venv'
    $venvPy  = Join-Path $venvDir 'Scripts\python.exe'
    if (-not (Test-Path $venvPy)) {
        & $Progress 36 'Creating isolated virtual environment (.venv)...'
        & $pyExe -m venv $venvDir
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create Python virtual environment (.venv).' }
    }

    & $Progress 44 'Upgrading pip & wheel...'
    & $venvPy -m pip install --upgrade pip wheel --quiet

    $hw = Get-HardwareSummary
    $useCuda = if ($ComputeChoice -eq 'cuda') { $true } elseif ($ComputeChoice -eq 'cpu') { $false } else { $hw.HasNvidia }
    $torchIndex = if ($useCuda) { 'https://download.pytorch.org/whl/cu124' } else { 'https://download.pytorch.org/whl/cpu' }
    $modeLabel = if ($useCuda) { 'NVIDIA CUDA GPU Acceleration' } else { 'CPU Compute Engine' }

    & $Progress 56 "Installing PyTorch ($modeLabel)..."
    & $Log "Installing PyTorch ($modeLabel) — this step may take 1-3 minutes on first install..."
    & $venvPy -m pip install torch --index-url $torchIndex
    if ($LASTEXITCODE -ne 0) {
        & $Log 'Fallback: installing default PyTorch wheel from PyPI...'
        & $venvPy -m pip install torch
        if ($LASTEXITCODE -ne 0) { throw 'Failed to install PyTorch.' }
    }

    & $Progress 72 'Installing ATLAS scientific & desktop packages (SciPy P2 FEM, FastAPI, pywebview)...'
    $reqFile = Join-Path $InstallDir 'requirements.txt'
    & $venvPy -m pip install -r $reqFile
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install packages from requirements.txt.' }

    if ($InstallFenicsWsl) {
        & $Progress 80 'Configuring WSL2 + genuine FEniCS 2019.1.0 conda environment...'
        if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
            & $Log 'WSL is not available on this Windows edition; Built-in Cross-Platform P2 FEM remains active.'
        } else {
            $raw = (& wsl.exe -l -q 2>$null) -join "`n"
            $distros = @($raw -split "`r?`n" | ForEach-Object { $_.Trim() -replace "`0",'' } | Where-Object { $_ -and $_ -notmatch '^docker-desktop' })
            if ($distros.Count -eq 0) {
                & $Log 'Installing WSL2 Ubuntu distribution (a Windows restart may be required afterwards to finish FEniCS setup)...'
                Start-Process -FilePath 'wsl.exe' -ArgumentList '--install -d Ubuntu --no-launch' -Wait -NoNewWindow
                & $Log 'WSL2 Ubuntu queued. After restarting Windows, re-run Setup-ATLAS.bat with Full FEniCS selected to finish creating the Atlas conda env.'
            } else {
                $distro = $distros[0]
                & $Log "Setting up Miniconda + FEniCS 2019.1.0 inside WSL2 ($distro)..."
                $wslSetup = @'
set -e
if [ ! -x "$HOME/miniconda3/bin/conda" ]; then
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  rm -f /tmp/miniconda.sh
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
if [ ! -x "$HOME/miniconda3/envs/Atlas/bin/python" ]; then
  conda create -y -n Atlas -c conda-forge python=3.9 fenics=2019.1.0 mshr numpy scipy matplotlib
fi
"$HOME/miniconda3/envs/Atlas/bin/python" -c "import dolfin; print('WSL2 FEniCS ready:', dolfin.__version__)"
'@
                & wsl.exe -d $distro bash -lc $wslSetup
                if ($LASTEXITCODE -eq 0) {
                    & $Log "WSL2 FEniCS 2019.1.0 environment ('Atlas') verified!"
                } else {
                    & $Log 'Note: WSL2 FEniCS setup did not complete cleanly; Built-in Cross-Platform P2 FEM is ready as fallback.'
                }
            }
        }
    }

    & $Progress 88 'Saving initial hardware & FEM preferences...'
    $settingsObj = @{
        device       = if ($ComputeChoice -in @('auto', 'cuda', 'cpu')) { $ComputeChoice } else { 'auto' }
        fem_backend  = 'auto'
        cpu_threads  = 4
        plot_dpi     = 140
        poll_ms      = 1500
        auto_open_log = $true
    }
    $settingsPath = Join-Path $InstallDir 'webapp\.app_settings.json'
    ($settingsObj | ConvertTo-Json -Depth 4) | Set-Content -Path $settingsPath -Encoding UTF8

    & $Progress 92 'Verifying Built-in Cross-Platform P2 FEM solver...'
    $verifyOut = (& $venvPy -c "import sys; sys.path.insert(0, r'$InstallDir'); from atlas.native_fem import solve_forward; r = solve_forward('cantilever'); print(f'Cantilever P2 FEM verified (compliance={r[\"compliance\"]:.2f} N*m)')" 2>&1) -join ' '
    if ($LASTEXITCODE -ne 0) { throw "Verification check failed: $verifyOut" }
    & $Log $verifyOut

    & $Progress 96 'Building native Windows launcher (ATLAS.exe) & shortcuts...'
    $buildBat = Join-Path $InstallDir 'build_exe.bat'
    if (Test-Path $buildBat) {
        & cmd.exe /c "`"$buildBat`"" | Out-Null
    }

    $exeTarget = Join-Path $InstallDir 'ATLAS.exe'
    $launchTarget = if (Test-Path $exeTarget) { $exeTarget } else { Join-Path $InstallDir 'ATLAS.bat' }
    $iconPath = Join-Path $InstallDir 'webapp\frontend\icon.ico'

    $wsh = New-Object -ComObject WScript.Shell
    if ($CreateDesktopShortcut) {
        $deskLnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'ATLAS Desktop Studio.lnk'
        $sc = $wsh.CreateShortcut($deskLnk)
        $sc.TargetPath = $launchTarget
        $sc.WorkingDirectory = $InstallDir
        $sc.Description = 'ATLAS — Structural Design & Finite-Element Studio'
        if (Test-Path $iconPath) { $sc.IconLocation = "$iconPath,0" }
        $sc.Save()
        & $Log "Created Desktop shortcut: $deskLnk"
    }
    if ($CreateStartMenuShortcut) {
        $programsDir = Join-Path ([Environment]::GetFolderPath('Programs')) 'ATLAS'
        if (-not (Test-Path $programsDir)) { New-Item -ItemType Directory -Path $programsDir -Force | Out-Null }
        $startLnk = Join-Path $programsDir 'ATLAS Desktop Studio.lnk'
        $sc2 = $wsh.CreateShortcut($startLnk)
        $sc2.TargetPath = $launchTarget
        $sc2.WorkingDirectory = $InstallDir
        $sc2.Description = 'ATLAS — Structural Design & Finite-Element Studio'
        if (Test-Path $iconPath) { $sc2.IconLocation = "$iconPath,0" }
        $sc2.Save()
        & $Log "Created Start Menu shortcut: $startLnk"
    }

    & $Progress 100 'Installation complete! ATLAS Desktop Studio is ready.'
    return $launchTarget
}

if ($CliMode) {
    Write-Host '===================================================================='
    Write-Host "  ATLAS Desktop Studio — Guided Setup ($APP_VERSION)"
    Write-Host '===================================================================='
    $hw = Get-HardwareSummary
    Write-Host ("  GPU detected : " + ($(if ($hw.GpuName) { $hw.GpuName } else { 'None (CPU mode)' })))
    Write-Host ("  Python       : " + ($(if ($hw.PythonExe) { $hw.PythonExe } else { 'Will install Python 3.11 via winget' })))
    Write-Host ("  Target folder: $TargetDir")
    if (-not $AutoMode) {
        Write-Host ''
        Write-Host '  Select FEM Edition:'
        Write-Host '    [1] Standard Cross-Platform Edition (Recommended — No WSL needed, Built-in P2 FEM)'
        Write-Host '    [2] Full FEniCS Edition (Also sets up WSL2 + legacy FEniCS 2019.1.0)'
        $ans = Read-Host '  Enter choice [1/2] (default: 1)'
        if ($ans -eq '2') { $WithFenics = $true }
    }
    $launcher = Invoke-AtlasInstall -InstallDir $TargetDir -InstallFenicsWsl $WithFenics -ComputeChoice $DevicePref `
        -CreateDesktopShortcut $true -CreateStartMenuShortcut $true `
        -Log { param($m) Write-Host "  -> $m" } `
        -Progress { param($pct, $msg) Write-Host ("[{0,3}%] {1}" -f $pct, $msg) }
    if (-not $NoLaunch) {
        Start-Process -FilePath $launcher -WorkingDirectory $TargetDir
    }
    exit 0
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
try {
    Add-Type -TypeDefinition 'using System.Runtime.InteropServices; public class DpiHelper { [DllImport("user32.dll")] public static extern bool SetProcessDPIAware(); }' -ErrorAction SilentlyContinue
    [DpiHelper]::SetProcessDPIAware() | Out-Null
} catch {}
[System.Windows.Forms.Application]::EnableVisualStyles()

$hw = Get-HardwareSummary

$form = New-Object System.Windows.Forms.Form
$form.Text = "ATLAS Desktop Studio — Setup Wizard ($APP_VERSION)"
$form.Size = New-Object System.Drawing.Size(680, 610)
$form.StartPosition = 'CenterScreen'
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
$form.BackColor = [System.Drawing.Color]::FromArgb(248, 250, 252)
$form.Font = New-Object System.Drawing.Font('Segoe UI', 9.5)

$iconCandidate = Join-Path $ScriptDir 'webapp\frontend\icon.ico'
if (Test-Path $iconCandidate) {
    $form.Icon = New-Object System.Drawing.Icon($iconCandidate)
}

$header = New-Object System.Windows.Forms.Panel
$header.Dock = 'Top'
$header.Height = 76
$header.BackColor = [System.Drawing.Color]::FromArgb(15, 23, 42)

$titleLbl = New-Object System.Windows.Forms.Label
$titleLbl.Text = 'ATLAS Desktop Studio'
$titleLbl.ForeColor = [System.Drawing.Color]::White
$titleLbl.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 15)
$titleLbl.Location = New-Object System.Drawing.Point(22, 13)
$titleLbl.AutoSize = $true

$subLbl = New-Object System.Windows.Forms.Label
$subLbl.Text = 'Deep-Energy Structural Solver & Dual Finite-Element Validation Suite'
$subLbl.ForeColor = [System.Drawing.Color]::FromArgb(148, 163, 184)
$subLbl.Font = New-Object System.Drawing.Font('Segoe UI', 9)
$subLbl.Location = New-Object System.Drawing.Point(24, 44)
$subLbl.AutoSize = $true

$header.Controls.AddRange(@($titleLbl, $subLbl))
$form.Controls.Add($header)

$hwText = 'Detected GPU: ' + ($(if ($hw.GpuName) { "$($hw.GpuName) (CUDA supported)" } else { 'No NVIDIA GPU detected (CPU mode ready)' })) +
          '   |   Python: ' + ($(if ($hw.PythonExe) { 'Installed' } else { 'Auto-install via winget' }))
$hwBanner = New-Object System.Windows.Forms.Label
$hwBanner.Text = $hwText
$hwBanner.Location = New-Object System.Drawing.Point(22, 88)
$hwBanner.Size = New-Object System.Drawing.Size(622, 22)
$hwBanner.ForeColor = [System.Drawing.Color]::FromArgb(30, 64, 175)
$hwBanner.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 8.8)
$form.Controls.Add($hwBanner)

$grpEdition = New-Object System.Windows.Forms.GroupBox
$grpEdition.Text = '1. Choose FEM Validation Edition'
$grpEdition.Location = New-Object System.Drawing.Point(22, 116)
$grpEdition.Size = New-Object System.Drawing.Size(622, 128)

$rbStandard = New-Object System.Windows.Forms.RadioButton
$rbStandard.Text = 'Standard Cross-Platform Edition (Recommended — No WSL Required)'
$rbStandard.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 9.5)
$rbStandard.Location = New-Object System.Drawing.Point(16, 24)
$rbStandard.Size = New-Object System.Drawing.Size(585, 22)
$rbStandard.Checked = (-not $WithFenics)

$lblStandard = New-Object System.Windows.Forms.Label
$lblStandard.Text = 'Uses Python + GPU/CPU PyTorch + Built-in Body-Fitted P2 FEM (SciPy). Works on every Windows PC cleanly.'
$lblStandard.ForeColor = [System.Drawing.Color]::FromArgb(71, 85, 105)
$lblStandard.Font = New-Object System.Drawing.Font('Segoe UI', 8.5)
$lblStandard.Location = New-Object System.Drawing.Point(36, 46)
$lblStandard.Size = New-Object System.Drawing.Size(570, 20)

$rbFenics = New-Object System.Windows.Forms.RadioButton
$rbFenics.Text = 'Full FEniCS Edition (WSL2 + Legacy FEniCS 2019.1.0 + Built-in P2 FEM)'
$rbFenics.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 9.5)
$rbFenics.Location = New-Object System.Drawing.Point(16, 72)
$rbFenics.Size = New-Object System.Drawing.Size(585, 22)
$rbFenics.Checked = $WithFenics

$lblFenics = New-Object System.Windows.Forms.Label
$lblFenics.Text = 'Also configures WSL2 Ubuntu + Miniconda Atlas env with genuine dolfin + mshr alongside Built-in FEM.'
$lblFenics.ForeColor = [System.Drawing.Color]::FromArgb(71, 85, 105)
$lblFenics.Font = New-Object System.Drawing.Font('Segoe UI', 8.5)
$lblFenics.Location = New-Object System.Drawing.Point(36, 94)
$lblFenics.Size = New-Object System.Drawing.Size(570, 20)

$grpEdition.Controls.AddRange(@($rbStandard, $lblStandard, $rbFenics, $lblFenics))
$form.Controls.Add($grpEdition)

$grpOpts = New-Object System.Windows.Forms.GroupBox
$grpOpts.Text = '2. Compute Hardware & Shortcuts'
$grpOpts.Location = New-Object System.Drawing.Point(22, 252)
$grpOpts.Size = New-Object System.Drawing.Size(622, 134)

$lblDev = New-Object System.Windows.Forms.Label
$lblDev.Text = 'Compute Engine:'
$lblDev.Location = New-Object System.Drawing.Point(16, 29)
$lblDev.Size = New-Object System.Drawing.Size(115, 22)

$cmbDev = New-Object System.Windows.Forms.ComboBox
$cmbDev.DropDownStyle = 'DropDownList'
$cmbDev.Location = New-Object System.Drawing.Point(135, 26)
$cmbDev.Size = New-Object System.Drawing.Size(468, 26)
[void]$cmbDev.Items.Add('Auto (Recommended — NVIDIA CUDA GPU if detected, else CPU)')
[void]$cmbDev.Items.Add('NVIDIA GPU Acceleration (CUDA 12.4)')
[void]$cmbDev.Items.Add('CPU Only (Smaller download, works on every PC/laptop)')
$cmbDev.SelectedIndex = 0

$lblDir = New-Object System.Windows.Forms.Label
$lblDir.Text = 'Install Folder:'
$lblDir.Location = New-Object System.Drawing.Point(16, 65)
$lblDir.Size = New-Object System.Drawing.Size(115, 22)

$txtDir = New-Object System.Windows.Forms.TextBox
$txtDir.Text = $TargetDir
$txtDir.Location = New-Object System.Drawing.Point(135, 62)
$txtDir.Size = New-Object System.Drawing.Size(382, 25)

$btnBrowse = New-Object System.Windows.Forms.Button
$btnBrowse.Text = 'Browse...'
$btnBrowse.Location = New-Object System.Drawing.Point(523, 60)
$btnBrowse.Size = New-Object System.Drawing.Size(80, 27)
$btnBrowse.Add_Click({
    $fbd = New-Object System.Windows.Forms.FolderBrowserDialog
    $fbd.SelectedPath = $txtDir.Text
    if ($fbd.ShowDialog() -eq 'OK') { $txtDir.Text = $fbd.SelectedPath }
})

$chkDesktop = New-Object System.Windows.Forms.CheckBox
$chkDesktop.Text = 'Desktop Shortcut'
$chkDesktop.Checked = $true
$chkDesktop.Location = New-Object System.Drawing.Point(18, 99)
$chkDesktop.Size = New-Object System.Drawing.Size(155, 24)

$chkStart = New-Object System.Windows.Forms.CheckBox
$chkStart.Text = 'Start Menu Shortcut'
$chkStart.Checked = $true
$chkStart.Location = New-Object System.Drawing.Point(185, 99)
$chkStart.Size = New-Object System.Drawing.Size(175, 24)

$chkLaunch = New-Object System.Windows.Forms.CheckBox
$chkLaunch.Text = 'Launch ATLAS when finished'
$chkLaunch.Checked = $true
$chkLaunch.Location = New-Object System.Drawing.Point(375, 99)
$chkLaunch.Size = New-Object System.Drawing.Size(230, 24)

$grpOpts.Controls.AddRange(@($lblDev, $cmbDev, $lblDir, $txtDir, $btnBrowse, $chkDesktop, $chkStart, $chkLaunch))
$form.Controls.Add($grpOpts)

$progBar = New-Object System.Windows.Forms.ProgressBar
$progBar.Location = New-Object System.Drawing.Point(22, 396)
$progBar.Size = New-Object System.Drawing.Size(622, 20)
$progBar.Minimum = 0
$progBar.Maximum = 100
$form.Controls.Add($progBar)

$logBox = New-Object System.Windows.Forms.TextBox
$logBox.Multiline = $true
$logBox.ScrollBars = 'Vertical'
$logBox.ReadOnly = $true
$logBox.BackColor = [System.Drawing.Color]::FromArgb(15, 23, 42)
$logBox.ForeColor = [System.Drawing.Color]::FromArgb(226, 232, 240)
$logBox.Font = New-Object System.Drawing.Font('Consolas', 8.5)
$logBox.Location = New-Object System.Drawing.Point(22, 424)
$logBox.Size = New-Object System.Drawing.Size(622, 92)
$logBox.Text = "Ready to install ATLAS Desktop Studio.`r`nSelect your preferred edition above and click 'Install ATLAS'."
$form.Controls.Add($logBox)

$btnInstall = New-Object System.Windows.Forms.Button
$btnInstall.Text = 'Install ATLAS'
$btnInstall.BackColor = [System.Drawing.Color]::FromArgb(37, 99, 235)
$btnInstall.ForeColor = [System.Drawing.Color]::White
$btnInstall.FlatStyle = 'Flat'
$btnInstall.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 10)
$btnInstall.Location = New-Object System.Drawing.Point(494, 526)
$btnInstall.Size = New-Object System.Drawing.Size(150, 34)

$btnClose = New-Object System.Windows.Forms.Button
$btnClose.Text = 'Close'
$btnClose.Location = New-Object System.Drawing.Point(398, 526)
$btnClose.Size = New-Object System.Drawing.Size(88, 34)
$btnClose.Add_Click({ $form.Close() })

$btnInstall.Add_Click({
    $btnInstall.Enabled = $false
    $grpEdition.Enabled = $false
    $grpOpts.Enabled = $false
    $logBox.Clear()

    $devMap = @('auto', 'cuda', 'cpu')
    $chosenDev = $devMap[$cmbDev.SelectedIndex]
    $chosenDir = $txtDir.Text.Trim()

    $appendLog = {
        param($msg)
        $logBox.AppendText("$msg`r`n")
        [System.Windows.Forms.Application]::DoEvents()
    }
    $updateProg = {
        param($pct, $msg)
        $progBar.Value = [Math]::Min(100, [Math]::Max(0, [int]$pct))
        $logBox.AppendText("[$pct%] $msg`r`n")
        [System.Windows.Forms.Application]::DoEvents()
    }

    try {
        $launcher = Invoke-AtlasInstall `
            -InstallDir $chosenDir `
            -InstallFenicsWsl $rbFenics.Checked `
            -ComputeChoice $chosenDev `
            -CreateDesktopShortcut $chkDesktop.Checked `
            -CreateStartMenuShortcut $chkStart.Checked `
            -Log $appendLog `
            -Progress $updateProg

        $btnInstall.Text = 'Installed!'
        if ($chkLaunch.Checked -and $launcher) {
            Start-Process -FilePath $launcher -WorkingDirectory $chosenDir
            $form.Close()
        } else {
            [System.Windows.Forms.MessageBox]::Show(
                "ATLAS Desktop Studio is installed and ready!`n`nFolder: $chosenDir",
                'ATLAS Setup Complete',
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Information
            ) | Out-Null
        }
    } catch {
        $appendLog.Invoke("ERROR: $($_.Exception.Message)")
        $btnInstall.Enabled = $true
        $btnInstall.Text = 'Retry Install'
        $grpEdition.Enabled = $true
        $grpOpts.Enabled = $true
        [System.Windows.Forms.MessageBox]::Show(
            $_.Exception.Message,
            'ATLAS Setup Error',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    }
})

$form.Controls.AddRange(@($btnInstall, $btnClose))
[void]$form.ShowDialog()
