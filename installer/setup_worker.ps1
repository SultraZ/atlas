$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$REPO_SLUG   = 'SultraZ/atlas'
$APP_VERSION = 'v1.1.0'
$PUBLISHER   = 'ATLAS'
$ZIP_URL_PRI = "https://github.com/$REPO_SLUG/releases/latest/download/ATLAS-Desktop-Studio.zip"
$ZIP_URL_ALT = "https://github.com/$REPO_SLUG/archive/refs/tags/$APP_VERSION.zip"

$SrcDir      = ($env:ATLAS_SRC_DIR).TrimEnd('\', '/')
$InstallDir  = ($env:ATLAS_DEST_DIR).TrimEnd('\', '/')
$Edition     = if ($env:ATLAS_EDITION) { $env:ATLAS_EDITION } else { 'standard' }
$ComputePref = if ($env:ATLAS_COMPUTE) { $env:ATLAS_COMPUTE } else { 'auto' }
$MakeDesktop = ($env:ATLAS_SHORTCUT_DESKTOP -ne '0')
$MakeStart   = ($env:ATLAS_SHORTCUT_STARTMENU -ne '0')
$WithFenics  = ($Edition -eq 'fenics')

function Emit-Step([int]$Pct, [string]$Msg) {
    Write-Output ("@@PROGRESS|{0}|{1}" -f $Pct, $Msg)
}

function Find-SystemPython {
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
    $localPy = Get-ChildItem -Path "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1
    if ($localPy) { return $localPy.FullName }
    return $null
}

function Test-NvidiaGpu {
    try {
        $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
        foreach ($g in $gpus) {
            if ($g.Name -match 'NVIDIA') { return $true }
        }
    } catch {}
    if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) { return $true }
    return $false
}

Emit-Step 5 "Preparing destination folder: $InstallDir"
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

$HasLocalFiles = $SrcDir -and (Test-Path (Join-Path $SrcDir 'webapp\desktop.py'))
$SameFolder = $false
if ($SrcDir) {
    $SameFolder = ([string]::Equals(
        [System.IO.Path]::GetFullPath($SrcDir).TrimEnd('\'),
        [System.IO.Path]::GetFullPath($InstallDir).TrimEnd('\'),
        [System.StringComparison]::OrdinalIgnoreCase
    ))
}

if ($HasLocalFiles) {
    if (-not $SameFolder) {
        Emit-Step 14 "Copying ATLAS application files into $InstallDir ..."
        & robocopy.exe $SrcDir $InstallDir /E /XD ".git" ".venv" "__pycache__" ".pytest_cache" "dist" /XF "*.pyc" /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) {
            throw "Failed while copying application files to $InstallDir (robocopy exit code $LASTEXITCODE)."
        }
    } else {
        Emit-Step 14 "Installing directly inside $InstallDir ..."
    }
} elseif (-not (Test-Path (Join-Path $InstallDir 'webapp\desktop.py'))) {
    Emit-Step 12 "Downloading ATLAS Desktop Studio ($APP_VERSION) from GitHub..."
    $tmpZip = Join-Path $env:TEMP ("atlas_pkg_" + [Guid]::NewGuid().ToString('N') + ".zip")
    $tmpExt = Join-Path $env:TEMP ("atlas_ext_" + [Guid]::NewGuid().ToString('N'))
    $okDl = $false
    foreach ($u in @($ZIP_URL_PRI, $ZIP_URL_ALT)) {
        try {
            Write-Output "Downloading package from $u ..."
            Invoke-WebRequest -Uri $u -OutFile $tmpZip -UseBasicParsing
            $okDl = $true
            break
        } catch {
            Write-Output "Mirror notice: $($_.Exception.Message)"
        }
    }
    if (-not $okDl) {
        throw "Could not download the ATLAS release package automatically. If this repository is private, download the full ZIP from GitHub Releases while signed in and run ATLAS-Setup.exe from that extracted folder."
    }
    Emit-Step 22 "Extracting ATLAS application files..."
    Expand-Archive -Path $tmpZip -DestinationPath $tmpExt -Force
    $inner = Get-ChildItem -Path $tmpExt -Directory | Select-Object -First 1
    $copyFrom = if ($inner -and (Test-Path (Join-Path $inner.FullName 'webapp\desktop.py'))) { $inner.FullName } else { $tmpExt }
    Copy-Item -Path (Join-Path $copyFrom '*') -Destination $InstallDir -Recurse -Force
    Remove-Item -Path $tmpZip -Force -ErrorAction SilentlyContinue
    Remove-Item -Path $tmpExt -Recurse -Force -ErrorAction SilentlyContinue
}

$UninstSrc = Join-Path $InstallDir 'installer\Uninstall-ATLAS.exe'
$UninstDst = Join-Path $InstallDir 'Uninstall-ATLAS.exe'
if ((Test-Path $UninstSrc) -and -not (Test-Path $UninstDst)) {
    Copy-Item -Path $UninstSrc -Destination $UninstDst -Force
}

try {
    Get-ChildItem -Path $InstallDir -Recurse -File -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue
} catch {}

Emit-Step 30 "Checking Python 3.10+ runtime..."
$pyExe = Find-SystemPython
if (-not $pyExe) {
    if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
        throw "Python 3.10+ was not found and winget is unavailable. Please install Python 3.11 from https://www.python.org/downloads/ and re-run ATLAS-Setup.exe."
    }
    Emit-Step 34 "Installing Python 3.11 automatically via Windows Package Manager (winget)..."
    Start-Process -FilePath 'winget.exe' -ArgumentList 'install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements' -Wait -NoNewWindow
    $env:Path = [System.Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path', 'User')
    $pyExe = Find-SystemPython
    if (-not $pyExe) {
        throw "Python 3.11 finished installing, please close and re-open ATLAS-Setup.exe once to refresh PATH."
    }
}
Write-Output "Detected Python executable: $pyExe"

$venvDir = Join-Path $InstallDir '.venv'
$venvPy  = Join-Path $venvDir 'Scripts\python.exe'
if (-not (Test-Path $venvPy)) {
    Emit-Step 40 "Creating dedicated Python virtual environment (.venv)..."
    & $pyExe -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create Python virtual environment in $venvDir." }
}

Emit-Step 46 "Upgrading pip package installer..."
& $venvPy -m pip install --upgrade pip wheel --quiet

$useCuda = if ($ComputePref -eq 'cuda') { $true } elseif ($ComputePref -eq 'cpu') { $false } else { Test-NvidiaGpu }
$torchIndex = if ($useCuda) { 'https://download.pytorch.org/whl/cu124' } else { 'https://download.pytorch.org/whl/cpu' }
$modeName = if ($useCuda) { 'NVIDIA CUDA GPU Acceleration' } else { 'Multi-Core CPU Engine' }

Emit-Step 58 "Installing PyTorch ($modeName)..."
Write-Output "Installing PyTorch ($modeName). First-time download may take 1-3 minutes..."
& $venvPy -m pip install torch --index-url $torchIndex
if ($LASTEXITCODE -ne 0) {
    Write-Output "Retrying PyTorch installation from default PyPI index..."
    & $venvPy -m pip install torch
    if ($LASTEXITCODE -ne 0) { throw "Failed to install PyTorch." }
}

Emit-Step 74 "Installing scientific & desktop runtime dependencies (SciPy P2 FEM, FastAPI, WebView)..."
$reqPath = Join-Path $InstallDir 'requirements.txt'
& $venvPy -m pip install -r $reqPath
if ($LASTEXITCODE -ne 0) { throw "Failed to install dependencies from requirements.txt." }

if ($WithFenics) {
    Emit-Step 82 "Configuring WSL2 + Legacy FEniCS 2019.1.0 ('Atlas' conda environment)..."
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        Write-Output "WSL2 is not enabled on this Windows installation; Built-in Cross-Platform P2 FEM remains active."
    } else {
        $raw = (& wsl.exe -l -q 2>$null) -join "`n"
        $distros = @($raw -split "`r?`n" | ForEach-Object { $_.Trim() -replace "`0",'' } | Where-Object { $_ -and $_ -notmatch '^docker-desktop' })
        if ($distros.Count -eq 0) {
            Write-Output "Installing Ubuntu WSL2 distribution (reboot Windows after setup completes, then run ATLAS-Setup.exe once more to finish creating the conda env)..."
            Start-Process -FilePath 'wsl.exe' -ArgumentList '--install -d Ubuntu --no-launch' -Wait -NoNewWindow
        } else {
            $distro = $distros[0]
            Write-Output "Provisioning Miniconda + FEniCS 2019.1.0 inside WSL2 ($distro)..."
            $wslCmd = @'
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
            & wsl.exe -d $distro bash -lc $wslCmd
        }
    }
}

Emit-Step 89 "Writing default hardware & FEM settings..."
$cfg = @{
    device        = if ($ComputePref -in @('auto', 'cuda', 'cpu')) { $ComputePref } else { 'auto' }
    fem_backend   = 'auto'
    cpu_threads   = 4
    plot_dpi      = 140
    poll_ms       = 1500
    auto_open_log = $true
}
($cfg | ConvertTo-Json -Depth 4) | Set-Content -Path (Join-Path $InstallDir 'webapp\.app_settings.json') -Encoding UTF8

Emit-Step 93 "Running Built-in P2 FEM verification self-test..."
$chk = (& $venvPy -c "import sys; sys.path.insert(0, r'$InstallDir'); from atlas.native_fem import solve_forward; r = solve_forward('cantilever'); print(f'Verified Built-in P2 FEM: cantilever compliance = {r[\"compliance\"]:.2f} N*m')" 2>&1) -join ' '
if ($LASTEXITCODE -ne 0) { throw "Self-test failed: $chk" }
Write-Output $chk

Emit-Step 97 "Registering application shortcuts & Windows Installed Apps metadata (Publisher: ATLAS)..."
$appExe   = Join-Path $InstallDir 'ATLAS.exe'
$iconPath = Join-Path $InstallDir 'webapp\frontend\icon.ico'

$regPath = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\ATLASDesktopStudio'
New-Item -Path $regPath -Force | Out-Null
New-ItemProperty -Path $regPath -Name 'DisplayName' -Value 'ATLAS Desktop Studio' -PropertyType String -Force | Out-Null
New-ItemProperty -Path $regPath -Name 'Publisher' -Value $PUBLISHER -PropertyType String -Force | Out-Null
New-ItemProperty -Path $regPath -Name 'DisplayVersion' -Value '1.0.0' -PropertyType String -Force | Out-Null
New-ItemProperty -Path $regPath -Name 'DisplayIcon' -Value "$appExe,0" -PropertyType String -Force | Out-Null
New-ItemProperty -Path $regPath -Name 'InstallLocation' -Value $InstallDir -PropertyType String -Force | Out-Null
New-ItemProperty -Path $regPath -Name 'UninstallString' -Value "`"$UninstDst`"" -PropertyType String -Force | Out-Null
New-ItemProperty -Path $regPath -Name 'URLInfoAbout' -Value "https://github.com/$REPO_SLUG" -PropertyType String -Force | Out-Null
New-ItemProperty -Path $regPath -Name 'NoModify' -Value 1 -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $regPath -Name 'NoRepair' -Value 1 -PropertyType DWord -Force | Out-Null

$wsh = New-Object -ComObject WScript.Shell
if ($MakeDesktop) {
    $deskLnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'ATLAS Desktop Studio.lnk'
    $sc = $wsh.CreateShortcut($deskLnk)
    $sc.TargetPath = $appExe
    $sc.WorkingDirectory = $InstallDir
    $sc.Description = 'ATLAS Desktop Studio — Structural Design & FEM Suite (Published by ATLAS)'
    if (Test-Path $appExe) { $sc.IconLocation = "$appExe,0" } elseif (Test-Path $iconPath) { $sc.IconLocation = "$iconPath,0" }
    $sc.Save()
    Write-Output "Created Desktop shortcut: $deskLnk"
}
if ($MakeStart) {
    $startDir = Join-Path ([Environment]::GetFolderPath('Programs')) 'ATLAS'
    if (-not (Test-Path $startDir)) { New-Item -ItemType Directory -Path $startDir -Force | Out-Null }
    $startLnk = Join-Path $startDir 'ATLAS Desktop Studio.lnk'
    $sc2 = $wsh.CreateShortcut($startLnk)
    $sc2.TargetPath = $appExe
    $sc2.WorkingDirectory = $InstallDir
    $sc2.Description = 'ATLAS Desktop Studio — Structural Design & FEM Suite'
    if (Test-Path $appExe) { $sc2.IconLocation = "$appExe,0" } elseif (Test-Path $iconPath) { $sc2.IconLocation = "$iconPath,0" }
    $sc2.Save()
    if (Test-Path $UninstDst) {
        $unLnk = Join-Path $startDir 'Uninstall ATLAS Desktop Studio.lnk'
        $sc3 = $wsh.CreateShortcut($unLnk)
        $sc3.TargetPath = $UninstDst
        $sc3.WorkingDirectory = $InstallDir
        $sc3.Description = 'Uninstall ATLAS Desktop Studio'
        $sc3.Save()
    }
    Write-Output "Created Start Menu shortcuts in: $startDir"
}

Emit-Step 100 "Installation complete! ATLAS Desktop Studio is ready to launch."
