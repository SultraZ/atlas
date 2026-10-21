# Keep ErrorActionPreference = 'Continue' so stderr notices from pip/python never trigger PowerShell 5.1 NativeCommandError.
# Every external command runs directly at top level (streaming output live) and checks $LASTEXITCODE as a plain integer.
$ErrorActionPreference = 'Continue'
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$REPO_SLUG   = 'SultraZ/atlas'
$APP_VERSION = 'v1.1.5'
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

function Find-SystemPython([string]$CheckScript) {
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        foreach ($ver in @('-3.12', '-3.13', '-3.11', '-3.14', '-3.10')) {
            $p = (& py.exe $ver $CheckScript 2>$null | Select-Object -First 1)
            if ($LASTEXITCODE -eq 0 -and $p -and (Test-Path $p.Trim())) { return $p.Trim() }
        }
    }
    foreach ($cmd in @('python.exe', 'python3.exe')) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found -and $found.Source -notmatch 'WindowsApps\\python') {
            $ok = (& $found.Source $CheckScript 2>$null | Select-Object -First 1)
            if ($LASTEXITCODE -eq 0 -and $ok -and (Test-Path $ok.Trim())) { return $ok.Trim() }
        }
    }
    $localCandidates = Get-ChildItem -Path "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending
    foreach ($cand in $localCandidates) {
        $ok = (& $cand.FullName $CheckScript 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -eq 0 -and $ok -and (Test-Path $ok.Trim())) { return $ok.Trim() }
    }
    return $null
}

function Get-NvidiaGpuInfo {
    $info = @{
        Present    = $false
        Name       = ''
        Driver     = ''
        MaxCuda    = 0.0
    }
    $smiCandidates = @()
    $smiCmd = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($smiCmd -and $smiCmd.Source) { $smiCandidates += $smiCmd.Source }
    $smiCandidates += "$env:SystemRoot\System32\nvidia-smi.exe"
    $smiCandidates += "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe"

    foreach ($smi in $smiCandidates) {
        if ($smi -and (Test-Path $smi)) {
            $info.Present = $true
            $hdrLines = (& $smi 2>$null)
            $hdr = ($hdrLines -join "`n")
            if ($hdr -match 'Driver Version:\s*([0-9]+\.[0-9]+)') {
                $info.Driver = $Matches[1]
            }
            if ($hdr -match 'CUDA Version:\s*([0-9]+\.[0-9]+)') {
                $val = 0.0
                if ([double]::TryParse($Matches[1], [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$val)) {
                    $info.MaxCuda = $val
                }
            }
            $gName = (& $smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
            if ($gName -and $gName.Trim()) {
                $info.Name = $gName.Trim()
            }
            break
        }
    }

    if (-not $info.Name) {
        try {
            $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
            foreach ($g in $gpus) {
                if ($g.Name -match 'NVIDIA|GeForce|RTX|Quadro|Tesla') {
                    $info.Present = $true
                    $info.Name = $g.Name.Trim()
                    break
                }
            }
        } catch {}
    }

    if (-not $info.Name) {
        try {
            $regItems = Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0*' -ErrorAction SilentlyContinue
            foreach ($item in $regItems) {
                if ($item.DriverDesc -and ($item.DriverDesc -match 'NVIDIA|GeForce|RTX|Quadro|Tesla')) {
                    $info.Present = $true
                    $info.Name = [string]$item.DriverDesc
                    break
                }
            }
        } catch {}
    }

    if ((-not $info.Present) -and (Test-Path "$env:SystemRoot\System32\nvcuda.dll")) {
        $info.Present = $true
        $info.Name = 'NVIDIA CUDA Adapter'
    }

    return $info
}

Emit-Step 5 "Preparing destination folder: $InstallDir"
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force -ErrorAction Stop | Out-Null
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

$NeedsPackageSync = $true
$InstalledWorker = Join-Path $InstallDir 'installer\setup_worker.ps1'
if ((Test-Path (Join-Path $InstallDir 'webapp\desktop.py')) -and
    (Test-Path (Join-Path $InstallDir 'installer\check_cuda.py')) -and
    (Test-Path $InstalledWorker)) {
    $existingWorkerText = [System.IO.File]::ReadAllText($InstalledWorker)
    if ($existingWorkerText.Contains("APP_VERSION = '$APP_VERSION'")) {
        $NeedsPackageSync = $false
    }
}

if ($HasLocalFiles) {
    if (-not $SameFolder) {
        Emit-Step 14 "Copying ATLAS application files into $InstallDir ..."
        & robocopy.exe $SrcDir $InstallDir /E /XD '.git' '.venv' '__pycache__' '.pytest_cache' 'dist' /XF '*.pyc' /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) {
            throw "Failed while copying application files to $InstallDir (robocopy exit code $LASTEXITCODE)."
        }
    } else {
        Emit-Step 14 "Installing directly inside $InstallDir ..."
    }
} elseif ($NeedsPackageSync) {
    Emit-Step 12 "Downloading ATLAS Desktop Studio ($APP_VERSION) from GitHub..."
    $tmpZip = Join-Path $env:TEMP ("atlas_pkg_" + [Guid]::NewGuid().ToString('N') + ".zip")
    $tmpExt = Join-Path $env:TEMP ("atlas_ext_" + [Guid]::NewGuid().ToString('N'))
    $okDl = $false
    foreach ($u in @($ZIP_URL_PRI, $ZIP_URL_ALT)) {
        try {
            Write-Output "Downloading package from $u ..."
            Invoke-WebRequest -Uri $u -OutFile $tmpZip -UseBasicParsing -ErrorAction Stop
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
    Expand-Archive -Path $tmpZip -DestinationPath $tmpExt -Force -ErrorAction Stop
    $inner = Get-ChildItem -Path $tmpExt -Directory | Select-Object -First 1
    $copyFrom = if ($inner -and (Test-Path (Join-Path $inner.FullName 'webapp\desktop.py'))) { $inner.FullName } else { $tmpExt }
    Copy-Item -Path (Join-Path $copyFrom '*') -Destination $InstallDir -Recurse -Force -ErrorAction Stop
    Remove-Item -Path $tmpZip -Force -ErrorAction SilentlyContinue
    Remove-Item -Path $tmpExt -Recurse -Force -ErrorAction SilentlyContinue
}

$UninstSrc = Join-Path $InstallDir 'installer\Uninstall-ATLAS.exe'
$UninstDst = Join-Path $InstallDir 'Uninstall-ATLAS.exe'
if (Test-Path $UninstSrc) {
    Copy-Item -Path $UninstSrc -Destination $UninstDst -Force -ErrorAction SilentlyContinue
}

$refDir = Join-Path $InstallDir 'fem\reference'
if (Test-Path $refDir) {
    Get-ChildItem -Path $refDir -File -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -notmatch '^(cantilever|l_bracket|plate_hole|steel_insert|bimaterial_c15|filled_slot)_'
    } | Remove-Item -Force -ErrorAction SilentlyContinue
}

try {
    Get-ChildItem -Path $InstallDir -Recurse -File -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue
} catch {}

Emit-Step 30 "Checking Python 3.10-3.14 runtime..."
$checkPyScript = Join-Path $InstallDir 'installer\check_python.py'
$pyExe = Find-SystemPython $checkPyScript
if (-not $pyExe) {
    if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
        throw "Compatible Python 3.10-3.14 runtime was not found and winget is unavailable. Please install Python 3.12 from https://www.python.org/downloads/ and re-run ATLAS-Setup.exe."
    }
    Emit-Step 34 "Installing Python 3.12 automatically via Windows Package Manager (winget)..."
    Start-Process -FilePath 'winget.exe' -ArgumentList 'install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements' -Wait -NoNewWindow
    $env:Path = [System.Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path', 'User')
    $pyExe = Find-SystemPython $checkPyScript
    if (-not $pyExe) {
        throw "Python 3.12 finished installing; please close and re-open ATLAS-Setup.exe once to refresh PATH."
    }
}
Write-Output "Selected Python runtime: $pyExe"

$venvDir = Join-Path $InstallDir '.venv'
$venvPy  = Join-Path $venvDir 'Scripts\python.exe'
if (Test-Path $venvPy) {
    & $venvPy $checkPyScript 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Output "Existing .venv uses an incompatible Python version; recreating .venv with $pyExe ..."
        Remove-Item -Path $venvDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
if (-not (Test-Path $venvPy)) {
    Emit-Step 40 "Creating dedicated Python virtual environment (.venv)..."
    & $pyExe -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create Python virtual environment in $venvDir." }
}

Emit-Step 46 "Upgrading pip, setuptools & wheel..."
& $venvPy -m pip install --upgrade pip setuptools wheel --quiet

$gpuInfo = Get-NvidiaGpuInfo
if ($gpuInfo.Present) {
    Write-Output ("Detected NVIDIA GPU: {0} (Driver {1}, Max CUDA {2})" -f $gpuInfo.Name, $(if ($gpuInfo.Driver) { $gpuInfo.Driver } else { 'detected' }), $(if ($gpuInfo.MaxCuda -gt 0) { $gpuInfo.MaxCuda.ToString('0.0', [Globalization.CultureInfo]::InvariantCulture) } else { 'auto' }))
} else {
    Write-Output "No NVIDIA GPU detected on system; configuring Multi-Core CPU runtime."
}

$useCuda  = if ($ComputePref -eq 'cuda') { $true } elseif ($ComputePref -eq 'cpu') { $false } else { [bool]$gpuInfo.Present }
$modeName = if ($useCuda) { 'NVIDIA CUDA GPU Acceleration' } else { 'Multi-Core CPU Engine' }

$checkCudaScript = Join-Path $InstallDir 'installer\check_cuda.py'
Emit-Step 58 "Configuring PyTorch ($modeName)..."
& $venvPy $checkCudaScript
$cudaCheckRc = $LASTEXITCODE

if ($useCuda) {
    if ($cudaCheckRc -eq 0) {
        Write-Output "Verified working CUDA PyTorch already present in .venv; reusing existing installation."
    } else {
        if ($cudaCheckRc -eq 2 -or $cudaCheckRc -eq 3) {
            Write-Output "Removing existing non-CUDA or mismatched PyTorch build before installing CUDA wheel..."
            & $venvPy -m pip uninstall -y torch torchvision torchaudio
        }

        $cudaTags = if ($gpuInfo.MaxCuda -ge 13.0) {
            @('cu130', 'cu128', 'cu126', 'cu124')
        } elseif ($gpuInfo.MaxCuda -ge 12.8) {
            @('cu128', 'cu126', 'cu124', 'cu130')
        } elseif ($gpuInfo.MaxCuda -ge 12.6) {
            @('cu126', 'cu124', 'cu128', 'cu130')
        } elseif ($gpuInfo.MaxCuda -gt 0.0) {
            @('cu124', 'cu121', 'cu126', 'cu128')
        } else {
            @('cu128', 'cu130', 'cu126', 'cu124')
        }

        $cudaVerified = $false
        $anyCudaWheelInstalled = $false
        foreach ($tag in $cudaTags) {
            $idxUrl = "https://download.pytorch.org/whl/$tag"
            Write-Output "Installing PyTorch CUDA wheel from $idxUrl ..."
            & $venvPy -m pip install --only-binary=:all: 'torch>=2.2' --index-url $idxUrl
            if ($LASTEXITCODE -eq 0) {
                $anyCudaWheelInstalled = $true
                & $venvPy $checkCudaScript
                $probeRc = $LASTEXITCODE
                if ($probeRc -eq 4 -and (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
                    Write-Output "Installing Microsoft Visual C++ Redistributable required by CUDA libraries..."
                    Start-Process -FilePath 'winget.exe' -ArgumentList 'install -e --id Microsoft.VCRedist.2015+.x64 --silent --accept-package-agreements --accept-source-agreements' -Wait -NoNewWindow
                    & $venvPy $checkCudaScript
                    $probeRc = $LASTEXITCODE
                }
                if ($probeRc -eq 0) {
                    $cudaVerified = $true
                    break
                }
                Write-Output "Wheel from $tag installed but CUDA runtime is not active (status $probeRc); trying next CUDA compatibility build..."
                & $venvPy -m pip uninstall -y torch
            }
        }

        if (-not $cudaVerified) {
            Write-Output "Notice: Could not activate CUDA wheel for current driver/Python; installing standard PyTorch wheel as fallback..."
            & $venvPy -m pip install --only-binary=:all: 'torch>=2.2'
            if ($LASTEXITCODE -ne 0) {
                & $venvPy -m pip install --prefer-binary 'torch>=2.2'
                if ($LASTEXITCODE -ne 0) { throw "Failed to install PyTorch wheel." }
            }
        }
    }
} else {
    if ($cudaCheckRc -eq 0 -or $cudaCheckRc -eq 2 -or $cudaCheckRc -eq 3) {
        Write-Output "PyTorch is already installed in .venv; reusing existing installation."
    } else {
        Write-Output "Installing CPU PyTorch wheel..."
        & $venvPy -m pip install --only-binary=:all: 'torch>=2.2' --index-url 'https://download.pytorch.org/whl/cpu'
        if ($LASTEXITCODE -ne 0) {
            & $venvPy -m pip install --only-binary=:all: 'torch>=2.2'
            if ($LASTEXITCODE -ne 0) { throw "Failed to install CPU PyTorch wheel." }
        }
    }
}

Emit-Step 72 "Installing scientific & server wheels (NumPy, SciPy P2 FEM, Matplotlib, FastAPI)..."
$reqPath = Join-Path $InstallDir 'requirements.txt'
& $venvPy -m pip install --only-binary=:all: -r $reqPath
if ($LASTEXITCODE -ne 0) {
    & $venvPy -m pip install --prefer-binary -r $reqPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to install dependencies from requirements.txt." }
}

Emit-Step 78 "Installing native Windows WebView2 shell support..."
& $venvPy -m pip install --prefer-binary 'pywebview>=5.0'
if ($LASTEXITCODE -ne 0) {
    Write-Output "Note: pywebview optional wheel skipped; native Microsoft Edge app-mode window remains ready."
}

if ($WithFenics) {
    Emit-Step 83 "Configuring WSL2 + Legacy FEniCS 2019.1.0 ('Atlas' conda environment)..."
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
$settingsJson = ($cfg | ConvertTo-Json -Depth 4)
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $InstallDir 'webapp\.app_settings.json'), $settingsJson, $utf8NoBom)

Emit-Step 93 "Running Built-in P2 FEM & compute verification self-test..."
$verifyScript = Join-Path $InstallDir 'installer\verify_install.py'
& $venvPy $verifyScript
if ($LASTEXITCODE -ne 0) {
    throw "ATLAS runtime verification check failed (exit code $LASTEXITCODE)."
}

Emit-Step 97 "Registering shortcuts & Windows Installed Apps entry (Publisher: ATLAS)..."
$appExe   = Join-Path $InstallDir 'ATLAS.exe'
$iconPath = Join-Path $InstallDir 'webapp\frontend\icon.ico'

$wsh = New-Object -ComObject WScript.Shell
if ($MakeDesktop) {
    $desktopDir = [Environment]::GetFolderPath('Desktop')
    $lnk = $wsh.CreateShortcut((Join-Path $desktopDir 'ATLAS Desktop Studio.lnk'))
    $lnk.TargetPath       = $appExe
    $lnk.WorkingDirectory = $InstallDir
    $lnk.IconLocation     = "$appExe,0"
    $lnk.Description      = 'ATLAS Structural Design Studio (Published by ATLAS)'
    $lnk.Save()
}
if ($MakeStart) {
    $startDir = Join-Path ([Environment]::GetFolderPath('Programs')) 'ATLAS'
    if (-not (Test-Path $startDir)) { New-Item -ItemType Directory -Path $startDir -Force | Out-Null }
    $lnk = $wsh.CreateShortcut((Join-Path $startDir 'ATLAS Desktop Studio.lnk'))
    $lnk.TargetPath       = $appExe
    $lnk.WorkingDirectory = $InstallDir
    $lnk.IconLocation     = "$appExe,0"
    $lnk.Description      = 'ATLAS Structural Design Studio (Published by ATLAS)'
    $lnk.Save()

    $ulnk = $wsh.CreateShortcut((Join-Path $startDir 'Uninstall ATLAS.lnk'))
    $ulnk.TargetPath       = $UninstDst
    $ulnk.WorkingDirectory = $InstallDir
    $ulnk.Description      = 'Uninstall ATLAS Desktop Studio'
    $ulnk.Save()
}

$regPath = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\ATLASDesktopStudio'
if (-not (Test-Path $regPath)) { New-Item -Path $regPath -Force | Out-Null }
Set-ItemProperty -Path $regPath -Name 'DisplayName'          -Value 'ATLAS Desktop Studio'
Set-ItemProperty -Path $regPath -Name 'DisplayVersion'       -Value '1.1.5'
Set-ItemProperty -Path $regPath -Name 'Publisher'            -Value $PUBLISHER
Set-ItemProperty -Path $regPath -Name 'InstallLocation'      -Value $InstallDir
Set-ItemProperty -Path $regPath -Name 'DisplayIcon'          -Value "$appExe,0"
Set-ItemProperty -Path $regPath -Name 'UninstallString'      -Value "`"$UninstDst`""
Set-ItemProperty -Path $regPath -Name 'QuietUninstallString' -Value "`"$UninstDst`" /S"
Set-ItemProperty -Path $regPath -Name 'NoModify'             -Value 1 -Type DWord
Set-ItemProperty -Path $regPath -Name 'NoRepair'             -Value 1 -Type DWord
Set-ItemProperty -Path $regPath -Name 'URLInfoAbout'         -Value "https://github.com/$REPO_SLUG"

Emit-Step 100 "Installation complete! ATLAS Desktop Studio ($APP_VERSION) is ready."
Write-Output "SUCCESS: Installed to $InstallDir (Publisher: $PUBLISHER)."
