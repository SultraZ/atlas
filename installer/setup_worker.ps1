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
$APP_VERSION = 'v1.1.5B'
$PUBLISHER   = 'ATLAS'
$API_LATEST  = "https://api.github.com/repos/$REPO_SLUG/releases/latest"
$ZIP_URL_PRI = "https://github.com/$REPO_SLUG/releases/latest/download/ATLAS-Desktop-Studio.zip"
$ZIP_URL_ALT = "https://github.com/$REPO_SLUG/archive/refs/tags/$APP_VERSION.zip"
$ZIP_URL_MAIN = "https://github.com/$REPO_SLUG/archive/refs/heads/main.zip"

$SrcDir      = ($env:ATLAS_SRC_DIR).TrimEnd('\', '/')
$InstallDir  = ($env:ATLAS_DEST_DIR).TrimEnd('\', '/')
$Action      = if ($env:ATLAS_ACTION -eq 'update') { 'update' } else { 'install' }
$IsUpdate    = ($Action -eq 'update')
$Edition     = if ($env:ATLAS_EDITION) { $env:ATLAS_EDITION } else { 'standard' }
$ComputePref = if ($env:ATLAS_COMPUTE) { $env:ATLAS_COMPUTE } else { 'auto' }
$MakeDesktop = ($env:ATLAS_SHORTCUT_DESKTOP -ne '0')
$MakeStart   = ($env:ATLAS_SHORTCUT_STARTMENU -ne '0')
$WithFenics  = ($Edition -eq 'fenics')

function Emit-Step([int]$Pct, [string]$Msg) {
    Write-Output ("@@PROGRESS|{0}|{1}" -f $Pct, $Msg)
}

function Test-PythonCandidate([string]$ExePath, [string]$CheckScript) {
    if (-not $ExePath) { return $null }
    if (-not (Test-Path -LiteralPath $ExePath)) { return $null }
    try {
        $outLines = @(& $ExePath $CheckScript 2>$null)
        $rc = $LASTEXITCODE
        if ($rc -eq 0 -and $outLines.Count -gt 0) {
            $resolved = ([string]$outLines[0]).Trim()
            if ($resolved -and (Test-Path -LiteralPath $resolved)) {
                return $resolved
            }
            return $ExePath
        }
    } catch {}
    return $null
}

function Find-SystemPython([string]$CheckScript, [switch]$IncludeStoreFallback) {
    # 1. PEP 773 Python Install Manager & standalone py.exe launchers (works even before PATH refreshes)
    $pyLaunchers = @()
    $pyCmd = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($pyCmd -and $pyCmd.Source) { $pyLaunchers += $pyCmd.Source }
    $pyLaunchers += @(
        "$env:LOCALAPPDATA\Python\bin\py.exe",
        "$env:LOCALAPPDATA\Programs\Python\Launcher\py.exe",
        "$env:SystemRoot\py.exe",
        "$env:ProgramFiles\Python Launcher\py.exe"
    )
    foreach ($launcher in ($pyLaunchers | Select-Object -Unique)) {
        if ($launcher -and (Test-Path -LiteralPath $launcher)) {
            foreach ($ver in @('-3.12', '-3.13', '-3.11', '-3.14', '-3.10')) {
                try {
                    $outLines = @(& $launcher $ver $CheckScript 2>$null)
                    $rc = $LASTEXITCODE
                    if ($rc -eq 0 -and $outLines.Count -gt 0) {
                        $resolved = ([string]$outLines[0]).Trim()
                        if ($resolved -and (Test-Path -LiteralPath $resolved) -and ($IncludeStoreFallback -or $resolved -notmatch 'WindowsApps\\PythonSoftwareFoundation')) {
                            return $resolved
                        }
                    }
                } catch {}
            }
        }
    }

    $candidates = New-Object System.Collections.Generic.List[string]

    # 2. PEP 773 Python Installation Manager directories (2025-2026 default on Windows)
    foreach ($glob in @(
        "$env:LOCALAPPDATA\Python\python*\python.exe",
        "$env:ProgramFiles\Python\python*\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe",
        "$env:ProgramFiles\Python3*\python.exe",
        "${env:ProgramFiles(x86)}\Python3*\python.exe",
        "$env:SystemDrive\Python3*\python.exe"
    )) {
        $items = Get-ChildItem -Path $glob -ErrorAction SilentlyContinue | Sort-Object FullName -Descending
        foreach ($it in $items) { $candidates.Add($it.FullName) }
    }

    # 3. PEP 514 Windows Registry keys (HKCU + HKLM — works regardless of install drive or PATH checkboxes)
    foreach ($regRoot in @(
        'HKCU:\Software\Python\PythonCore',
        'HKLM:\Software\Python\PythonCore',
        'HKLM:\Software\WOW6432Node\Python\PythonCore'
    )) {
        if (Test-Path $regRoot) {
            $verKeys = Get-ChildItem -Path $regRoot -ErrorAction SilentlyContinue | Sort-Object PSChildName -Descending
            foreach ($vk in $verKeys) {
                $ipKey = Join-Path $vk.PSPath 'InstallPath'
                if (Test-Path $ipKey) {
                    $props = Get-ItemProperty -Path $ipKey -ErrorAction SilentlyContinue
                    if ($props.ExecutablePath) {
                        $candidates.Add([string]$props.ExecutablePath)
                    }
                    $defDir = $props.'(default)'
                    if ($defDir) {
                        $candidates.Add((Join-Path ([string]$defDir) 'python.exe'))
                    }
                }
            }
        }
    }

    # 4. Commands on PATH (skipping bare WindowsApps\python.exe & python3.exe Store stubs; versioned python3.1x.exe never exist as uninstalled stubs)
    foreach ($cmd in @('python3.12.exe', 'python3.13.exe', 'python3.11.exe', 'python3.14.exe', 'python3.10.exe', 'python.exe', 'python3.exe')) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found -and $found.Source) {
            $src = [string]$found.Source
            if ($src -notmatch 'WindowsApps\\python3?\.exe$') {
                $candidates.Add($src)
            }
        }
    }

    # 5. UV / Pyenv-Win / Conda installations
    foreach ($glob2 in @(
        "$env:LOCALAPPDATA\uv\python\cpython-3.*\python.exe",
        "$env:APPDATA\uv\python\cpython-3.*\python.exe",
        "$env:USERPROFILE\.pyenv\pyenv-win\versions\3.*\python.exe",
        "$env:USERPROFILE\miniconda3\python.exe",
        "$env:USERPROFILE\anaconda3\python.exe",
        "$env:USERPROFILE\miniforge3\python.exe",
        "$env:ProgramData\miniconda3\python.exe",
        "$env:ProgramData\anaconda3\python.exe"
    )) {
        $items2 = Get-ChildItem -Path $glob2 -ErrorAction SilentlyContinue | Sort-Object FullName -Descending
        foreach ($it2 in $items2) { $candidates.Add($it2.FullName) }
    }

    # 6. Installed Microsoft Store Appx packages (queried cleanly via Get-AppxPackage — never opens Microsoft Store)
    if ($IncludeStoreFallback) {
        try {
            $appxPkgs = Get-AppxPackage -Name 'PythonSoftwareFoundation.Python.3*' -ErrorAction SilentlyContinue | Sort-Object Version -Descending
            foreach ($pkg in $appxPkgs) {
                if ($pkg.InstallLocation) {
                    $candidates.Add((Join-Path $pkg.InstallLocation 'python.exe'))
                }
            }
        } catch {}
    }

    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($candPath in $candidates) {
        if (-not $candPath) { continue }
        if (-not $seen.Add($candPath)) { continue }
        $verified = Test-PythonCandidate $candPath $CheckScript
        if ($verified) {
            if ((-not $IncludeStoreFallback) -and ($verified -match 'WindowsApps\\PythonSoftwareFoundation')) {
                continue
            }
            return $verified
        }
    }
    return $null
}

function Refresh-ProcessPath {
    $mPath = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
    $uPath = [System.Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = ($mPath + ';' + $uPath).Trim(';')
}

function Install-OfficialPython312([string]$CheckScript) {
    # Step 1: Try winget with --force so stale/existing package registrations still reinstall cleanly
    if (Get-Command winget.exe -ErrorAction SilentlyContinue) {
        Emit-Step 37 "Provisioning Python 3.12 runtime via Windows Package Manager..."
        Start-Process -FilePath 'winget.exe' -ArgumentList 'install -e --id Python.Python.3.12 --source winget --scope user --force --accept-package-agreements --accept-source-agreements' -Wait -NoNewWindow
        Refresh-ProcessPath
        $foundAfterWinget = Find-SystemPython $CheckScript
        if ($foundAfterWinget) { return $foundAfterWinget }
    }

    # Step 2: Direct official python.org x64 installer into deterministic per-user directory (zero PATH/winget dependency)
    $directPyDir = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312'
    $directPyExe = Join-Path $directPyDir 'python.exe'
    $pyUrls = @(
        'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe',
        'https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe',
        'https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe'
    )
    $tmpInstaller = Join-Path $env:TEMP ("atlas_python312_" + [Guid]::NewGuid().ToString('N') + ".exe")
    $downloaded = $false
    foreach ($u in $pyUrls) {
        try {
            Emit-Step 39 "Downloading official Python 3.12 x64 installer from python.org..."
            Invoke-WebRequest -Uri $u -Headers @{ 'User-Agent' = 'ATLAS-Setup-Wizard' } -OutFile $tmpInstaller -UseBasicParsing -ErrorAction Stop
            if ((Test-Path -LiteralPath $tmpInstaller) -and ((Get-Item -LiteralPath $tmpInstaller).Length -gt 5000000)) {
                $downloaded = $true
                break
            }
        } catch {
            Write-Output "Python mirror notice: $($_.Exception.Message)"
        }
    }

    if ($downloaded) {
        Emit-Step 41 "Installing standalone Python 3.12 runtime into $directPyDir ..."
        Start-Process -FilePath $tmpInstaller -ArgumentList "/quiet InstallAllUsers=0 TargetDir=`"$directPyDir`" PrependPath=1 Include_test=0 Include_launcher=0" -Wait
        Refresh-ProcessPath
        $found = Test-PythonCandidate $directPyExe $CheckScript
        if (-not $found) { $found = Find-SystemPython $CheckScript }

        # Self-heal stale MSI state (e.g., if user previously installed & manually deleted folder without running uninstaller)
        if (-not $found) {
            Write-Output "Repairing existing Python 3.12 Windows Installer registration..."
            Start-Process -FilePath $tmpInstaller -ArgumentList '/repair /quiet' -Wait
            Refresh-ProcessPath
            $found = Test-PythonCandidate $directPyExe $CheckScript
            if (-not $found) { $found = Find-SystemPython $CheckScript }
        }
        if (-not $found) {
            Write-Output "Performing clean per-user Python 3.12 reinstallation..."
            Start-Process -FilePath $tmpInstaller -ArgumentList '/uninstall /quiet' -Wait
            Start-Process -FilePath $tmpInstaller -ArgumentList "/quiet InstallAllUsers=0 TargetDir=`"$directPyDir`" PrependPath=1 Include_test=0 Include_launcher=0" -Wait
            Refresh-ProcessPath
            $found = Test-PythonCandidate $directPyExe $CheckScript
            if (-not $found) { $found = Find-SystemPython $CheckScript -IncludeStoreFallback }
        }
        Remove-Item -LiteralPath $tmpInstaller -Force -ErrorAction SilentlyContinue
        if ($found) { return $found }
    }

    $lastTry = Find-SystemPython $CheckScript -IncludeStoreFallback
    if ($lastTry) { return $lastTry }
    throw "Could not automatically install or locate a 64-bit Python 3.10-3.14 runtime. Check internet connectivity or install Python 3.12 (64-bit) from https://www.python.org/downloads/."
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

$verbLabel = if ($IsUpdate) { 'Updating' } else { 'Preparing' }
Emit-Step 5 "$verbLabel ATLAS folder: $InstallDir"
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force -ErrorAction Stop | Out-Null
}

# Remove leftover old self-update backup if unlocked and cleanly stop any running ATLAS Python worker inside $InstallDir
Remove-Item -Path (Join-Path $InstallDir 'ATLAS-Setup.old.exe') -Force -ErrorAction SilentlyContinue
try {
    $venvPrefix = [System.IO.Path]::GetFullPath((Join-Path $InstallDir '.venv')) + '\'
    $runningProcs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue
    foreach ($rp in $runningProcs) {
        if ($rp.ExecutablePath -and $rp.ExecutablePath.StartsWith($venvPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            Write-Output "Stopping running ATLAS background process (PID $($rp.ProcessId))..."
            Stop-Process -Id $rp.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
} catch {}

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
if ((-not $IsUpdate) -and
    (Test-Path (Join-Path $InstallDir 'webapp\desktop.py')) -and
    (Test-Path (Join-Path $InstallDir 'installer\check_cuda.py')) -and
    (Test-Path $InstalledWorker)) {
    $existingWorkerText = [System.IO.File]::ReadAllText($InstalledWorker)
    if ($existingWorkerText.Contains("python-3.12.10-amd64.exe")) {
        $NeedsPackageSync = $false
    }
}

$ResolvedVersion = $APP_VERSION

if ($HasLocalFiles -and (-not $SameFolder)) {
    Emit-Step 16 "Syncing updated ATLAS files from package folder into $InstallDir ..."
    $destSetupExe = Join-Path $InstallDir 'ATLAS-Setup.exe'
    if (Test-Path $destSetupExe) {
        Move-Item -Path $destSetupExe -Destination (Join-Path $InstallDir 'ATLAS-Setup.old.exe') -Force -ErrorAction SilentlyContinue
    }
    & robocopy.exe $SrcDir $InstallDir /E /XD '.git' '.venv' '__pycache__' '.pytest_cache' 'dist' /XF '*.pyc' '.app_settings.json' /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) {
        throw "Failed while copying application files to $InstallDir (robocopy exit code $LASTEXITCODE)."
    }
} elseif ($IsUpdate -or (-not $HasLocalFiles -and $NeedsPackageSync)) {
    Emit-Step 14 "Checking GitHub for the latest ATLAS Desktop Studio release..."
    $downloadUrls = @()
    try {
        $rel = Invoke-RestMethod -Uri $API_LATEST -Headers @{ 'User-Agent' = 'ATLAS-Setup-Updater' } -UseBasicParsing -ErrorAction Stop
        if ($rel -and $rel.tag_name) {
            $ResolvedVersion = [string]$rel.tag_name
            Write-Output "Latest release on GitHub: $ResolvedVersion"
            $downloadUrls += "https://github.com/$REPO_SLUG/archive/refs/tags/$ResolvedVersion.zip"
        }
        if ($rel -and $rel.zipball_url) {
            $downloadUrls += [string]$rel.zipball_url
        }
    } catch {
        Write-Output "Release metadata lookup note: $($_.Exception.Message)"
    }
    $downloadUrls += @($ZIP_URL_PRI, $ZIP_URL_ALT, $ZIP_URL_MAIN)

    $tmpZip = Join-Path $env:TEMP ("atlas_pkg_" + [Guid]::NewGuid().ToString('N') + ".zip")
    $tmpExt = Join-Path $env:TEMP ("atlas_ext_" + [Guid]::NewGuid().ToString('N'))
    $okDl = $false
    foreach ($u in $downloadUrls) {
        if (-not $u) { continue }
        try {
            Write-Output "Downloading ATLAS package from $u ..."
            Invoke-WebRequest -Uri $u -Headers @{ 'User-Agent' = 'ATLAS-Setup-Updater' } -OutFile $tmpZip -UseBasicParsing -ErrorAction Stop
            $okDl = $true
            break
        } catch {
            Write-Output "Mirror notice: $($_.Exception.Message)"
        }
    }
    if (-not $okDl) {
        if ($SameFolder -and (Test-Path (Join-Path $InstallDir 'webapp\desktop.py'))) {
            Write-Output "Offline notice: keeping local application files in $InstallDir."
        } else {
            throw "Could not download the ATLAS release package from GitHub. Check your internet connection or extract the release ZIP locally."
        }
    } else {
        Emit-Step 26 "Applying updated ATLAS application files..."
        Expand-Archive -Path $tmpZip -DestinationPath $tmpExt -Force -ErrorAction Stop
        $inner = Get-ChildItem -Path $tmpExt -Directory | Select-Object -First 1
        $copyFrom = if ($inner -and (Test-Path (Join-Path $inner.FullName 'webapp\desktop.py'))) { $inner.FullName } else { $tmpExt }
        $destSetupExe = Join-Path $InstallDir 'ATLAS-Setup.exe'
        if (Test-Path $destSetupExe) {
            Move-Item -Path $destSetupExe -Destination (Join-Path $InstallDir 'ATLAS-Setup.old.exe') -Force -ErrorAction SilentlyContinue
        }
        & robocopy.exe $copyFrom $InstallDir /E /XD '.git' '.venv' '__pycache__' '.pytest_cache' 'dist' /XF '*.pyc' '.app_settings.json' /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) {
            Copy-Item -Path (Join-Path $copyFrom '*') -Destination $InstallDir -Recurse -Force -ErrorAction Stop
        }
        Remove-Item -Path $tmpZip -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $tmpExt -Recurse -Force -ErrorAction SilentlyContinue
    }
} else {
    Emit-Step 16 "Installing directly inside $InstallDir ..."
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

$gpuInfo = Get-NvidiaGpuInfo
if ($gpuInfo.Present) {
    Write-Output ("Detected NVIDIA GPU: {0} (Driver {1}, Max CUDA {2})" -f $gpuInfo.Name, $(if ($gpuInfo.Driver) { $gpuInfo.Driver } else { 'detected' }), $(if ($gpuInfo.MaxCuda -gt 0) { $gpuInfo.MaxCuda.ToString('0.0', [Globalization.CultureInfo]::InvariantCulture) } else { 'auto' }))
} else {
    Write-Output "No NVIDIA GPU detected on system; configuring Multi-Core CPU runtime."
}

$useCuda  = if ($ComputePref -eq 'cuda') { $true } elseif ($ComputePref -eq 'cpu') { $false } else { [bool]$gpuInfo.Present }
$modeName = if ($useCuda) { 'NVIDIA CUDA GPU Acceleration' } else { 'Multi-Core CPU Engine' }

$checkPyScript   = Join-Path $InstallDir 'installer\check_python.py'
$checkCudaScript = Join-Path $InstallDir 'installer\check_cuda.py'
$verifyScript    = Join-Path $InstallDir 'installer\verify_install.py'
$venvDir         = Join-Path $InstallDir '.venv'
$venvPy          = Join-Path $venvDir 'Scripts\python.exe'

# Fast-path Update: if existing .venv already passes Python + CUDA/CPU + FEM self-test, skip all pip downloads!
$FastUpdatedEnv = $false
if ($IsUpdate -and (Test-Path $venvPy) -and (-not $WithFenics)) {
    Emit-Step 52 "Checking existing Python & compute runtime for fast update..."
    $pyOk = [bool](Test-PythonCandidate $venvPy $checkPyScript)
    $cudaNeedOk = $true
    if ($pyOk -and $useCuda) {
        & $venvPy $checkCudaScript
        $cudaNeedOk = ($LASTEXITCODE -eq 0)
    }
    if ($pyOk -and $cudaNeedOk) {
        & $venvPy $verifyScript
        if ($LASTEXITCODE -eq 0) {
            Write-Output "Existing .venv runtime & packages verified — skipping package downloads for instant update!"
            $FastUpdatedEnv = $true
        }
    }
}

if (-not $FastUpdatedEnv) {
    Emit-Step 34 "Checking 64-bit Python 3.10-3.14 runtime..."
    $existingVenvOk = $false
    if (Test-Path $venvPy) {
        $existingVenvOk = [bool](Test-PythonCandidate $venvPy $checkPyScript)
        if (-not $existingVenvOk) {
            Write-Output "Existing .venv uses an incompatible or missing base Python; recreating .venv ..."
            Remove-Item -Path $venvDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    if ($existingVenvOk) {
        Write-Output "Reusing existing virtual environment Python: $venvPy"
    } else {
        $pyExe = Find-SystemPython $checkPyScript
        if (-not $pyExe) {
            $pyExe = Install-OfficialPython312 $checkPyScript
        }
        Write-Output "Selected Python runtime: $pyExe"

        Emit-Step 42 "Creating dedicated Python virtual environment (.venv)..."
        & $pyExe -m venv $venvDir
        if ($LASTEXITCODE -ne 0 -or (-not (Test-Path $venvPy))) {
            Write-Output "Primary Python candidate could not create .venv cleanly; installing standalone Python 3.12 runtime..."
            Remove-Item -Path $venvDir -Recurse -Force -ErrorAction SilentlyContinue
            $pyExe = Install-OfficialPython312 $checkPyScript
            Write-Output "Retrying virtual environment creation with: $pyExe"
            & $pyExe -m venv $venvDir
            if ($LASTEXITCODE -ne 0 -or (-not (Test-Path $venvPy))) {
                throw "Failed to create Python virtual environment in $venvDir."
            }
        }
    }

    Emit-Step 48 "Upgrading pip, setuptools & wheel..."
    & $venvPy -m pip install --upgrade pip setuptools wheel --quiet

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
            foreach ($tag in $cudaTags) {
                $idxUrl = "https://download.pytorch.org/whl/$tag"
                Write-Output "Installing PyTorch CUDA wheel from $idxUrl ..."
                & $venvPy -m pip install --only-binary=:all: 'torch>=2.2' --index-url $idxUrl
                if ($LASTEXITCODE -eq 0) {
                    & $venvPy $checkCudaScript
                    $probeRc = $LASTEXITCODE
                    if ($probeRc -eq 4 -and (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
                        Write-Output "Installing Microsoft Visual C++ Redistributable required by CUDA libraries..."
                        Start-Process -FilePath 'winget.exe' -ArgumentList 'install -e --id Microsoft.VCRedist.2015+.x64 --force --silent --accept-package-agreements --accept-source-agreements' -Wait -NoNewWindow
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

    Emit-Step 93 "Running Built-in P2 FEM & compute verification self-test..."
    & $venvPy $verifyScript
    if ($LASTEXITCODE -ne 0) {
        throw "ATLAS runtime verification check failed (exit code $LASTEXITCODE)."
    }
}

$settingsPath = Join-Path $InstallDir 'webapp\.app_settings.json'
if ((-not $IsUpdate) -or (-not (Test-Path $settingsPath)) -or ($ComputePref -in @('cuda', 'cpu'))) {
    Emit-Step 89 "Writing hardware & FEM settings..."
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
    [System.IO.File]::WriteAllText($settingsPath, $settingsJson, $utf8NoBom)
}

Emit-Step 97 "Updating shortcuts & Windows Installed Apps entry (Publisher: ATLAS)..."
$appExe   = Join-Path $InstallDir 'ATLAS.exe'
$dispVer  = ($ResolvedVersion -replace '^v', '')

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
Set-ItemProperty -Path $regPath -Name 'DisplayVersion'       -Value $dispVer
Set-ItemProperty -Path $regPath -Name 'Publisher'            -Value $PUBLISHER
Set-ItemProperty -Path $regPath -Name 'InstallLocation'      -Value $InstallDir
Set-ItemProperty -Path $regPath -Name 'DisplayIcon'          -Value "$appExe,0"
Set-ItemProperty -Path $regPath -Name 'UninstallString'      -Value "`"$UninstDst`""
Set-ItemProperty -Path $regPath -Name 'QuietUninstallString' -Value "`"$UninstDst`" /S"
Set-ItemProperty -Path $regPath -Name 'NoModify'             -Value 1 -Type DWord
Set-ItemProperty -Path $regPath -Name 'NoRepair'             -Value 1 -Type DWord
Set-ItemProperty -Path $regPath -Name 'URLInfoAbout'         -Value "https://github.com/$REPO_SLUG"

$doneWord = if ($IsUpdate) { 'Update' } else { 'Installation' }
Emit-Step 100 "$doneWord complete! ATLAS Desktop Studio ($ResolvedVersion) is ready."
Write-Output "SUCCESS: $doneWord finished at $InstallDir (Publisher: $PUBLISHER)."
