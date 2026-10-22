#!/usr/bin/env python3
"""
Cross-compile genuine native x86_64 Windows GUI (.exe) binaries for ATLAS:
  1. ATLAS-Setup.exe            — Native Win32 Graphical Setup Wizard
  2. ATLAS.exe                  — Native Win32 Application Launcher
  3. installer/Uninstall-ATLAS.exe — Native Win32 Uninstaller

All binaries embed:
  - Multi-resolution application icon (webapp/frontend/icon.ico)
  - Windows Common Controls v6 + Per-Monitor High-DPI v2 XML Manifest
  - Win32 VS_VERSIONINFO resource table with Publisher / Company / Author = "ATLAS"
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLER_DIR = ROOT / "installer"
ICON_ICO = ROOT / "webapp" / "frontend" / "icon.ico"

MANIFEST_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <assemblyIdentity
    version="1.0.0.0"
    processorArchitecture="amd64"
    name="ATLAS.DesktopStudio"
    type="win32"
  />
  <description>ATLAS Desktop Studio — Published by ATLAS</description>
  <dependency>
    <dependentAssembly>
      <assemblyIdentity
        type="win32"
        name="Microsoft.Windows.Common-Controls"
        version="6.0.0.0"
        processorArchitecture="*"
        publicKeyToken="6595b64144ccf1df"
        language="*"
      />
    </dependentAssembly>
  </dependency>
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="asInvoker" uiAccess="false" />
      </requestedPrivileges>
    </security>
  </trustInfo>
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings>
      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true/pm</dpiAware>
      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2</dpiAwareness>
    </windowsSettings>
  </application>
</assembly>
"""


def write_worker_header() -> Path:
    ps1_bytes = (INSTALLER_DIR / "setup_worker.ps1").read_bytes()
    hex_items = ", ".join(f"0x{b:02x}" for b in ps1_bytes)
    header = (
        "/* Auto-generated from installer/setup_worker.ps1 */\n"
        "#pragma once\n"
        "#include <stddef.h>\n"
        f"static const unsigned char ATLAS_WORKER_PS1[] = {{ {hex_items}, 0x00 }};\n"
        f"static const size_t ATLAS_WORKER_PS1_LEN = {len(ps1_bytes)}u;\n"
    )
    out = INSTALLER_DIR / "worker_payload.h"
    out.write_text(header, encoding="utf-8")
    return out


def compile_win32_exe(
    c_source: Path,
    out_exe: Path,
    file_desc: str,
    orig_filename: str,
    extra_libs: list[str],
) -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        manifest_path = td_path / "app.manifest"
        manifest_path.write_text(MANIFEST_XML, encoding="utf-8")

        rc_path = td_path / "app.rc"
        res_path = td_path / "app.res"
        ico_posix = ICON_ICO.as_posix()
        manifest_posix = manifest_path.as_posix()

        rc_content = f'''#include <windows.h>

1 ICON "{ico_posix}"
1 24 "{manifest_posix}"

1 VERSIONINFO
FILEVERSION 1,1,7,0
PRODUCTVERSION 1,1,7,0
FILEFLAGSMASK 0x3fL
FILEFLAGS 0x0L
FILEOS 0x40004L
FILETYPE 0x1L
FILESUBTYPE 0x0L
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904B0"
    BEGIN
      VALUE "CompanyName", "ATLAS\\0"
      VALUE "Publisher", "ATLAS\\0"
      VALUE "Author", "ATLAS\\0"
      VALUE "FileDescription", "{file_desc}\\0"
      VALUE "FileVersion", "1.1.7\\0"
      VALUE "InternalName", "{orig_filename}\\0"
      VALUE "LegalCopyright", "Copyright (c) 2026 ATLAS. All rights reserved.\\0"
      VALUE "OriginalFilename", "{orig_filename}\\0"
      VALUE "ProductName", "ATLAS Desktop Studio\\0"
      VALUE "ProductVersion", "1.1.7\\0"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x0409, 1200
  END
END
'''
        rc_path.write_text(rc_content, encoding="utf-8")

        subprocess.run(
            [sys.executable, "-m", "ziglang", "rc", str(rc_path), str(res_path)],
            cwd=ROOT,
            check=True,
        )

        out_exe.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "ziglang",
            "cc",
            "-target",
            "x86_64-windows-gnu",
            "-Os",
            "-s",
            "-municode",
            f"-I{INSTALLER_DIR}",
            "-Wl,--subsystem,windows",
            str(c_source),
            str(res_path),
            *extra_libs,
            "-o",
            str(out_exe),
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)


def verify_exe_metadata(exe_path: Path, expected_desc: str) -> dict[str, str]:
    import pefile

    pe = pefile.PE(str(exe_path))
    assert pe.FILE_HEADER.Machine == 0x8664, f"Expected x86_64 PE32+, got {hex(pe.FILE_HEADER.Machine)}"
    assert pe.OPTIONAL_HEADER.Subsystem == 2, f"Expected WINDOWS_GUI subsystem (2), got {pe.OPTIONAL_HEADER.Subsystem}"

    strings: dict[str, str] = {}
    for fileinfo in pe.FileInfo[0]:
        if fileinfo.Key.decode(errors="ignore") == "StringFileInfo":
            for st in fileinfo.StringTable:
                for k, v in st.entries.items():
                    strings[k.decode(errors="ignore")] = v.decode(errors="ignore")

    assert strings.get("CompanyName") == "ATLAS", f"CompanyName mismatch: {strings}"
    assert strings.get("Author") == "ATLAS", f"Author mismatch: {strings}"
    assert strings.get("Publisher") == "ATLAS", f"Publisher mismatch: {strings}"
    assert strings.get("FileDescription") == expected_desc, f"FileDescription mismatch: {strings}"
    pe.close()
    return strings


def main() -> int:
    write_worker_header()

    targets = [
        (
            INSTALLER_DIR / "setup_wizard.c",
            ROOT / "ATLAS-Setup.exe",
            "ATLAS Setup Wizard",
            "ATLAS-Setup.exe",
            ["-lcomctl32", "-lshell32", "-lole32", "-lgdi32", "-ladvapi32"],
        ),
        (
            INSTALLER_DIR / "atlas_launcher.c",
            ROOT / "ATLAS.exe",
            "ATLAS Desktop Studio",
            "ATLAS.exe",
            ["-lshell32", "-lshlwapi", "-luser32"],
        ),
        (
            INSTALLER_DIR / "uninstaller.c",
            INSTALLER_DIR / "Uninstall-ATLAS.exe",
            "ATLAS Uninstaller",
            "Uninstall-ATLAS.exe",
            ["-lshell32", "-lole32", "-ladvapi32", "-luser32"],
        ),
    ]

    for c_src, out_exe, desc, orig, libs in targets:
        compile_win32_exe(c_src, out_exe, desc, orig, libs)
        meta = verify_exe_metadata(out_exe, desc)
        print(
            f"[OK] Built {out_exe.relative_to(ROOT)} "
            f"({out_exe.stat().st_size / 1024:.1f} KB) — "
            f"Publisher='{meta['Publisher']}', Author='{meta['Author']}', Description='{meta['FileDescription']}'"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
