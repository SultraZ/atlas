/*
 * ATLAS.exe — Native Win32 Desktop Application Launcher
 * Publisher / Author: ATLAS
 *
 * Launches ATLAS Desktop Studio directly without any console popup or .bat script:
 *   1. Native Windows virtual environment (<AppDir>\.venv\Scripts\python.exe webapp\desktop.py)
 *   2. WSL2 'Atlas' environment when run inside \\wsl.localhost\... or WSL-configured host
 *   3. Prompts to launch ATLAS-Setup.exe automatically if run prior to first-time setup
 */
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#include <shlwapi.h>
#include <wchar.h>

static BOOL file_exists(const wchar_t *p) {
    DWORD a = GetFileAttributesW(p);
    return (a != INVALID_FILE_ATTRIBUTES) && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

static int run_hidden_process(wchar_t *cmdLine, const wchar_t *workDir, BOOL waitProc) {
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;

    if (!CreateProcessW(NULL, cmdLine, NULL, NULL, FALSE,
                        CREATE_NO_WINDOW, NULL, workDir, &si, &pi)) {
        return -1;
    }
    DWORD exitCode = 0;
    if (waitProc) {
        WaitForSingleObject(pi.hProcess, INFINITE);
        GetExitCodeProcess(pi.hProcess, &exitCode);
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return (int)exitCode;
}

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE hPrev, PWSTR lpCmdLine, int nShowCmd) {
    (void)hInst; (void)hPrev; (void)nShowCmd;

    wchar_t exePath[MAX_PATH * 2] = {0};
    GetModuleFileNameW(NULL, exePath, MAX_PATH * 2 - 1);

    wchar_t appDir[MAX_PATH * 2] = {0};
    wcsncpy(appDir, exePath, MAX_PATH * 2 - 1);
    wchar_t *slash = wcsrchr(appDir, L'\\');
    if (slash) *slash = L'\0';

    wchar_t venvPyw[MAX_PATH * 2] = {0};
    wchar_t venvPy[MAX_PATH * 2]  = {0};
    wchar_t desktopPy[MAX_PATH * 2] = {0};
    wchar_t setupExe[MAX_PATH * 2]  = {0};

    _snwprintf(venvPyw, MAX_PATH * 2 - 1, L"%ls\\.venv\\Scripts\\pythonw.exe", appDir);
    _snwprintf(venvPy,  MAX_PATH * 2 - 1, L"%ls\\.venv\\Scripts\\python.exe", appDir);
    _snwprintf(desktopPy, MAX_PATH * 2 - 1, L"%ls\\webapp\\desktop.py", appDir);
    _snwprintf(setupExe,  MAX_PATH * 2 - 1, L"%ls\\ATLAS-Setup.exe", appDir);

    /* 1. Native Windows virtual environment */
    const wchar_t *chosenPy = file_exists(venvPy) ? venvPy : (file_exists(venvPyw) ? venvPyw : NULL);
    if (chosenPy && file_exists(desktopPy)) {
        wchar_t cmd[4096] = {0};
        if (lpCmdLine && lpCmdLine[0]) {
            _snwprintf(cmd, 4095, L"\"%ls\" \"%ls\" %ls", chosenPy, desktopPy, lpCmdLine);
        } else {
            _snwprintf(cmd, 4095, L"\"%ls\" \"%ls\"", chosenPy, desktopPy);
        }
        return run_hidden_process(cmd, appDir, TRUE);
    }

    /* 2. Check if located on a WSL UNC path (\\wsl.localhost\Distro\... or \\wsl$\Distro\...) */
    if (wcsncmp(appDir, L"\\\\wsl.localhost\\", 16) == 0 || wcsncmp(appDir, L"\\\\wsl$\\", 7) == 0) {
        const wchar_t *p = (appDir[5] == L'$') ? (appDir + 7) : (appDir + 16);
        const wchar_t *nextSlash = wcschr(p, L'\\');
        if (nextSlash) {
            wchar_t distro[128] = {0};
            size_t dlen = (size_t)(nextSlash - p);
            if (dlen < 127) {
                wcsncpy(distro, p, dlen);
                wchar_t linuxPath[2048] = {0};
                size_t j = 0;
                for (const wchar_t *c = nextSlash; *c && j < 2046; ++c) {
                    linuxPath[j++] = (*c == L'\\') ? L'/' : *c;
                }
                wchar_t wslCmd[4096] = {0};
                _snwprintf(wslCmd, 4095,
                    L"wsl.exe -d \"%ls\" bash -lc \"cd '%ls' && chmod +x launch_atlas.sh && ./launch_atlas.sh\"",
                    distro, linuxPath);
                wchar_t sysDir[MAX_PATH] = {0};
                GetSystemDirectoryW(sysDir, MAX_PATH);
                return run_hidden_process(wslCmd, sysDir, TRUE);
            }
        }
    }

    /* 3. Not configured yet — offer to run ATLAS-Setup.exe right away */
    if (file_exists(setupExe)) {
        int ans = MessageBoxW(
            NULL,
            L"ATLAS Desktop Studio needs a quick one-time setup on this computer before its first launch.\n\n"
            L"Would you like to open the ATLAS Setup Wizard now?",
            L"ATLAS Desktop Studio",
            MB_YESNO | MB_ICONQUESTION
        );
        if (ans == IDYES) {
            ShellExecuteW(NULL, L"open", setupExe, NULL, appDir, SW_SHOWNORMAL);
        }
        return 0;
    }

    MessageBoxW(
        NULL,
        L"Could not find the configured ATLAS runtime (.venv) or ATLAS-Setup.exe.\n\n"
        L"Please run ATLAS-Setup.exe to install ATLAS Desktop Studio.",
        L"ATLAS Desktop Studio — Setup Required",
        MB_OK | MB_ICONINFORMATION
    );
    return 1;
}
