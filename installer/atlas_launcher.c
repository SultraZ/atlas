/*
 * ATLAS.exe — Native Win32 Desktop Application Launcher
 * Publisher / Author: ATLAS
 *
 * Launches ATLAS Desktop Studio directly without any console popup or .bat script:
 *   1. Native Windows virtual environment (<AppDir>\.venv\Scripts\python.exe webapp\desktop.py)
 *      with UTF-8 stream configuration + startup error diagnostics from atlas_desktop.log
 *   2. WSL2 'Atlas' environment when run inside \\wsl.localhost\... or \\wsl$\...
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
#include <stdlib.h>
#include <wchar.h>

static BOOL file_exists(const wchar_t *p) {
    DWORD a = GetFileAttributesW(p);
    return (a != INVALID_FILE_ATTRIBUTES) && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

static void show_startup_error(const wchar_t *logPath) {
    wchar_t msg[4096] = L"ATLAS Desktop Studio exited unexpectedly during startup.\n\n";
    HANDLE hFile = CreateFileW(logPath, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE) {
        DWORD sz = GetFileSize(hFile, NULL);
        if (sz > 0 && sz != INVALID_FILE_SIZE) {
            DWORD readLen = (sz > 3000) ? 3000 : sz;
            if (sz > readLen) SetFilePointer(hFile, (LONG)(sz - readLen), NULL, FILE_BEGIN);
            char buf[3008] = {0};
            DWORD got = 0;
            if (ReadFile(hFile, buf, readLen, &got, NULL) && got > 0) {
                buf[got] = '\0';
                wchar_t wtail[3200] = {0};
                MultiByteToWideChar(CP_UTF8, 0, buf, -1, wtail, 3190);
                wcsncat(msg, wtail, 3800);
            }
        }
        CloseHandle(hFile);
    }
    MessageBoxW(NULL, msg, L"ATLAS Desktop Studio — Startup Error", MB_OK | MB_ICONERROR);
}

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE hPrev, PWSTR lpCmdLine, int nShowCmd) {
    (void)hInst; (void)hPrev; (void)nShowCmd;

    wchar_t exePath[MAX_PATH * 2] = {0};
    GetModuleFileNameW(NULL, exePath, MAX_PATH * 2 - 1);

    wchar_t appDir[MAX_PATH * 2] = {0};
    wcsncpy(appDir, exePath, MAX_PATH * 2 - 1);
    wchar_t *slash = wcsrchr(appDir, L'\\');
    if (slash) *slash = L'\0';

    wchar_t venvPy[MAX_PATH * 2]    = {0};
    wchar_t desktopPy[MAX_PATH * 2] = {0};
    wchar_t setupExe[MAX_PATH * 2]  = {0};
    wchar_t logPath[MAX_PATH * 2]   = {0};

    _snwprintf(venvPy,    MAX_PATH * 2 - 1, L"%ls\\.venv\\Scripts\\python.exe", appDir);
    _snwprintf(desktopPy, MAX_PATH * 2 - 1, L"%ls\\webapp\\desktop.py", appDir);
    _snwprintf(setupExe,  MAX_PATH * 2 - 1, L"%ls\\ATLAS-Setup.exe", appDir);
    _snwprintf(logPath,   MAX_PATH * 2 - 1, L"%ls\\webapp\\atlas_desktop.log", appDir);

    /* 1. Native Windows virtual environment */
    if (file_exists(venvPy) && file_exists(desktopPy)) {
        SetEnvironmentVariableW(L"PYTHONUTF8", L"1");
        SetEnvironmentVariableW(L"PYTHONIOENCODING", L"utf-8");
        SetEnvironmentVariableW(L"ATLAS_LOG_FILE", logPath);

        wchar_t cmd[4096] = {0};
        if (lpCmdLine && lpCmdLine[0]) {
            _snwprintf(cmd, 4095, L"\"%ls\" \"%ls\" %ls", venvPy, desktopPy, lpCmdLine);
        } else {
            _snwprintf(cmd, 4095, L"\"%ls\" \"%ls\"", venvPy, desktopPy);
        }

        STARTUPINFOW si;
        PROCESS_INFORMATION pi;
        ZeroMemory(&si, sizeof(si));
        ZeroMemory(&pi, sizeof(pi));
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESHOWWINDOW;
        si.wShowWindow = SW_HIDE;

        if (!CreateProcessW(NULL, cmd, NULL, NULL, FALSE,
                            CREATE_NO_WINDOW, NULL, appDir, &si, &pi)) {
            MessageBoxW(NULL, L"Failed to launch ATLAS Python runtime (.venv\\Scripts\\python.exe).",
                        L"ATLAS Desktop Studio", MB_OK | MB_ICONERROR);
            return 1;
        }

        /* Observe early startup window (4 seconds); if healthy & alive, hand off cleanly */
        DWORD waitRes = WaitForSingleObject(pi.hProcess, 4000);
        if (waitRes == WAIT_OBJECT_0) {
            DWORD exitCode = 0;
            GetExitCodeProcess(pi.hProcess, &exitCode);
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
            if (exitCode != 0) {
                show_startup_error(logPath);
                return (int)exitCode;
            }
            return 0;
        }
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        return 0;
    }

    /* 2. WSL UNC path support (\\wsl.localhost\Distro\... or \\wsl$\Distro\...) */
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
                STARTUPINFOW si;
                PROCESS_INFORMATION pi;
                ZeroMemory(&si, sizeof(si));
                ZeroMemory(&pi, sizeof(pi));
                si.cb = sizeof(si);
                si.dwFlags = STARTF_USESHOWWINDOW;
                si.wShowWindow = SW_HIDE;
                if (CreateProcessW(NULL, wslCmd, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, sysDir, &si, &pi)) {
                    CloseHandle(pi.hThread);
                    CloseHandle(pi.hProcess);
                    return 0;
                }
            }
        }
    }

    /* 3. Offer one-click Setup Wizard if runtime isn't set up yet */
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
