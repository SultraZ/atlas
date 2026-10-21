/*
 * Uninstall-ATLAS.exe — Native Win32 Uninstaller
 * Publisher / Author: ATLAS
 *
 * Cleanly removes Desktop & Start Menu shortcuts, Windows Installed Apps registry
 * metadata, and schedules removal of the application directory upon confirmation.
 */
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shlobj.h>
#include <shellapi.h>
#include <wchar.h>

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE hPrev, PWSTR lpCmdLine, int nShowCmd) {
    (void)hInst; (void)hPrev; (void)lpCmdLine; (void)nShowCmd;

    int confirm = MessageBoxW(
        NULL,
        L"Are you sure you want to completely remove ATLAS Desktop Studio and all of its shortcuts from your computer?",
        L"Uninstall ATLAS Desktop Studio",
        MB_YESNO | MB_ICONQUESTION
    );
    if (confirm != IDYES) return 0;

    wchar_t exePath[MAX_PATH * 2] = {0};
    GetModuleFileNameW(NULL, exePath, MAX_PATH * 2 - 1);

    wchar_t appDir[MAX_PATH * 2] = {0};
    wcsncpy(appDir, exePath, MAX_PATH * 2 - 1);
    wchar_t *slash = wcsrchr(appDir, L'\\');
    if (slash) *slash = L'\0';

    /* Remove Desktop shortcut */
    wchar_t deskDir[MAX_PATH] = {0};
    if (SUCCEEDED(SHGetFolderPathW(NULL, CSIDL_DESKTOPDIRECTORY, NULL, 0, deskDir))) {
        wchar_t lnk[MAX_PATH * 2] = {0};
        _snwprintf(lnk, MAX_PATH * 2 - 1, L"%ls\\ATLAS Desktop Studio.lnk", deskDir);
        DeleteFileW(lnk);
    }

    /* Remove Start Menu folder */
    wchar_t progDir[MAX_PATH] = {0};
    if (SUCCEEDED(SHGetFolderPathW(NULL, CSIDL_PROGRAMS, NULL, 0, progDir))) {
        wchar_t startFolder[MAX_PATH * 2] = {0};
        wchar_t lnk1[MAX_PATH * 2] = {0};
        wchar_t lnk2[MAX_PATH * 2] = {0};
        _snwprintf(startFolder, MAX_PATH * 2 - 1, L"%ls\\ATLAS", progDir);
        _snwprintf(lnk1, MAX_PATH * 2 - 1, L"%ls\\ATLAS Desktop Studio.lnk", startFolder);
        _snwprintf(lnk2, MAX_PATH * 2 - 1, L"%ls\\Uninstall ATLAS Desktop Studio.lnk", startFolder);
        DeleteFileW(lnk1);
        DeleteFileW(lnk2);
        RemoveDirectoryW(startFolder);
    }

    /* Remove Windows Installed Apps Registry Entry */
    RegDeleteKeyW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\ATLASDesktopStudio");

    /* Schedule self-cleaning directory removal if inside Programs\ATLAS */
    wchar_t tempDir[MAX_PATH] = {0};
    GetTempPathW(MAX_PATH, tempDir);
    wchar_t cmdLine[4096] = {0};
    _snwprintf(cmdLine, 4095,
        L"cmd.exe /c ping 127.0.0.1 -n 3 >nul & rmdir /s /q \"%ls\"",
        appDir);

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    if (CreateProcessW(NULL, cmdLine, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, tempDir, &si, &pi)) {
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }

    MessageBoxW(
        NULL,
        L"ATLAS Desktop Studio has been uninstalled from your computer.",
        L"ATLAS Uninstaller",
        MB_OK | MB_ICONINFORMATION
    );
    return 0;
}
