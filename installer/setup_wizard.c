/*
 * ATLAS-Setup.exe — Native Win32 Graphical Setup Wizard
 * Publisher / Author: ATLAS
 * Product: ATLAS Desktop Studio v1.0.0
 */
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commctrl.h>
#include <shlobj.h>
#include <shellapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "worker_payload.h"

#define WM_INSTALL_LOG  (WM_APP + 10)
#define WM_INSTALL_PROG (WM_APP + 11)
#define WM_INSTALL_DONE (WM_APP + 12)

#define ID_RB_STANDARD  101
#define ID_RB_FENICS    102
#define ID_CMB_COMPUTE  103
#define ID_TXT_DIR      104
#define ID_BTN_BROWSE   105
#define ID_CHK_DESKTOP  106
#define ID_CHK_START    107
#define ID_CHK_LAUNCH   108
#define ID_BTN_INSTALL  109
#define ID_BTN_CANCEL   110
#define ID_PROG_BAR     111
#define ID_TXT_LOG      112

static int g_dpi = 96;
static int SX(int v) { return MulDiv(v, g_dpi, 96); }

static HWND g_hWnd = NULL;
static HWND g_hRbStd, g_hRbFen, g_hCmbDev, g_hTxtDir, g_hBtnBrowse;
static HWND g_hChkDesk, g_hChkStart, g_hChkLaunch;
static HWND g_hProg, g_hLog, g_hBtnInstall, g_hBtnCancel, g_hGrp1, g_hGrp2;
static HFONT g_fTitle, g_fSub, g_fBold, g_fNorm, g_fSmall, g_fMono;
static HBRUSH g_brBg, g_brLog;
static HICON g_hAppIcon = NULL;
static volatile BOOL g_installing = FALSE;
static wchar_t g_srcDir[MAX_PATH * 2] = {0};
static wchar_t g_gpuInfo[256] = L"Detecting graphics hardware...";

typedef struct {
    wchar_t destDir[MAX_PATH * 2];
    BOOL withFenics;
    int computeIdx; /* 0 = auto, 1 = cuda, 2 = cpu */
    BOOL makeDesktop;
    BOOL makeStart;
    BOOL launchAfter;
} InstallParams;

static void detect_gpu_summary(void) {
    DISPLAY_DEVICEW dd;
    ZeroMemory(&dd, sizeof(dd));
    dd.cb = sizeof(dd);
    wchar_t firstAdapter[128] = {0};
    for (DWORD i = 0; EnumDisplayDevicesW(NULL, i, &dd, 0); ++i) {
        if (dd.DeviceString[0]) {
            if (!firstAdapter[0]) wcsncpy(firstAdapter, dd.DeviceString, 127);
            if (wcsstr(dd.DeviceString, L"NVIDIA") || wcsstr(dd.DeviceString, L"GeForce") ||
                wcsstr(dd.DeviceString, L"RTX") || wcsstr(dd.DeviceString, L"Quadro")) {
                _snwprintf(g_gpuInfo, 255, L"Detected GPU: %ls (CUDA Acceleration Supported)", dd.DeviceString);
                return;
            }
        }
        ZeroMemory(&dd, sizeof(dd));
        dd.cb = sizeof(dd);
    }
    if (firstAdapter[0]) {
        _snwprintf(g_gpuInfo, 255, L"Graphics: %ls  |  Multi-Core CPU Mode Ready", firstAdapter);
    } else {
        wcsncpy(g_gpuInfo, L"System Hardware Ready (Auto-detects CUDA GPU or Multi-Core CPU)", 255);
    }
}

static void append_log_line(const wchar_t *msg) {
    if (!g_hLog) return;
    int len = GetWindowTextLengthW(g_hLog);
    SendMessageW(g_hLog, EM_SETSEL, (WPARAM)len, (LPARAM)len);
    SendMessageW(g_hLog, EM_REPLACESEL, FALSE, (LPARAM)msg);
    SendMessageW(g_hLog, EM_REPLACESEL, FALSE, (LPARAM)L"\r\n");
    SendMessageW(g_hLog, EM_SCROLLCARET, 0, 0);
}

static void handle_worker_line(const char *utf8Line) {
    if (!utf8Line || !utf8Line[0]) return;
    int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8Line, -1, NULL, 0);
    if (wlen <= 0) return;
    wchar_t *wbuf = (wchar_t *)calloc((size_t)wlen + 4, sizeof(wchar_t));
    if (!wbuf) return;
    MultiByteToWideChar(CP_UTF8, 0, utf8Line, -1, wbuf, wlen);

    if (wcsncmp(wbuf, L"@@PROGRESS|", 11) == 0) {
        int pct = _wtoi(wbuf + 11);
        wchar_t *bar = wcschr(wbuf + 11, L'|');
        if (bar && bar[1]) {
            wchar_t *copy = _wcsdup(bar + 1);
            PostMessageW(g_hWnd, WM_INSTALL_PROG, (WPARAM)pct, (LPARAM)copy);
            free(wbuf);
            return;
        }
    }
    PostMessageW(g_hWnd, WM_INSTALL_LOG, 0, (LPARAM)wbuf);
}

static DWORD WINAPI install_worker_thread(LPVOID lpParam) {
    InstallParams *p = (InstallParams *)lpParam;

    wchar_t tempDir[MAX_PATH] = {0};
    GetTempPathW(MAX_PATH, tempDir);
    wchar_t ps1Path[MAX_PATH * 2] = {0};
    _snwprintf(ps1Path, MAX_PATH * 2 - 1, L"%lsatlas_setup_worker_%lu.ps1", tempDir, GetCurrentProcessId());

    HANDLE hFile = CreateFileW(ps1Path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        PostMessageW(g_hWnd, WM_INSTALL_LOG, 0, (LPARAM)_wcsdup(L"ERROR: Could not write temporary installer script."));
        PostMessageW(g_hWnd, WM_INSTALL_DONE, 1, (LPARAM)p);
        return 1;
    }
    static const unsigned char bom[3] = {0xEF, 0xBB, 0xBF};
    DWORD written = 0;
    WriteFile(hFile, bom, 3, &written, NULL);
    WriteFile(hFile, ATLAS_WORKER_PS1, (DWORD)ATLAS_WORKER_PS1_LEN, &written, NULL);
    CloseHandle(hFile);

    SetEnvironmentVariableW(L"ATLAS_SRC_DIR", g_srcDir);
    SetEnvironmentVariableW(L"ATLAS_DEST_DIR", p->destDir);
    SetEnvironmentVariableW(L"ATLAS_EDITION", p->withFenics ? L"fenics" : L"standard");
    SetEnvironmentVariableW(L"ATLAS_COMPUTE",
        p->computeIdx == 1 ? L"cuda" : (p->computeIdx == 2 ? L"cpu" : L"auto"));
    SetEnvironmentVariableW(L"ATLAS_SHORTCUT_DESKTOP", p->makeDesktop ? L"1" : L"0");
    SetEnvironmentVariableW(L"ATLAS_SHORTCUT_STARTMENU", p->makeStart ? L"1" : L"0");

    SECURITY_ATTRIBUTES sa = { sizeof(SECURITY_ATTRIBUTES), NULL, TRUE };
    HANDLE hRead = NULL, hWrite = NULL;
    CreatePipe(&hRead, &hWrite, &sa, 0);
    SetHandleInformation(hRead, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.hStdOutput = hWrite;
    si.hStdError  = hWrite;
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
    si.wShowWindow = SW_HIDE;

    wchar_t cmd[4096] = {0};
    _snwprintf(cmd, 4095,
        L"powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"%ls\"",
        ps1Path);

    DWORD exitCode = 1;
    if (CreateProcessW(NULL, cmd, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        CloseHandle(hWrite);
        hWrite = NULL;

        char chunk[1024];
        char lineBuf[8192];
        size_t lineLen = 0;
        DWORD bytesRead = 0;
        while (ReadFile(hRead, chunk, sizeof(chunk), &bytesRead, NULL) && bytesRead > 0) {
            for (DWORD i = 0; i < bytesRead; ++i) {
                char ch = chunk[i];
                if (ch == '\n' || ch == '\r') {
                    if (lineLen > 0) {
                        lineBuf[lineLen] = '\0';
                        handle_worker_line(lineBuf);
                        lineLen = 0;
                    }
                } else if (lineLen + 1 < sizeof(lineBuf)) {
                    lineBuf[lineLen++] = ch;
                }
            }
        }
        if (lineLen > 0) {
            lineBuf[lineLen] = '\0';
            handle_worker_line(lineBuf);
        }
        WaitForSingleObject(pi.hProcess, INFINITE);
        GetExitCodeProcess(pi.hProcess, &exitCode);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    } else {
        if (hWrite) CloseHandle(hWrite);
        PostMessageW(g_hWnd, WM_INSTALL_LOG, 0, (LPARAM)_wcsdup(L"ERROR: Failed to start Windows PowerShell host."));
    }

    if (hRead) CloseHandle(hRead);
    DeleteFileW(ps1Path);

    PostMessageW(g_hWnd, WM_INSTALL_DONE, (WPARAM)exitCode, (LPARAM)p);
    return exitCode;
}

static void start_installation(void) {
    if (g_installing) return;

    InstallParams *p = (InstallParams *)calloc(1, sizeof(InstallParams));
    GetWindowTextW(g_hTxtDir, p->destDir, MAX_PATH * 2 - 1);
    p->withFenics  = (SendMessageW(g_hRbFen, BM_GETCHECK, 0, 0) == BST_CHECKED);
    p->computeIdx  = (int)SendMessageW(g_hCmbDev, CB_GETCURSEL, 0, 0);
    p->makeDesktop = (SendMessageW(g_hChkDesk, BM_GETCHECK, 0, 0) == BST_CHECKED);
    p->makeStart   = (SendMessageW(g_hChkStart, BM_GETCHECK, 0, 0) == BST_CHECKED);
    p->launchAfter = (SendMessageW(g_hChkLaunch, BM_GETCHECK, 0, 0) == BST_CHECKED);

    if (!p->destDir[0]) {
        free(p);
        MessageBoxW(g_hWnd, L"Please choose a destination folder.", L"ATLAS Setup Wizard", MB_OK | MB_ICONWARNING);
        return;
    }

    g_installing = TRUE;
    EnableWindow(g_hBtnInstall, FALSE);
    SetWindowTextW(g_hBtnInstall, L"Installing...");
    EnableWindow(g_hRbStd, FALSE);
    EnableWindow(g_hRbFen, FALSE);
    EnableWindow(g_hCmbDev, FALSE);
    EnableWindow(g_hTxtDir, FALSE);
    EnableWindow(g_hBtnBrowse, FALSE);
    EnableWindow(g_hChkDesk, FALSE);
    EnableWindow(g_hChkStart, FALSE);
    SetWindowTextW(g_hLog, L"");
    SendMessageW(g_hProg, PBM_SETPOS, 2, 0);

    append_log_line(L"Starting ATLAS Desktop Studio Setup Wizard (Publisher: ATLAS)...");
    HANDLE hThread = CreateThread(NULL, 0, install_worker_thread, p, 0, NULL);
    if (hThread) CloseHandle(hThread);
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_CREATE: {
        wchar_t defaultDest[MAX_PATH * 2] = {0};
        wchar_t localApp[MAX_PATH] = {0};
        if (SUCCEEDED(SHGetFolderPathW(NULL, CSIDL_LOCAL_APPDATA, NULL, 0, localApp))) {
            _snwprintf(defaultDest, MAX_PATH * 2 - 1, L"%ls\\Programs\\ATLAS", localApp);
        } else {
            wcsncpy(defaultDest, L"C:\\ATLAS", MAX_PATH * 2 - 1);
        }

        HWND hGpu = CreateWindowExW(0, L"STATIC", g_gpuInfo,
            WS_CHILD | WS_VISIBLE,
            SX(24), SX(84), SX(620), SX(20), hwnd, NULL, NULL, NULL);
        SendMessageW(hGpu, WM_SETFONT, (WPARAM)g_fBold, TRUE);

        g_hGrp1 = CreateWindowExW(0, L"BUTTON", L"1. Select FEM Validation Edition",
            WS_CHILD | WS_VISIBLE | BS_GROUPBOX,
            SX(22), SX(110), SX(624), SX(128), hwnd, NULL, NULL, NULL);
        SendMessageW(g_hGrp1, WM_SETFONT, (WPARAM)g_fBold, TRUE);

        g_hRbStd = CreateWindowExW(0, L"BUTTON",
            L"Standard Cross-Platform Edition  (Recommended — No WSL Required)",
            WS_CHILD | WS_VISIBLE | BS_AUTORADIOBUTTON | WS_GROUP,
            SX(38), SX(134), SX(590), SX(22), hwnd, (HMENU)ID_RB_STANDARD, NULL, NULL);
        SendMessageW(g_hRbStd, WM_SETFONT, (WPARAM)g_fBold, TRUE);
        SendMessageW(g_hRbStd, BM_SETCHECK, BST_CHECKED, 0);

        HWND hSub1 = CreateWindowExW(0, L"STATIC",
            L"Python + GPU/CPU PyTorch + Built-in Body-Fitted P2 FEM (SciPy). Runs natively on every Windows PC.",
            WS_CHILD | WS_VISIBLE,
            SX(58), SX(156), SX(572), SX(18), hwnd, NULL, NULL, NULL);
        SendMessageW(hSub1, WM_SETFONT, (WPARAM)g_fSmall, TRUE);

        g_hRbFen = CreateWindowExW(0, L"BUTTON",
            L"Full FEniCS Edition  (WSL2 + Legacy FEniCS 2019.1.0 + Built-in P2 FEM)",
            WS_CHILD | WS_VISIBLE | BS_AUTORADIOBUTTON,
            SX(38), SX(182), SX(590), SX(22), hwnd, (HMENU)ID_RB_FENICS, NULL, NULL);
        SendMessageW(g_hRbFen, WM_SETFONT, (WPARAM)g_fBold, TRUE);

        HWND hSub2 = CreateWindowExW(0, L"STATIC",
            L"Also configures WSL2 Ubuntu + Miniconda 'Atlas' env with genuine dolfin + mshr alongside Built-in FEM.",
            WS_CHILD | WS_VISIBLE,
            SX(58), SX(204), SX(572), SX(18), hwnd, NULL, NULL, NULL);
        SendMessageW(hSub2, WM_SETFONT, (WPARAM)g_fSmall, TRUE);

        g_hGrp2 = CreateWindowExW(0, L"BUTTON", L"2. Compute Hardware & Installation Options",
            WS_CHILD | WS_VISIBLE | BS_GROUPBOX,
            SX(22), SX(246), SX(624), SX(136), hwnd, NULL, NULL, NULL);
        SendMessageW(g_hGrp2, WM_SETFONT, (WPARAM)g_fBold, TRUE);

        HWND hLblDev = CreateWindowExW(0, L"STATIC", L"Compute Engine:",
            WS_CHILD | WS_VISIBLE,
            SX(38), SX(275), SX(118), SX(22), hwnd, NULL, NULL, NULL);
        SendMessageW(hLblDev, WM_SETFONT, (WPARAM)g_fNorm, TRUE);

        g_hCmbDev = CreateWindowExW(0, L"COMBOBOX", L"",
            WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP,
            SX(160), SX(271), SX(468), SX(200), hwnd, (HMENU)ID_CMB_COMPUTE, NULL, NULL);
        SendMessageW(g_hCmbDev, WM_SETFONT, (WPARAM)g_fNorm, TRUE);
        SendMessageW(g_hCmbDev, CB_ADDSTRING, 0, (LPARAM)L"Auto-Detect Hardware (Recommended — CUDA GPU if present, else CPU)");
        SendMessageW(g_hCmbDev, CB_ADDSTRING, 0, (LPARAM)L"NVIDIA GPU Acceleration (CUDA 12.4)");
        SendMessageW(g_hCmbDev, CB_ADDSTRING, 0, (LPARAM)L"CPU Only (Smaller download, works on every PC/laptop)");
        SendMessageW(g_hCmbDev, CB_SETCURSEL, 0, 0);

        HWND hLblDir = CreateWindowExW(0, L"STATIC", L"Install Location:",
            WS_CHILD | WS_VISIBLE,
            SX(38), SX(312), SX(118), SX(22), hwnd, NULL, NULL, NULL);
        SendMessageW(hLblDir, WM_SETFONT, (WPARAM)g_fNorm, TRUE);

        g_hTxtDir = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", defaultDest,
            WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL | WS_TABSTOP,
            SX(160), SX(309), SX(380), SX(25), hwnd, (HMENU)ID_TXT_DIR, NULL, NULL);
        SendMessageW(g_hTxtDir, WM_SETFONT, (WPARAM)g_fNorm, TRUE);

        g_hBtnBrowse = CreateWindowExW(0, L"BUTTON", L"Browse...",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON | WS_TABSTOP,
            SX(548), SX(308), SX(80), SX(27), hwnd, (HMENU)ID_BTN_BROWSE, NULL, NULL);
        SendMessageW(g_hBtnBrowse, WM_SETFONT, (WPARAM)g_fNorm, TRUE);

        g_hChkDesk = CreateWindowExW(0, L"BUTTON", L"Desktop shortcut",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX | WS_TABSTOP,
            SX(38), SX(348), SX(148), SX(22), hwnd, (HMENU)ID_CHK_DESKTOP, NULL, NULL);
        SendMessageW(g_hChkDesk, WM_SETFONT, (WPARAM)g_fNorm, TRUE);
        SendMessageW(g_hChkDesk, BM_SETCHECK, BST_CHECKED, 0);

        g_hChkStart = CreateWindowExW(0, L"BUTTON", L"Start Menu & Installed Apps",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX | WS_TABSTOP,
            SX(196), SX(348), SX(212), SX(22), hwnd, (HMENU)ID_CHK_START, NULL, NULL);
        SendMessageW(g_hChkStart, WM_SETFONT, (WPARAM)g_fNorm, TRUE);
        SendMessageW(g_hChkStart, BM_SETCHECK, BST_CHECKED, 0);

        g_hChkLaunch = CreateWindowExW(0, L"BUTTON", L"Launch ATLAS when complete",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX | WS_TABSTOP,
            SX(418), SX(348), SX(212), SX(22), hwnd, (HMENU)ID_CHK_LAUNCH, NULL, NULL);
        SendMessageW(g_hChkLaunch, WM_SETFONT, (WPARAM)g_fNorm, TRUE);
        SendMessageW(g_hChkLaunch, BM_SETCHECK, BST_CHECKED, 0);

        g_hProg = CreateWindowExW(0, PROGRESS_CLASSW, NULL,
            WS_CHILD | WS_VISIBLE | PBS_SMOOTH,
            SX(22), SX(394), SX(624), SX(20), hwnd, (HMENU)ID_PROG_BAR, NULL, NULL);
        SendMessageW(g_hProg, PBM_SETRANGE, 0, MAKELPARAM(0, 100));

        g_hLog = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT",
            L"Ready to install ATLAS Desktop Studio (Published by ATLAS).\r\n"
            L"Click 'Install ATLAS' to configure everything automatically.",
            WS_CHILD | WS_VISIBLE | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY | WS_VSCROLL,
            SX(22), SX(422), SX(624), SX(98), hwnd, (HMENU)ID_TXT_LOG, NULL, NULL);
        SendMessageW(g_hLog, WM_SETFONT, (WPARAM)g_fMono, TRUE);

        g_hBtnCancel = CreateWindowExW(0, L"BUTTON", L"Close",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON | WS_TABSTOP,
            SX(396), SX(532), SX(92), SX(34), hwnd, (HMENU)ID_BTN_CANCEL, NULL, NULL);
        SendMessageW(g_hBtnCancel, WM_SETFONT, (WPARAM)g_fNorm, TRUE);

        g_hBtnInstall = CreateWindowExW(0, L"BUTTON", L"Install ATLAS",
            WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON | WS_TABSTOP,
            SX(498), SX(532), SX(148), SX(34), hwnd, (HMENU)ID_BTN_INSTALL, NULL, NULL);
        SendMessageW(g_hBtnInstall, WM_SETFONT, (WPARAM)g_fBold, TRUE);
        return 0;
    }

    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        RECT banner = { 0, 0, SX(680), SX(74) };
        HBRUSH hNavy = CreateSolidBrush(RGB(15, 23, 42));
        FillRect(hdc, &banner, hNavy);
        DeleteObject(hNavy);

        RECT accent = { 0, SX(72), SX(680), SX(74) };
        HBRUSH hBlue = CreateSolidBrush(RGB(37, 99, 235));
        FillRect(hdc, &accent, hBlue);
        DeleteObject(hBlue);

        if (g_hAppIcon) {
            DrawIconEx(hdc, SX(22), SX(17), g_hAppIcon, SX(40), SX(40), 0, NULL, DI_NORMAL);
        }

        SetBkMode(hdc, TRANSPARENT);
        HFONT oldF = (HFONT)SelectObject(hdc, g_fTitle);
        SetTextColor(hdc, RGB(255, 255, 255));
        RECT rcT = { SX(74), SX(13), SX(650), SX(42) };
        DrawTextW(hdc, L"ATLAS Desktop Studio — Setup Wizard", -1, &rcT, DT_LEFT | DT_SINGLELINE);

        SelectObject(hdc, g_fSub);
        SetTextColor(hdc, RGB(148, 163, 184));
        RECT rcS = { SX(75), SX(42), SX(650), SX(66) };
        DrawTextW(hdc, L"Publisher: ATLAS   \x2022   Version 1.0.0   \x2022   Deep-Energy & Dual FEM Suite", -1, &rcS, DT_LEFT | DT_SINGLELINE);
        SelectObject(hdc, oldF);

        EndPaint(hwnd, &ps);
        return 0;
    }

    case WM_CTLCOLORSTATIC: {
        HDC hdc = (HDC)wParam;
        HWND hCtl = (HWND)lParam;
        if (hCtl == g_hLog) {
            SetBkColor(hdc, RGB(15, 23, 42));
            SetTextColor(hdc, RGB(226, 232, 240));
            return (LRESULT)g_brLog;
        }
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, RGB(30, 41, 59));
        return (LRESULT)g_brBg;
    }

    case WM_COMMAND: {
        int id = LOWORD(wParam);
        if (id == ID_BTN_BROWSE) {
            BROWSEINFOW bi;
            ZeroMemory(&bi, sizeof(bi));
            bi.hwndOwner = hwnd;
            bi.lpszTitle = L"Choose destination folder for ATLAS Desktop Studio:";
            bi.ulFlags   = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
            LPITEMIDLIST pidl = SHBrowseForFolderW(&bi);
            if (pidl) {
                wchar_t chosen[MAX_PATH] = {0};
                if (SHGetPathFromIDListW(pidl, chosen) && chosen[0]) {
                    SetWindowTextW(g_hTxtDir, chosen);
                }
                CoTaskMemFree(pidl);
            }
            return 0;
        }
        if (id == ID_BTN_INSTALL) {
            start_installation();
            return 0;
        }
        if (id == ID_BTN_CANCEL) {
            if (g_installing) {
                int q = MessageBoxW(hwnd,
                    L"Installation is currently in progress. Are you sure you want to exit?",
                    L"ATLAS Setup Wizard", MB_YESNO | MB_ICONWARNING);
                if (q != IDYES) return 0;
            }
            DestroyWindow(hwnd);
            return 0;
        }
        break;
    }

    case WM_INSTALL_LOG: {
        wchar_t *s = (wchar_t *)lParam;
        if (s) {
            append_log_line(s);
            free(s);
        }
        return 0;
    }

    case WM_INSTALL_PROG: {
        int pct = (int)wParam;
        wchar_t *s = (wchar_t *)lParam;
        SendMessageW(g_hProg, PBM_SETPOS, (WPARAM)pct, 0);
        if (s) {
            wchar_t fmt[2048] = {0};
            _snwprintf(fmt, 2047, L"[%d%%] %ls", pct, s);
            append_log_line(fmt);
            free(s);
        }
        return 0;
    }

    case WM_INSTALL_DONE: {
        DWORD code = (DWORD)wParam;
        InstallParams *p = (InstallParams *)lParam;
        g_installing = FALSE;

        if (code == 0) {
            SendMessageW(g_hProg, PBM_SETPOS, 100, 0);
            SetWindowTextW(g_hBtnInstall, L"Installed!");
            SetWindowTextW(g_hBtnCancel, L"Finish");

            wchar_t appExe[MAX_PATH * 2] = {0};
            _snwprintf(appExe, MAX_PATH * 2 - 1, L"%ls\\ATLAS.exe", p->destDir);
            if (p->launchAfter) {
                ShellExecuteW(NULL, L"open", appExe, NULL, p->destDir, SW_SHOWNORMAL);
                free(p);
                DestroyWindow(hwnd);
                return 0;
            }
            MessageBoxW(hwnd,
                L"ATLAS Desktop Studio has been installed successfully!\n\n"
                L"You can launch it anytime from your Desktop or Start Menu.",
                L"ATLAS Setup Complete",
                MB_OK | MB_ICONINFORMATION);
        } else {
            EnableWindow(g_hBtnInstall, TRUE);
            SetWindowTextW(g_hBtnInstall, L"Retry Install");
            EnableWindow(g_hRbStd, TRUE);
            EnableWindow(g_hRbFen, TRUE);
            EnableWindow(g_hCmbDev, TRUE);
            EnableWindow(g_hTxtDir, TRUE);
            EnableWindow(g_hBtnBrowse, TRUE);
            EnableWindow(g_hChkDesk, TRUE);
            EnableWindow(g_hChkStart, TRUE);
            MessageBoxW(hwnd,
                L"Setup encountered an error. Check the log box in the wizard window for details.",
                L"ATLAS Setup Error",
                MB_OK | MB_ICONERROR);
        }
        free(p);
        return 0;
    }

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE hPrev, PWSTR lpCmdLine, int nShowCmd) {
    (void)hPrev;
    CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);

    INITCOMMONCONTROLSEX icc = { sizeof(INITCOMMONCONTROLSEX), ICC_STANDARD_CLASSES | ICC_PROGRESS_CLASS };
    InitCommonControlsEx(&icc);

    HDC screen = GetDC(NULL);
    if (screen) {
        g_dpi = GetDeviceCaps(screen, LOGPIXELSX);
        ReleaseDC(NULL, screen);
    }

    wchar_t selfPath[MAX_PATH * 2] = {0};
    GetModuleFileNameW(NULL, selfPath, MAX_PATH * 2 - 1);
    wcsncpy(g_srcDir, selfPath, MAX_PATH * 2 - 1);
    wchar_t *sl = wcsrchr(g_srcDir, L'\\');
    if (sl) *sl = L'\0';

    detect_gpu_summary();

    g_hAppIcon = (HICON)LoadImageW(hInst, MAKEINTRESOURCEW(1), IMAGE_ICON, SX(40), SX(40), LR_DEFAULTCOLOR);
    g_brBg  = CreateSolidBrush(RGB(248, 250, 252));
    g_brLog = CreateSolidBrush(RGB(15, 23, 42));

    g_fTitle = CreateFontW(-SX(20), 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
    g_fSub   = CreateFontW(-SX(12), 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
    g_fBold  = CreateFontW(-SX(13), 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
    g_fNorm  = CreateFontW(-SX(13), 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
    g_fSmall = CreateFontW(-SX(11), 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
    g_fMono  = CreateFontW(-SX(12), 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Consolas");

    WNDCLASSEXW wc;
    ZeroMemory(&wc, sizeof(wc));
    wc.cbSize        = sizeof(wc);
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = hInst;
    wc.hIcon         = LoadIconW(hInst, MAKEINTRESOURCEW(1));
    wc.hIconSm       = (HICON)LoadImageW(hInst, MAKEINTRESOURCEW(1), IMAGE_ICON, 16, 16, LR_DEFAULTCOLOR);
    wc.hCursor       = LoadCursorW(NULL, IDC_ARROW);
    wc.hbrBackground = g_brBg;
    wc.lpszClassName = L"AtlasSetupWizardWnd";
    RegisterClassExW(&wc);

    DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    RECT wr = { 0, 0, SX(668), SX(584) };
    AdjustWindowRect(&wr, style, FALSE);
    int winW = wr.right - wr.left;
    int winH = wr.bottom - wr.top;
    int scrW = GetSystemMetrics(SM_CXSCREEN);
    int scrH = GetSystemMetrics(SM_CYSCREEN);

    g_hWnd = CreateWindowExW(
        0, L"AtlasSetupWizardWnd",
        L"ATLAS Setup Wizard  —  ATLAS Desktop Studio v1.0.0",
        style,
        (scrW - winW) / 2, (scrH - winH) / 2, winW, winH,
        NULL, NULL, hInst, NULL
    );

    ShowWindow(g_hWnd, nShowCmd);
    UpdateWindow(g_hWnd);

    if (lpCmdLine && wcsstr(lpCmdLine, L"--unattended")) {
        start_installation();
    }

    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0) > 0) {
        if (!IsDialogMessageW(g_hWnd, &msg)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }

    DeleteObject(g_fTitle);
    DeleteObject(g_fSub);
    DeleteObject(g_fBold);
    DeleteObject(g_fNorm);
    DeleteObject(g_fSmall);
    DeleteObject(g_fMono);
    DeleteObject(g_brBg);
    DeleteObject(g_brLog);
    CoUninitialize();
    return (int)msg.wParam;
}
