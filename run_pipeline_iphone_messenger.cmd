@echo off
REM ============================================================================
REM vidsmash pipeline wrapper -- iOS Messenger source variant.
REM
REM   Per-input wrapper. The naming convention is one wrapper per video
REM   archetype so we can pin input-specific defaults without bloating the
REM   stitch flag surface for everyone.
REM
REM Per-video output layout:
REM   out\<video_basename>\
REM     keyframes.json       (from detect_pauses)
REM     timeline.png         (from detect_pauses)
REM     keyframe_chunk_*.png (from stitch_keyframes)
REM     report.json          (from stitch_keyframes)
REM     view.html            (from make_view_html)
REM
REM   Multiple input videos thus produce side-by-side, fully self-contained
REM   directories. To A/B a tuning change, pass an override out_dir as the
REM   second positional arg (e.g. `out\lexi_iphone_messenger_all_test`).
REM
REM Canonical stitch behaviour is the DEFAULT in tools\stitch_keyframes.py
REM (no opt-in flags required here). To opt OUT of a canonical default for
REM diagnostics pass --no-clear-beyond-bubble-extent / --no-seam-line-fix /
REM --no-rescue-refine-offset / etc. as extra args -- they forward to stitch.
REM
REM Usage:
REM   run_pipeline_iphone_messenger.cmd [input.mp4] [out_dir] [extra_args...]
REM
REM Defaults:
REM   input    : lexi_iphone_messenger_all.mp4
REM   out_dir  : out\<input_basename_without_extension>\
REM ============================================================================

setlocal ENABLEDELAYEDEXPANSION
set "PYTHONIOENCODING=utf-8"

set "INPUT=%~1"
if "%INPUT%"=="" set "INPUT=lexi_iphone_messenger_all.mp4"

set "OUTDIR=%~2"
if "%OUTDIR%"=="" (
    REM Derive out dir from input basename (strip extension).
    for %%F in ("%INPUT%") do set "OUTDIR=out\%%~nF"
)

REM Drop the first two positional args; forward the rest to stitch.
if not "%~1"=="" shift
if not "%~1"=="" shift
set "EXTRA="
:collect_extra
if "%~1"=="" goto done_collect
set "EXTRA=%EXTRA% %~1"
shift
goto collect_extra
:done_collect

set "KEYFRAMES=%OUTDIR%\keyframes.json"

echo [run] input    = %INPUT%
echo [run] out      = %OUTDIR%
echo [run] keyframes= %KEYFRAMES%
echo [run] extra    =%EXTRA%
echo.

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

if not exist "%KEYFRAMES%" (
    echo [run] %KEYFRAMES% missing - running detect_pauses
    python tools\detect_pauses.py --input "%INPUT%" --out "%OUTDIR%"
    if errorlevel 1 goto fail
)

echo [run] stitch_keyframes ...
python tools\stitch_keyframes.py --input "%INPUT%" --keyframes "%KEYFRAMES%" --out "%OUTDIR%" %EXTRA%
if errorlevel 1 goto fail

echo [run] validate_stitch (non-fatal) ...
python tools\validate_stitch.py --input "%INPUT%" --out "%OUTDIR%"
if errorlevel 1 echo [run] WARN: validate_stitch failed (pre-existing schema mismatch, ignored)

echo [run] make_view_html ...
python tools\make_view_html.py --dir "%OUTDIR%" --glob "keyframe_chunk_*.png" --title "vidsmash: %~n1"
if errorlevel 1 goto fail

echo.
echo [run] done - view: %OUTDIR%\view.html
endlocal
exit /b 0

:fail
echo [run] FAILED (exit %ERRORLEVEL%)
endlocal
exit /b 1
