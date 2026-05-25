@echo off
REM ============================================================================
REM vidsmash canonical pipeline wrapper.
REM
REM   The "working" stitch behaviour is the DEFAULT in stitch_keyframes.py
REM   (no opt-in flags required). This wrapper only orchestrates the four
REM   stages and tolerates the pre-existing validate_stitch schema mismatch.
REM
REM   If a future regression needs to opt OUT of one of the canonical
REM   behaviours for diagnostics, pass --no-clear-beyond-bubble-extent,
REM   --no-bubble-extent-synthetic-aa, --no-mask-detected-circles, or
REM   --no-rescue-starved-circles as extra args (they forward to stitch).
REM
REM Usage:
REM   run_pipeline.cmd ^<input.mp4^> [out_dir] [extra_args...]
REM
REM Defaults:
REM   input    : lexiconv.mp4
REM   out_dir  : out\stitch
REM
REM Extra flags after out_dir pass through to stitch_keyframes.py
REM (e.g. --rescue-max-frames 32, --no-clear-beyond-bubble-extent).
REM ============================================================================

setlocal ENABLEDELAYEDEXPANSION
set "PYTHONIOENCODING=utf-8"

set "INPUT=%~1"
if "%INPUT%"=="" set "INPUT=lexiconv.mp4"

set "OUTDIR=%~2"
if "%OUTDIR%"=="" set "OUTDIR=out\stitch"

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

set "KEYFRAMES=out\keyframes.json"

echo [run] input    = %INPUT%
echo [run] out      = %OUTDIR%
echo [run] keyframes= %KEYFRAMES%
echo [run] extra    =%EXTRA%
echo.

if not exist "%KEYFRAMES%" (
    echo [run] %KEYFRAMES% missing - running detect_pauses
    python tools\detect_pauses.py --input "%INPUT%" --out out
    if errorlevel 1 goto fail
)

echo [run] stitch_keyframes ...
python tools\stitch_keyframes.py --input "%INPUT%" --keyframes "%KEYFRAMES%" --out "%OUTDIR%" %EXTRA%
if errorlevel 1 goto fail

echo [run] validate_stitch (non-fatal) ...
python tools\validate_stitch.py --input "%INPUT%" --out "%OUTDIR%"
if errorlevel 1 echo [run] WARN: validate_stitch failed (pre-existing schema mismatch, ignored)

echo [run] make_view_html ...
python tools\make_view_html.py --dir "%OUTDIR%" --glob "keyframe_chunk_*.png" --title "vidsmash"
if errorlevel 1 goto fail

echo.
echo [run] done - view: %OUTDIR%\view.html
endlocal
exit /b 0

:fail
echo [run] FAILED (exit %ERRORLEVEL%)
endlocal
exit /b 1
