@echo off
REM Run all tests and checks for roop-cam
REM 1. Linting with flake8
REM 2. Type checking with mypy
REM 3. Integration test with example files

setlocal enabledelayedexpansion

echo.
echo roop-cam Test Suite
echo ===================
echo.

REM Check if venv is activated
if "%VIRTUAL_ENV%"=="" (
    echo [WARNING] Virtual environment not activated
    echo Consider running: venv\Scripts\activate.bat
    echo.
)

REM Run flake8
echo 1. Running flake8 ^(linting^)...
flake8 pipeline.py roop
if %errorlevel% neq 0 (
    echo [FAILED] flake8 failed
    exit /b 1
)
echo [OK] flake8 passed
echo.

REM Run mypy
echo 2. Running mypy ^(type checking^)...
mypy pipeline.py roop
if %errorlevel% neq 0 (
    echo [FAILED] mypy failed
    exit /b 1
)
echo [OK] mypy passed
echo.

REM Run integration test
echo 3. Running integration test...
if not exist ".github\examples\source.jpg" (
    echo [WARNING] Example files not found, skipping integration test
    echo Expected: .github\examples\source.jpg and .github\examples\target.mp4
) else if not exist ".github\examples\target.mp4" (
    echo [WARNING] Example files not found, skipping integration test
    echo Expected: .github\examples\source.jpg and .github\examples\target.mp4
) else (
    set TEST_OUTPUT=.test_output.mp4
    echo Processing example files...
    python pipeline.py -s .github\examples\source.jpg -t .github\examples\target.mp4 -o "!TEST_OUTPUT!"
    if %errorlevel% neq 0 (
        echo [FAILED] Integration test failed
        exit /b 1
    )
    echo [OK] Integration test passed

    REM Clean up test output
    if exist "!TEST_OUTPUT!" (
        del "!TEST_OUTPUT!"
    )
)
echo.

echo [OK] All tests passed!
echo.
