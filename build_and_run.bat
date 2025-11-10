@echo off
setlocal
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
"%CUDA_PATH%\bin\nvcc.exe" -arch=sm_61 matrix_mul.cu -o matrix_mul
if %errorlevel% neq 0 exit /b %errorlevel%
matrix_mul.exe
endlocal
