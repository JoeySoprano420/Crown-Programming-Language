@echo off
clang -c crown_runtime.c -o crown_runtime.o
clang -c hello.ll -o hello.o
clang crown_runtime.o hello.o -o hello.exe
echo Build complete: hello.exe
