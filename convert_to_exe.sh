clang C:\Users\420up\Crown-Programming-Language\crown_runtime.o hello.o -o hello.exe

clang C:\Users\420up\Crown-Programming-Language\crown_runtime.o -o crown_runtime.exe

cl crown_runtime.c /Fo:crown_runtime.obj /Fe:crown_runtime.exe

cl crown_runtime.obj hello.obj /Fe:hello.exe

