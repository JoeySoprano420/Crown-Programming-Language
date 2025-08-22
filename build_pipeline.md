---

## 🧱 1. Clang Build Pipeline (.crown → .exe)

Assuming:
- `crown_runtime.c` is your runtime
- `hello.ll` is the LLVM IR emitted from `hello.crown`

### 🔧 Compile & Link with Clang

```bash
clang -O3 -c crown_runtime.c -o crown_runtime.o
clang -O3 hello.ll crown_runtime.o -o hello.exe
```

You can also use `.bc` or `.o` from your IR:

```bash
clang -O3 hello.bc crown_runtime.o -o hello.exe
```

---

## 🧰 2. MSVC Build Pipeline

If you want to use MSVC’s `cl.exe`:

### ✅ Runtime Compilation

```bat
cl /TC /c crown_runtime.c /Fo:crown_runtime.obj
```

### ✅ Link with IR (converted to `.obj`)

Assuming you’ve converted `hello.ll` to `hello.obj` via `llc` or `llvm-as`:

```bat
cl crown_runtime.obj hello.obj /Fe:hello.exe
```

> Tip: Use `/link /SUBSYSTEM:CONSOLE` if you want to force console output.

---

## 📦 3. Batch Script (`build.bat`)

Here’s a Windows-friendly batch file that compiles everything:

```bat
@echo off
setlocal

echo 🔧 Compiling runtime...
clang -O3 -c crown_runtime.c -o crown_runtime.o

echo 🔧 Compiling IR...
clang -O3 hello.ll -o hello.o

echo 🔗 Linking...
clang hello.o crown_runtime.o -o hello.exe

echo ✅ Build complete: hello.exe
```

---

## 🛠️ 4. Makefile (Linux/macOS or Windows with Make)

```makefile
CFLAGS = -O3
RUNTIME = crown_runtime.c
IR = hello.ll
OUT = hello.exe

all: $(OUT)

crown_runtime.o: $(RUNTIME)
	clang $(CFLAGS) -c $(RUNTIME) -o crown_runtime.o

hello.o: $(IR)
	clang $(CFLAGS) -c $(IR) -o hello.o

$(OUT): crown_runtime.o hello.o
	clang crown_runtime.o hello.o -o $(OUT)

clean:
	del *.o *.exe
```

---

## 🔮 5. Ritual CLI Shell with Glyph Animation

Here’s a symbolic CLI shell that animates the Phoenix glyph during compilation:

```python
# phoenix_shell.py
import time, sys

glyph = [
    "   /\\   ",
    "  /__\\  ",
    " /\\  /\\ ",
    "/__\\/__\\"
]

def pulse():
    for i in range(3):
        for line in glyph:
            print(f"\033[1;35m{line}\033[0m")
        time.sleep(0.2)
        print("\033c", end="")  # clear screen

def compile():
    print("🧬 Summoning Phoenix...")
    pulse()
    print("🔧 Linking capsules...")
    time.sleep(1)
    print("✅ Ritual complete: hello.exe")

if __name__ == "__main__":
    compile()
```

Run it with:

```bash
python phoenix_shell.py
```

---

## 🧠 6. Trace Overlay Hook (Crown Runtime)

Inject this into `crown_runtime.c` to log symbolic traces:

```c
void crown_trace(const char* msg) {
    printf("🔮 [Phoenix Trace] %s\n", msg);
}
```

Use it like:

```c
crown_trace("Array pushed");
```

---

## 🧬 7. Guardian Capsule Confirmation

Add this to your runtime:

```c
void crown_guardian_confirm(const char* capsule_name) {
    printf("🛡️ Guardian Capsule '%s' confirmed execution.\n", capsule_name);
}
```

Call it post-compilation:

```c
crown_guardian_confirm("hello.crown");
```

---

