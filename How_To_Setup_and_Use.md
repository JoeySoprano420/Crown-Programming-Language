To set up and use Crown Script, you’ve got two main execution paths: interpreted (via Python) and compiled (via LLVM + native runtime). Here's a full setup guide tailored to your repo and tooling:

---

## 🧠 Option 1: Interpreted Execution (Python VM)

This is the fastest way to run `.crown` scripts and test language features.

### ✅ Requirements
- Python 3.8+
- `llvmlite` (for optional JIT/AOT)
```bash
pip install llvmlite
```

### ▶️ Run a Script
```bash
python crown_script.py your_script.crown
```

This will:
- Parse the `.crown` file
- Build an AST
- Execute it via the VM
- Optionally compile to LLVM IR if JIT/AOT is triggered

---

## ⚙️ Option 2: AOT Compilation (LLVM → Native Binary)

This path compiles `.crown` to LLVM IR, then links with the native runtime to produce a standalone executable.

### ✅ Requirements
- Clang (LLVM toolchain)
- `crown_script.py` (your compiler frontend)
- `crown_runtime.c` and `crown_runtime.h` (native runtime)
- `llvm_declares_functions.ll` (runtime function declarations)

### 🛠️ Build Steps

1. **Generate LLVM IR**
```bash
python crown_script.py your_script.crown --emit-ir -o your_script.ll
```

2. **Compile Runtime**
```bash
clang -c crown_runtime.c -o crown_runtime.o
```

3. **Link IR + Runtime**
```bash
clang crown_runtime.o your_script.ll -o your_program
```

Now you can run `./your_program` as a native binary.

---

## 🧪 Optional: JIT Execution

If you want to run hot paths via LLVM JIT:
```bash
python crown_script.py your_script.crown --jit
```

This uses `llvmlite` to compile and execute in-memory.

---

## 🧩 File Layout Suggestion

```
Crown-Programming-Language/
├── crown_script.py         # Interpreter + compiler
├── crown_runtime.c         # Native runtime
├── crown_runtime.h         # Runtime header
├── llvm_declares_functions.ll  # LLVM function declarations
├── examples/
│   └── hello.crown         # Sample script
├── build/
│   └── your_script.ll      # Generated IR
```

---

## 🧙 Bonus: Try a Sample Script

Create `examples/hello.crown`:
```crown
say "Hello from Crown"
make x = 5
say x * x
```

Then run:
```bash
python crown_script.py examples/hello.crown
```

Or compile:
```bash
python crown_script.py examples/hello.crown --emit-ir -o build/hello.ll
clang -c crown_runtime.c -o build/crown_runtime.o
clang build/crown_runtime.o build/hello.ll -o hello
./hello
```

---

Would you like a Makefile or shell script to automate this flow? I can scaffold one that builds, runs, and even watches `.crown` files for changes.
