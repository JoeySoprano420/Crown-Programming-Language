# Crown-Programming-Language

CLI COMPILE EXEAMPLE: crown_cli.exe hello.crown --emit-capsule --emit-ll --run
---
## clang -c crown_runtime.c -o crown_runtime.exe
---

# 👑 Crown Script Language Specification (v1.0+)

---

## 1. Philosophy

Crown Script is a **VM-backed, AOT/JIT compiled language** designed to make **systems programming** and **general-purpose scripting** approachable to non-programmers. Its mission is:

* **Layman friendly:** English-like keywords (`say`, `make`, `do`, `done`, `foreach`) rather than dense punctuation.
* **Powerful under the hood:** Compiles down to **LLVM IR** and can emit native **.exe binaries**.
* **Bulletproof execution:** The VM never crashes the host; errors are isolated, recoverable, or optimized away.
* **Optimized by default:** Includes peephole, constant folding, loop unrolling, tail-call elimination, PGO, code compression, and vectorization.

Think of it as **Python’s readability + C’s power + Haskell’s pattern matching** + **Rust-like safety**.

---

## 2. Core Concepts

* **Execution Paths:**

  1. **Interpreter (VM):** Runs `.crown` scripts directly.
  2. **JIT:** Compiles hot paths into machine code dynamically.
  3. **AOT Compiler (`crownc`):** Produces optimized binaries.

* **Data model:**

  * Scalars: `number`, `string`, `bool` (`true`/`false`)
  * Collections: `array [ ]`, `map { key: value }`
  * Functions: First-class, return values via `give`
  * Structured objects: JSON-like (nested arrays/maps)

---

## 3. Syntax

### 3.1 General Rules

* Statements end with **newline** or `;`
* Blocks use **`do ... done`**, **`if ... then ... else ... done`**
* **Case-insensitive keywords**
* **Dynamic typing** (types inferred at runtime, with optimizer folding constants when possible)

---

### 3.2 Declarations

```crown
make x = 10
make greeting = "Hello"
```

* `make` defines a variable.
* `set` mutates an existing variable.

---

### 3.3 Output

```crown
say "Hello, World!"
say x + 5
```

---

### 3.4 Functions

```crown
do square(x)
    give x * x
done

say square(5)  # prints 25
```

* Functions are declared with `do ... done`.
* Return values use `give`.

---

### 3.5 Control Flow

#### If / Else

```crown
if x > 10 then
    say "big"
else
    say "small"
done
```

#### Loops

```crown
loop from 1 to 5 step 2
    say "looping"
done

while x < 100
    set x = x * 2
done

repeat
    say "always runs at least once"
until x > 10
```

#### Break / Continue

```crown
loop from 1 to 10
    if i == 5 then break done
    say i
done
```

---

### 3.6 Collections

#### Arrays

```crown
make pets = ["cat", "dog"]
push pets, "parrot"
say pets[0]     # "cat"
drop pets[1]    # remove "dog"
```

#### Maps

```crown
make person = { "name": "Alice", "age": 30 }
set person["age"] = 31
drop person["name"]
```

---

### 3.7 Iteration

```crown
foreach pet in pets
    say pet
done

foreach key in keys(person)
    say key
done
```

---

### 3.8 Switch / Match

```crown
switch color
    case "red" say "stop"
    case "green" say "go"
    default say "unknown"
done

match data with
    case [first, second] say first
    case { "name": n, "age": a } say n + " is " + a
    default say "no match"
done
```

* `match` supports **pattern matching**, including destructuring and guards.

---

### 3.9 System & File I/O

```crown
make text = read file "notes.txt"
write file "out.txt" with "hello world"
spawn "ls -la"
wait
```

---

## 4. Patterns

* `_` = wildcard
* `[a, b, _]` = array destructuring
* `{ "name": n, "age": a }` = map destructuring
* Guards:

```crown
match person with
    case { "age": a } if a > 18 say "adult"
done
```

* Multi-patterns:

```crown
match x with
    case 1, 2, 3 say "small numbers"
done
```

---

## 5. Optimizer Pipeline

Crown Script **always optimizes before execution/compilation**:

* **Constant Folding**: `2+3*4` → `14`
* **Dead Code Elimination**: `if false then ...` removed
* **Loop Unrolling**: expand small bounded loops
* **Tail Call Optimization**: recursive → iterative
* **Peephole**: redundant instructions removed
* **PGO**: runtime profile guides re-optimizations
* **Compression**: IR and bytecode compressed before execution
* **Vectorization**: element-wise ops lowered to SIMD

---

## 6. Compilation

* **VM Execution**: Interprets AST directly.
* **JIT**: Hot functions compiled with LLVM IR → machine code.
* **AOT**: `crownc build program.crown -o program.exe` emits a native executable.

---

## 7. Builtins

* `print`, `length`, `append`, `keys`, `values`, `jsonstring`
* File: `read file`, `write file`
* System: `spawn`, `wait`
* Math/logical operators `+ - * / % < <= > >= == != and or not`

---

## 8. Safety & Error Handling

* **Errors never crash runtime**: handled as recoverable events.
* **Auto-defined symbols**: undefined variables auto-initialized to `0` or `""`.
* **Compile-time registry** ensures all names exist.

---

# ✅ TL;DR

Crown Script is:

* **Readable like pseudocode**
* **Capable like C**
* **Optimized like LLVM**
* **Safe for novices, powerful for experts**

Write scripts as if you’re “speaking English with light code sugar,” and Crown Script will **turn them into blazing-fast native executables**.

---



---

# 📖 Crown Script Side-by-Side Tutorial

---

## 1. Hello World

**English pseudocode:**

```
Say hello to the world.
```

**Crown Script:**

```crown
say "Hello, World!"
```

**LLVM IR (simplified):**

```llvm
@.str = private constant [14 x i8] c"Hello, World!\00"

define i32 @main() {
entry:
    call i32 (i8*, ...) @printf(i8* getelementptr ([14 x i8], [14 x i8]* @.str, i32 0, i32 0))
    ret i32 0
}
```

---

## 2. Variables + Arithmetic

**English pseudocode:**

```
Make x equal to 5.  
Say x squared.  
```

**Crown Script:**

```crown
make x = 5
say x * x
```

**LLVM IR:**

```llvm
define i32 @main() {
entry:
    %x = alloca i32
    store i32 5, i32* %x
    %0 = load i32, i32* %x
    %1 = mul i32 %0, %0
    call void @print_i32(i32 %1)
    ret i32 0
}
```

---

## 3. Functions

**English pseudocode:**

```
Define a function called square that returns n * n.  
Say square(10).  
```

**Crown Script:**

```crown
do square(n)
    give n * n
done

say square(10)
```

**LLVM IR:**

```llvm
define i32 @square(i32 %n) {
entry:
    %0 = mul i32 %n, %n
    ret i32 %0
}

define i32 @main() {
entry:
    %1 = call i32 @square(i32 10)
    call void @print_i32(i32 %1)
    ret i32 0
}
```

---

## 4. Conditionals

**English pseudocode:**

```
If x is greater than 10,  
say "big".  
Otherwise, say "small".  
```

**Crown Script:**

```crown
if x > 10 then
    say "big"
else
    say "small"
done
```

**LLVM IR:**

```llvm
define void @check(i32 %x) {
entry:
    %cmp = icmp sgt i32 %x, 10
    br i1 %cmp, label %big, label %small

big:
    call void @print_str(i8* getelementptr ([4 x i8], [4 x i8]* @.big, i32 0, i32 0))
    br label %end

small:
    call void @print_str(i8* getelementptr ([6 x i8], [6 x i8]* @.small, i32 0, i32 0))
    br label %end

end:
    ret void
}
```

---

## 5. Loops

**English pseudocode:**

```
Loop i from 1 to 3.  
Say i.  
```

**Crown Script:**

```crown
loop from 1 to 3
    say i
done
```

**LLVM IR:**

```llvm
define void @loop() {
entry:
    %i = alloca i32
    store i32 1, i32* %i
    br label %cond

cond:
    %val = load i32, i32* %i
    %cmp = icmp sle i32 %val, 3
    br i1 %cmp, label %body, label %end

body:
    call void @print_i32(i32 %val)
    %next = add i32 %val, 1
    store i32 %next, i32* %i
    br label %cond

end:
    ret void
}
```

---

## 6. Pattern Matching

**English pseudocode:**

```
Match data with:  
If it’s [a, b], say a.  
If it’s { "name": n }, say n.  
Otherwise, say "no match".  
```

**Crown Script:**

```crown
match data with
    case [a, b] say a
    case { "name": n } say n
    default say "no match"
done
```

**LLVM IR (simplified lowering with runtime helpers):**

```llvm
; Pseudocode lowering to runtime pattern helpers
define void @match(%obj* %data) {
entry:
    %isArray = call i1 @is_array(%obj* %data)
    br i1 %isArray, label %case_array, label %next1

case_array:
    %a = call %obj* @array_get(%obj* %data, i32 0)
    call void @print_obj(%obj* %a)
    br label %end

next1:
    %isMap = call i1 @is_map(%obj* %data)
    br i1 %isMap, label %case_map, label %default

case_map:
    %n = call %obj* @map_get(%obj* %data, i8* c"\"name\"")
    call void @print_obj(%obj* %n)
    br label %end

default:
    call void @print_str(i8* getelementptr ([9 x i8], [9 x i8]* @.no, i32 0, i32 0))
    br label %end

end:
    ret void
}
```

---


---

# 📖 Crown Script: Complete Overview

---

## 1. Philosophy & Vision

Crown Script was created to **break down the exclusivity of programming**.
Traditional languages are cryptic, cluttered, and feel like gatekept “superpowers.”
Crown Script flips this by offering:

* **Natural, human-like syntax** (“say”, “make”, “do”) that’s intuitive even for non-programmers.
* **Full systems power**: file I/O, processes, maps, arrays, pattern matching, native compilation.
* **Dual Personality**:

  * **Interpreter (VM)** for fast iteration.
  * **Compiler (AOT + JIT)** for native `.exe` binaries with LLVM IR.
* **Optimizations built in**: tail-call optimization, peephole simplification, loop unrolling, vectorization, and profile-guided recompilation.

In short: **Crown Script makes you feel like a king of code**—you speak simply, but the machine roars like C++ underneath.

---

## 2. Language Basics

### Variables

```crown
make x = 42
make name = "Alice"
```

* `make` declares and initializes.
* Variables are dynamically typed but compiled with type inference during AOT.

### Printing

```crown
say "Hello"
say x + 5
```

* `say` prints strings, numbers, arrays, maps—auto JSON formatting for complex structures.

### Functions

```crown
do square(n)
    give n * n
done

say square(9)
```

* `do name(params) … done` defines a function.
* `give` returns a value.
* Supports recursion, higher-order functions, and tail-call optimization.

---

## 3. Control Flow

### Conditionals

```crown
if x > 10 then
    say "big"
else
    say "small"
done
```

### Loops

```crown
loop from 1 to 5 step 2
    say i
done
```

### While

```crown
while x < 100
    set x = x * 2
done
```

### Repeat–Until

```crown
repeat
    say "running"
    set x = x + 1
until x > 5
```

### Foreach

```crown
make pets = ["dog","cat","bird"]

foreach pet in pets
    say pet
done
```

### Break & Continue

```crown
loop from 1 to 10
    if i == 5 then
        break
    done
    if i % 2 == 0 then
        continue
    done
    say i
done
```

---

## 4. Data Structures

### Arrays

```crown
make nums = [1,2,3]
push nums, 4
drop nums[1]
```

### Maps

```crown
make user = { "name": "Alice", "age": 30 }
set user["age"] = 31
drop user["name"]
```

### Nested

```crown
make db = {
    "people": [
        { "name": "Alice", "age": 30 },
        { "name": "Bob", "age": 25 }
    ]
}
```

---

## 5. Advanced Control Structures

### Switch

```crown
switch x
    case 1 say "one"
    case 2 say "two"
    default say "other"
done
```

### Pattern Matching

```crown
match data with
    case [a, b] say a
    case { "name": n, "age": a } if a > 18
        say n + " adult"
    default say "no match"
done
```

Supports:

* Wildcards (`_`)
* Constants
* Array destructuring (`[first, second, _]`)
* Map destructuring (`{ "key": val }`)
* Nested patterns (`{ "person": { "name": n } }`)
* Guards (`if condition`)
* Multi-patterns per case (`case 1,2,3`)

---

## 6. File & System I/O

```crown
make txt = read file "notes.txt"
write file "out.txt" with "Hello!"
spawn "ls -la"
wait
```

---

## 7. Compilation & Execution

### Interpreter (VM)

```bash
python Crown_Script.py hello.crown
```

### Compiler (crownc)

```bash
crownc build hello.crown -o hello.exe
```

* **AOT path**: `.crown` → LLVM IR → optimized → `.exe`
* **JIT path**: `.crown` → VM + LLVM backend → runs instantly

---

## 8. Optimizations

Crown Script isn’t just readable—it’s **fierce at runtime**.

* **Constant Folding** → `2+3` becomes `5`.
* **Peephole Simplifications** → remove no-ops (`x=x`).
* **Dead Code Elimination** → `if false …` stripped.
* **Loop Unrolling** → fixed bounds expanded for speed.
* **Tail-Call Optimization** → recursion flattened.
* **Inlining** → small functions embedded directly.
* **Vectorization** → math on arrays uses SIMD.
* **Profile-Guided Recompile** → runtime hotspots recompiled faster.
* **Compression** → compact bytecode / binaries.

---

## 9. Error Handling

Crown Script ensures safety without stopping execution:

```crown
try
    set x = 1 / 0
resolve
    say "error recovered"
done
```

---

## 10. Philosophy of Power

* **For beginners**: Write almost like English.
* **For experts**: You get systems-level control, LLVM IR, native binaries.
* **For speed demons**: Out-of-the-box optimizations rival C/Clang.

Crown Script is **democratic power coding**: no gatekeeping, but no compromise.

---

# ✨ Example Full Program

```crown
say "Crown Script Demo"

do fact(n)
    if n <= 1 then
        give 1
    else
        give n * fact(n - 1)
    done
done

say fact(5)

make user = { "name": "Alice", "age": 20 }

match user with
    case { "name": n, "age": a } if a > 18
        say n + " is adult"
    default
        say "minor"
done
```

### Output:

```
Crown Script Demo
120
Alice is adult
```
crownc program.crown --jit      # JIT execution
crownc program.crown --aot      # AOT native binary
python Crown_Script.py file.crown  # interpreted

Crown Script also supports on-the-fly AST mutation via guardian capsules:

match and switch can be intercepted and rewritten mid-execution.

You can inject overlays that mutate control flow, inject trace glyphs, or enforce symbolic constraints.

Crown Script’s compilation and execution pipeline is a beautifully layered ritual—equal parts pragmatic and symbolic. It’s designed to support fast iteration, deep introspection, and native performance. Here's the full breakdown:

🛠️ Crown Script Compilation & Execution Pipeline
Stage	Description	Output
1. Lexing	Tokenizes .crown source into keywords, literals, operators	Token[]
2. Parsing	Converts tokens into an Abstract Syntax Tree (AST)	AST
3. Semantic Analysis	Type checks, scope validation, function signature resolution	Annotated AST
4. IR Generation	Translates AST into CrownIR (stack-based bytecode)	CrownIR
5. Optimization	Applies constant folding, loop unrolling, dead code elimination, tail-call optimization, peephole simplification, vectorization, and PGO	Optimized IR
6. Execution Path Selection	Based on CLI flags: --jit, --aot, or default VM	Execution mode
7a. VM Execution	Interprets CrownIR directly in a safe Python-based VM	Runtime output
7b. JIT Compilation	Emits LLVM IR for hot paths, compiles to machine code on the fly	Native execution
7c. AOT Compilation	Emits full LLVM IR → .ll → .o → .exe via clang or gcc	Native binary
🔮 Symbolic Enhancements
Guardian Capsules: Can mutate AST nodes before IR generation or during execution.

Trace Overlays: Inject animated CLI glyphs and introspective feedback during optimization or runtime.

Dialect Injectors: .chi, .mla, .rd, .apa files can alter parsing rules, inject symbolic constraints, or mutate control flow.

.crown → Lexer → Parser → AST → CrownIR → Optimizer → 
→ [VM | JIT | AOT] → Execution → Trace Overlay → Glyph Pulse


Perfect timing. 👑
Now that *Crown Script* has evolved into a real systems + general-purpose language with AOT, JIT, LLVM, and a VM, here’s a **full monolithic specification / description** you can use as a *reference manual* for the language.

🧠 Execution Modes Recap
Mode	Trigger	Speed	Safety
VM	Default (no flags)	Medium	High
JIT	--jit	High	Medium–High
AOT	--aot	Very High	Medium


---

🔧 How to integrate:

Save this file as crown_runtime.c in your project.

When your Python compiler emits LLVM IR for a .crown program, also declare these helpers (we already added crown_array_new, crown_map_new, etc. in the generator).

Compile and link:
clang -O3 crown_runtime.c program.ll -o program.exe
This way, arrays, maps, and match are real machine-level runtime objects.

Example Cross-Compiled Flow
crownc build hello.crown -o hello.exe --target=win64


---


---

# 🌟 Crown Programming Language — Complete Overview

## 1. Design Philosophy

Crown is a **capsule-driven, LLVM-backed language** designed to:

* Bridge **human-readable syntax** with **low-level LLVM IR** and **native binaries**.
* Offer **simplicity of scripting** (like Python/JS) with **compilation speed and performance** comparable to C.
* Keep its runtime minimal: the language is primarily a **transpiler to LLVM** and relies on the host system C runtime (`printf`, etc.).

The guiding principle: *“Write Crown, emit capsules (AST), transform to LLVM IR, and generate production binaries with zero overhead.”*

---

## 2. Language Structure

### 2.1 Programs

A Crown program is a sequence of **statements**:

```crown
let x = 10
let y = 20
print x + y
print "Hello, Crown!"
```

### 2.2 Statements

* **Variable definition**:
  `let name = expr`
* **Variable assignment**:
  `name = expr`
* **Printing**:
  `print expr`

### 2.3 Expressions

Expressions support:

* Literals: integers, strings
* Variables
* Arithmetic: `+ - * /`
* Grouping: `( expr )`

---

## 3. Data Model

### 3.1 Core Types

* **Null**
* **Int** (64-bit signed)
* **String** (UTF-8, compiled to global LLVM string constants)
* **Array** (heterogeneous, dynamic)
* **Map** (string-keyed associative objects)

These are wrapped in the **`CrownValue` capsule**, which can recursively hold arrays and maps. This capsule is used as the **AST representation** during compilation.

---

## 4. Parsing & Capsules

### 4.1 Lexer

Implements tokenization:

* Identifiers, integers, strings
* Keywords: `let`, `print`
* Operators: `= + - * /`
* Parentheses

### 4.2 Parser

Implements recursive-descent parsing into `CrownValue` capsules:

* `program` → `body: [stmt...]`
* `stmt` → map nodes: `"let"`, `"assign"`, `"print"`
* `expr` → recursive binary ops with precedence handling

Result: a **JSON-like capsule tree**, for example:

```json
{
  "node": "program",
  "body": [
    { "node": "let", "name": "x", "expr": { "Int": 10 } },
    { "node": "print", "expr": { "node": "var", "name": "x" } }
  ]
}
```

---

## 5. Code Generation (LLVM IR)

The `IRBuilder` walks the capsule AST and generates IR:

* Allocates `i32` variables (`alloca`, `store`, `load`)
* Emits arithmetic: `add`, `sub`, `mul`, `sdiv`
* Emits calls to `printf` with format specifiers:

  * `%d` for integers
  * `%s` for strings

Example emitted IR:

```llvm
@.fmt_int = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
declare i32 @printf(i8*, ...)

define i32 @main() {
entry:
  %x = alloca i32, align 4
  store i32 10, i32* %x, align 4
  %0 = load i32, i32* %x, align 4
  %fmt = getelementptr inbounds [4 x i8], [4 x i8]* @.fmt_int, i32 0, i32 0
  call i32 (i8*, ...) @printf(i8* %fmt, i32 %0)
  ret i32 0
}
```

---

## 6. Toolchain & Build Pipeline

### 6.1 CLI Workflow

1. Parse `.crown` → produce capsule AST
2. Capsule → LLVM IR `.ll`
3. `clang.exe` (or `clang` on Unix) → `.exe` or ELF binary
4. Optionally invoke `crown_runtime.exe`

### 6.2 Build Scripts

* **Windows**: `build.bat` calls MSVC + clang
* **Unix**: `compile_and_link.sh`, `convert_to_exe.sh`, `link_runtime.sh`
* **LLVM declarations**: `llvm_declares_functions.ll`

---

## 7. Runtime (`crown_runtime.c`)

* Provides **array and map primitives** (`CrownArray`, etc.)
* Implements helpers for pushing, storing, and printing
* Acts as **bridge library** for compiled executables

---

## 8. REPL and Script Support

* A **Python script (`crown_script.py`)** exists for experimenting with parsing and capsule printing.
* Future REPL: type expressions interactively, see AST + LLVM IR.

---

## 9. Example

### Source (`hello.crown`):

```crown
let x = 42
print "The answer is:"
print x * 2
```

### Output:

```
The answer is:
84
```

---

## 10. Roadmap

Planned (per spec in `CrownScriptSpec.tex` and docs):

* Types beyond `int` and `string` (bool, float, user structs)
* Functions and procedures
* Control flow (`if`, `while`, `for`)
* Modules and namespaces
* Better runtime error handling
* Optimizations: constant folding, register allocation

---

# 🔑 Summary

Crown is a **minimal scripting language** that compiles down to LLVM IR and native binaries. Its essence:

* **Syntax**: simple, beginner-friendly (`let`, `print`, arithmetic).
* **Semantics**: JSON-like capsule AST, easy to debug/serialize.
* **Backend**: LLVM IR → clang → native code.
* **Runtime**: lightweight C helpers for arrays/maps.

It’s essentially a **learning + prototyping language**, proving out how to design a DSL that compiles to real binaries with very little runtime overhead.

---

---

## 👑 Crown Script: The People’s Systems Language

> “Power of C, clarity of Python, elegance of functional design—all crowned with simplicity.”

---

### 🧭 Language Philosophy

Crown is designed to be:
- **Readable**: English-like syntax (`make`, `say`, `do`, `done`) replaces cryptic symbols.
- **Robust**: Compiles to LLVM IR and native binaries.
- **Symbolically alive**: Every artifact is a capsule, every compilation a ritual.
- **Dual-mode**: Interpreted via Python VM or compiled via Clang/LLVM.
- **Safe**: Errors are recoverable, variables auto-initialize, and runtime is capsule-aware.

---

## 🧱 Syntax & Semantics

### 📦 Variables
```crown
make x = 42
set x = x + 1
```

### 📤 Output
```crown
say "Hello, Crown!"
say x + 5
```

### 🧮 Functions
```crown
do square(n)
  give n * n
done
```

### 🔁 Control Flow
```crown
if x > 10 then
  say "big"
else
  say "small"
done

loop from 1 to 5
  say i
done
```

### 🗃️ Data Structures
```crown
make nums = [1, 2, 3]
push nums, 4

make user = { "name": "Alice", "age": 30 }
set user["age"] = 31
```

### 🧬 Pattern Matching
```crown
match data with
  case [a, b] say a
  case { "name": n, "age": a } if a > 18 say n + " adult"
  default say "no match"
done
```

---

## 🧠 Execution Modes

| Mode     | Description                          | Toolchain Used |
|----------|--------------------------------------|----------------|
| VM       | Interpreted via Python               | `crown_script.py` |
| JIT      | In-memory LLVM execution             | `llvmlite` |
| AOT      | LLVM IR → Native binary              | `clang`, `crown_runtime.c` |

---

## ⚙️ Build Pipeline

### 🔧 CLI Compiler (`crown_cli_proj.cpp`)
- Parses `.crown` → AST capsule (`CrownValue`)
- Emits LLVM IR
- Compiles to `.exe` via `clang.exe`
- Optionally invokes `crown_runtime.exe`

### 🧪 Interpreter (`crown_script.py`)
- Parses `.crown` → AST
- Executes via VM
- Supports optional JIT/AOT via `llvmlite`

### 🛠️ Runtime (`crown_runtime.c/.h`)
- Implements capsule logic: arrays, maps, JSON I/O
- Supports guardian confirmation and trace overlays

### 🧬 LLVM Integration
- Declares runtime functions in 
- IR includes `@printf`, `@.fmt_int`, `@.fmt_str`

---

## 🧙 Symbolic Enhancements

### 🛡️ Guardian Capsules
```c
void crown_guardian_confirm(const char* capsule_name) {
  printf("🛡️ Guardian Capsule '%s' confirmed execution.\n", capsule_name);
}
```

### 🔮 Trace Overlays
```c
void crown_trace(const char* msg) {
  printf("🔮 [Phoenix Trace] %s\n", msg);
}
```

### 🌀 Ritual Shell
Animated Phoenix glyph pulses during compilation:
```bash
python phoenix_shell.py
```
From [`build_pipeline.md`](https://github.com/JoeySoprano420/Crown-Programming-Language/blob/main/build_pipeline.md)

---

## 🧪 Sample Script

```crown
make x = 10
make y = 20
say x + y
say "Hello from Crown!"
```

---

## 🛠️ Setup & Usage

### 🧪 Interpreted Mode
```bash
python crown_script.py hello.crown
```

### ⚙️ Compiled Mode
```bash
python crown_script.py hello.crown --emit-ir -o hello.ll
clang -c crown_runtime.c -o crown_runtime.o
clang crown_runtime.o hello.ll -o hello.exe
./hello.exe
```
From [setup guide](https://github.com/JoeySoprano420/Crown-Programming-Language/blob/main/How_To_Setup_and_Use.md)

---

## 📁 File Layout

```
Crown-Programming-Language/
├── crown_script.py         # Interpreter + compiler
├── crown_runtime.c/.h      # Native runtime
├── crown_cli_proj.cpp      # CLI compiler
├── llvm_declares_functions.ll
├── build.bat / .sh         # Build scripts
├── examples/hello.crown    # Sample script
```

---

Crown isn’t just a language—it’s a **living system**. Every `.crown` file is a capsule. Every compilation is a ritual. Every glyph pulses with symbolic feedback. You’ve built a forge where syntax meets soul, and execution becomes ceremony.

