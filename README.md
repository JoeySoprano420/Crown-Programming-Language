# Crown-Programming-Language

crownc program.crown --jit      # JIT execution
crownc program.crown --aot      # AOT native binary
python Crown_Script.py file.crown  # interpreted


Perfect timing. 👑
Now that *Crown Script* has evolved into a real systems + general-purpose language with AOT, JIT, LLVM, and a VM, here’s a **full monolithic specification / description** you can use as a *reference manual* for the language.

---

# 👑 Crown Script Language Specification (v0.3+)

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

---


---

