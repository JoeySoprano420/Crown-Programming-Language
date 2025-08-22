
# 📘 The Crown Book  
**The People’s Systems Language**

---

## Preface

Crown is a multi-paradigm language that combines **readability** with **systems-level power**.  
Its mission: **make code approachable like English**, yet **compile down to efficient machine-native executables** using LLVM.  

This book is written to be *hands-on*. Each chapter gives:
- **Examples** (code you can run),
- **Explanations** (what’s happening),
- **Exercises + Solutions** (so you can self-learn).

---

# Part I: Getting Started

---

## Chapter 1 – Saying Hello

### First program
```crown
say "Hello, World!"
```

Run with the interpreter:
```sh
python crown_script.py hello.crown
```

Compile to an executable:
```sh
crownc build hello.crown -o hello.exe
./hello.exe
```

Output:
```
Hello, World!
```

### Exercises & Solutions
**1.1 Print your name, favorite number, and a fun fact**
```crown
say "My name is Violet"
say "My favorite number is 7"
say "Fun fact: I love programming languages!"
```

**1.2 Print a phrase five times**
```crown
loop from 1 to 5
    say "Crown is amazing!"
done
```

---

## Chapter 2 – Variables & Expressions

### Declaring & Reassigning
```crown
make x = 42
make y = 7
say x + y       # 49
set x = x * 2
say x           # 84
```

### Strings
```crown
make name = "Alice"
say "Hello, " + name
```

### Exercises & Solutions
**2.1 Swap two numbers**
```crown
make a = 10
make b = 20
say "Before: a=" + a + ", b=" + b
make temp = a
set a = b
set b = temp
say "After: a=" + a + ", b=" + b
```

**2.2 Concatenate two strings**
```crown
make s = "Crown " + "Language"
say s
```

---

## Chapter 3 – Control Flow

### If/Else
```crown
make age = 17
if age >= 18 then
    say "Adult"
else
    say "Minor"
done
```

### Loops
```crown
loop from 1 to 5
    say i
done
```

### Break & Continue
```crown
loop from 1 to 10
    if i == 5 then break done
    if i % 2 == 0 then continue done
    say i
done
```

### Exercises & Solutions
**3.1 Odd numbers 1–15**
```crown
loop from 1 to 15
    if (i % 2) == 1 then say i done
done
```

**3.2 Countdown + Blast off**
```crown
loop from 5 to 1 step -1
    say i
done
say "Blast off!"
```

---

# Part II: Building Power

---

## Chapter 4 – Functions

### Defining Functions
```crown
do square(n)
    give n * n
done
```

### Recursion
```crown
do factorial(n)
    if n <= 1 then give 1 done
    give n * factorial(n - 1)
done
```

### Exercises & Solutions
**4.1 Triple a number**
```crown
do triple(n)
    give n * 3
done
say triple(4)
```

**4.2 Fibonacci recursion**
```crown
do fib(n)
    if n <= 1 then give n done
    give fib(n-1) + fib(n-2)
done
say fib(6)
```

---

## Chapter 5 – Data Structures

### Arrays & Maps
```crown
make animals = ["cat", "dog", "owl"]
make person = { "name": "Alice", "age": 25 }
```

### Iteration
```crown
foreach a in animals
    say "Animal: " + a
done
```

### Exercises & Solutions
**5.1 Shopping list**
```crown
make list = ["milk", "bread", "apples"]
foreach item in list
    say "Buy: " + item
done
```

**5.2 Student record**
```crown
make student = { "name": "Alice", "age": 25 }
say "Name: " + student["name"] + ", Age: " + student["age"]
```

---

## Chapter 6 – Pattern Matching

```crown
make data = [
    ["red", "blue"],
    { "name": "Bob", "age": 20 },
    42
]

foreach item in data
    match item with
        case [a, b]              say "Pair: " + a + ", " + b
        case { "name": n, "age": a } if a >= 18
            say n + " adult"
        default
            say "Other"
    done
done
```

### Exercises & Solutions
**6.1 Classify a number**
```crown
make nums = [0, -3, 10]
foreach n in nums
    match n with
        case 0             say "zero"
        case v if v > 0    say "positive"
        case v if v < 0    say "negative"
    done
done
```

**6.2 Admin role**
```crown
make user = { "role": "admin", "name": "Jane" }
match user with
    case { "role": "admin", "name": n } say "Welcome admin " + n
    default say "Access denied"
done
```

---

## Chapter 7 – Modules & Organization

**shapes.crown**
```crown
do area_square(n)
    give n * n
done
```

**main.crown**
```crown
import "shapes.crown"
say area_square(4)
```

---

# Part III: Advanced Crown

---

## Chapter 8 – Optimizations

Crown uses LLVM optimizations: constant folding, dead code elimination, loop unrolling, tail calls, vectorization.

**Exercise & Solution**
```crown
do sum(n)
    if n <= 0 then give 0 done
    give n + sum(n - 1)
done
say sum(10)
```

---

## Chapter 9 – Running Crown Everywhere

### Run in VM
```sh
python crown_script.py program.crown
```

### Compile AOT
```sh
crownc build program.crown -o program.exe
```

**program.crown**
```crown
make total = 0
loop from 1 to 1000000
    set total = total + i
done
say total
```

---

## Chapter 10 – Text Adventure

```crown
make rooms = {
  "start": { "desc": "You are in a dark room.", "exits": ["north"] },
  "north": { "desc": "You see a locked chest.", "exits": ["south"] },
  "east":  { "desc": "There is a shiny key here.", "exits": ["west"] }
}

make current = "start"
make inventory = []

while true
    say rooms[current]["desc"]
    say "Exits: " + rooms[current]["exits"]
    make cmd = read()
    if cmd == "quit" then break done

    if cmd == "take key" and current == "east"
        say "You picked up the key."
        set inventory = inventory + ["key"]
    else if cmd == "open chest" and current == "north"
        if "key" in inventory then
            say "You unlock the chest! Treasure is yours!"
            break
        else
            say "It’s locked..."
        done
    else if cmd in rooms[current]["exits"]
        set current = cmd
    else
        say "You can’t go that way."
    done
done
```

---

## Chapter 11 – Errors & Messaging (New)

**Goals:** Keep programs running, surface issues clearly, and communicate with users, files, and processes.

### 11.1 Error Philosophy
- Runtime errors are **contained by the VM**—your script retains control.
- **Beginner-friendly defaults**: uninitialized names read as safe defaults (e.g., `0`, `""`) so first runs don’t crash.
- Prefer **defensive programming**: validate inputs, use `match` with guards, and return sentinel values when needed.

### 11.2 Console Messages
```crown
say "Starting job"
say "progress=" + 42
say "done."
```

### 11.3 File Messages
```crown
make txt = read file "inbox.txt"
say "inbox len=" + length(txt)
write file "outbox.txt" with ("processed:\n" + txt)
```

### 11.4 Process Messages
```crown
spawn "echo Hello from Crown"
wait
```

### 11.5 Defensive Patterns
**Safe division**
```crown
do safe_div(a, b)
    if b == 0 then give { "ok": false, "err": "divide by zero" } done
    give { "ok": true, "val": a / b }
done

make r = safe_div(10, 0)
match r with
  case { "ok": true, "val": v }  say v
  case { "ok": false, "err": e } say "error: " + e
done
```

**Shape checks via match**
```crown
do greet(person)
    match person with
        case { "name": n } if n != ""  say "Hello, " + n
        default                        say "error: bad person record"
    done
done
```

**Warnings for defaults**
```crown
say x
if x == 0 then say "warning: x is default-initialized"
```

---

## Chapter 12 – Memory Model & Resource Handling (New)

**Goals:** Write efficient code without manual memory hazards.

### 12.1 The High-Level Model
- Crown’s runtime manages memory automatically for primitive values, arrays, and maps.
- You don’t call `malloc/free`; focus on **value flow** and **lifetime by scope**.
- Large data structures are passed by reference under the hood; treat them as values from code’s perspective.

### 12.2 Practical Guidance
**Reuse buffers**
```crown
make acc = ""
foreach line in ["a","b","c"]
    set acc = acc + line + "\n"
done
say acc
```

**Avoid giant temporaries**
```crown
# Good: process per-chunk
foreach chunk in read file "big.txt"  # assume iterator provided by runtime
    say "len=" + length(chunk)
done
```

**Mutate in place when sensible**
```crown
make xs = [1,2,3,4]
loop from 0 to length(xs)-1
    set xs[i] = xs[i] * 2
done
say xs
```

### 12.3 Files, Handles, and Processes
- File reads/writes are **scoped ops**; the runtime closes file handles after the operation.
- `spawn` launches a process; `wait` ensures cleanup of process resources.

```crown
spawn "mytool --version"
wait
```

### 12.4 Common Pitfalls & Fixes
- **Holding on to giant strings**: stream instead; process in chunks.
- **Accidental quadratic concatenation**: build arrays then `join`, or pre-size if available.
- **Dangling state**: centralize state in one map and pass it to helpers; return updated state.

```crown
make state = { "count": 0 }

do step(state)
    set state["count"] = state["count"] + 1
    give state
done

set state = step(state)
say state["count"]
```

---

# Epilogue

With these chapters you now understand **how Crown reports errors**, **how to message users/files/processes**, and **how the runtime manages memory** so you can focus on **clarity + performance**.
