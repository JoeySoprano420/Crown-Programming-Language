# Crown Script — Clean Integrated Build (v0.3)
# Run: python Crown_Script.py program.crown
import sys, re, json
from typing import List, Dict, Any, Optional

# ---------- LEXER ----------
class Token:
    def __init__(self, kind, value, line, col):
        self.kind, self.value, self.line, self.col = kind, value, line, col
    def __repr__(self): return f"{self.kind}:{self.value}"

KEYWORDS = {
    "say","make","give","do","done","loop","from","to","step",
    "if","else","then","true","false","and","or","not",
    "read","file","write","with","spawn","wait",
    "set","push","drop","foreach","in","break","continue",
    "while","repeat","until","switch","case","default",
    "match","_"
}

TOKEN_SPEC = [
    ("NUMBER",   r"\d+"),
    ("STRING",   r'"[^"\n]*"'),
    ("IDENT",    r"[A-Za-z_][A-Za-z0-9_]*"),
    ("OP",       r"==|!=|<=|>=|[+\-*/%<>=]"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("LBRACK",   r"\["),
    ("RBRACK",   r"\]"),
    ("LBRACE",   r"\{"),
    ("RBRACE",   r"\}"),
    ("COMMA",    r","),
    ("COLON",    r":"),
    ("NEWLINE",  r"\n"),
    ("SKIP",     r"[ \t]+"),
    ("COMMENT",  r"\#.*"),
    ("MISMATCH", r"."),
]

def tokenize(src: str) -> List[Token]:
    regex = "|".join(f"(?P<{n}>{p})" for n,p in TOKEN_SPEC)
    line=1; col=0; tokens=[]
    for m in re.finditer(regex, src):
        kind=m.lastgroup; value=m.group()
        if kind=="NEWLINE":
            line+=1; col=0; continue
        if kind in ("SKIP","COMMENT"):
            col+=len(value); continue
        col+=len(value)
        if kind=="NUMBER": tokens.append(Token("NUMBER", int(value), line, col))
        elif kind=="STRING": tokens.append(Token("STRING", value[1:-1], line, col))
        elif kind=="IDENT":
            low=value.lower()
            if low in KEYWORDS: tokens.append(Token(low.upper(), low, line, col))
            else: tokens.append(Token("IDENT", value, line, col))
        elif kind=="MISMATCH":
            raise SyntaxError(f"Unexpected {value!r} at line {line}")
        else:
            tokens.append(Token(kind, value, line, col))
    return tokens

# ---------- AST ----------
class Node: pass
class Program(Node):
    def __init__(self, stmts): self.stmts=stmts
class Say(Node):
    def __init__(self, expr): self.expr=expr
class Make(Node):
    def __init__(self, name, expr): self.name=name; self.expr=expr
class Assign(Node):
    def __init__(self, name, expr): self.name=name; self.expr=expr
class FuncDef(Node):
    def __init__(self, name, params, body): self.name=name; self.params=params; self.body=body
class Return(Node):
    def __init__(self, expr): self.expr=expr
class If(Node):
    def __init__(self, cond, then, other): self.cond=cond; self.then=then; self.other=other
class Loop(Node):
    def __init__(self, start,end,step,body): self.start=start; self.end=end; self.step=step; self.body=body
class While(Node):
    def __init__(self, cond, body): self.cond=cond; self.body=body
class RepeatUntil(Node):
    def __init__(self, body, cond): self.body=body; self.cond=cond
class Foreach(Node):
    def __init__(self, var, iterable, body): self.var=var; self.iterable=iterable; self.body=body
class Break(Node): pass
class Continue(Node): pass
class Switch(Node):
    def __init__(self, expr, cases, default): self.expr=expr; self.cases=cases; self.default=default
class Match(Node):
    def __init__(self, expr, cases, default): self.expr=expr; self.cases=cases; self.default=default

# Expressions
class Var(Node):
    def __init__(self, name): self.name=name
class Literal(Node):
    def __init__(self, value): self.value=value
class BinOp(Node):
    def __init__(self,l,o,r): self.l=l; self.o=o; self.r=r
class UnaryOp(Node):
    def __init__(self,o,e): self.o=o; self.e=e
class Call(Node):
    def __init__(self,name,args): self.name=name; self.args=args
class ArrayLit(Node):
    def __init__(self, items): self.items=items
class MapLit(Node):
    def __init__(self, pairs): self.pairs=pairs
class Index(Node):
    def __init__(self, base, idx): self.base=base; self.idx=idx
class SetIndex(Node):
    def __init__(self, target, idx, expr): self.target=target; self.idx=idx; self.expr=expr
class SetMapKey(Node):
    def __init__(self, target, key, expr): self.target=target; self.key=key; self.expr=expr
class Push(Node):
    def __init__(self, target, expr): self.target=target; self.expr=expr
class Drop(Node):
    def __init__(self, target): self.target=target

# Pattern nodes (simplified)
class WildcardPat(Node): pass
class ConstPat(Node):
    def __init__(self, expr): self.expr=expr
class ArrayPat(Node):
    def __init__(self, length_any=True): self.length_any=length_any
class MapPat(Node):
    def __init__(self, keys): self.keys=keys

# ---------- PARSER ----------
class Parser:
    def __init__(self,tokens): self.t=tokens; self.p=0; self.func_stack=[]
    def peek(self): return self.t[self.p] if self.p<len(self.t) else None
    def eat(self, kind=None):
        tok=self.peek()
        if not tok: raise SyntaxError("Unexpected EOF")
        if kind and tok.kind!=kind: raise SyntaxError(f"Expected {kind}, got {tok.kind}")
        self.p+=1; return tok
    def accept(self, kind):
        if self.peek() and self.peek().kind==kind:
            return self.eat(kind)
        return None

    def parse(self):
        stmts=[]
        while self.peek():
            stmts.append(self.statement())
        return Program(stmts)

    def statement(self):
        tok=self.peek()
        if tok.kind=="SAY":
            self.eat("SAY"); return Say(self.expression())
        if tok.kind=="MAKE":
            self.eat("MAKE")
            name=self.eat("IDENT").value
            self.eat("OP")  # '='
            return Make(name, self.expression())
        if tok.kind=="GIVE" and self.func_stack:
            self.eat("GIVE")
            return Return(self.expression())
        if tok.kind=="GIVE":  # assignment alias
            self.eat("GIVE")
            name=self.eat("IDENT").value
            self.eat("OP")
            return Assign(name, self.expression())
        if tok.kind=="SET":
            self.eat("SET")
            target=self.primary(lvalue=True)
            self.eat("OP")
            expr=self.expression()
            if isinstance(target, Index): return SetIndex(target.base, target.idx, expr)
            if isinstance(target, Var): return Assign(target.name, expr)
            raise SyntaxError("Unsupported set target")
        if tok.kind=="PUSH":
            self.eat("PUSH")
            arr=self.expression()
            self.eat("COMMA")
            val=self.expression()
            return Push(arr,val)
        if tok.kind=="DROP":
            self.eat("DROP")
            tgt=self.primary(lvalue=True)
            return Drop(tgt)
        if tok.kind=="DO":
            self.eat("DO")
            name=self.eat("IDENT").value
            self.eat("LPAREN")
            params=[]
            if not self.accept("RPAREN"):
                params.append(self.eat("IDENT").value)
                while self.accept("COMMA"):
                    params.append(self.eat("IDENT").value)
                self.eat("RPAREN")
            body=[]
            self.func_stack.append(name)
            while self.peek() and self.peek().kind!="DONE":
                body.append(self.statement())
            self.eat("DONE")
            self.func_stack.pop()
            return FuncDef(name, params, body)
        if tok.kind=="IF":
            self.eat("IF"); cond=self.expression(); self.eat("THEN")
            then=[]
            while self.peek() and self.peek().kind not in {"ELSE","DONE"}:
                then.append(self.statement())
            other=[]
            if self.accept("ELSE"):
                while self.peek() and self.peek().kind!="DONE":
                    other.append(self.statement())
            self.eat("DONE")
            return If(cond,then,other)
        if tok.kind=="LOOP":
            self.eat("LOOP"); self.eat("FROM"); start=self.expression()
            self.eat("TO"); end=self.expression()
            step=Literal(1)
            if self.accept("STEP"): step=self.expression()
            body=[]
            while self.peek() and self.peek().kind!="DONE":
                body.append(self.statement())
            self.eat("DONE")
            return Loop(start,end,step,body)
        if tok.kind=="WHILE":
            self.eat("WHILE")
            cond=self.expression()
            body=[]
            while self.peek() and self.peek().kind!="DONE":
                body.append(self.statement())
            self.eat("DONE")
            return While(cond,body)
        if tok.kind=="REPEAT":
            self.eat("REPEAT")
            body=[]
            while self.peek() and self.peek().kind!="UNTIL":
                body.append(self.statement())
            self.eat("UNTIL")
            cond=self.expression()
            return RepeatUntil(body,cond)
        if tok.kind=="FOREACH":
            self.eat("FOREACH")
            var=self.eat("IDENT").value
            self.eat("IN")
            it=self.expression()
            body=[]
            while self.peek() and self.peek().kind!="DONE":
                body.append(self.statement())
            self.eat("DONE")
            return Foreach(var,it,body)
        if tok.kind=="BREAK":
            self.eat("BREAK"); return Break()
        if tok.kind=="CONTINUE":
            self.eat("CONTINUE"); return Continue()
        if tok.kind=="SWITCH":
            self.eat("SWITCH"); expr=self.expression()
            cases=[]; default=[]
            while self.peek() and self.peek().kind!="DONE":
                if self.peek().kind=="CASE":
                    self.eat("CASE")
                    cv=self.expression()
                    cbody=[]
                    while self.peek() and self.peek().kind not in {"CASE","DEFAULT","DONE"}:
                        cbody.append(self.statement())
                    cases.append((cv,cbody))
                elif self.peek().kind=="DEFAULT":
                    self.eat("DEFAULT")
                    while self.peek() and self.peek().kind not in {"CASE","DONE"}:
                        default.append(self.statement())
            self.eat("DONE")
            return Switch(expr,cases,default)
        if tok.kind=="MATCH":
            self.eat("MATCH"); expr=self.expression()
            self.eat("WITH")
            cases=[]; default=[]
            while self.peek() and self.peek().kind!="DONE":
                if self.peek().kind=="CASE":
                    self.eat("CASE")
                    pat=self.parse_pattern()
                    body=[]
                    while self.peek() and self.peek().kind not in {"CASE","DEFAULT","DONE"}:
                        body.append(self.statement())
                    cases.append((pat,body))
                elif self.peek().kind=="DEFAULT":
                    self.eat("DEFAULT")
                    while self.peek() and self.peek().kind not in {"CASE","DONE"}:
                        default.append(self.statement())
            self.eat("DONE")
            return Match(expr,cases,default)
        # expression-only call
        expr=self.expression()
        if isinstance(expr,Call):
            return expr
        raise SyntaxError(f"Unexpected start of statement: {tok.kind}")

    def parse_pattern(self):
        if self.accept("LBRACK"):
            self.eat("RBRACK"); return ArrayPat()
        if self.accept("LBRACE"):
            keys=[]
            if not self.accept("RBRACE"):
                if self.peek().kind!="RBRACE":
                    keys.append(self.pattern_key())
                    while self.accept("COMMA"):
                        keys.append(self.pattern_key())
                self.eat("RBRACE")
            return MapPat(keys)
        if self.accept("IDENT") and self.t[self.p-1].value=="_":
            return WildcardPat()
        # constant pattern
        return ConstPat(self.expression())

    def pattern_key(self):
        tok=self.peek()
        if tok.kind=="STRING":
            return self.eat("STRING").value
        raise SyntaxError("Expected string key in map pattern")

    # ----- Expressions (precedence) -----
    def expression(self): return self.logic_or()
    def logic_or(self):
        node=self.logic_and()
        while self.peek() and self.peek().value=="or":
            op=self.eat().value; node=BinOp(node,op,self.logic_and())
        return node
    def logic_and(self):
        node=self.equality()
        while self.peek() and self.peek().value=="and":
            op=self.eat().value; node=BinOp(node,op,self.equality())
        return node
    def equality(self):
        node=self.comparison()
        while self.peek() and self.peek().value in ("==","!="):
            op=self.eat().value; node=BinOp(node,op,self.comparison())
        return node
    def comparison(self):
        node=self.term()
        while self.peek() and self.peek().value in ("<","<=",">",">="):
            op=self.eat().value; node=BinOp(node,op,self.term())
        return node
    def term(self):
        node=self.factor()
        while self.peek() and self.peek().value in ("+","-"):
            op=self.eat().value; node=BinOp(node,op,self.factor())
        return node
    def factor(self):
        node=self.unary()
        while self.peek() and self.peek().value in ("*","/","%"):
            op=self.eat().value; node=BinOp(node,op,self.unary())
        return node
    def unary(self):
        if self.peek() and self.peek().value in ("-","not"):
            op=self.eat().value; return UnaryOp(op,self.unary())
        return self.primary()

    def primary(self, lvalue=False):
        tok=self.peek()
        if not tok: raise SyntaxError("EOF in expression")
        if tok.kind=="NUMBER": self.eat(); return Literal(tok.value)
        if tok.kind=="STRING": self.eat(); return Literal(tok.value)
        if tok.kind=="TRUE": self.eat(); return Literal(True)
        if tok.kind=="FALSE": self.eat(); return Literal(False)
        if tok.kind=="IDENT":
            name=self.eat("IDENT").value
            node=Var(name)
            if self.accept("LPAREN"):
                args=[]
                if not self.accept("RPAREN"):
                    args.append(self.expression())
                    while self.accept("COMMA"):
                        args.append(self.expression())
                    self.eat("RPAREN")
                node=Call(name,args)
            # indexing chain
            while self.accept("LBRACK"):
                idx=self.expression(); self.eat("RBRACK")
                node=Index(node,idx)
            return node
        if tok.kind=="LBRACK":
            self.eat("LBRACK")
            items=[]
            if not self.accept("RBRACK"):
                items.append(self.expression())
                while self.accept("COMMA"):
                    items.append(self.expression())
                self.eat("RBRACK")
            return ArrayLit(items)
        if tok.kind=="LBRACE":
            self.eat("LBRACE")
            pairs=[]
            if not self.accept("RBRACE"):
                key=self.expression(); self.eat("COLON"); val=self.expression()
                pairs.append((key,val))
                while self.accept("COMMA"):
                    key=self.expression(); self.eat("COLON"); val=self.expression()
                    pairs.append((key,val))
                self.eat("RBRACE")
            return MapLit(pairs)
        if tok.kind=="LPAREN":
            self.eat("LPAREN"); e=self.expression(); self.eat("RPAREN"); return e
        raise SyntaxError(f"Unexpected token {tok.kind} in expression")

# ---------- VM ----------
class ReturnSignal(Exception):
    def __init__(self,value): self.value=value
class BreakSignal(Exception): pass
class ContinueSignal(Exception): pass

class VM:
    def __init__(self, ast: Program):
        self.ast=ast
        self.globals: Dict[str, Any]={}
        self.funcs: Dict[str, FuncDef]={}
        self.builtins={
            "print":lambda *a: print(*a),
            "length":lambda x: len(x),
            "append":lambda arr,val: (arr.append(val) or arr),
            "keys":lambda m: list(m.keys()),
            "values":lambda m: list(m.values()),
            "jsonstring":lambda x: json.dumps(x, ensure_ascii=False)
        }

    def run(self):
        for s in self.ast.stmts:
            self.exec_stmt(s)

    # ---- Execution ----
    def exec_block(self, stmts):
        for s in stmts:
            self.exec_stmt(s)

    def exec_stmt(self, s: Node):
        if isinstance(s,Say):
            val=self.eval(s.expr)
            if isinstance(val,(list,dict)):
                print(json.dumps(val, ensure_ascii=False))
            else:
                print(val)
        elif isinstance(s,Make):
            self.globals.setdefault(s.name, self.eval(s.expr))
        elif isinstance(s,Assign):
            self.globals[s.name]=self.eval(s.expr)
        elif isinstance(s,FuncDef):
            self.funcs[s.name]=s
        elif isinstance(s,Return):
            raise ReturnSignal(self.eval(s.expr))
        elif isinstance(s,If):
            if self.eval(s.cond): self.exec_block(s.then)
            else: self.exec_block(s.other)
        elif isinstance(s,Loop):
            start=self.eval(s.start); end=self.eval(s.end); step=self.eval(s.step)
            i=start
            while i<=end:
                try:
                    self.exec_block(s.body)
                except ContinueSignal:
                    pass
                except BreakSignal:
                    break
                i+=step
        elif isinstance(s,While):
            while self.eval(s.cond):
                try:
                    self.exec_block(s.body)
                except ContinueSignal:
                    continue
                except BreakSignal:
                    break
        elif isinstance(s,RepeatUntil):
            while True:
                try:
                    self.exec_block(s.body)
                except ContinueSignal:
                    pass
                except BreakSignal:
                    break
                if self.eval(s.cond): break
        elif isinstance(s,Foreach):
            it=self.eval(s.iterable)
            if isinstance(it,dict):
                itr=list(it.keys())
            else:
                itr=it
            for item in itr:
                self.globals[s.var]=item
                try:
                    self.exec_block(s.body)
                except ContinueSignal:
                    continue
                except BreakSignal:
                    break
        elif isinstance(s,Break):
            raise BreakSignal()
        elif isinstance(s,Continue):
            raise ContinueSignal()
        elif isinstance(s,Switch):
            val=self.eval(s.expr)
            executed=False
            for cv, body in s.cases:
                if self.eval(cv)==val:
                    self.exec_block(body); executed=True; break
            if not executed and s.default:
                self.exec_block(s.default)
        elif isinstance(s,Match):
            val=self.eval(s.expr)
            matched=False
            for pat, body in s.cases:
                if self.match(pat,val):
                    self.exec_block(body); matched=True; break
            if not matched and s.default:
                self.exec_block(s.default)
        elif isinstance(s,Call):
            self.eval(s)
        elif isinstance(s,SetIndex):
            base=self.eval(s.target)
            idx=self.eval(s.idx)
            val=self.eval(s.expr)
            base[idx]=val
        elif isinstance(s,Push):
            arr=self.eval(s.target)
            arr.append(self.eval(s.expr))
        elif isinstance(s,Drop):
            tgt=s.target
            if isinstance(tgt,Var):
                self.globals.pop(tgt.name, None)
            elif isinstance(tgt,Index):
                base=self.eval(tgt.base); idx=self.eval(tgt.idx)
                del base[idx]
            else:
                raise RuntimeError("Unsupported drop target")
        else:
            raise RuntimeError(f"Unhandled statement {type(s).__name__}")

    # ---- Eval expressions ----
    def eval(self, node: Node):
        if isinstance(node,Literal): return node.value
        if isinstance(node,Var): return self.globals.get(node.name,0)
        if isinstance(node,BinOp):
            l=self.eval(node.l); r=self.eval(node.r); o=node.o
            if o=="+": return l+r
            if o=="-": return l-r
            if o=="*": return l*r
            if o=="/": return l//r if isinstance(l,int) and isinstance(r,int) else l/r
            if o=="%": return l%r
            if o=="<": return l<r
            if o==">": return l>r
            if o=="<=": return l<=r
            if o==">=": return l>=r
            if o=="==": return l==r
            if o=="!=": return l!=r
            if o=="and": return bool(l) and bool(r)
            if o=="or": return bool(l) or bool(r)
            raise RuntimeError(f"Unknown op {o}")
        if isinstance(node,UnaryOp):
            v=self.eval(node.e)
            if node.o=="-": return -v
            if node.o=="not": return not v
        if isinstance(node,Call):
            # builtin
            if node.name in self.builtins:
                return self.builtins[node.name](*[self.eval(a) for a in node.args])
            if node.name not in self.funcs:
                raise RuntimeError(f"Unknown function {node.name}")
            f=self.funcs[node.name]
            if len(f.params)!=len(node.args):
                raise RuntimeError("Arity mismatch")
            snapshot=self.globals.copy()
            try:
                for p,arg in zip(f.params,node.args):
                    self.globals[p]=self.eval(arg)
                for st in f.body:
                    self.exec_stmt(st)
            except ReturnSignal as r:
                return r.value
            finally:
                self.globals=snapshot
            return None
        if isinstance(node,ArrayLit):
            return [self.eval(i) for i in node.items]
        if isinstance(node,MapLit):
            return { self.eval(k): self.eval(v) for k,v in node.pairs }
        if isinstance(node,Index):
            base=self.eval(node.base); idx=self.eval(node.idx); return base[idx]
        return None

    # ---- Pattern matching ----
    def match(self, pat: Node, val):
        if isinstance(pat,WildcardPat): return True
        if isinstance(pat,ConstPat): return self.eval(pat.expr)==val
        if isinstance(pat,ArrayPat): return isinstance(val,list)
        if isinstance(pat,MapPat):
            if not isinstance(val,dict): return False
            return all(k in val for k in pat.keys)
        return False

# ---------- DRIVER ----------
def main():
    if len(sys.argv)<2:
        print("Usage: python Crown_Script.py file.crown")
        sys.exit(1)
    with open(sys.argv[1], encoding="utf-8") as f:
        src=f.read()
    tokens=tokenize(src)
    ast=Parser(tokens).parse()
    VM(ast).run()

if __name__=="__main__":
    main()

# ---------- COMPILER BACKEND (AOT + JIT) ----------
try:
    from llvmlite import ir, binding
except ImportError:
    ir = None
    binding = None

class CrownCompiler:
    def __init__(self, ast: Program):
        self.ast = ast
        self.module = ir.Module(name="crown_module") if ir else None
        self.funcs = {}

    def compile(self):
        if not ir:
            raise RuntimeError("llvmlite not installed, cannot compile")
        # Define a main() function
        fnty = ir.FunctionType(ir.IntType(32), [])
        main_fn = ir.Function(self.module, fnty, name="main")
        block = main_fn.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        # For now, simply call into a builtin "printf"
        printf_ty = ir.FunctionType(ir.IntType(32), [ir.PointerType(ir.IntType(8))], var_arg=True)
        printf = ir.Function(self.module, printf_ty, name="printf")

        fmt = "%s\n\0"
        c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt)),
                            bytearray(fmt.encode("utf8")))
        global_fmt = ir.GlobalVariable(self.module, c_fmt.type, name="fstr")
        global_fmt.linkage = "internal"
        global_fmt.global_constant = True
        global_fmt.initializer = c_fmt
        fmt_ptr = builder.bitcast(global_fmt, ir.PointerType(ir.IntType(8)))

        # Walk AST and emit trivial say statements as printf
        for s in self.ast.stmts:
            if isinstance(s, Say):
                if isinstance(s.expr, Literal) and isinstance(s.expr.value, str):
                    s_val = s.expr.value + "\0"
                    c_str = ir.Constant(ir.ArrayType(ir.IntType(8), len(s_val)),
                                        bytearray(s_val.encode("utf8")))
                    gv = ir.GlobalVariable(self.module, c_str.type, name=f"str{len(self.funcs)}")
                    gv.linkage = "internal"
                    gv.global_constant = True
                    gv.initializer = c_str
                    str_ptr = builder.bitcast(gv, ir.PointerType(ir.IntType(8)))
                    builder.call(printf, [fmt_ptr, str_ptr])

        builder.ret(ir.Constant(ir.IntType(32), 0))
        return str(self.module)

    def jit_run(self):
        if not binding:
            raise RuntimeError("llvmlite binding not installed")
        llvm_ir = self.compile()
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()
        target = binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        backing_mod = binding.parse_assembly("")
        engine = binding.create_mcjit_compiler(backing_mod, target_machine)

        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        engine.add_module(mod)
        engine.finalize_object()
        engine.run_static_constructors()

        # Run main()
        func_ptr = engine.get_function_address("main")
        import ctypes
        cfunc = ctypes.CFUNCTYPE(ctypes.c_int)(func_ptr)
        return cfunc()

    def aot_build(self, outfile="a.out"):
        if not binding:
            raise RuntimeError("llvmlite binding not installed")
        llvm_ir = self.compile()
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()
        target = binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        with open(outfile + ".ll", "w") as f:
            f.write(str(mod))
        obj = target_machine.emit_object(mod)
        with open(outfile + ".o", "wb") as f:
            f.write(obj)
        import subprocess, os
        exe = outfile if outfile.endswith(".exe") else outfile + ".exe"
        # Use clang or gcc to link
        cc = "clang" if shutil.which("clang") else "gcc"
        subprocess.run([cc, outfile + ".o", "-o", exe])
        return exe

# ---------- crownc DRIVER ----------
import argparse, shutil

def crownc():
    parser = argparse.ArgumentParser(prog="crownc")
    parser.add_argument("file", help="Crown source file (.crown)")
    parser.add_argument("-o", "--output", help="Output file", default="a.out")
    parser.add_argument("--jit", action="store_true", help="Run with JIT instead of interpret")
    parser.add_argument("--aot", action="store_true", help="Compile to native executable")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as f:
        src = f.read()
    tokens = tokenize(src)
    ast = Parser(tokens).parse()

    if args.jit:
        print("[crownc] Running with JIT backend...")
        result = CrownCompiler(ast).jit_run()
        print(f"[crownc] Program returned {result}")
    elif args.aot:
        print("[crownc] Building AOT executable...")
        exe = CrownCompiler(ast).aot_build(args.output)
        print(f"[crownc] Built {exe}")
    else:
        print("[crownc] Interpreting...")
        VM(ast).run()

if __name__ == "__main__" and sys.argv[0].endswith("crownc.py"):
    crownc()

    # ---------- FULL LLVM CODEGEN ----------
if ir and binding:
    class CodegenLLVM:
        def __init__(self, ast: Program):
            self.ast = ast
            self.module = ir.Module(name="crown_module")
            self.builder = None
            self.funcs = {}
            self.globals = {}

        # ---- Entry ----
        def build(self):
            main_ty = ir.FunctionType(ir.IntType(32), [])
            main_fn = ir.Function(self.module, main_ty, name="main")
            block = main_fn.append_basic_block("entry")
            self.builder = ir.IRBuilder(block)
            for stmt in self.ast.stmts:
                self.codegen_stmt(stmt)
            self.builder.ret(ir.Constant(ir.IntType(32), 0))
            return str(self.module)

        # ---- Helpers ----
        def to_ir_val(self, val):
            if isinstance(val, bool):
                return ir.Constant(ir.IntType(1), int(val))
            if isinstance(val, int):
                return ir.Constant(ir.IntType(64), val)
            raise RuntimeError(f"Unsupported literal {val}")

        # ---- Codegen for statements ----
        def codegen_stmt(self, stmt):
            if isinstance(stmt, Say):
                val = self.codegen_expr(stmt.expr)
                self.printf_val(val)
            elif isinstance(stmt, Make):
                v = self.codegen_expr(stmt.expr)
                self.globals[stmt.name] = v
            elif isinstance(stmt, Assign):
                v = self.codegen_expr(stmt.expr)
                self.globals[stmt.name] = v
            elif isinstance(stmt, If):
                self.codegen_if(stmt)
            elif isinstance(stmt, Loop):
                self.codegen_loop(stmt)
            elif isinstance(stmt, While):
                self.codegen_while(stmt)
            elif isinstance(stmt, RepeatUntil):
                self.codegen_repeat(stmt)
            elif isinstance(stmt, FuncDef):
                self.codegen_func(stmt)
            elif isinstance(stmt, Call):
                self.codegen_expr(stmt)
            # (Add other stmts as needed)

        def codegen_if(self, stmt: If):
            cond = self.codegen_expr(stmt.cond)
            then_bb = self.builder.append_basic_block("then")
            else_bb = self.builder.append_basic_block("else")
            merge_bb = self.builder.append_basic_block("ifend")
            self.builder.cbranch(cond, then_bb, else_bb)

            self.builder.position_at_start(then_bb)
            for s in stmt.then:
                self.codegen_stmt(s)
            self.builder.branch(merge_bb)

            self.builder.position_at_start(else_bb)
            for s in stmt.other:
                self.codegen_stmt(s)
            self.builder.branch(merge_bb)

            self.builder.position_at_start(merge_bb)

        def codegen_loop(self, stmt: Loop):
            start = self.codegen_expr(stmt.start)
            end = self.codegen_expr(stmt.end)
            step = self.codegen_expr(stmt.step)
            var = ir.GlobalVariable(self.module, ir.IntType(64), "loopi")
            var.initializer = start
            var.linkage = "internal"

            loop_bb = self.builder.append_basic_block("loop")
            after_bb = self.builder.append_basic_block("afterloop")

            self.builder.branch(loop_bb)
            self.builder.position_at_start(loop_bb)

            # body
            for s in stmt.body:
                self.codegen_stmt(s)

            # increment
            cur = self.builder.load(var)
            nxt = self.builder.add(cur, step)
            self.builder.store(nxt, var)
            cmp = self.builder.icmp_signed("<=", nxt, end)
            self.builder.cbranch(cmp, loop_bb, after_bb)

            self.builder.position_at_start(after_bb)

        def codegen_while(self, stmt: While):
            cond_bb = self.builder.append_basic_block("while.cond")
            body_bb = self.builder.append_basic_block("while.body")
            end_bb = self.builder.append_basic_block("while.end")

            self.builder.branch(cond_bb)
            self.builder.position_at_start(cond_bb)
            cond_val = self.codegen_expr(stmt.cond)
            self.builder.cbranch(cond_val, body_bb, end_bb)

            self.builder.position_at_start(body_bb)
            for s in stmt.body:
                self.codegen_stmt(s)
            self.builder.branch(cond_bb)

            self.builder.position_at_start(end_bb)

        def codegen_repeat(self, stmt: RepeatUntil):
            body_bb = self.builder.append_basic_block("repeat.body")
            cond_bb = self.builder.append_basic_block("repeat.cond")
            end_bb = self.builder.append_basic_block("repeat.end")

            self.builder.branch(body_bb)
            self.builder.position_at_start(body_bb)
            for s in stmt.body:
                self.codegen_stmt(s)
            self.builder.branch(cond_bb)

            self.builder.position_at_start(cond_bb)
            cond_val = self.codegen_expr(stmt.cond)
            self.builder.cbranch(cond_val, end_bb, body_bb)

            self.builder.position_at_start(end_bb)

        def codegen_func(self, f: FuncDef):
            arg_types = [ir.IntType(64)] * len(f.params)
            fnty = ir.FunctionType(ir.IntType(64), arg_types)
            fn = ir.Function(self.module, fnty, name=f.name)
            block = fn.append_basic_block("entry")
            saved_builder = self.builder
            self.builder = ir.IRBuilder(block)
            # body
            for s in f.body:
                self.codegen_stmt(s)
            self.builder.ret(ir.Constant(ir.IntType(64), 0))
            self.funcs[f.name] = fn
            self.builder = saved_builder

        # ---- Codegen for expressions ----
        def codegen_expr(self, expr):
            if isinstance(expr, Literal):
                return self.to_ir_val(expr.value)
            if isinstance(expr, Var):
                return self.globals.get(expr.name, ir.Constant(ir.IntType(64), 0))
            if isinstance(expr, BinOp):
                l = self.codegen_expr(expr.l)
                r = self.codegen_expr(expr.r)
                if expr.o == "+": return self.builder.add(l, r)
                if expr.o == "-": return self.builder.sub(l, r)
                if expr.o == "*": return self.builder.mul(l, r)
                if expr.o == "/": return self.builder.sdiv(l, r)
                if expr.o in ("<", ">", "<=", ">=", "==", "!="):
                    return self.builder.icmp_signed(expr.o, l, r)
            if isinstance(expr, Call):
                if expr.name in self.funcs:
                    fn = self.funcs[expr.name]
                    args = [self.codegen_expr(a) for a in expr.args]
                    return self.builder.call(fn, args)
                else:
                    raise RuntimeError(f"Unknown function {expr.name}")
            return ir.Constant(ir.IntType(64), 0)

        # ---- Print support ----
        def printf_val(self, val):
            printf_ty = ir.FunctionType(ir.IntType(32), [ir.PointerType(ir.IntType(8))], var_arg=True)
            printf = self.module.globals.get("printf")
            if not printf:
                printf = ir.Function(self.module, printf_ty, name="printf")
            fmt = "%lld\n\0"
            c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt)),
                                bytearray(fmt.encode("utf8")))
            gv = ir.GlobalVariable(self.module, c_fmt.type, name=f"fstr{len(self.globals)}")
            gv.linkage = "internal"
            gv.global_constant = True
            gv.initializer = c_fmt
            fmt_ptr = self.builder.bitcast(gv, ir.PointerType(ir.IntType(8)))
            self.builder.call(printf, [fmt_ptr, val])

    # ---------- Compiler Integration ----------
    class CrownCompilerFull:
        def __init__(self, ast: Program):
            self.ast = ast

        def compile_ir(self):
            return CodegenLLVM(self.ast).build()

        def jit_run(self):
            llvm_ir = self.compile_ir()
            binding.initialize()
            binding.initialize_native_target()
            binding.initialize_native_asmprinter()
            target = binding.Target.from_default_triple()
            target_machine = target.create_target_machine()
            backing_mod = binding.parse_assembly("")
            engine = binding.create_mcjit_compiler(backing_mod, target_machine)
            mod = binding.parse_assembly(llvm_ir)
            mod.verify()
            engine.add_module(mod)
            engine.finalize_object()
            engine.run_static_constructors()
            func_ptr = engine.get_function_address("main")
            import ctypes
            cfunc = ctypes.CFUNCTYPE(ctypes.c_int)(func_ptr)
            return cfunc()

        def aot_build(self, outfile="a.out"):
            llvm_ir = self.compile_ir()
            binding.initialize()
            binding.initialize_native_target()
            binding.initialize_native_asmprinter()
            target = binding.Target.from_default_triple()
            target_machine = target.create_target_machine()
            mod = binding.parse_assembly(llvm_ir)
            mod.verify()
            with open(outfile + ".ll", "w") as f:
                f.write(str(mod))
            obj = target_machine.emit_object(mod)
            with open(outfile + ".o", "wb") as f:
                f.write(obj)
            import subprocess, shutil
            exe = outfile if outfile.endswith(".exe") else outfile + ".exe"
            cc = "clang" if shutil.which("clang") else "gcc"
            subprocess.run([cc, outfile + ".o", "-o", exe])
            return exe

        # ---------- Parser and AST Nodes ----------
        def parse(self, src: str):
            tokens = tokenize(src)
            return Parser(tokens).parse()

        # ---------- OPTIMIZATION PIPELINE ----------
class CrownOptimizer:
    def __init__(self, ast):
        self.ast = ast

    def run(self):
        self.ast = self.constant_fold(self.ast)
        self.ast = self.dead_code_elim(self.ast)
        self.ast = self.peephole(self.ast)
        self.ast = self.loop_unroll(self.ast)
        self.ast = self.tail_calls(self.ast)
        return self.ast

    def constant_fold(self, node):
        if isinstance(node, BinOp):
            l = self.constant_fold(node.l)
            r = self.constant_fold(node.r)
            if isinstance(l, Literal) and isinstance(r, Literal):
                if node.o == "+": return Literal(l.value + r.value)
                if node.o == "-": return Literal(l.value - r.value)
                if node.o == "*": return Literal(l.value * r.value)
                if node.o == "/": return Literal(l.value // r.value)
            return BinOp(l, node.o, r)
        if isinstance(node, Program):
            return Program([self.constant_fold(s) for s in node.stmts])
        return node

    def dead_code_elim(self, node):
        if isinstance(node, If):
            cond = node.cond
            if isinstance(cond, Literal):
                return Program(node.then if cond.value else node.other)
        if isinstance(node, Program):
            return Program([self.dead_code_elim(s) for s in node.stmts])
        return node

    def peephole(self, node):
        if isinstance(node, Assign) and isinstance(node.expr, Var) and node.name == node.expr.name:
            return None  # redundant self-assignment
        if isinstance(node, Program):
            return Program([x for s in node.stmts if (x:=self.peephole(s))])
        return node

    def loop_unroll(self, node):
        if isinstance(node, Loop):
            start, end, step = (node.start, node.end, node.step)
            if all(isinstance(x, Literal) for x in [start,end,step]):
                unrolled = []
                i = start.value
                while i <= end.value:
                    for s in node.body:
                        unrolled.append(s)
                    i += step.value
                return Program(unrolled)
        if isinstance(node, Program):
            return Program([self.loop_unroll(s) for s in node.stmts])
        return node

    def tail_calls(self, node):
        # Convert recursive tail calls to loop form (simplified)
        return node

# ---------- RUNTIME HELPERS FOR LLVM ----------
if ir and binding:
    from llvmlite import ir as llvm_ir

    # Runtime helpers (to be linked with libc)
    # arrays = malloc + index
    # maps   = dict-like (stubbed with Python C API, here just placeholder)

    # Extended CodegenLLVM
    class CodegenLLVM(CodegenLLVM):  # extend previous one
        def codegen_foreach(self, stmt: Foreach):
            # foreach var in iterable: body
            iterable = self.codegen_expr(stmt.iterable)
            idx_ptr = self.builder.alloca(ir.IntType(64), name="i")
            self.builder.store(ir.Constant(ir.IntType(64), 0), idx_ptr)

            loop_bb = self.builder.append_basic_block("foreach.loop")
            end_bb = self.builder.append_basic_block("foreach.end")

            self.builder.branch(loop_bb)
            self.builder.position_at_start(loop_bb)
            idx = self.builder.load(idx_ptr)
            # NOTE: real impl requires knowing len(iterable), stub = 10
            length = ir.Constant(ir.IntType(64), 10)
            cond = self.builder.icmp_signed("<", idx, length)
            self.builder.cbranch(cond, loop_bb, end_bb)

            for s in stmt.body:
                self.codegen_stmt(s)

            nxt = self.builder.add(idx, ir.Constant(ir.IntType(64), 1))
            self.builder.store(nxt, idx_ptr)
            self.builder.branch(loop_bb)

            self.builder.position_at_start(end_bb)

        def codegen_match(self, stmt: Match):
            # lower match into if-else cascades
            val = self.codegen_expr(stmt.expr)
            end_bb = self.builder.append_basic_block("match.end")
            next_bb = None
            for pat, body in stmt.cases:
                bb = self.builder.append_basic_block("case")
                if next_bb: self.builder.position_at_start(next_bb)
                cond = self.codegen_expr(pat.expr)
                self.builder.cbranch(cond, bb, end_bb)
                self.builder.position_at_start(bb)
                for s in body: self.codegen_stmt(s)
                self.builder.branch(end_bb)
                next_bb = bb
            if stmt.default:
                for s in stmt.default:
                    self.codegen_stmt(s)
            self.builder.position_at_start(end_bb)

        def codegen_stmt(self, stmt):
            if isinstance(stmt, Foreach): return self.codegen_foreach(stmt)
            if isinstance(stmt, Match): return self.codegen_match(stmt)
            return super().codegen_stmt(stmt)

# ---------- EXTREME JIT/AOT SPEED ----------
class CrownCompilerExtreme(CrownCompilerFull):
    def __init__(self, ast: Program):
        super().__init__(ast)

    def compile_ir(self):
        # run optimizer passes before LLVM
        opt_ast = CrownOptimizer(self.ast).run()
        return CodegenLLVM(opt_ast).build()

    def jit_run(self):
        # PGO hook: count node visits
        result = super().jit_run()
        return result

    def aot_build(self, outfile="a.out"):
        exe = super().aot_build(outfile)
        # Compression stub: upx-like step could go here
        return exe

    # ================================================================
# Crown Script Supreme LLVM Codegen + Compiler CLI
# ================================================================
try:
    from llvmlite import ir, binding
    LLVM_AVAILABLE = True
except ImportError:
    LLVM_AVAILABLE = False

# --------- LLVM CODEGEN ---------
class CrownLLVMCodegen:
    def __init__(self):
        self.module = ir.Module(name="crown_module")
        self.funcs = {}
        self.builder = None
        self.printf = None
        self.declare_runtime()

    def declare_runtime(self):
        # Declare printf
        voidptr_ty = ir.IntType(8).as_pointer()
        self.printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        self.printf = ir.Function(self.module, self.printf_ty, name="printf")

    def codegen(self, ast: Program):
        for stmt in ast.stmts:
            self.codegen_stmt(stmt)
        return str(self.module)

    def codegen_stmt(self, stmt):
        if isinstance(stmt, Say):
            fmt = "%d\n\0"
            c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt)),
                                bytearray(fmt.encode("utf8")))
            global_fmt = ir.GlobalVariable(self.module, c_fmt.type, name="fstr")
            global_fmt.linkage = 'internal'
            global_fmt.global_constant = True
            global_fmt.initializer = c_fmt
            fmt_ptr = self.builder.bitcast(global_fmt, ir.IntType(8).as_pointer())
            val = self.codegen_expr(stmt.expr)
            self.builder.call(self.printf, [fmt_ptr, val])
        elif isinstance(stmt, Make):
            val = self.codegen_expr(stmt.expr)
            gv = ir.GlobalVariable(self.module, val.type, name=stmt.name)
            gv.initializer = val
            self.funcs[stmt.name] = gv
        elif isinstance(stmt, FuncDef):
            fn_ty = ir.FunctionType(ir.IntType(64), [ir.IntType(64)] * len(stmt.params))
            fn = ir.Function(self.module, fn_ty, name=stmt.name)
            self.funcs[stmt.name] = fn
            block = fn.append_basic_block(name="entry")
            self.builder = ir.IRBuilder(block)
            for st in stmt.body:
                self.codegen_stmt(st)
            if not self.builder.block.is_terminated:
                self.builder.ret(ir.Constant(ir.IntType(64), 0))
        elif isinstance(stmt, Return):
            val = self.codegen_expr(stmt.expr)
            self.builder.ret(val)
        elif isinstance(stmt, If):
            cond_val = self.codegen_expr(stmt.cond)
            zero = ir.Constant(ir.IntType(1), 0)
            cond_bool = self.builder.icmp_signed('!=', cond_val, zero, "ifcond")
            then_bb = self.builder.append_basic_block("then")
            else_bb = self.builder.append_basic_block("else")
            merge_bb = self.builder.append_basic_block("ifcont")
            self.builder.cbranch(cond_bool, then_bb, else_bb)

            self.builder.position_at_start(then_bb)
            for st in stmt.then:
                self.codegen_stmt(st)
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_bb)

            self.builder.position_at_start(else_bb)
            for st in stmt.other:
                self.codegen_stmt(st)
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_bb)

            self.builder.position_at_start(merge_bb)

        elif isinstance(stmt, Loop):
            start = self.codegen_expr(stmt.start)
            end = self.codegen_expr(stmt.end)
            step = self.codegen_expr(stmt.step)
            preheader = self.builder.block
            loop_bb = self.builder.append_basic_block("loop")
            after_bb = self.builder.append_basic_block("afterloop")

            self.builder.branch(loop_bb)
            self.builder.position_at_start(loop_bb)
            phi = self.builder.phi(ir.IntType(64), "i")
            phi.add_incoming(start, preheader)

            for st in stmt.body:
                self.codegen_stmt(st)

            nextval = self.builder.add(phi, step, "nextval")
            cmp = self.builder.icmp_signed("<=", nextval, end, "loopcond")
            self.builder.cbranch(cmp, loop_bb, after_bb)
            phi.add_incoming(nextval, self.builder.block)

            self.builder.position_at_start(after_bb)

    def codegen_expr(self, expr):
        if isinstance(expr, Literal):
            if isinstance(expr.value, bool):
                return ir.Constant(ir.IntType(1), int(expr.value))
            if isinstance(expr.value, int):
                return ir.Constant(ir.IntType(64), expr.value)
        elif isinstance(expr, BinOp):
            l = self.codegen_expr(expr.l)
            r = self.codegen_expr(expr.r)
            if expr.o == "+": return self.builder.add(l, r, "addtmp")
            if expr.o == "-": return self.builder.sub(l, r, "subtmp")
            if expr.o == "*": return self.builder.mul(l, r, "multmp")
            if expr.o == "/": return self.builder.sdiv(l, r, "divtmp")
        return ir.Constant(ir.IntType(64), 0)


# --------- OPTIMIZER SUPREME ---------
class SupremeOptimizer:
    def __init__(self, llvm_ir: str):
        self.llvm_ir = llvm_ir

    def optimize(self, level=3):
        if not LLVM_AVAILABLE: return self.llvm_ir
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()
        llvm_mod = binding.parse_assembly(self.llvm_ir)
        llvm_mod.verify()
        pmb = binding.PassManagerBuilder()
        pmb.opt_level = level
        pm = binding.ModulePassManager()
        pmb.populate(pm)
        pm.run(llvm_mod)
        return str(llvm_mod)


# --------- COMPILER CLI ---------
def crownc_cli():
    import argparse, os
    p = argparse.ArgumentParser(prog="crownc", description="Crown Script Compiler Supreme")
    p.add_argument("file", help="Source .crown file")
    p.add_argument("-o", "--output", help="Output binary/exe")
    p.add_argument("--jit", action="store_true", help="Run with JIT instead of AOT")
    p.add_argument("-O", type=int, default=2, help="Optimization level (0–3)")
    args = p.parse_args()

    with open(args.file, encoding="utf-8") as f:
        src = f.read()
    tokens = tokenize(src)
    ast = Parser(tokens).parse()

    if not LLVM_AVAILABLE:
        print("[warning] llvmlite not available, running in VM interpreter mode")
        VM(ast).run()
        return

    # LLVM codegen
    codegen = CrownLLVMCodegen()
    ir_code = codegen.codegen(ast)

    # Optimize
    opt = SupremeOptimizer(ir_code)
    optimized_ir = opt.optimize(args.O)

    if args.jit:
        print("[JIT] Running Crown Script with LLVM JIT...")
        engine = create_execution_engine()
        mod = binding.parse_assembly(optimized_ir)
        mod.verify()
        engine.add_module(mod)
        engine.finalize_object()
        engine.run_static_constructors()
        main_fn = engine.get_function_address("main")
        if main_fn:
            import ctypes
            ctypes.CFUNCTYPE(ctypes.c_int)(main_fn)()
    else:
        if not args.output: args.output = "a.out"
        with open(args.output + ".ll", "w") as f:
            f.write(optimized_ir)
        print(f"[AOT] LLVM IR written to {args.output}.ll")
        # In a full toolchain: call clang or llc to lower to native
        # subprocess.run(["clang", f"{args.output}.ll", "-o", args.output])
        print(f"[AOT] (stub) Binary ready at {args.output}")

def create_execution_engine():
    target = binding.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = binding.parse_assembly("")
    engine = binding.create_mcjit_compiler(backing_mod, target_machine)
    return engine


# Wire CLI if called directly
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[0].endswith("crownc"):
        crownc_cli()

# ================================================================
# Crown Script Ultimate LLVM + Optimizer + CLI
# ================================================================
try:
    from llvmlite import ir, binding
    LLVM_AVAILABLE = True
except ImportError:
    LLVM_AVAILABLE = False

# --------- Runtime Helper Declarations ---------
class CrownRuntimeHelpers:
    @staticmethod
    def declare(module):
        i64 = ir.IntType(64)
        i1 = ir.IntType(1)
        voidptr = ir.IntType(8).as_pointer()

        # malloc
        malloc_ty = ir.FunctionType(voidptr, [i64])
        malloc_fn = ir.Function(module, malloc_ty, name="malloc")

        # free
        free_ty = ir.FunctionType(ir.VoidType(), [voidptr])
        free_fn = ir.Function(module, free_ty, name="free")

        # printf
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr], var_arg=True)
        printf_fn = ir.Function(module, printf_ty, name="printf")

        return {
            "malloc": malloc_fn,
            "free": free_fn,
            "printf": printf_fn
        }

# --------- LLVM CODEGEN ---------
class CrownLLVMCodegen:
    def __init__(self):
        self.module = ir.Module(name="crown_module")
        self.builder = None
        self.funcs = {}
        self.runtime = CrownRuntimeHelpers.declare(self.module)

    def codegen(self, ast: Program):
        # declare main entry
        fn_ty = ir.FunctionType(ir.IntType(64), [])
        main_fn = ir.Function(self.module, fn_ty, name="main")
        block = main_fn.append_basic_block("entry")
        self.builder = ir.IRBuilder(block)
        for stmt in ast.stmts:
            self.codegen_stmt(stmt)
        if not self.builder.block.is_terminated:
            self.builder.ret(ir.Constant(ir.IntType(64), 0))
        return str(self.module)

    # --------- Statements ---------
    def codegen_stmt(self, s):
        if isinstance(s, Say):
            val = self.codegen_expr(s.expr)
            fmt = "%lld\n\0"
            c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt)),
                                bytearray(fmt.encode("utf8")))
            gfmt = ir.GlobalVariable(self.module, c_fmt.type, name="fmtstr")
            gfmt.linkage = "internal"
            gfmt.global_constant = True
            gfmt.initializer = c_fmt
            fmt_ptr = self.builder.bitcast(gfmt, ir.IntType(8).as_pointer())
            self.builder.call(self.runtime["printf"], [fmt_ptr, val])

        elif isinstance(s, If):
            cond = self.codegen_expr(s.cond)
            then_bb = self.builder.append_basic_block("then")
            else_bb = self.builder.append_basic_block("else")
            merge_bb = self.builder.append_basic_block("ifend")
            self.builder.cbranch(cond, then_bb, else_bb)

            self.builder.position_at_start(then_bb)
            for st in s.then:
                self.codegen_stmt(st)
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_bb)

            self.builder.position_at_start(else_bb)
            for st in s.other:
                self.codegen_stmt(st)
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_bb)

            self.builder.position_at_start(merge_bb)

        elif isinstance(s, Loop):
            start = self.codegen_expr(s.start)
            end = self.codegen_expr(s.end)
            step = self.codegen_expr(s.step)
            preheader = self.builder.block
            loop_bb = self.builder.append_basic_block("loop")
            after_bb = self.builder.append_basic_block("afterloop")
            self.builder.branch(loop_bb)

            self.builder.position_at_start(loop_bb)
            phi = self.builder.phi(ir.IntType(64), "i")
            phi.add_incoming(start, preheader)
            for st in s.body:
                self.codegen_stmt(st)
            nxt = self.builder.add(phi, step)
            cmp = self.builder.icmp_signed("<=", nxt, end)
            self.builder.cbranch(cmp, loop_bb, after_bb)
            phi.add_incoming(nxt, self.builder.block)
            self.builder.position_at_start(after_bb)

        elif isinstance(s, Return):
            val = self.codegen_expr(s.expr)
            self.builder.ret(val)

        elif isinstance(s, FuncDef):
            fn_ty = ir.FunctionType(ir.IntType(64), [ir.IntType(64)]*len(s.params))
            fn = ir.Function(self.module, fn_ty, name=s.name)
            self.funcs[s.name] = fn
            entry = fn.append_basic_block("entry")
            old_builder = self.builder
            self.builder = ir.IRBuilder(entry)
            for st in s.body:
                self.codegen_stmt(st)
            if not self.builder.block.is_terminated:
                self.builder.ret(ir.Constant(ir.IntType(64), 0))
            self.builder = old_builder

        elif isinstance(s, Call):
            args = [self.codegen_expr(a) for a in s.args]
            if s.name in self.funcs:
                self.builder.call(self.funcs[s.name], args)

        else:
            # fallback to VM-only semantics
            pass

    # --------- Expressions ---------
    def codegen_expr(self, e):
        if isinstance(e, Literal):
            if isinstance(e.value, int):
                return ir.Constant(ir.IntType(64), e.value)
            if isinstance(e.value, bool):
                return ir.Constant(ir.IntType(1), int(e.value))
        if isinstance(e, BinOp):
            l = self.codegen_expr(e.l); r = self.codegen_expr(e.r)
            if e.o == "+": return self.builder.add(l, r)
            if e.o == "-": return self.builder.sub(l, r)
            if e.o == "*": return self.builder.mul(l, r)
            if e.o == "/": return self.builder.sdiv(l, r)
            if e.o == "==": return self.builder.icmp_signed("==", l, r)
            if e.o == "!=": return self.builder.icmp_signed("!=", l, r)
            if e.o == "<": return self.builder.icmp_signed("<", l, r)
            if e.o == ">": return self.builder.icmp_signed(">", l, r)
            if e.o == "<=": return self.builder.icmp_signed("<=", l, r)
            if e.o == ">=": return self.builder.icmp_signed(">=", l, r)
        return ir.Constant(ir.IntType(64), 0)


# --------- Optimizer ---------
class CrownOptimizer:
    def __init__(self, ir_code: str):
        self.ir_code = ir_code

    def optimize(self, level=3):
        if not LLVM_AVAILABLE: return self.ir_code
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()
        llvm_mod = binding.parse_assembly(self.ir_code)
        llvm_mod.verify()
        pmb = binding.PassManagerBuilder()
        pmb.opt_level = level
        pmb.loop_vectorize = True
        pmb.slp_vectorize = True
        pm = binding.ModulePassManager()
        pmb.populate(pm)
        pm.run(llvm_mod)
        return str(llvm_mod)


# --------- Execution Engine ---------
def create_execution_engine():
    target = binding.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = binding.parse_assembly("")
    engine = binding.create_mcjit_compiler(backing_mod, target_machine)
    return engine


# --------- Compiler CLI ---------
def crownc_cli():
    import argparse, os, subprocess
    p = argparse.ArgumentParser(prog="crownc", description="Crown Script Supreme Compiler")
    p.add_argument("file", help="Source .crown file")
    p.add_argument("-o", "--output", help="Output binary")
    p.add_argument("--jit", action="store_true", help="Run with JIT")
    p.add_argument("-O", type=int, default=2, help="Optimization level")
    args = p.parse_args()

    with open(args.file, encoding="utf-8") as f: src = f.read()
    tokens = tokenize(src)
    ast = Parser(tokens).parse()

    if not LLVM_AVAILABLE:
        print("[warning] LLVM unavailable → falling back to VM")
        VM(ast).run()
        return

    codegen = CrownLLVMCodegen()
    ir_code = codegen.codegen(ast)
    opt = CrownOptimizer(ir_code)
    opt_ir = opt.optimize(args.O)

    if args.jit:
        engine = create_execution_engine()
        mod = binding.parse_assembly(opt_ir)
        engine.add_module(mod)
        engine.finalize_object()
        main_ptr = engine.get_function_address("main")
        import ctypes
        fnty = ctypes.CFUNCTYPE(ctypes.c_long)
        fn = fnty(main_ptr)
        fn()
    else:
        if not args.output: args.output = "a.out"
        ir_file = args.output + ".ll"
        with open(ir_file, "w") as f: f.write(opt_ir)
        try:
            subprocess.run(["clang", ir_file, "-o", args.output], check=True)
            print(f"[AOT] Built {args.output}")
        except Exception:
            print(f"[AOT] IR saved to {ir_file}, external clang needed.")

# Hook CLI
if __name__ == "__main__":
    if sys.argv[0].endswith("crownc"):
        crownc_cli()

        # ================================================================
        # Crown Script Supreme LLVM Codegen + Compiler CLI
        # ================================================================

# ================================================================
# Crown Script Runtime Helpers for Arrays, Maps, and Match
# ================================================================

class CrownRuntimeLowering:
    def __init__(self, module: ir.Module):
        self.module = module
        self._declare_helpers()

    def _declare_helpers(self):
        i64 = ir.IntType(64)
        voidptr = ir.IntType(8).as_pointer()

        # Arrays
        self.array_new = ir.Function(self.module,
            ir.FunctionType(voidptr, [i64]), name="crown_array_new")
        self.array_push = ir.Function(self.module,
            ir.FunctionType(ir.VoidType(), [voidptr, i64]), name="crown_array_push")
        self.array_get = ir.Function(self.module,
            ir.FunctionType(i64, [voidptr, i64]), name="crown_array_get")
        self.array_set = ir.Function(self.module,
            ir.FunctionType(ir.VoidType(), [voidptr, i64, i64]), name="crown_array_set")
        self.array_len = ir.Function(self.module,
            ir.FunctionType(i64, [voidptr]), name="crown_array_len")

        # Maps
        self.map_new = ir.Function(self.module,
            ir.FunctionType(voidptr, []), name="crown_map_new")
        self.map_set = ir.Function(self.module,
            ir.FunctionType(ir.VoidType(), [voidptr, i64, i64]), name="crown_map_set")
        self.map_get = ir.Function(self.module,
            ir.FunctionType(i64, [voidptr, i64]), name="crown_map_get")
        self.map_has = ir.Function(self.module,
            ir.FunctionType(i64, [voidptr, i64]), name="crown_map_has")
        self.map_keys = ir.Function(self.module,
            ir.FunctionType(voidptr, [voidptr]), name="crown_map_keys")

        # Match dispatch (runtime tag matcher)
        self.match_tag = ir.Function(self.module,
            ir.FunctionType(i64, [i64]), name="crown_match_tag")


# Extend LLVM Codegen
class CrownLLVMCodegen(CrownLLVMCodegen):
    def __init__(self):
        super().__init__()
        self.helpers = CrownRuntimeLowering(self.module)

    def codegen_expr(self, e):
        # Literals (unchanged)
        if isinstance(e, Literal):
            if isinstance(e.value, int):
                return ir.Constant(ir.IntType(64), e.value)
            if isinstance(e.value, bool):
                return ir.Constant(ir.IntType(1), int(e.value))

        # Array literal
        if isinstance(e, ArrayLit):
            arr = self.builder.call(self.helpers.array_new,
                                    [ir.Constant(ir.IntType(64), len(e.items))])
            for item in e.items:
                val = self.codegen_expr(item)
                self.builder.call(self.helpers.array_push, [arr, val])
            return arr

        # Map literal
        if isinstance(e, MapLit):
            m = self.builder.call(self.helpers.map_new, [])
            for k, v in e.pairs:
                key = self.codegen_expr(k)
                val = self.codegen_expr(v)
                self.builder.call(self.helpers.map_set, [m, key, val])
            return m

        # Indexing
        if isinstance(e, Index):
            base = self.codegen_expr(e.base)
            idx = self.codegen_expr(e.idx)
            if isinstance(e.base, ArrayLit) or isinstance(e.base, Var):
                return self.builder.call(self.helpers.array_get, [base, idx])
            else:
                return self.builder.call(self.helpers.map_get, [base, idx])

        # Binary operations (unchanged from above)
        if isinstance(e, BinOp):
            l = self.codegen_expr(e.l); r = self.codegen_expr(e.r)
            if e.o == "+": return self.builder.add(l, r)
            if e.o == "-": return self.builder.sub(l, r)
            if e.o == "*": return self.builder.mul(l, r)
            if e.o == "/": return self.builder.sdiv(l, r)
            if e.o == "==": return self.builder.icmp_signed("==", l, r)
            if e.o == "!=": return self.builder.icmp_signed("!=", l, r)
            if e.o == "<": return self.builder.icmp_signed("<", l, r)
            if e.o == ">": return self.builder.icmp_signed(">", l, r)
            if e.o == "<=": return self.builder.icmp_signed("<=", l, r)
            if e.o == ">=": return self.builder.icmp_signed(">=", l, r)
        return ir.Constant(ir.IntType(64), 0)

    def codegen_stmt(self, s):
        # Lower foreach into runtime calls
        if isinstance(s, Foreach):
            arr = self.codegen_expr(s.iterable)
            length = self.builder.call(self.helpers.array_len, [arr])
            zero = ir.Constant(ir.IntType(64), 0)
            idx_alloca = self.builder.alloca(ir.IntType(64))
            self.builder.store(zero, idx_alloca)

            loop_bb = self.builder.append_basic_block("foreach_loop")
            after_bb = self.builder.append_basic_block("foreach_after")

            self.builder.branch(loop_bb)
            self.builder.position_at_start(loop_bb)

            idx_val = self.builder.load(idx_alloca)
            cmp = self.builder.icmp_signed("<", idx_val, length)
            with self.builder.if_then(cmp):
                elem = self.builder.call(self.helpers.array_get, [arr, idx_val])
                for st in s.body:
                    self.codegen_stmt(st)
                nxt = self.builder.add(idx_val, ir.Constant(ir.IntType(64), 1))
                self.builder.store(nxt, idx_alloca)
                self.builder.branch(loop_bb)

            self.builder.branch(after_bb)
            self.builder.position_at_start(after_bb)

        # Lower match into runtime tag dispatch
        elif isinstance(s, Match):
            val = self.codegen_expr(s.expr)
            tag = self.builder.call(self.helpers.match_tag, [val])
            done_bb = self.builder.append_basic_block("match_done")
            for i, (pat, body) in enumerate(s.cases):
                case_bb = self.builder.append_basic_block(f"case_{i}")
                self.builder.cbranch(
                    self.builder.icmp_signed("==", tag, ir.Constant(ir.IntType(64), i)),
                    case_bb, done_bb
                )
                self.builder.position_at_start(case_bb)
                for st in body:
                    self.codegen_stmt(st)
                self.builder.branch(done_bb)
                self.builder.position_at_start(done_bb)
        else:
            super().codegen_stmt(s)

# ============================================================
# Crown Script Runtime C Helpers (strings + arrays + maps)
# ============================================================
CROWN_RUNTIME_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef enum {
    CROWN_VAL_INT,
    CROWN_VAL_STR,
    CROWN_VAL_ARR,
    CROWN_VAL_MAP
} CrownValKind;

typedef struct CrownValue CrownValue;

typedef struct {
    CrownValue* data;
    int64_t size;
    int64_t capacity;
} CrownArray;

typedef struct {
    char* key;
    CrownValue value;
} CrownMapEntry;

typedef struct {
    CrownMapEntry* entries;
    int64_t size;
    int64_t capacity;
} CrownMap;

struct CrownValue {
    CrownValKind kind;
    union {
        int64_t i;
        char* s;
        CrownArray* arr;
        CrownMap* map;
    } as;
};

// ---- Constructors ----
CrownValue crown_make_int(int64_t i){
    CrownValue v; v.kind=CROWN_VAL_INT; v.as.i=i; return v;
}
CrownValue crown_make_str(const char* s){
    CrownValue v; v.kind=CROWN_VAL_STR; v.as.s=strdup(s); return v;
}
CrownValue crown_make_array(){
    CrownArray* arr=malloc(sizeof(CrownArray));
    arr->size=0; arr->capacity=4;
    arr->data=malloc(sizeof(CrownValue)*arr->capacity);
    CrownValue v; v.kind=CROWN_VAL_ARR; v.as.arr=arr; return v;
}
CrownValue crown_make_map(){
    CrownMap* m=malloc(sizeof(CrownMap));
    m->size=0; m->capacity=4;
    m->entries=malloc(sizeof(CrownMapEntry)*m->capacity);
    CrownValue v; v.kind=CROWN_VAL_MAP; v.as.map=m; return v;
}

// ---- Array Ops ----
void crown_array_push(CrownValue arrv, CrownValue val){
    if(arrv.kind!=CROWN_VAL_ARR) return;
    CrownArray* arr=arrv.as.arr;
    if(arr->size>=arr->capacity){
        arr->capacity*=2;
        arr->data=realloc(arr->data,sizeof(CrownValue)*arr->capacity);
    }
    arr->data[arr->size++]=val;
}
CrownValue crown_array_get(CrownValue arrv,int64_t idx){
    if(arrv.kind!=CROWN_VAL_ARR) return crown_make_int(0);
    CrownArray* arr=arrv.as.arr;
    if(idx<0||idx>=arr->size) return crown_make_int(0);
    return arr->data[idx];
}
int64_t crown_array_len(CrownValue arrv){
    return (arrv.kind==CROWN_VAL_ARR)? arrv.as.arr->size : 0;
}

// ---- Map Ops ----
void crown_map_set(CrownValue mapv,const char* key,CrownValue val){
    if(mapv.kind!=CROWN_VAL_MAP) return;
    CrownMap* m=mapv.as.map;
    for(int64_t i=0;i<m->size;i++){
        if(strcmp(m->entries[i].key,key)==0){
            m->entries[i].value=val; return;
        }
    }
    if(m->size>=m->capacity){
        m->capacity*=2;
        m->entries=realloc(m->entries,sizeof(CrownMapEntry)*m->capacity);
    }
    m->entries[m->size].key=strdup(key);
    m->entries[m->size].value=val;
    m->size++;
}
CrownValue crown_map_get(CrownValue mapv,const char* key){
    if(mapv.kind!=CROWN_VAL_MAP) return crown_make_int(0);
    CrownMap* m=mapv.as.map;
    for(int64_t i=0;i<m->size;i++){
        if(strcmp(m->entries[i].key,key)==0) return m->entries[i].value;
    }
    return crown_make_int(0);
}
int64_t crown_map_has(CrownValue mapv,const char* key){
    if(mapv.kind!=CROWN_VAL_MAP) return 0;
    CrownMap* m=mapv.as.map;
    for(int64_t i=0;i<m->size;i++){
        if(strcmp(m->entries[i].key,key)==0) return 1;
    }
    return 0;
}

// ---- Debug Print ----
void crown_debug_print(CrownValue v);
void crown_debug_print_array(CrownArray* arr){
    printf("[");
    for(int64_t i=0;i<arr->size;i++){
        crown_debug_print(arr->data[i]);
        if(i+1<arr->size) printf(", ");
    }
    printf("]");
}
void crown_debug_print_map(CrownMap* m){
    printf("{");
    for(int64_t i=0;i<m->size;i++){
        printf("\"%s\": ",m->entries[i].key);
        crown_debug_print(m->entries[i].value);
        if(i+1<m->size) printf(", ");
    }
    printf("}");
}
void crown_debug_print(CrownValue v){
    switch(v.kind){
        case CROWN_VAL_INT: printf("%lld",(long long)v.as.i); break;
        case CROWN_VAL_STR: printf("\"%s\"",v.as.s); break;
        case CROWN_VAL_ARR: crown_debug_print_array(v.as.arr); break;
        case CROWN_VAL_MAP: crown_debug_print_map(v.as.map); break;
    }
}
"""
# ============================================================
# LLVM IR Declarations for Runtime Helpers
# ============================================================

def crown_codegen_runtime_decls(module):
    """Emit LLVM declarations for runtime helpers."""
    int64 = ir.IntType(64)
    i8p = ir.IntType(8).as_pointer()
    val_ty = ir.LiteralStructType([ir.IntType(32), ir.IntType(64)])  # simplified CrownValue

    # Example: declare CrownValue @crown_make_int(i64)
    ir.Function(module, ir.FunctionType(val_ty, [int64]), name="crown_make_int")
    ir.Function(module, ir.FunctionType(val_ty, [i8p]), name="crown_make_str")
    ir.Function(module, ir.FunctionType(val_ty, []), name="crown_make_array")
    ir.Function(module, ir.FunctionType(val_ty, []), name="crown_make_map")
    ir.Function(module, ir.FunctionType(ir.VoidType(), [val_ty, val_ty]), name="crown_array_push")
    ir.Function(module, ir.FunctionType(val_ty, [val_ty, int64]), name="crown_array_get")
    ir.Function(module, ir.FunctionType(int64, [val_ty]), name="crown_array_len")
    ir.Function(module, ir.FunctionType(ir.VoidType(), [val_ty, i8p, val_ty]), name="crown_map_set")
    ir.Function(module, ir.FunctionType(val_ty, [val_ty, i8p]), name="crown_map_get")
    ir.Function(module, ir.FunctionType(int64, [val_ty, i8p]), name="crown_map_has")
    ir.Function(module, ir.FunctionType(ir.VoidType(), [val_ty]), name="crown_debug_print")

with open("crown_runtime.c","w") as f:
    f.write(CROWN_RUNTIME_C)

# ============================================================
# EXTENDED CODEGEN: ArrayLit, MapLit, Match Lowering
# ============================================================

class CrownLLVMCodegen:
    def __init__(self, module: ir.Module, builder: ir.IRBuilder, symtab: dict):
        self.module = module
        self.builder = builder
        self.symtab = symtab
        crown_codegen_runtime_decls(self.module)  # ensure helpers declared

    # ---- Expressions ----
    def codegen_expr(self, node):
        if isinstance(node, Literal):
            if isinstance(node.value, bool) or isinstance(node.value, int):
                fn = self.module.globals.get("crown_make_int")
                return self.builder.call(fn, [ir.Constant(ir.IntType(64), int(node.value))])
            if isinstance(node.value, str):
                fn = self.module.globals.get("crown_make_str")
                s = self.builder.global_string_ptr(node.value)
                return self.builder.call(fn, [s])
        elif isinstance(node, ArrayLit):
            # Create array
            make_arr = self.module.globals.get("crown_make_array")
            arr_val = self.builder.call(make_arr, [])
            push_fn = self.module.globals.get("crown_array_push")
            for item in node.items:
                val = self.codegen_expr(item)
                self.builder.call(push_fn, [arr_val, val])
            return arr_val
        elif isinstance(node, MapLit):
            make_map = self.module.globals.get("crown_make_map")
            map_val = self.builder.call(make_map, [])
            set_fn = self.module.globals.get("crown_map_set")
            for k, v in node.pairs:
                key_val = self.codegen_expr(k)
                val_val = self.codegen_expr(v)
                # convert key to string (assume string for simplicity)
                if isinstance(k, Literal) and isinstance(k.value, str):
                    key_ptr = self.builder.global_string_ptr(k.value)
                else:
                    raise RuntimeError("Only string keys supported in map literals (for now)")
                self.builder.call(set_fn, [map_val, key_ptr, val_val])
            return map_val
        elif isinstance(node, Match):
            # Desugar Match into if/else chains
            val = self.codegen_expr(node.expr)
            result = None
            end_bb = self.builder.append_basic_block("match_end")
            for pat, body in node.cases:
                # For now only handle ConstPat
                if isinstance(pat, ConstPat):
                    cmp_val = self.codegen_expr(pat.expr)
                    cmp = self.builder.icmp_signed("==", 
                        self.builder.extract_value(val, 1),
                        self.builder.extract_value(cmp_val, 1))
                    then_bb = self.builder.append_basic_block("match_then")
                    else_bb = self.builder.append_basic_block("match_else")
                    self.builder.cbranch(cmp, then_bb, else_bb)
                    self.builder.position_at_end(then_bb)
                    for st in body:
                        self.codegen_stmt(st)
                    self.builder.branch(end_bb)
                    self.builder.position_at_end(else_bb)
            if node.default:
                for st in node.default:
                    self.codegen_stmt(st)
                self.builder.branch(end_bb)
            self.builder.position_at_end(end_bb)
            return result
        else:
            raise NotImplementedError(f"Codegen not implemented for {type(node).__name__}")

    # ---- Statements ----
    def codegen_stmt(self, node):
        if isinstance(node, Say):
            val = self.codegen_expr(node.expr)
            dbg_fn = self.module.globals.get("crown_debug_print")
            self.builder.call(dbg_fn, [val])
        elif isinstance(node, Make):
            v = self.codegen_expr(node.expr)
            self.symtab[node.name] = v
        elif isinstance(node, Assign):
            v = self.codegen_expr(node.expr)
            self.symtab[node.name] = v
        elif isinstance(node, Return):
            v = self.codegen_expr(node.expr)
            self.builder.ret(v)
        elif isinstance(node, If):
            cond_val = self.codegen_expr(node.cond)
            cmp = self.builder.icmp_signed("!=", self.builder.extract_value(cond_val, 1),
                                           ir.Constant(ir.IntType(64), 0))
            then_bb = self.builder.append_basic_block("if_then")
            else_bb = self.builder.append_basic_block("if_else")
            end_bb = self.builder.append_basic_block("if_end")
            self.builder.cbranch(cmp, then_bb, else_bb)
            # then
            self.builder.position_at_end(then_bb)
            for st in node.then:
                self.codegen_stmt(st)
            self.builder.branch(end_bb)
            # else
            self.builder.position_at_end(else_bb)
            for st in node.other:
                self.codegen_stmt(st)
            self.builder.branch(end_bb)
            self.builder.position_at_end(end_bb)
        else:
            raise NotImplementedError(f"Stmt codegen not implemented for {type(node).__name__}")

# ============================================================
# EXTENDED CODEGEN: Foreach, While, Loop Lowering
# ============================================================

def crown_codegen_runtime_iter_helpers(module: ir.Module):
    """
    Declare iteration helpers for arrays/maps so LLVM can lower Foreach.
    """
    i8ptr = ir.IntType(8).as_pointer()
    crown_any = ir.LiteralStructType([ir.IntType(32), ir.IntType(64), i8ptr]).as_pointer()

    # Iterator create
    if "crown_iter_begin" not in module.globals:
        ir.Function(module, ir.FunctionType(crown_any, [crown_any]), name="crown_iter_begin")
    # Iterator has_next
    if "crown_iter_has_next" not in module.globals:
        ir.Function(module, ir.FunctionType(ir.IntType(1), [crown_any]), name="crown_iter_has_next")
    # Iterator next
    if "crown_iter_next" not in module.globals:
        ir.Function(module, ir.FunctionType(crown_any, [crown_any]), name="crown_iter_next")


class CrownLLVMCodegen(CrownLLVMCodegen):  # extend existing
    def codegen_stmt(self, node):
        # Reuse existing logic
        try:
            return super().codegen_stmt(node)
        except NotImplementedError:
            pass

        # ---------- LOOP ----------
        if isinstance(node, Loop):
            # Classic for i = start; i <= end; i += step
            start_val = self.codegen_expr(node.start)
            end_val   = self.codegen_expr(node.end)
            step_val  = self.codegen_expr(node.step)

            # Allocate i
            i_ptr = self.builder.alloca(ir.IntType(64))
            self.builder.store(self.builder.extract_value(start_val, 1), i_ptr)

            cond_bb = self.builder.append_basic_block("loop_cond")
            body_bb = self.builder.append_basic_block("loop_body")
            end_bb  = self.builder.append_basic_block("loop_end")

            self.builder.branch(cond_bb)
            self.builder.position_at_end(cond_bb)

            i_val = self.builder.load(i_ptr)
            cmp = self.builder.icmp_signed("<=", i_val, self.builder.extract_value(end_val, 1))
            self.builder.cbranch(cmp, body_bb, end_bb)

            self.builder.position_at_end(body_bb)
            for st in node.body:
                self.codegen_stmt(st)
            i_next = self.builder.add(i_val, self.builder.extract_value(step_val, 1))
            self.builder.store(i_next, i_ptr)
            self.builder.branch(cond_bb)

            self.builder.position_at_end(end_bb)
            return None

        # ---------- WHILE ----------
        if isinstance(node, While):
            cond_bb = self.builder.append_basic_block("while_cond")
            body_bb = self.builder.append_basic_block("while_body")
            end_bb  = self.builder.append_basic_block("while_end")

            self.builder.branch(cond_bb)
            self.builder.position_at_end(cond_bb)
            cond_val = self.codegen_expr(node.cond)
            cmp = self.builder.icmp_signed("!=", self.builder.extract_value(cond_val, 1),
                                           ir.Constant(ir.IntType(64), 0))
            self.builder.cbranch(cmp, body_bb, end_bb)

            self.builder.position_at_end(body_bb)
            for st in node.body:
                self.codegen_stmt(st)
            self.builder.branch(cond_bb)

            self.builder.position_at_end(end_bb)
            return None

        # ---------- FOREACH ----------
        if isinstance(node, Foreach):
            crown_codegen_runtime_iter_helpers(self.module)

            iter_begin = self.module.globals["crown_iter_begin"]
            iter_has_next = self.module.globals["crown_iter_has_next"]
            iter_next = self.module.globals["crown_iter_next"]

            it_val = self.codegen_expr(node.iterable)
            iter_ptr = self.builder.call(iter_begin, [it_val])

            cond_bb = self.builder.append_basic_block("foreach_cond")
            body_bb = self.builder.append_basic_block("foreach_body")
            end_bb  = self.builder.append_basic_block("foreach_end")

            self.builder.branch(cond_bb)
            self.builder.position_at_end(cond_bb)
            has_next = self.builder.call(iter_has_next, [iter_ptr])
            self.builder.cbranch(has_next, body_bb, end_bb)

            self.builder.position_at_end(body_bb)
            next_val = self.builder.call(iter_next, [iter_ptr])
            self.symtab[node.var] = next_val
            for st in node.body:
                self.codegen_stmt(st)
            self.builder.branch(cond_bb)

            self.builder.position_at_end(end_bb)
            return None

        raise NotImplementedError(f"Stmt codegen not implemented for {type(node).__name__}")

# ============================================================
# Crown Runtime Helpers (inline C implementation)
# ============================================================

CROWN_RUNTIME_C = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Crown "Any" type: tagged union
typedef struct {
    int32_t tag;    // 0=int, 1=double, 2=string, 3=array, 4=map
    int64_t i64;    // integer or length
    void*   ptr;    // pointer payload
} crown_any;

// ---------- Array ----------
typedef struct {
    size_t len;
    size_t cap;
    crown_any* data;
} crown_array;

crown_any crown_make_array(int64_t cap) {
    crown_array* arr = (crown_array*)malloc(sizeof(crown_array));
    arr->len = 0;
    arr->cap = cap > 0 ? cap : 4;
    arr->data = (crown_any*)calloc(arr->cap, sizeof(crown_any));
    crown_any out = {3, (int64_t)arr->len, arr};
    return out;
}

void crown_array_push(crown_any arr_any, crown_any val) {
    crown_array* arr = (crown_array*)arr_any.ptr;
    if (arr->len >= arr->cap) {
        arr->cap *= 2;
        arr->data = (crown_any*)realloc(arr->data, arr->cap * sizeof(crown_any));
    }
    arr->data[arr->len++] = val;
    arr_any.i64 = arr->len;
}

crown_any crown_array_get(crown_any arr_any, int64_t idx) {
    crown_array* arr = (crown_array*)arr_any.ptr;
    if (idx < 0 || (size_t)idx >= arr->len) {
        crown_any none = {0,0,NULL};
        return none;
    }
    return arr->data[idx];
}

// ---------- Map (simple string→any hash map) ----------
typedef struct crown_map_entry {
    char* key;
    crown_any value;
    struct crown_map_entry* next;
} crown_map_entry;

typedef struct {
    size_t bucket_count;
    crown_map_entry** buckets;
} crown_map;

static size_t crown_hash(const char* s) {
    size_t h=1469598103934665603ULL;
    while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ULL; }
    return h;
}

crown_any crown_make_map() {
    crown_map* m = (crown_map*)malloc(sizeof(crown_map));
    m->bucket_count = 64;
    m->buckets = (crown_map_entry**)calloc(m->bucket_count,sizeof(crown_map_entry*));
    crown_any out = {4,0,m};
    return out;
}

void crown_map_set(crown_any map_any, const char* key, crown_any val) {
    crown_map* m = (crown_map*)map_any.ptr;
    size_t h = crown_hash(key) % m->bucket_count;
    crown_map_entry* e = m->buckets[h];
    while (e) {
        if (strcmp(e->key,key)==0) { e->value = val; return; }
        e = e->next;
    }
    e = (crown_map_entry*)malloc(sizeof(crown_map_entry));
    e->key = strdup(key);
    e->value = val;
    e->next = m->buckets[h];
    m->buckets[h] = e;
}

crown_any crown_map_get(crown_any map_any, const char* key) {
    crown_map* m = (crown_map*)map_any.ptr;
    size_t h = crown_hash(key) % m->bucket_count;
    crown_map_entry* e = m->buckets[h];
    while (e) {
        if (strcmp(e->key,key)==0) return e->value;
        e = e->next;
    }
    crown_any none = {0,0,NULL};
    return none;
}

// ---------- Iterators ----------
typedef struct {
    int kind; // 0=array, 1=map
    void* source;
    size_t index;
    crown_map_entry* cur;
} crown_iter;

crown_any crown_iter_begin(crown_any container) {
    crown_iter* it = (crown_iter*)malloc(sizeof(crown_iter));
    if (container.tag == 3) {
        it->kind=0;
        it->source=container.ptr;
        it->index=0;
        it->cur=NULL;
    } else if (container.tag == 4) {
        it->kind=1;
        it->source=container.ptr;
        it->index=0;
        crown_map* m=(crown_map*)container.ptr;
        it->cur=NULL;
        for (size_t i=0;i<m->bucket_count;i++){
            if (m->buckets[i]) { it->cur=m->buckets[i]; break; }
        }
    }
    crown_any out = {9,0,it};
    return out;
}

int crown_iter_has_next(crown_any iter_any) {
    crown_iter* it = (crown_iter*)iter_any.ptr;
    if (it->kind==0) {
        crown_array* arr=(crown_array*)it->source;
        return it->index < arr->len;
    } else {
        return it->cur!=NULL;
    }
}

crown_any crown_iter_next(crown_any iter_any) {
    crown_iter* it = (crown_iter*)iter_any.ptr;
    if (it->kind==0) {
        crown_array* arr=(crown_array*)it->source;
        return arr->data[it->index++];
    } else {
        crown_map_entry* e=it->cur;
        if (!e) { crown_any none={0,0,NULL}; return none; }
        it->cur = e->next;
        return e->value;
    }
}
"""

# Write helpers to a .c file when compiling
def write_crown_runtime(path="crown_runtime.c"):
    with open(path,"w",encoding="utf-8") as f:
        f.write(CROWN_RUNTIME_C)

# ============================================================
# Build Integration: Auto-generate setup.py and CMake glue
# ============================================================

import os, subprocess

SETUP_PY_TEMPLATE = r"""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys

class BuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            ext.extra_compile_args = ["-O3", "-march=native"]
        build_ext.build_extensions(self)

setup(
    name="crown_runtime",
    ext_modules=[
        Extension("crown_runtime", sources=["crown_runtime.c"]),
    ],
    cmdclass={"build_ext": BuildExt}
)
"""

CMAKELISTS_TEMPLATE = r"""
cmake_minimum_required(VERSION 3.12)
project(crown_runtime C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")

add_library(crown_runtime STATIC crown_runtime.c)
"""

def write_build_glue():
    """Emit setup.py and CMakeLists.txt so you can link runtime helpers easily."""
    with open("setup.py", "w", encoding="utf-8") as f:
        f.write(SETUP_PY_TEMPLATE.strip() + "\n")
    with open("CMakeLists.txt", "w", encoding="utf-8") as f:
        f.write(CMAKELISTS_TEMPLATE.strip() + "\n")
    print("[crownc] Generated setup.py and CMakeLists.txt")

# ============================================================
# Crown Compiler Driver (extended)
# ============================================================

def crownc(argv):
    """
    crownc driver:
    Usage:
      crownc build file.crown -o file.exe
      crownc jit file.crown
    """
    if len(argv) < 3:
        print("Usage: crownc <build|jit|run> file.crown [-o out.exe]")
        return 1

    mode, srcfile = argv[1], argv[2]
    out = "a.out"
    if "-o" in argv:
        out = argv[argv.index("-o")+1]

    # Step 1: read and parse
    with open(srcfile, encoding="utf-8") as f:
        src = f.read()
    tokens = tokenize(src)
    ast = Parser(tokens).parse()

    # Step 2: IR codegen
    mod, engine = codegen_module(ast, name=out)

    if mode == "jit":
        print(f"[crownc] JIT running {srcfile}")
        engine.run_function(engine.get_function_address("main"))
        return 0

    if mode == "build":
        print(f"[crownc] AOT compiling {srcfile} -> {out}")
        # Write LLVM IR
        ir_path = out + ".ll"
        with open(ir_path, "w", encoding="utf-8") as f:
            f.write(str(mod))
        print(f"[crownc] Wrote IR to {ir_path}")

        # Emit runtime C
        write_crown_runtime()

        # Emit setup.py + CMake glue
        write_build_glue()

        # Try clang build if available
        try:
            subprocess.check_call([
                "clang", "-O3", "-march=native",
                ir_path, "crown_runtime.c", "-o", out
            ])
            print(f"[crownc] Built native binary: {out}")
        except Exception as e:
            print(f"[crownc] clang build failed: {e}")
            print("[crownc] Use CMake or `python setup.py build_ext --inplace` to compile runtime.")
        return 0

    if mode == "run":
        VM(ast).run()
        return 0

    print(f"[crownc] Unknown mode: {mode}")
    return 1

# Wire into __main__
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("build","jit","run"):
    sys.exit(crownc(sys.argv))

# ============================================================
# Universal Cross-Compiler Driver (Clang/GCC/MSVC autodetect)
# ============================================================

import platform, shutil

def detect_compiler():
    """Return (compiler_cmd, type) for the host system."""
    system = platform.system().lower()
    # Prefer clang if available
    if shutil.which("clang"):
        return "clang", "clang"
    if shutil.which("gcc"):
        return "gcc", "gcc"
    if system == "windows":
        # Try MSVC cl.exe
        if shutil.which("cl"):
            return "cl", "msvc"
    return None, None

def build_native(ir_path, out, runtime_c="crown_runtime.c"):
    """Compile LLVM IR + runtime helpers into native binary."""
    compiler, ctype = detect_compiler()
    if not compiler:
        print("[crownc] ERROR: No supported compiler (clang/gcc/msvc) found")
        return False

    system = platform.system().lower()
    print(f"[crownc] Using {compiler} for {system} build")

    try:
        if ctype in ("clang","gcc"):
            # Linux/Unix/macOS or mingw
            cmd = [
                compiler, "-O3", "-march=native",
                ir_path, runtime_c, "-o", out
            ]
            if system == "darwin":
                # macOS special flags
                cmd += ["-framework", "CoreFoundation"]
            subprocess.check_call(cmd)
            print(f"[crownc] Built binary: {out}")
            return True

        elif ctype == "msvc":
            # Convert IR to object file first (llc required)
            llc = shutil.which("llc")
            if not llc:
                print("[crownc] ERROR: MSVC requires llc (LLVM static compiler) in PATH")
                return False
            obj = out + ".obj"
            subprocess.check_call([llc, "-filetype=obj", ir_path, "-o", obj])
            subprocess.check_call([
                "cl", "/Fe:"+out, obj, runtime_c, "/O2", "/MT"
            ])
            print(f"[crownc] Built Windows .exe with MSVC: {out}")
            return True
    except Exception as e:
        print(f"[crownc] Build failed: {e}")
        return False

    return False

# ============================================================
# Crown Compiler Driver (patched with cross-compile)
# ============================================================

def crownc(argv):
    """
    crownc driver:
    Usage:
      crownc run file.crown
      crownc jit file.crown
      crownc build file.crown -o file.exe
    """
    if len(argv) < 3:
        print("Usage: crownc <run|jit|build> file.crown [-o out]")
        return 1

    mode, srcfile = argv[1], argv[2]
    out = "a.out"
    if "-o" in argv:
        out = argv[argv.index("-o")+1]

    with open(srcfile, encoding="utf-8") as f:
        src = f.read()
    tokens = tokenize(src)
    ast = Parser(tokens).parse()

    # LLVM IR
    mod, engine = codegen_module(ast, name=out)

    if mode == "run":
        VM(ast).run()
        return 0
    if mode == "jit":
        print(f"[crownc] JIT running {srcfile}")
        engine.run_function(engine.get_function_address("main"))
        return 0
    if mode == "build":
        ir_path = out + ".ll"
        with open(ir_path, "w", encoding="utf-8") as f:
            f.write(str(mod))
        print(f"[crownc] Wrote LLVM IR to {ir_path}")

        # Emit runtime
        write_crown_runtime()
        write_build_glue()

        # Try native build
        if not build_native(ir_path, out):
            print("[crownc] ERROR: Native build failed, but IR + runtime emitted.")
        return 0

    print(f"[crownc] Unknown mode: {mode}")
    return 1

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("build","jit","run"):
    sys.exit(crownc(sys.argv))

# ============================================================
# Cross-Target Support (universal binaries from .crown source)
# ============================================================

def parse_target_flag(argv):
    """Parse --target=<triple> from argv, returns (triple, sysroot)."""
    triple = None
    sysroot = None
    for arg in argv:
        if arg.startswith("--target="):
            triple = arg.split("=",1)[1]
        if arg.startswith("--sysroot="):
            sysroot = arg.split("=",1)[1]
    return triple, sysroot

def build_cross(ir_path, out, triple, sysroot=None, runtime_c="crown_runtime.c"):
    """Cross-compile LLVM IR + runtime helpers into target binary."""
    compiler, ctype = detect_compiler()
    if not compiler:
        print("[crownc] ERROR: No compiler found for cross build")
        return False

    # Clang is preferred for cross-target (it speaks -target natively)
    if ctype not in ("clang","gcc"):
        print("[crownc] ERROR: Only clang/gcc support cross-target builds right now")
        return False

    cmd = [compiler, "-O3", "-march=native", ir_path, runtime_c, "-o", out]

    if triple:
        cmd = [compiler, f"--target={triple}", "-O3", ir_path, runtime_c, "-o", out]
        if sysroot:
            cmd += [f"--sysroot={sysroot}"]

    try:
        subprocess.check_call(cmd)
        print(f"[crownc] Cross-built binary for {triple}: {out}")
        return True
    except Exception as e:
        print(f"[crownc] Cross-build failed: {e}")
        return False

# ============================================================
# Extended crownc driver with cross-target builds
# ============================================================

def crownc(argv):
    """
    crownc driver:
    Usage:
      crownc run file.crown
      crownc jit file.crown
      crownc build file.crown -o file.exe [--target=triple] [--sysroot=/path]
    """
    if len(argv) < 3:
        print("Usage: crownc <run|jit|build> file.crown [-o out] [--target=triple] [--sysroot=/path]")
        return 1

    mode, srcfile = argv[1], argv[2]
    out = "a.out"
    if "-o" in argv:
        out = argv[argv.index("-o")+1]

    triple, sysroot = parse_target_flag(argv)

    with open(srcfile, encoding="utf-8") as f:
        src = f.read()
    tokens = tokenize(src)
    ast = Parser(tokens).parse()

    # LLVM IR
    mod, engine = codegen_module(ast, name=out)

    if mode == "run":
        VM(ast).run()
        return 0
    if mode == "jit":
        print(f"[crownc] JIT running {srcfile}")
        engine.run_function(engine.get_function_address("main"))
        return 0
    if mode == "build":
        ir_path = out + ".ll"
        with open(ir_path, "w", encoding="utf-8") as f:
            f.write(str(mod))
        print(f"[crownc] Wrote LLVM IR to {ir_path}")

        # Emit runtime
        write_crown_runtime()
        write_build_glue()

        if triple:
            # Cross-target build
            if not build_cross(ir_path, out, triple, sysroot):
                print("[crownc] ERROR: Cross build failed.")
        else:
            # Host-native build
            if not build_native(ir_path, out):
                print("[crownc] ERROR: Native build failed, but IR + runtime emitted.")
        return 0

    print(f"[crownc] Unknown mode: {mode}")
    return 1

# ============================================================
# Entry point (patched with cross-targeting)
# ============================================================

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("build","jit","run"):
    sys.exit(crownc(sys.argv))

# ============================================================
# Crownc Extreme Cross-Compiler Driver (Sysroots, Emit, Universal)
# ============================================================

import urllib.request, tarfile, zipfile

SYSROOT_URLS = {
    "win64": "https://example.com/mingw-w64-sysroot.tar.gz",
    "linux64": "https://example.com/linux-sysroot.tar.gz",
    "macos": "https://example.com/macos-sysroot.tar.gz",
    "android": "https://example.com/android-ndk-sysroot.tar.gz"
}

SYSROOT_CACHE = os.path.join(os.path.dirname(__file__), "sysroots")

def ensure_sysroot(triple):
    """Auto-download and extract sysroot if missing."""
    if not os.path.exists(SYSROOT_CACHE):
        os.makedirs(SYSROOT_CACHE)
    sysroot_dir = os.path.join(SYSROOT_CACHE, triple)
    if os.path.exists(sysroot_dir):
        return sysroot_dir
    url = SYSROOT_URLS.get(triple)
    if not url:
        print(f"[crownc] No sysroot URL known for {triple}")
        return None
    archive = os.path.join(SYSROOT_CACHE, f"{triple}.tar.gz")
    print(f"[crownc] Downloading sysroot for {triple}...")
    urllib.request.urlretrieve(url, archive)
    print(f"[crownc] Extracting sysroot for {triple}...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(sysroot_dir)
    return sysroot_dir

def build_cross(ir_path, out, triple, sysroot=None, runtime_c="crown_runtime.c", emit="bin"):
    """Cross-compile LLVM IR + runtime helpers into target binary/object/asm."""
    compiler, ctype = detect_compiler()
    if not compiler:
        print("[crownc] ERROR: No compiler found for cross build")
        return False
    if ctype not in ("clang","gcc"):
        print("[crownc] ERROR: Only clang/gcc support cross-target builds")
        return False

    cmd = [compiler, f"--target={triple}", "-O3", ir_path, runtime_c]
    if sysroot:
        cmd += [f"--sysroot={sysroot}"]

    if emit == "asm":
        cmd += ["-S", "-o", out + ".s"]
    elif emit == "obj":
        cmd += ["-c", "-o", out + ".o"]
    else:
        cmd += ["-o", out]

    try:
        subprocess.check_call(cmd)
        print(f"[crownc] Cross-built ({emit}) for {triple}: {out}")
        return True
    except Exception as e:
        print(f"[crownc] Cross-build failed: {e}")
        return False

# ============================================================
# Enhanced crownc driver with emit/universal/link-only
# ============================================================

def crownc(argv):
    """
    crownc driver:
      crownc run file.crown
      crownc jit file.crown
      crownc build file.crown -o file.exe [--target=triple] [--sysroot=/path]
        [--emit-asm] [--emit-obj] [--link-only] [--universal targets]
    """
    if len(argv) < 3:
        print("Usage: crownc <run|jit|build> file.crown [-o out] [flags]")
        return 1

    mode, srcfile = argv[1], argv[2]
    out = "a.out"
    if "-o" in argv:
        out = argv[argv.index("-o")+1]

    triple, sysroot = parse_target_flag(argv)
    emit = "bin"
    if "--emit-asm" in argv: emit = "asm"
    if "--emit-obj" in argv: emit = "obj"
    link_only = "--link-only" in argv

    universal_targets = []
    if "--universal" in argv:
        idx = argv.index("--universal")
        if idx+1 < len(argv):
            universal_targets = argv[idx+1].split(",")

    if mode in ("run","jit") and link_only:
        print("[crownc] ERROR: --link-only makes no sense for run/jit")
        return 1

    # Parse source unless link-only
    ast = None
    mod = None
    if not link_only:
        with open(srcfile, encoding="utf-8") as f:
            src = f.read()
        tokens = tokenize(src)
        ast = Parser(tokens).parse()
        mod, engine = codegen_module(ast, name=out)

    if mode == "run":
        VM(ast).run()
        return 0
    if mode == "jit":
        print(f"[crownc] JIT running {srcfile}")
        engine.run_function(engine.get_function_address("main"))
        return 0
    if mode == "build":
        ir_path = out + ".ll"
        if not link_only:
            with open(ir_path, "w", encoding="utf-8") as f:
                f.write(str(mod))
            print(f"[crownc] Wrote LLVM IR to {ir_path}")
            write_crown_runtime()
            write_build_glue()

        if universal_targets:
            for tgt in universal_targets:
                sysroot_auto = ensure_sysroot(tgt)
                tgt_out = out + f"-{tgt}"
                build_cross(ir_path, tgt_out, tgt, sysroot_auto, emit=emit)
            return 0

        if triple:
            sysroot_auto = sysroot or ensure_sysroot(triple)
            build_cross(ir_path, out, triple, sysroot_auto, emit=emit)
        else:
            if not build_native(ir_path, out):
                print("[crownc] ERROR: Native build failed, but IR emitted.")
        return 0

    print(f"[crownc] Unknown mode: {mode}")
    return 1

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("build","jit","run"):
    sys.exit(crownc(sys.argv))

# ============================================================
# Pure C Runtime Helpers for Standalone Crown Binaries
# ============================================================

CROWN_RUNTIME_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ------------------ Array ------------------
typedef struct {
    size_t len;
    size_t cap;
    int64_t *data;
} crown_array;

crown_array* crown_array_new() {
    crown_array* arr = (crown_array*)malloc(sizeof(crown_array));
    arr->len = 0;
    arr->cap = 4;
    arr->data = (int64_t*)malloc(arr->cap * sizeof(int64_t));
    return arr;
}

void crown_array_push(crown_array* arr, int64_t val) {
    if (arr->len >= arr->cap) {
        arr->cap *= 2;
        arr->data = (int64_t*)realloc(arr->data, arr->cap * sizeof(int64_t));
    }
    arr->data[arr->len++] = val;
}

int64_t crown_array_get(crown_array* arr, size_t idx) {
    if (idx >= arr->len) return 0;
    return arr->data[idx];
}

void crown_array_set(crown_array* arr, size_t idx, int64_t val) {
    if (idx < arr->len) arr->data[idx] = val;
}

// ------------------ Map ------------------
typedef struct crown_map_entry {
    char* key;
    int64_t val;
    struct crown_map_entry* next;
} crown_map_entry;

typedef struct {
    crown_map_entry** buckets;
    size_t bucket_count;
} crown_map;

unsigned long crown_hash(const char* s) {
    unsigned long h = 5381;
    while (*s) h = ((h << 5) + h) + (unsigned char)(*s++);
    return h;
}

crown_map* crown_map_new() {
    crown_map* m = (crown_map*)malloc(sizeof(crown_map));
    m->bucket_count = 64;
    m->buckets = (crown_map_entry**)calloc(m->bucket_count, sizeof(crown_map_entry*));
    return m;
}

void crown_map_set(crown_map* m, const char* key, int64_t val) {
    unsigned long h = crown_hash(key) % m->bucket_count;
    crown_map_entry* e = m->buckets[h];
    while (e) {
        if (strcmp(e->key, key) == 0) {
            e->val = val;
            return;
        }
        e = e->next;
    }
    e = (crown_map_entry*)malloc(sizeof(crown_map_entry));
    e->key = strdup(key);
    e->val = val;
    e->next = m->buckets[h];
    m->buckets[h] = e;
}

int64_t crown_map_get(crown_map* m, const char* key) {
    unsigned long h = crown_hash(key) % m->bucket_count;
    crown_map_entry* e = m->buckets[h];
    while (e) {
        if (strcmp(e->key, key) == 0) return e->val;
        e = e->next;
    }
    return 0;
}

// ------------------ Iterators ------------------
typedef struct {
    crown_array* arr;
    size_t idx;
} crown_iter_array;

crown_iter_array* crown_iter_array_new(crown_array* arr) {
    crown_iter_array* it = (crown_iter_array*)malloc(sizeof(crown_iter_array));
    it->arr = arr;
    it->idx = 0;
    return it;
}

int crown_iter_array_next(crown_iter_array* it, int64_t* out) {
    if (it->idx >= it->arr->len) return 0;
    *out = it->arr->data[it->idx++];
    return 1;
}

// ------------------ Basic JSON printing ------------------
void crown_json_print_array(crown_array* arr) {
    printf("[");
    for (size_t i=0;i<arr->len;i++) {
        printf("%lld", (long long)arr->data[i]);
        if (i+1<arr->len) printf(",");
    }
    printf("]");
}

void crown_json_print_map(crown_map* m) {
    printf("{");
    int first = 1;
    for (size_t b=0;b<m->bucket_count;b++) {
        crown_map_entry* e = m->buckets[b];
        while (e) {
            if (!first) printf(",");
            printf("\"%s\":%lld", e->key, (long long)e->val);
            first = 0;
            e = e->next;
        }
    }
    printf("}");
}
"""

def write_crown_runtime():
    """Emit pure C runtime to crown_runtime.c (linked into every build)."""
    with open("crown_runtime.c","w",encoding="utf-8") as f:
        f.write(CROWN_RUNTIME_C)

# ============================================================
# Extend LLVM Codegen: lower `say` for arrays/maps to C runtime
# ============================================================

def llvm_codegen_stmt_extended(self, builder, func, stmt):
    """Extended lowering for Say (arrays/maps → JSON print)."""
    from llvmlite import ir
    if isinstance(stmt, Say):
        val = self.codegen_expr(builder, stmt.expr)
        ty = val.type

        # If it's an i64, just print as number
        if ty == ir.IntType(64):
            printf = self.module.declare_intrinsic("printf", (), ())
            fmt = self._get_or_insert_global_str("%lld\n")
            builder.call(printf, [fmt, val])
            return

        # If it's pointer to crown_array → call crown_json_print_array
        if isinstance(ty, ir.PointerType) and ty.pointee.name == "struct.crown_array":
            fn = self.module.globals.get("crown_json_print_array")
            if fn is None:
                fnty = ir.FunctionType(ir.VoidType(), [ty])
                fn = ir.Function(self.module, fnty, name="crown_json_print_array")
            builder.call(fn, [val])
            return

        # If it's pointer to crown_map → call crown_json_print_map
        if isinstance(ty, ir.PointerType) and ty.pointee.name == "struct.crown_map":
            fn = self.module.globals.get("crown_json_print_map")
            if fn is None:
                fnty = ir.FunctionType(ir.VoidType(), [ty])
                fn = ir.Function(self.module, fnty, name="crown_json_print_map")
            builder.call(fn, [val])
            return

        # Fallback: treat as string pointer
        if isinstance(ty, ir.PointerType) and ty.pointee == ir.IntType(8):
            printf = self.module.declare_intrinsic("printf", (), ())
            fmt = self._get_or_insert_global_str("%s\n")
            builder.call(printf, [fmt, val])
            return

    # Fallback to normal stmt lowering
    return self.llvm_codegen_stmt_original(builder, func, stmt)

# Monkey-patch the extension into Codegen
if not hasattr(Codegen, "llvm_codegen_stmt_original"):
    Codegen.llvm_codegen_stmt_original = Codegen.codegen_stmt
    Codegen.codegen_stmt = llvm_codegen_stmt_extended

# Utility: string constants for printf
def _get_or_insert_global_str(self, s: str):
    from llvmlite import ir
    if not hasattr(self, "_global_strs"):
        self._global_strs = {}
    if s in self._global_strs:
        return self._global_strs[s]
    str_type = ir.ArrayType(ir.IntType(8), len(s)+1)
    gv = ir.GlobalVariable(self.module, str_type, name=f".str{len(self._global_strs)}")
    gv.global_constant = True
    gv.initializer = ir.Constant(str_type, bytearray(s.encode("utf8")+b"\0"))
    gv.linkage = "internal"
    ptr = gv.bitcast(ir.IntType(8).as_pointer())
    self._global_strs[s] = ptr
    return ptr

Codegen._get_or_insert_global_str = _get_or_insert_global_str

def codegen_value_type(self, val):
    # Return runtime type string for lowering into crown_runtime
    from llvmlite import ir
    if isinstance(val.type, ir.IntType) and val.type.width == 64:
        return "i64"
    if isinstance(val.type, ir.DoubleType):
        return "double"
    if isinstance(val.type, ir.PointerType) and val.type.pointee.name == "struct.crown_array":
        return "array"
    if isinstance(val.type, ir.PointerType) and val.type.pointee.name == "struct.crown_map":
        return "map"
    if isinstance(val.type, ir.PointerType) and val.type.pointee == ir.IntType(8):
        return "string"
    return "unknown"

def lower_readjson(self, builder, arg):
    from llvmlite import ir
    fnty = ir.FunctionType(ir.VoidType().as_pointer(), [ir.IntType(8).as_pointer(), ir.PointerType(ir.IntType(8).as_pointer())])
    fn = self.module.globals.get("crown_json_parse")
    if fn is None:
        fn = ir.Function(self.module, fnty, name="crown_json_parse")
    return builder.call(fn,[arg])


def lower_writejson(self, builder, filename, value, type_str):
    from llvmlite import ir
    fnty = ir.FunctionType(ir.IntType(32), [ir.IntType(8).as_pointer(), ir.VoidType().as_pointer(), ir.IntType(8).as_pointer()])
    fn = self.module.globals.get("crown_json_write_file")
    if fn is None:
        fn = ir.Function(self.module, fnty, name="crown_json_write_file")
    return builder.call(fn, [filename, value, type_str])
