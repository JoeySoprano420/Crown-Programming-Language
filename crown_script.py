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
