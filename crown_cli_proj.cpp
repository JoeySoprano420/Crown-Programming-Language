// crown_cli.cpp
// Crown CLI: parse .crown -> CrownValue capsule -> LLVM IR -> binary -> (optional) run crown_runtime.exe
// C++14, Windows (Visual Studio 2022). Uses clang.exe to compile LLVM IR to an .exe if available.

#include <windows.h>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>

// ---------------------------- Utilities ----------------------------
static inline void ltrim_inplace(std::string& s) { size_t i=0; while (i<s.size() && std::isspace((unsigned char)s[i])) ++i; if (i) s.erase(0,i); }
static inline void rtrim_inplace(std::string& s) { while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back(); }
static inline std::string trim(std::string s) { ltrim_inplace(s); rtrim_inplace(s); return s; }
static inline bool starts_with(const std::string& s, const std::string& p) { return s.size()>=p.size() && std::equal(p.begin(),p.end(),s.begin()); }

static std::string read_text(const std::string& path, std::string& error) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) { error = "Failed to open: " + path; return {}; }
    std::ostringstream oss; oss<<ifs.rdbuf();
    if (!ifs.good() && !ifs.eof()) { error = "I/O error reading: " + path; return {}; }
    return oss.str();
}

static bool write_text(const std::string& path, const std::string& text, std::string& error) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) { error = "Failed to write: " + path; return false; }
    ofs<<text;
    if (!ofs.good()) { error = "I/O error writing: " + path; return false; }
    return true;
}

static std::string replace_ext(const std::string& path, const std::string& newExt) {
    size_t slash = path.find_last_of("/\\");
    size_t dot = path.find_last_of('.');
    if (dot==std::string::npos || (slash!=std::string::npos && dot<slash)) return path + newExt;
    return path.substr(0,dot) + newExt;
}

static std::wstring widen(const std::string& s) {
    if (s.empty()) return std::wstring();
    int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    if (len <= 0) return std::wstring();
    std::wstring w; w.resize((size_t)len);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], len);
    return w;
}

static std::wstring find_on_path(const wchar_t* exe) {
    DWORD need = SearchPathW(nullptr, exe, nullptr, 0, nullptr, nullptr);
    if (!need) return L"";
    std::vector<wchar_t> buf(need);
    DWORD got = SearchPathW(nullptr, exe, nullptr, (DWORD)buf.size(), buf.data(), nullptr);
    if (!got) return L"";
    return std::wstring(buf.data(), got);
}

static bool run_command(const std::wstring& cmd, std::string& error, DWORD* exitCode) {
    STARTUPINFOW si{}; si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> wcmd(cmd.begin(), cmd.end()); wcmd.push_back(L'\0');
    if (!CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        DWORD e = GetLastError();
        std::ostringstream oss; oss << "CreateProcessW failed, Win32 error " << e;
        error = oss.str(); return false;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code=0; GetExitCodeProcess(pi.hProcess, &code);
    if (exitCode) *exitCode = code;
    CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
    return true;
}

// ---------------------------- CrownValue (capsule) ----------------------------
struct CrownValue {
    enum Kind { Null, Int, Str, Arr, Map } kind = Null;
    int64_t i = 0;
    std::string s;
    std::vector<CrownValue> a;
    std::map<std::string, CrownValue> m;

    static CrownValue make_null() { return CrownValue(); }
    static CrownValue make_int(int64_t v){ CrownValue cv; cv.kind=Int; cv.i=v; return cv; }
    static CrownValue make_str(const std::string& v){ CrownValue cv; cv.kind=Str; cv.s=v; return cv; }
    static CrownValue make_arr(){ CrownValue cv; cv.kind=Arr; return cv; }
    static CrownValue make_map(){ CrownValue cv; cv.kind=Map; return cv; }

    CrownValue& map_set(const std::string& k, const CrownValue& v){ kind=Map; m[k]=v; return *this; }
    CrownValue& arr_push(const CrownValue& v){ kind=Arr; a.push_back(v); return *this; }

    bool is_null() const { return kind==Null; }
};

static void json_escape_print(const std::string& s, std::ostream& os) {
    os << '"';
    for (char c : s) {
        switch (c) {
            case '\"': os<<"\\\""; break;
            case '\\': os<<"\\\\"; break;
            case '\n': os<<"\\n"; break;
            case '\r': os<<"\\r"; break;
            case '\t': os<<"\\t"; break;
            default: os<<c;
        }
    }
    os << '"';
}

static void crown_debug_print(const CrownValue& v, std::ostream& os = std::cout) {
    switch (v.kind) {
        case CrownValue::Null: os<<"null"; break;
        case CrownValue::Int: os<<v.i; break;
        case CrownValue::Str: json_escape_print(v.s, os); break;
        case CrownValue::Arr: {
            os<<'[';
            for (size_t i=0;i<v.a.size();++i){ crown_debug_print(v.a[i], os); if (i+1<v.a.size()) os<<','; }
            os<<']';
        } break;
        case CrownValue::Map: {
            os<<'{'; size_t n=0;
            for (auto& kv : v.m) { json_escape_print(kv.first, os); os<<':'; crown_debug_print(kv.second, os); if (++n<v.m.size()) os<<','; }
            os<<'}';
        } break;
    }
}

// ---------------------------- Minimal Crown parser ----------------------------
// Grammar (tiny subset):
// program  := { stmt }
// stmt     := "let" ident "=" expr
//           | ident "=" expr
//           | "print" expr
// expr     := term { ('+'|'-') term }
// term     := factor { ('*'|'/') factor }
// factor   := int | string | ident | '(' expr ')'

struct Lexer {
    enum T { End, Ident, Int, String, Let, Print, Assign, LParen, RParen, Plus, Minus, Star, Slash } t = End;
    std::string lex; int64_t ival=0;
    const char* p=nullptr;

    explicit Lexer(const std::string& s) { p = s.c_str(); next(); }

    static bool isIdentStart(char c){ return std::isalpha((unsigned char)c) || c=='_'; }
    static bool isIdent(char c){ return std::isalnum((unsigned char)c) || c=='_'; }

    void skipws(){ while (*p && std::isspace((unsigned char)*p)) ++p; }

    void next() {
        skipws();
        if (!*p) { t = End; return; }
        char c=*p;
        if (isIdentStart(c)) {
            const char* s = p++; while (isIdent(*p)) ++p;
            lex.assign(s, p-s);
            if (lex=="let") t=Let;
            else if (lex=="print") t=Print;
            else t=Ident;
            return;
        }
        if (std::isdigit((unsigned char)c)) {
            char* end=nullptr;
            long long v = std::strtoll(p,&end,10);
            ival = (int64_t)v; p=end; t=Int; return;
        }
        if (c=='"') {
            ++p; std::string out;
            while (*p && *p!='"') {
                char ch=*p++;
                if (ch=='\\' && *p) {
                    char esc=*p++;
                    switch (esc) {
                        case 'n': out.push_back('\n'); break;
                        case 't': out.push_back('\t'); break;
                        case 'r': out.push_back('\r'); break;
                        case '\\': out.push_back('\\'); break;
                        case '"': out.push_back('"'); break;
                        default: out.push_back(esc); break;
                    }
                } else out.push_back(ch);
            }
            if (*p=='"') ++p;
            lex = out; t=String; return;
        }
        ++p;
        switch (c) {
            case '=': t=Assign; break;
            case '(': t=LParen; break;
            case ')': t=RParen; break;
            case '+': t=Plus; break;
            case '-': t=Minus; break;
            case '*': t=Star; break;
            case '/': t=Slash; break;
            default: t=End; break;
        }
    }
};

struct Parser {
    Lexer lx;

    explicit Parser(const std::string& src) : lx(src) {}

    CrownValue parseProgram() {
        CrownValue prog = CrownValue::make_map();
        CrownValue body = CrownValue::make_arr();
        while (lx.t != Lexer::End) {
            CrownValue s = parseStmt();
            if (s.is_null()) break;
            body.arr_push(s);
        }
        prog.map_set("node", CrownValue::make_str("program"));
        prog.map_set("body", body);
        return prog;
    }

    CrownValue parseStmt() {
        if (lx.t == Lexer::Let) {
            lx.next();
            if (lx.t != Lexer::Ident) return CrownValue::make_null();
            std::string name = lx.lex; lx.next();
            if (lx.t != Lexer::Assign) return CrownValue::make_null();
            lx.next();
            CrownValue e = parseExpr();
            CrownValue st = CrownValue::make_map();
            st.map_set("node", CrownValue::make_str("let"));
            st.map_set("name", CrownValue::make_str(name));
            st.map_set("expr", e);
            return st;
        }
        if (lx.t == Lexer::Print) {
            lx.next();
            CrownValue e = parseExpr();
            CrownValue st = CrownValue::make_map();
            st.map_set("node", CrownValue::make_str("print"));
            st.map_set("expr", e);
            return st;
        }
        if (lx.t == Lexer::Ident) {
            std::string name = lx.lex; lx.next();
            if (lx.t != Lexer::Assign) return CrownValue::make_null();
            lx.next();
            CrownValue e = parseExpr();
            CrownValue st = CrownValue::make_map();
            st.map_set("node", CrownValue::make_str("assign"));
            st.map_set("name", CrownValue::make_str(name));
            st.map_set("expr", e);
            return st;
        }
        return CrownValue::make_null();
    }

    CrownValue parseExpr() { return parseAddSub(); }

    CrownValue parseAddSub() {
        CrownValue left = parseMulDiv();
        while (lx.t==Lexer::Plus || lx.t==Lexer::Minus) {
            std::string op = (lx.t==Lexer::Plus ? "add" : "sub");
            lx.next();
            CrownValue right = parseMulDiv();
            CrownValue n = CrownValue::make_map();
            n.map_set("node", CrownValue::make_str(op)).map_set("left", left).map_set("right", right);
            left = n;
        }
        return left;
    }

    CrownValue parseMulDiv() {
        CrownValue left = parseFactor();
        while (lx.t==Lexer::Star || lx.t==Lexer::Slash) {
            std::string op = (lx.t==Lexer::Star ? "mul" : "div");
            lx.next();
            CrownValue right = parseFactor();
            CrownValue n = CrownValue::make_map();
            n.map_set("node", CrownValue::make_str(op)).map_set("left", left).map_set("right", right);
            left = n;
        }
        return left;
    }

    CrownValue parseFactor() {
        if (lx.t == Lexer::Int) { auto v = CrownValue::make_int(lx.ival); lx.next(); return v; }
        if (lx.t == Lexer::String) { auto v = CrownValue::make_str(lx.lex); lx.next(); return v; }
        if (lx.t == Lexer::Ident) { CrownValue v = CrownValue::make_map(); v.map_set("node", CrownValue::make_str("var")).map_set("name", CrownValue::make_str(lx.lex)); lx.next(); return v; }
        if (lx.t == Lexer::LParen) { lx.next(); CrownValue e = parseExpr(); if (lx.t==Lexer::RParen) lx.next(); return e; }
        return CrownValue::make_null();
    }
};

// ---------------------------- LLVM IR generation ----------------------------
struct IRBuilder {
    std::vector<std::string> globals;
    std::vector<std::string> body;
    std::set<std::string> vars;
    std::map<std::string, std::string> strGlobals;
    int tmp = 0, strId = 0;

    std::string nextTmp() { std::ostringstream oss; oss << "%t" << (++tmp); return oss.str(); }

    static std::string llvm_escape_cstr(const std::string& s, int& outLenNoNull) {
        std::ostringstream oss; outLenNoNull = 0;
        for (unsigned char c : s) {
            switch (c) {
                case '\\': oss << "\\5C"; break;
                case '\"': oss << "\\22"; break;
                case '\n': oss << "\\0A"; break;
                case '\r': oss << "\\0D"; break;
                case '\t': oss << "\\09"; break;
                default:
                    if (c < 32 || c >= 127) { char buf[8]; std::snprintf(buf, sizeof(buf), "\\%02X", (unsigned)c); oss << buf; }
                    else { oss << (char)c; }
            }
            outLenNoNull++;
        }
        return oss.str();
    }

    std::string ensureGlobalString(const std::string& s) {
        auto it = strGlobals.find(s);
        if (it != strGlobals.end()) return it->second;
        int lenNoNull = 0;
        std::string payload = llvm_escape_cstr(s, lenNoNull);
        std::ostringstream name; name << "@.str." << strId++;
        int totalLen = lenNoNull + 1; // add null
        std::ostringstream g;
        g << name.str() << " = private unnamed_addr constant [" << totalLen << " x i8] c\"" << payload << "\\00\", align 1";
        globals.push_back(g.str());
        strGlobals.emplace(s, name.str());
        return name.str();
    }

    void collectVarsFromExpr(const CrownValue& e) {
        if (e.kind==CrownValue::Map) {
            auto itNode = e.m.find("node");
            if (itNode!=e.m.end() && itNode->second.kind==CrownValue::Str) {
                const std::string& k = itNode->second.s;
                if (k=="var") {
                    auto itName = e.m.find("name");
                    if (itName!=e.m.end() && itName->second.kind==CrownValue::Str) vars.insert(itName->second.s);
                } else if (k=="add"||k=="sub"||k=="mul"||k=="div") {
                    collectVarsFromExpr(e.m.at("left"));
                    collectVarsFromExpr(e.m.at("right"));
                }
            }
        }
    }

    void collectVarsFromStmt(const CrownValue& s) {
        auto itNode = s.m.find("node");
        if (itNode==s.m.end() || itNode->second.kind!=CrownValue::Str) return;
        const std::string& k = itNode->second.s;
        if (k=="let" || k=="assign") {
            auto itName = s.m.find("name");
            if (itName!=s.m.end() && itName->second.kind==CrownValue::Str) vars.insert(itName->second.s);
            collectVarsFromExpr(s.m.at("expr"));
        } else if (k=="print") {
            collectVarsFromExpr(s.m.at("expr"));
        }
    }

    std::string emitExpr(const CrownValue& e) {
        if (e.kind==CrownValue::Int) {
            std::ostringstream oss; oss<<e.i; return oss.str(); // i32 literal inline
        }
        if (e.kind==CrownValue::Str) {
            // produce i8* pointer to global string
            std::string g = ensureGlobalString(e.s);
            std::string t = nextTmp();
            std::ostringstream gep;
            int lenNoNull = 0; (void)llvm_escape_cstr(e.s, lenNoNull);
            gep << "  " << t << " = getelementptr inbounds [" << (lenNoNull+1) << " x i8], [" << (lenNoNull+1) << " x i8]* " << g << ", i32 0, i32 0";
            body.push_back(gep.str());
            return t; // i8*
        }
        if (e.kind==CrownValue::Map) {
            auto itNode = e.m.find("node");
            if (itNode!=e.m.end() && itNode->second.kind==CrownValue::Str) {
                const std::string& k = itNode->second.s;
                if (k=="var") {
                    const std::string& name = e.m.at("name").s;
                    std::string t = nextTmp();
                    std::ostringstream ld;
                    ld << "  " << t << " = load i32, i32* %" << name << ", align 4";
                    body.push_back(ld.str());
                    return t; // i32
                }
                if (k=="add"||k=="sub"||k=="mul"||k=="div") {
                    std::string a = emitExpr(e.m.at("left"));
                    std::string b = emitExpr(e.m.at("right"));
                    std::string t = nextTmp();
                    const char* op = (k=="add"?"add":k=="sub"?"sub":k=="mul"?"mul":"sdiv");
                    std::ostringstream ir;
                    ir << "  " << t << " = " << op << " i32 " << a << ", " << b;
                    body.push_back(ir.str());
                    return t;
                }
            }
        }
        // Fallback null as 0
        return "0";
    }

    std::string build(const CrownValue& program) {
        // Collect variables (alloca in entry)
        const CrownValue& bodyArr = program.m.at("body");
        for (const auto& st : bodyArr.a) collectVarsFromStmt(st);

        // Common globals
        globals.push_back("declare i32 @printf(i8*, ...)");
        globals.push_back("@.fmt_int = private unnamed_addr constant [4 x i8] c\"%d\\0A\\00\", align 1");
        globals.push_back("@.fmt_str = private unnamed_addr constant [4 x i8] c\"%s\\0A\\00\", align 1");

        // Emit allocas
        for (const auto& v : vars) {
            std::ostringstream a; a << "  %" << v << " = alloca i32, align 4";
            body.push_back(a.str());
            // initialize to 0
            std::ostringstream z; z << "  store i32 0, i32* %" << v << ", align 4";
            body.push_back(z.str());
        }

        // Emit statements
        for (const auto& st : bodyArr.a) {
            const std::string& kind = st.m.at("node").s;
            if (kind=="let" || kind=="assign") {
                std::string name = st.m.at("name").s;
                std::string v = emitExpr(st.m.at("expr"));
                std::ostringstream s; s << "  store i32 " << v << ", i32* %" << name << ", align 4";
                body.push_back(s.str());
            } else if (kind=="print") {
                const CrownValue& e = st.m.at("expr");
                if (e.kind==CrownValue::Str) {
                    // printf("%s\n", str)
                    int dummy=0; (void)dummy;
                    std::string strPtr = emitExpr(e); // i8*
                    std::string fmt = nextTmp();
                    body.push_back("  " + fmt + " = getelementptr inbounds [4 x i8], [4 x i8]* @.fmt_str, i32 0, i32 0");
                    std::ostringstream call;
                    call << "  call i32 (i8*, ...) @printf(i8* " << fmt << ", i8* " << strPtr << ")";
                    body.push_back(call.str());
                } else {
                    // printf("%d\n", i32)
                    std::string val = emitExpr(e);
                    std::string fmt = nextTmp();
                    body.push_back("  " + fmt + " = getelementptr inbounds [4 x i8], [4 x i8]* @.fmt_int, i32 0, i32 0");
                    std::ostringstream call;
                    call << "  call i32 (i8*, ...) @printf(i8* " << fmt << ", i32 " << val << ")";
                    body.push_back(call.str());
                }
            }
        }

        // Assemble module text
        std::ostringstream ir;
        ir << "; ModuleID = 'crown'\n";
        for (auto& g : globals) ir << g << "\n";
        ir << "define i32 @main() {\n";
        ir << "entry:\n";
        for (auto& s : body) ir << s << "\n";
        ir << "  ret i32 0\n";
        ir << "}\n";
        return ir.str();
    }
};

// ---------------------------- Build pipeline ----------------------------
static bool compile_ll_to_exe(const std::string& llPath, const std::string& exePath, std::string& error) {
    std::wstring clang = find_on_path(L"clang.exe");
    if (clang.empty()) { error = "clang.exe not found on PATH. Install LLVM or use 'Desktop development with C++' incl. LLVM tools."; return false; }
    std::wostringstream cmd;
    cmd << L"\"" << clang << L"\" -O2 \"" << widen(llPath) << L"\" -o \"" << widen(exePath) << L"\"";
    DWORD exitCode = 0;
    if (!run_command(cmd.str(), error, &exitCode)) return false;
    if (exitCode != 0) { std::ostringstream oss; oss<<"clang exited with code "<<exitCode; error = oss.str(); return false; }
    return true;
}

static bool invoke_runtime(const std::string& exePath, std::string& error) {
    // Default runtime path; change as needed
    std::wstring runtime = L"C:\\Users\\420up\\Crown-Programming-Language\\crown_runtime.exe";
    // If not present, try PATH
    DWORD attrib = GetFileAttributesW(runtime.c_str());
    if (attrib==INVALID_FILE_ATTRIBUTES) {
        std::wstring onPath = find_on_path(L"crown_runtime.exe");
        if (!onPath.empty()) runtime = onPath;
    }
    attrib = GetFileAttributesW(runtime.c_str());
    if (attrib==INVALID_FILE_ATTRIBUTES) { error = "crown_runtime.exe not found (checked default path and PATH)."; return false; }

    std::wostringstream cmd;
    cmd << L"\"" << runtime << L"\" \"" << widen(exePath) << L"\"";
    DWORD exitCode = 0;
    if (!run_command(cmd.str(), error, &exitCode)) return false;
    if (exitCode != 0) { std::ostringstream oss; oss<<"crown_runtime exited with code "<<exitCode; error = oss.str(); return false; }
    return true;
}

// ---------------------------- Main ----------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: crown_cli <file.crown> [--emit-capsule] [--emit-ll] [--no-rt] [--run]\n";
        return 1;
    }

    std::string inPath = argv[1];
    bool emitCapsule = false, emitLL = true, runRT = true, runExe = false;

    for (int i=2;i<argc;i++) {
        std::string a = argv[i];
        if (a=="--emit-capsule") emitCapsule = true;
        else if (a=="--emit-ll") emitLL = true;
        else if (a=="--no-rt") runRT = false;
        else if (a=="--run") runExe = true;
    }

    std::string error;
    std::string src = read_text(inPath, error);
    if (!error.empty()) { std::cerr << error << "\n"; return 2; }

    // Parse .crown into CrownValue capsule (AST)
    Parser parser(src);
    CrownValue capsule = parser.parseProgram();

    if (emitCapsule) {
        std::cout << "Capsule (AST):\n";
        crown_debug_print(capsule); std::cout << "\n";
    }

    // Transpile to LLVM IR
    IRBuilder irb;
    std::string ir = irb.build(capsule);
    std::string llPath = replace_ext(inPath, ".ll");
    if (emitLL) {
        if (!write_text(llPath, ir, error)) { std::cerr << error << "\n"; return 3; }
        std::cout << "Emitted LLVM IR: " << llPath << "\n";
    }

    // Compile to native binary using clang
    std::string exePath = replace_ext(inPath, ".exe");
    if (!compile_ll_to_exe(llPath, exePath, error)) { std::cerr << error << "\n"; return 4; }
    std::cout << "Built binary: " << exePath << "\n";

    // Optionally run the produced binary
    if (runExe) {
        std::wostringstream cmd;
        cmd << L"\"" << widen(exePath) << L"\"";
        DWORD exitCode = 0;
        if (!run_command(cmd.str(), error, &exitCode)) { std::cerr << error << "\n"; return 5; }
        std::cout << "Program exited with code " << exitCode << "\n";
    }

    // Optionally invoke crown_runtime.exe with the binary
    if (runRT) {
        if (!invoke_runtime(exePath, error)) { std::cerr << error << "\n"; return 6; }
        std::cout << "crown_runtime completed successfully.\n";
    }

    return 0;
}

// ---------------------------- Extra: Simple REPL ----------------------------
static void repl() {
    std::cout << "Crown REPL (type 'exit' to quit)\n";
    std::string line;
    while (true) {
        std::cout << ">> ";
        if (!std::getline(std::cin, line)) break;
        if (line=="exit" || line=="quit") break;
        if (line.empty()) continue;

        Parser parser(line);
        CrownValue capsule = parser.parseProgram();
        if (capsule.is_null()) {
            std::cerr << "Parse error.\n";
            continue;
        }

        std::cout << "AST: ";
        crown_debug_print(capsule);
        std::cout << "\n";

        IRBuilder irb;
        std::string ir = irb.build(capsule);
        std::cout << "LLVM IR:\n" << ir << "\n";
    }
}

// ---------------------------- Extra: Test harness ----------------------------
#ifdef _DEBUG
static void run_demo() {
    std::string demoSrc = R"(
        let x = 10
        let y = 20
        print x + y
        print "Hello, Crown!"
    )";

    Parser parser(demoSrc);
    CrownValue capsule = parser.parseProgram();

    std::cout << "[Demo AST]\n";
    crown_debug_print(capsule); std::cout << "\n";

    IRBuilder irb;
    std::string ir = irb.build(capsule);
    std::cout << "[Demo LLVM IR]\n" << ir << "\n";
}
#endif

// ---------------------------- Post-Main Hook ----------------------------
int post_main() {
#ifdef _DEBUG
    run_demo();
#endif
    repl();
    return 0;
}

