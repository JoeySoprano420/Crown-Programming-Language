// crown_runtime.c + demo main
// Build on MSVC: cl /TC crown_runtime.c /Fe:crown.exe
// Build on clang: clang crown_runtime.c -o crown

#define _CRT_SECURE_NO_WARNINGS
#ifdef _MSC_VER
#define strdup _strdup
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

    // ============================================================
    // Tagged Value System
    // ============================================================

    typedef enum {
        CROWN_VAL_INT,
        CROWN_VAL_STR,
        CROWN_VAL_ARR,
        CROWN_VAL_MAP,
        CROWN_VAL_NULL
    } CrownValKind;

    struct CrownArray;
    struct CrownMap;

    typedef struct CrownValue {
        CrownValKind kind;
        union {
            int64_t i;
            char* s;
            struct CrownArray* arr;
            struct CrownMap* map;
        } as;
    } CrownValue;

    typedef struct CrownArray {
        CrownValue* data;
        int64_t size;
        int64_t capacity;
    } CrownArray;

    typedef struct {
        char* key;
        CrownValue value;
    } CrownMapEntry;

    typedef struct CrownMap {
        CrownMapEntry* entries;
        int64_t size;
        int64_t capacity;
    } CrownMap;

    // ============================================================
    // Helpers
    // ============================================================

    static CrownValue crown_null_value(void) {
        CrownValue v;
        v.kind = CROWN_VAL_NULL;
        return v;
    }

    // ============================================================
    // Constructors
    // ============================================================

    CrownValue crown_make_int(int64_t i) {
        CrownValue v; v.kind = CROWN_VAL_INT; v.as.i = i; return v;
    }

    CrownValue crown_make_str(const char* s) {
        CrownValue v; v.kind = CROWN_VAL_STR;
        v.as.s = s ? strdup(s) : strdup("");
        return v;
    }

    CrownValue crown_make_array() {
        CrownArray* arr = (CrownArray*)malloc(sizeof(CrownArray));
        if (!arr) return crown_null_value();
        arr->size = 0; arr->capacity = 4;
        arr->data = (CrownValue*)malloc(sizeof(CrownValue) * arr->capacity);
        if (!arr->data) { free(arr); return crown_null_value(); }
        CrownValue v; v.kind = CROWN_VAL_ARR; v.as.arr = arr; return v;
    }

    CrownValue crown_make_map() {
        CrownMap* m = (CrownMap*)malloc(sizeof(CrownMap));
        if (!m) return crown_null_value();
        m->size = 0; m->capacity = 4;
        m->entries = (CrownMapEntry*)malloc(sizeof(CrownMapEntry) * m->capacity);
        if (!m->entries) { free(m); return crown_null_value(); }
        CrownValue v; v.kind = CROWN_VAL_MAP; v.as.map = m; return v;
    }

    // ============================================================
    // Array Ops
    // ============================================================

    void crown_array_push(CrownValue arrv, CrownValue val) {
        if (arrv.kind != CROWN_VAL_ARR || !arrv.as.arr) return;
        CrownArray* arr = arrv.as.arr;
        if (arr->size >= arr->capacity) {
            int64_t newcap = arr->capacity * 2;
            void* tmp = realloc(arr->data, sizeof(CrownValue) * newcap);
            if (!tmp) return;
            arr->data = (CrownValue*)tmp;
            arr->capacity = newcap;
        }
        arr->data[arr->size++] = val;
    }

    CrownValue crown_array_get(CrownValue arrv, int64_t idx) {
        if (arrv.kind != CROWN_VAL_ARR || !arrv.as.arr) return crown_null_value();
        CrownArray* arr = arrv.as.arr;
        if (idx < 0 || idx >= arr->size) return crown_null_value();
        return arr->data[idx];
    }

    int64_t crown_array_len(CrownValue arrv) {
        if (arrv.kind != CROWN_VAL_ARR || !arrv.as.arr) return 0;
        return arrv.as.arr->size;
    }

    // ============================================================
    // Map Ops
    // ============================================================

    void crown_map_set(CrownValue mapv, const char* key, CrownValue val) {
        if (mapv.kind != CROWN_VAL_MAP || !mapv.as.map) return;
        CrownMap* m = mapv.as.map;
        for (int64_t i = 0; i < m->size; i++) {
            if (strcmp(m->entries[i].key, key) == 0) {
                m->entries[i].value = val; return;
            }
        }
        if (m->size >= m->capacity) {
            int64_t newcap = m->capacity * 2;
            void* tmp = realloc(m->entries, sizeof(CrownMapEntry) * newcap);
            if (!tmp) return;
            m->entries = (CrownMapEntry*)tmp;
            m->capacity = newcap;
        }
        m->entries[m->size].key = strdup(key ? key : "");
        m->entries[m->size].value = val;
        m->size++;
    }

    CrownValue crown_map_get(CrownValue mapv, const char* key) {
        if (mapv.kind != CROWN_VAL_MAP || !mapv.as.map) return crown_null_value();
        CrownMap* m = mapv.as.map;
        for (int64_t i = 0; i < m->size; i++) {
            if (strcmp(m->entries[i].key, key) == 0) return m->entries[i].value;
        }
        return crown_null_value();
    }

    int64_t crown_map_has(CrownValue mapv, const char* key) {
        if (mapv.kind != CROWN_VAL_MAP || !mapv.as.map) return 0;
        CrownMap* m = mapv.as.map;
        for (int64_t i = 0; i < m->size; i++) {
            if (strcmp(m->entries[i].key, key) == 0) return 1;
        }
        return 0;
    }

    // ============================================================
    // JSON Debug Printing
    // ============================================================

    static void crown_json_escape_and_print(const char* s, FILE* f) {
        fputc('"', f);
        for (const char* p = s; p && *p; p++) {
            switch (*p) {
            case '\"': fputs("\\\"", f); break;
            case '\\': fputs("\\\\", f); break;
            case '\n': fputs("\\n", f); break;
            case '\r': fputs("\\r", f); break;
            case '\t': fputs("\\t", f); break;
            default: fputc(*p, f);
            }
        }
        fputc('"', f);
    }

    void crown_json_fprint(FILE* f, CrownValue v);

    static void crown_json_fprint_array(FILE* f, CrownArray* arr) {
        fputc('[', f);
        for (int64_t i = 0; i < arr->size; i++) {
            crown_json_fprint(f, arr->data[i]);
            if (i + 1 < arr->size) fputc(',', f);
        }
        fputc(']', f);
    }

    static void crown_json_fprint_map(FILE* f, CrownMap* m) {
        fputc('{', f);
        for (int64_t i = 0; i < m->size; i++) {
            crown_json_escape_and_print(m->entries[i].key, f);
            fputc(':', f);
            crown_json_fprint(f, m->entries[i].value);
            if (i + 1 < m->size) fputc(',', f);
        }
        fputc('}', f);
    }

    void crown_json_fprint(FILE* f, CrownValue v) {
        switch (v.kind) {
        case CROWN_VAL_INT: fprintf(f, "%lld", (long long)v.as.i); break;
        case CROWN_VAL_STR: crown_json_escape_and_print(v.as.s, f); break;
        case CROWN_VAL_ARR: crown_json_fprint_array(f, v.as.arr); break;
        case CROWN_VAL_MAP: crown_json_fprint_map(f, v.as.map); break;
        case CROWN_VAL_NULL: fputs("null", f); break;
        }
    }

    void crown_debug_print(CrownValue v) {
        crown_json_fprint(stdout, v);
        fputc('\n', stdout);
    }

    // ============================================================
    // JSON Parser
    // ============================================================

    static const char* skip_ws(const char* s) {
        while (*s && isspace((unsigned char)*s)) s++;
        return s;
    }

    static CrownValue parse_value(const char** sp);

    static char* parse_string(const char** sp) {
        const char* s = *sp; s++;
        size_t cap = 16, len = 0;
        char* buf = (char*)malloc(cap);
        if (!buf) return NULL;
        while (*s && *s != '"') {
            char c = *s;
            if (c == '\\') {
                s++;
                if (*s == 'n') c = '\n';
                else if (*s == 't') c = '\t';
                else if (*s == 'r') c = '\r';
                else c = *s;
            }
            if (len + 1 >= cap) {
                size_t newcap = cap * 2;
                void* tmp = realloc(buf, newcap);
                if (!tmp) { free(buf); return NULL; }
                buf = (char*)tmp;
                cap = newcap;
            }
            buf[len++] = c; s++;
        }
        buf[len] = 0;
        if (*s == '"') s++;
        *sp = s;
        return buf;
    }

    static CrownValue parse_number(const char** sp) {
        char* end;
        long long v = strtoll(*sp, &end, 10);
        *sp = end;
        return crown_make_int(v);
    }

    static CrownValue parse_array(const char** sp) {
        const char* s = *sp; s++;
        CrownValue arrv = crown_make_array();
        if (arrv.kind != CROWN_VAL_ARR) { *sp = s; return crown_null_value(); }
        s = skip_ws(s);
        while (*s && *s != ']') {
            CrownValue val = parse_value(&s);
            crown_array_push(arrv, val);
            s = skip_ws(s);
            if (*s == ',') { s++; s = skip_ws(s); }
        }
        if (*s == ']') s++;
        *sp = s;
        return arrv;
    }

    static CrownValue parse_object(const char** sp) {
        const char* s = *sp; s++;
        CrownValue mapv = crown_make_map();
        if (mapv.kind != CROWN_VAL_MAP) { *sp = s; return crown_null_value(); }
        s = skip_ws(s);
        while (*s && *s != '}') {
            char* key = parse_string(&s);
            if (!key) { *sp = s; return crown_null_value(); }
            s = skip_ws(s); if (*s == ':') s++;
            s = skip_ws(s);
            CrownValue val = parse_value(&s);
            crown_map_set(mapv, key, val);
            free(key);
            s = skip_ws(s);
            if (*s == ',') { s++; s = skip_ws(s); }
        }
        if (*s == '}') s++;
        *sp = s;
        return mapv;
    }

    static CrownValue parse_value(const char** sp) {
        const char* s = skip_ws(*sp);
        if (*s == '"') { char* str = parse_string(&s); *sp = s; CrownValue v = crown_make_str(str); free(str); return v; }
        if (*s == '{') { CrownValue v = parse_object(&s); *sp = s; return v; }
        if (*s == '[') { CrownValue v = parse_array(&s); *sp = s; return v; }
        if (isdigit((unsigned char)*s) || *s == '-') { CrownValue v = parse_number(&s); *sp = s; return v; }
        if (!strncmp(s, "true", 4)) { *sp = s + 4; return crown_make_int(1); }
        if (!strncmp(s, "false", 5)) { *sp = s + 5; return crown_make_int(0); }
        if (!strncmp(s, "null", 4)) { *sp = s + 4; return crown_null_value(); }
        *sp = s; return crown_null_value();
    }

    CrownValue crown_json_parse(const char* text) {
        const char* s = text;
        return parse_value(&s);
    }

    // ============================================================
    // JSON File I/O
    // ============================================================

    int crown_json_write_file(const char* filename, CrownValue v) {
        FILE* f = fopen(filename, "w");
        if (!f) return errno;
        crown_json_fprint(f, v);
        fclose(f);
        return 0;
    }

    CrownValue crown_json_read_file(const char* filename) {
        FILE* f = fopen(filename, "rb");
        if (!f) return crown_null_value();
        if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return crown_null_value(); }
        long sz = ftell(f);
        if (sz < 0) { fclose(f); return crown_null_value(); }
        rewind(f);

        if (sz > SIZE_MAX - 1) { fclose(f); return crown_null_value(); }
        char* buf = (char*)malloc((size_t)sz + 1);
        if (!buf) { fclose(f); return crown_null_value(); }
        size_t nread = fread(buf, 1, (size_t)sz, f);
        if (ferror(f)) { free(buf); fclose(f); return crown_null_value(); }
        buf[nread] = 0;
        fclose(f);

        CrownValue v = crown_json_parse(buf);
        free(buf);
        return v;
    }

#ifdef __cplusplus
} // extern "C"
#endif

// ============================================================
// Demo Main (so MSVC has an entry point)
// ============================================================

int main(void) {
    // Array demo
    CrownValue arr = crown_make_array();
    crown_array_push(arr, crown_make_int(42));
    crown_array_push(arr, crown_make_str("hello crown"));
    printf("Array demo:\n");
    crown_debug_print(arr);

    // Map demo
    CrownValue map = crown_make_map();
    crown_map_set(map, "answer", crown_make_int(42));
    crown_map_set(map, "greet", crown_make_str("hi!"));
    printf("Map demo:\n");
    crown_debug_print(map);

    // JSON round trip
    crown_json_write_file("out.json", map);
    CrownValue parsed = crown_json_read_file("out.json");
    printf("JSON round trip:\n");
    crown_debug_print(parsed);

    return 0;
}
