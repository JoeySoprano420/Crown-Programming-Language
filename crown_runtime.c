// Crown Script Runtime (crown_runtime.c)
// Compile with: clang -c crown_runtime.c -o crown_runtime.o
// Link with: clang crown_runtime.o your_ir.bc -o yourprog
//
// Provides arrays, maps, strings, JSON parse/print, and file I/O
// for compiled Crown programs.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <errno.h>

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

// ============================================================
// Constructors
// ============================================================

CrownValue crown_make_int(int64_t i) {
    CrownValue v; v.kind = CROWN_VAL_INT; v.as.i = i; return v;
}

CrownValue crown_make_str(const char* s) {
    CrownValue v; v.kind = CROWN_VAL_STR;
    v.as.s = strdup(s);
    return v;
}

CrownValue crown_make_array() {
    CrownArray* arr = (CrownArray*)malloc(sizeof(CrownArray));
    arr->size = 0; arr->capacity = 4;
    arr->data = (CrownValue*)malloc(sizeof(CrownValue) * arr->capacity);
    CrownValue v; v.kind = CROWN_VAL_ARR; v.as.arr = arr; return v;
}

CrownValue crown_make_map() {
    CrownMap* m = (CrownMap*)malloc(sizeof(CrownMap));
    m->size = 0; m->capacity = 4;
    m->entries = (CrownMapEntry*)malloc(sizeof(CrownMapEntry) * m->capacity);
    CrownValue v; v.kind = CROWN_VAL_MAP; v.as.map = m; return v;
}

CrownValue crown_make_null() {
    CrownValue v; v.kind = CROWN_VAL_NULL; return v;
}

// ============================================================
// Array Ops
// ============================================================

void crown_array_push(CrownValue arrv, CrownValue val) {
    if (arrv.kind != CROWN_VAL_ARR) return;
    CrownArray* arr = arrv.as.arr;
    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        arr->data = (CrownValue*)realloc(arr->data, sizeof(CrownValue) * arr->capacity);
    }
    arr->data[arr->size++] = val;
}

CrownValue crown_array_get(CrownValue arrv, int64_t idx) {
    if (arrv.kind != CROWN_VAL_ARR) return crown_make_null();
    CrownArray* arr = arrv.as.arr;
    if (idx < 0 || idx >= arr->size) return crown_make_null();
    return arr->data[idx];
}

int64_t crown_array_len(CrownValue arrv) {
    if (arrv.kind != CROWN_VAL_ARR) return 0;
    return arrv.as.arr->size;
}

// ============================================================
// Map Ops
// ============================================================

void crown_map_set(CrownValue mapv, const char* key, CrownValue val) {
    if (mapv.kind != CROWN_VAL_MAP) return;
    CrownMap* m = mapv.as.map;
    for (int64_t i = 0; i < m->size; i++) {
        if (strcmp(m->entries[i].key, key) == 0) {
            m->entries[i].value = val; return;
        }
    }
    if (m->size >= m->capacity) {
        m->capacity *= 2;
        m->entries = (CrownMapEntry*)realloc(m->entries, sizeof(CrownMapEntry) * m->capacity);
    }
    m->entries[m->size].key = strdup(key);
    m->entries[m->size].value = val;
    m->size++;
}

CrownValue crown_map_get(CrownValue mapv, const char* key) {
    if (mapv.kind != CROWN_VAL_MAP) return crown_make_null();
    CrownMap* m = mapv.as.map;
    for (int64_t i = 0; i < m->size; i++) {
        if (strcmp(m->entries[i].key, key) == 0) return m->entries[i].value;
    }
    return crown_make_null();
}

int64_t crown_map_has(CrownValue mapv, const char* key) {
    if (mapv.kind != CROWN_VAL_MAP) return 0;
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
    for (const char* p = s; *p; p++) {
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

// Shortcut to print to stdout
void crown_debug_print(CrownValue v) {
    crown_json_fprint(stdout, v);
    fputc('\n', stdout);
}

// ============================================================
// JSON Parser (string -> CrownValue)
// ============================================================

static const char* skip_ws(const char* s) {
    while (*s && isspace((unsigned char)*s)) s++;
    return s;
}

static CrownValue parse_value(const char** sp);

static char* parse_string(const char** sp) {
    const char* s = *sp; s++;
    char* buf = (char*)malloc(1); size_t len = 0, cap = 1;
    while (*s && *s != '"') {
        char c = *s;
        if (c == '\\') {
            s++;
            if (*s == 'n') c = '\n';
            else if (*s == 't') c = '\t';
            else if (*s == 'r') c = '\r';
            else c = *s;
        }
        if (len + 1 >= cap) { cap *= 2; buf : realloc(buf, cap); }
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
    s = skip_ws(s);
    while (*s && *s != '}') {
        char* key = parse_string(&s);
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
    if (!strncmp(s, "null", 4)) { *sp = s + 4; return crown_make_null(); }
    *sp = s; return crown_make_null();
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
    if (!f) return crown_make_null();
    fseek(f, 0, SEEK_END); long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = 0;
    fclose(f);
    CrownValue v = crown_json_parse(buf);
    free(buf);
    return v;
}
