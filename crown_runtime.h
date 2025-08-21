// Crown Script Runtime API (crown_runtime.h)
// Use with crown_runtime.c
// Provides tagged values, arrays, maps, JSON parse/print, and file I/O.

#ifndef CROWN_RUNTIME_H
#define CROWN_RUNTIME_H

#include <stdint.h>
#include <stdio.h>

// ============================================================
// Value System
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

CrownValue crown_make_int(int64_t i);
CrownValue crown_make_str(const char* s);
CrownValue crown_make_array();
CrownValue crown_make_map();
CrownValue crown_make_null();

// ============================================================
// Array Ops
// ============================================================

void crown_array_push(CrownValue arrv, CrownValue val);
CrownValue crown_array_get(CrownValue arrv, int64_t idx);
int64_t crown_array_len(CrownValue arrv);

// ============================================================
// Map Ops
// ============================================================

void crown_map_set(CrownValue mapv, const char* key, CrownValue val);
CrownValue crown_map_get(CrownValue mapv, const char* key);
int64_t crown_map_has(CrownValue mapv, const char* key);

// ============================================================
// Debug Printing
// ============================================================

void crown_debug_print(CrownValue v);        // pretty-print to stdout
void crown_json_fprint(FILE* f, CrownValue v); // print to file

// ============================================================
// JSON Parser
// ============================================================

CrownValue crown_json_parse(const char* text);

// ============================================================
// JSON File I/O
// ============================================================

int crown_json_write_file(const char* filename, CrownValue v);
CrownValue crown_json_read_file(const char* filename);

#endif // CROWN_RUNTIME_H
