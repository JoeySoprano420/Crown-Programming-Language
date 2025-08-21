// Crown Script Runtime Helpers (crown_runtime.c)
// Compile with: clang -c crown_runtime.c -o crown_runtime.o
// Link with your generated LLVM IR to produce a full .exe

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// --------------------- Array ---------------------

typedef struct {
    int64_t *data;
    int64_t size;
    int64_t capacity;
} CrownArray;

void* crown_array_new(int64_t initial_size) {
    CrownArray* arr = (CrownArray*)malloc(sizeof(CrownArray));
    arr->capacity = (initial_size > 0) ? initial_size : 4;
    arr->size = 0;
    arr->data = (int64_t*)malloc(sizeof(int64_t) * arr->capacity);
    return arr;
}

void crown_array_push(void* arr_ptr, int64_t value) {
    CrownArray* arr = (CrownArray*)arr_ptr;
    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        arr->data = (int64_t*)realloc(arr->data, sizeof(int64_t) * arr->capacity);
    }
    arr->data[arr->size++] = value;
}

int64_t crown_array_get(void* arr_ptr, int64_t index) {
    CrownArray* arr = (CrownArray*)arr_ptr;
    if (index < 0 || index >= arr->size) return 0;
    return arr->data[index];
}

void crown_array_set(void* arr_ptr, int64_t index, int64_t value) {
    CrownArray* arr = (CrownArray*)arr_ptr;
    if (index >= 0 && index < arr->size) {
        arr->data[index] = value;
    }
}

int64_t crown_array_len(void* arr_ptr) {
    CrownArray* arr = (CrownArray*)arr_ptr;
    return arr->size;
}

// --------------------- Map ---------------------

typedef struct {
    int64_t key;
    int64_t value;
} CrownMapEntry;

typedef struct {
    CrownMapEntry* entries;
    int64_t size;
    int64_t capacity;
} CrownMap;

void* crown_map_new() {
    CrownMap* m = (CrownMap*)malloc(sizeof(CrownMap));
    m->capacity = 8;
    m->size = 0;
    m->entries = (CrownMapEntry*)malloc(sizeof(CrownMapEntry) * m->capacity);
    return m;
}

void crown_map_set(void* m_ptr, int64_t key, int64_t value) {
    CrownMap* m = (CrownMap*)m_ptr;
    for (int64_t i=0; i<m->size; i++) {
        if (m->entries[i].key == key) {
            m->entries[i].value = value;
            return;
        }
    }
    if (m->size >= m->capacity) {
        m->capacity *= 2;
        m->entries = (CrownMapEntry*)realloc(m->entries, sizeof(CrownMapEntry) * m->capacity);
    }
    m->entries[m->size].key = key;
    m->entries[m->size].value = value;
    m->size++;
}

int64_t crown_map_get(void* m_ptr, int64_t key) {
    CrownMap* m = (CrownMap*)m_ptr;
    for (int64_t i=0; i<m->size; i++) {
        if (m->entries[i].key == key) {
            return m->entries[i].value;
        }
    }
    return 0;
}

int64_t crown_map_has(void* m_ptr, int64_t key) {
    CrownMap* m = (CrownMap*)m_ptr;
    for (int64_t i=0; i<m->size; i++) {
        if (m->entries[i].key == key) return 1;
    }
    return 0;
}

void* crown_map_keys(void* m_ptr) {
    CrownMap* m = (CrownMap*)m_ptr;
    CrownArray* arr = (CrownArray*)crown_array_new(m->size);
    for (int64_t i=0; i<m->size; i++) {
        crown_array_push(arr, m->entries[i].key);
    }
    return arr;
}

// --------------------- Match Helper ---------------------

int64_t crown_match_tag(int64_t value) {
    // Very simplistic: identity mapping for now
    // Later you can implement hashing, tagging, or object kind IDs
    return value;
}

// --------------------- Debug Helpers ---------------------

void crown_debug_print_array(void* arr_ptr) {
    CrownArray* arr = (CrownArray*)arr_ptr;
    printf("[");
    for (int64_t i=0; i<arr->size; i++) {
        printf("%lld", (long long)arr->data[i]);
        if (i+1 < arr->size) printf(", ");
    }
    printf("]\n");
}

void crown_debug_print_map(void* m_ptr) {
    CrownMap* m = (CrownMap*)m_ptr;
    printf("{");
    for (int64_t i=0; i<m->size; i++) {
        printf("%lld: %lld", (long long)m->entries[i].key, (long long)m->entries[i].value);
        if (i+1 < m->size) printf(", ");
    }
    printf("}\n");
}

// Crown Script Runtime Helpers (Extended)
// Supports arrays, maps, strings, and JSON-style nested structures.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// --------------------- Tagged Value ---------------------
// Instead of raw int64_t, we introduce a tagged union to hold
// numbers, strings, arrays, maps.

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

// --------------------- Value Constructors ---------------------

CrownValue crown_make_int(int64_t i) {
    CrownValue v; v.kind=CROWN_VAL_INT; v.as.i=i; return v;
}

CrownValue crown_make_str(const char* s) {
    CrownValue v; v.kind=CROWN_VAL_STR;
    v.as.s = strdup(s);
    return v;
}

CrownValue crown_make_array() {
    CrownArray* arr=(CrownArray*)malloc(sizeof(CrownArray));
    arr->size=0; arr->capacity=4;
    arr->data=(CrownValue*)malloc(sizeof(CrownValue)*arr->capacity);
    CrownValue v; v.kind=CROWN_VAL_ARR; v.as.arr=arr; return v;
}

CrownValue crown_make_map() {
    CrownMap* m=(CrownMap*)malloc(sizeof(CrownMap));
    m->size=0; m->capacity=4;
    m->entries=(CrownMapEntry*)malloc(sizeof(CrownMapEntry)*m->capacity);
    CrownValue v; v.kind=CROWN_VAL_MAP; v.as.map=m; return v;
}

// --------------------- Array Ops ---------------------

void crown_array_push(CrownValue arrv, CrownValue val) {
    if (arrv.kind!=CROWN_VAL_ARR) return;
    CrownArray* arr=arrv.as.arr;
    if (arr->size>=arr->capacity) {
        arr->capacity*=2;
        arr->data=(CrownValue*)realloc(arr->data,sizeof(CrownValue)*arr->capacity);
    }
    arr->data[arr->size++]=val;
}

CrownValue crown_array_get(CrownValue arrv,int64_t idx) {
    if (arrv.kind!=CROWN_VAL_ARR) return crown_make_int(0);
    CrownArray* arr=arrv.as.arr;
    if (idx<0||idx>=arr->size) return crown_make_int(0);
    return arr->data[idx];
}

int64_t crown_array_len(CrownValue arrv) {
    if (arrv.kind!=CROWN_VAL_ARR) return 0;
    return arrv.as.arr->size;
}

// --------------------- Map Ops ---------------------

void crown_map_set(CrownValue mapv,const char* key,CrownValue val) {
    if (mapv.kind!=CROWN_VAL_MAP) return;
    CrownMap* m=mapv.as.map;
    for(int64_t i=0;i<m->size;i++){
        if(strcmp(m->entries[i].key,key)==0){
            m->entries[i].value=val; return;
        }
    }
    if(m->size>=m->capacity){
        m->capacity*=2;
        m->entries=(CrownMapEntry*)realloc(m->entries,sizeof(CrownMapEntry)*m->capacity);
    }
    m->entries[m->size].key=strdup(key);
    m->entries[m->size].value=val;
    m->size++;
}

CrownValue crown_map_get(CrownValue mapv,const char* key) {
    if(mapv.kind!=CROWN_VAL_MAP) return crown_make_int(0);
    CrownMap* m=mapv.as.map;
    for(int64_t i=0;i<m->size;i++){
        if(strcmp(m->entries[i].key,key)==0) return m->entries[i].value;
    }
    return crown_make_int(0);
}

int64_t crown_map_has(CrownValue mapv,const char* key) {
    if(mapv.kind!=CROWN_VAL_MAP) return 0;
    CrownMap* m=mapv.as.map;
    for(int64_t i=0;i<m->size;i++){
        if(strcmp(m->entries[i].key,key)==0) return 1;
    }
    return 0;
}

// --------------------- Debug Printing ---------------------

void crown_debug_print(CrownValue v);

void crown_debug_print_array(CrownArray* arr) {
    printf("[");
    for(int64_t i=0;i<arr->size;i++){
        crown_debug_print(arr->data[i]);
        if(i+1<arr->size) printf(", ");
    }
    printf("]");
}

void crown_debug_print_map(CrownMap* m) {
    printf("{");
    for(int64_t i=0;i<m->size;i++){
        printf("\"%s\": ", m->entries[i].key);
        crown_debug_print(m->entries[i].value);
        if(i+1<m->size) printf(", ");
    }
    printf("}");
}

void crown_debug_print(CrownValue v) {
    switch(v.kind){
        case CROWN_VAL_INT: printf("%lld",(long long)v.as.i); break;
        case CROWN_VAL_STR: printf("\"%s\"",v.as.s); break;
        case CROWN_VAL_ARR: crown_debug_print_array(v.as.arr); break;
        case CROWN_VAL_MAP: crown_debug_print_map(v.as.map); break;
    }
}

// ============================================================
// Crown Runtime: Recursive JSON Printers
// ============================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Forward declarations for recursion
void crown_json_print_value(void *val, const char *type);

void crown_json_print_array(struct crown_array *arr) {
    printf("[");
    for (size_t i = 0; i < arr->len; i++) {
        crown_json_print_value(arr->items[i], arr->types[i]);
        if (i + 1 < arr->len) printf(",");
    }
    printf("]");
}

void crown_json_print_map(struct crown_map *map) {
    printf("{");
    for (size_t i = 0; i < map->len; i++) {
        printf("\"%s\":", map->keys[i]);
        crown_json_print_value(map->values[i], map->types[i]);
        if (i + 1 < map->len) printf(",");
    }
    printf("}");
}

static void crown_json_escape_and_print(const char *s) {
    printf("\"");
    for (const char *p = s; *p; p++) {
        switch (*p) {
            case '\"': printf("\\\""); break;
            case '\\': printf("\\\\"); break;
            case '\n': printf("\\n"); break;
            case '\r': printf("\\r"); break;
            case '\t': printf("\\t"); break;
            default: putchar(*p);
        }
    }
    printf("\"");
}

void crown_json_print_value(void *val, const char *type) {
    if (!val || !type) {
        printf("null");
        return;
    }
    if (strcmp(type, "i64") == 0) {
        printf("%lld", *(long long*)val);
    } else if (strcmp(type, "double") == 0) {
        printf("%g", *(double*)val);
    } else if (strcmp(type, "string") == 0) {
        crown_json_escape_and_print((const char*)val);
    } else if (strcmp(type, "array") == 0) {
        crown_json_print_array((struct crown_array*)val);
    } else if (strcmp(type, "map") == 0) {
        crown_json_print_map((struct crown_map*)val);
    } else {
        printf("\"<unknown>\"");
    }
}

struct crown_array {
    size_t len;
    void **items;
    const char **types; // type tag per item
};

struct crown_map {
    size_t len;
    char **keys;
    void **values;
    const char **types; // type tag per value
};

