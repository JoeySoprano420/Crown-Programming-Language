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

// ============================================================
// Crown Runtime: JSON Parser (string -> arrays/maps)
// ============================================================

#include <ctype.h>

// Forward declarations
void* crown_json_parse_value(const char **s, const char **type);

// Skip whitespace
static void skip_ws(const char **s) {
    while (isspace(**s)) (*s)++;
}

// Parse string literal
static char* parse_string(const char **s) {
    (*s)++; // skip opening "
    const char *start = *s;
    size_t cap = 16, len = 0;
    char *buf = malloc(cap);
    while (**s && **s != '"') {
        char c = **s;
        if (c == '\\') {
            (*s)++;
            if (**s == 'n') c = '\n';
            else if (**s == 't') c = '\t';
            else if (**s == 'r') c = '\r';
            else c = **s;
        }
        if (len+1 >= cap) { cap*=2; buf=realloc(buf,cap); }
        buf[len++] = c;
        (*s)++;
    }
    buf[len] = '\0';
    if (**s == '"') (*s)++;
    return buf;
}

// Parse number
static long long parse_number(const char **s) {
    long long val = 0;
    int sign = 1;
    if (**s == '-') { sign=-1; (*s)++; }
    while (isdigit(**s)) {
        val = val*10 + (**s - '0');
        (*s)++;
    }
    return val*sign;
}

void* crown_json_parse_array(const char **s, const char **type) {
    (*s)++; // skip [
    struct crown_array *arr = malloc(sizeof(struct crown_array));
    arr->len=0; arr->items=NULL; arr->types=NULL;
    skip_ws(s);
    while (**s && **s != ']') {
        const char *elem_type=NULL;
        void *elem = crown_json_parse_value(s,&elem_type);
        arr->len++;
        arr->items = realloc(arr->items, arr->len*sizeof(void*));
        arr->types = realloc(arr->types, arr->len*sizeof(char*));
        arr->items[arr->len-1]=elem;
        arr->types[arr->len-1]=elem_type;
        skip_ws(s);
        if (**s == ',') { (*s)++; skip_ws(s); }
    }
    if (**s == ']') (*s)++;
    *type="array";
    return arr;
}

void* crown_json_parse_map(const char **s, const char **type) {
    (*s)++; // skip {
    struct crown_map *map = malloc(sizeof(struct crown_map));
    map->len=0; map->keys=NULL; map->values=NULL; map->types=NULL;
    skip_ws(s);
    while (**s && **s != '}') {
        char *key=parse_string(s);
        skip_ws(s); if (**s==':') (*s)++;
        skip_ws(s);
        const char *val_type=NULL;
        void *val=crown_json_parse_value(s,&val_type);
        map->len++;
        map->keys=realloc(map->keys,map->len*sizeof(char*));
        map->values=realloc(map->values,map->len*sizeof(void*));
        map->types=realloc(map->types,map->len*sizeof(char*));
        map->keys[map->len-1]=key;
        map->values[map->len-1]=val;
        map->types[map->len-1]=val_type;
        skip_ws(s);
        if (**s==',') { (*s)++; skip_ws(s); }
    }
    if (**s=='}') (*s)++;
    *type="map";
    return map;
}

void* crown_json_parse_value(const char **s, const char **type) {
    skip_ws(s);
    if (**s == '"') {
        *type="string";
        return parse_string(s);
    }
    if (**s == '{') {
        return crown_json_parse_map(s,type);
    }
    if (**s == '[') {
        return crown_json_parse_array(s,type);
    }
    if (isdigit(**s) || **s=='-') {
        *type="i64";
        long long *n=malloc(sizeof(long long));
        *n=parse_number(s);
        return n;
    }
    if (strncmp(*s,"true",4)==0) {
        *type="i64"; long long *n=malloc(sizeof(long long)); *n=1;
        *s+=4; return n;
    }
    if (strncmp(*s,"false",5)==0) {
        *type="i64"; long long *n=malloc(sizeof(long long)); *n=0;
        *s+=5; return n;
    }
    if (strncmp(*s,"null",4)==0) {
        *type="null"; *s+=4; return NULL;
    }
    *type="unknown";
    return NULL;
}

// Public entry point
void* crown_json_parse(const char *src, const char **type) {
    return crown_json_parse_value(&src,type);
}

make pets = readjson("{\"pets\": [\"dog\",\"cat\"]}")
say pets

// ============================================================
// Crown Runtime: JSON Writing to File
// ============================================================

#include <errno.h>

// Recursively print JSON to file
static void crown_json_fprint_value(FILE *f, void *val, const char *type);

static void crown_json_fprint_string(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') {
            fputc('\\', f);
            fputc(*s, f);
        } else if (*s == '\n') {
            fputs("\\n", f);
        } else if (*s == '\t') {
            fputs("\\t", f);
        } else {
            fputc(*s, f);
        }
    }
    fputc('"', f);
}

static void crown_json_fprint_array(FILE *f, struct crown_array *arr) {
    fputc('[', f);
    for (size_t i=0; i<arr->len; i++) {
        if (i>0) fputc(',', f);
        crown_json_fprint_value(f, arr->items[i], arr->types[i]);
    }
    fputc(']', f);
}

static void crown_json_fprint_map(FILE *f, struct crown_map *map) {
    fputc('{', f);
    for (size_t i=0; i<map->len; i++) {
        if (i>0) fputc(',', f);
        crown_json_fprint_string(f, map->keys[i]);
        fputc(':', f);
        crown_json_fprint_value(f, map->values[i], map->types[i]);
    }
    fputc('}', f);
}

static void crown_json_fprint_value(FILE *f, void *val, const char *type) {
    if (!val) { fputs("null", f); return; }
    if (strcmp(type,"string")==0) {
        crown_json_fprint_string(f, (char*)val);
    } else if (strcmp(type,"i64")==0) {
        fprintf(f, "%lld", *(long long*)val);
    } else if (strcmp(type,"array")==0) {
        crown_json_fprint_array(f, (struct crown_array*)val);
    } else if (strcmp(type,"map")==0) {
        crown_json_fprint_map(f, (struct crown_map*)val);
    } else if (strcmp(type,"null")==0) {
        fputs("null", f);
    } else {
        fputs("\"<unknown>\"", f);
    }
}

// Public entry point
int crown_json_write_file(const char *filename, void *val, const char *type) {
    FILE *f = fopen(filename, "w");
    if (!f) return errno;
    crown_json_fprint_value(f, val, type);
    fclose(f);
    return 0; // success
}

// ============================================================
// JSON Read from File
// ============================================================
#include <ctype.h>

static char* crown_read_file_to_str(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    fseek(f,0,SEEK_END);
    long sz = ftell(f);
    fseek(f,0,SEEK_SET);
    char *buf = (char*)malloc(sz+1);
    fread(buf,1,sz,f);
    buf[sz]=0;
    fclose(f);
    return buf;
}

// Recursive descent parser helpers (reuse earlier JSON parser)
void* crown_json_parse(const char *text); // assume we already defined earlier

// Entry: read JSON file into runtime value
void* crown_json_read_file(const char *filename) {
    char *src = crown_read_file_to_str(filename);
    if (!src) return NULL;
    void *val = crown_json_parse(src);
    free(src);
    return val;
}

// ============================================================
// JSON Runtime Integration
// ============================================================

extern void* crown_json_parse(const char* text);

void* crown_json_read_file(const char* filename) {
    FILE* f = fopen(filename,"rb");
    if(!f) return NULL;
    fseek(f,0,SEEK_END);
    long sz=ftell(f);
    fseek(f,0,SEEK_SET);
    char* buf=(char*)malloc(sz+1);
    fread(buf,1,sz,f);
    buf[sz]=0;
    fclose(f);
    void* val = crown_json_parse(buf);
    free(buf);
    return val;
}

int crown_json_write_file(const char* filename, void* value) {
    FILE* f=fopen(filename,"wb");
    if(!f) return -1;
    char* str = crown_json_stringify(value); // assume implemented earlier
    fputs(str,f);
    fclose(f);
    free(str);
    return 0;
}

// ============================================================
// JSON RUNTIME IMPLEMENTATION
// ============================================================

#include <string.h>
#include <ctype.h>

// Forward declarations from crown_runtime.c
extern void* crown_array_new(int n);
extern void  crown_array_set(void* arr, int idx, void* val);
extern void* crown_map_new();
extern void  crown_map_set(void* map, const char* key, void* val);
extern char* crown_json_stringify(void* value);

// --------- JSON Parser Helpers ---------

static const char* skip_ws(const char* s) {
    while (*s && isspace((unsigned char)*s)) s++;
    return s;
}

static void* parse_value(const char** sp);

static void* parse_array(const char** sp) {
    const char* s = skip_ws(*sp);
    if (*s != '[') return NULL;
    s++;
    void* arr = crown_array_new(0);
    int idx=0;
    while (1) {
        s = skip_ws(s);
        if (*s == ']') { s++; break; }
        void* val = parse_value(&s);
        crown_array_set(arr, idx++, val);
        s = skip_ws(s);
        if (*s == ',') { s++; continue; }
        if (*s == ']') { s++; break; }
    }
    *sp = s;
    return arr;
}

static void* parse_string(const char** sp) {
    const char* s = *sp;
    if (*s != '"') return NULL;
    s++;
    const char* start = s;
    while (*s && *s != '"') s++;
    size_t len = s - start;
    char* out = (char*)malloc(len+1);
    memcpy(out, start, len);
    out[len]=0;
    if (*s=='"') s++;
    *sp = s;
    return out;
}

static void* parse_object(const char** sp) {
    const char* s = skip_ws(*sp);
    if (*s != '{') return NULL;
    s++;
    void* map = crown_map_new();
    while (1) {
        s = skip_ws(s);
        if (*s == '}') { s++; break; }
        void* key = parse_string(&s);
        s = skip_ws(s);
        if (*s == ':') s++;
        void* val = parse_value(&s);
        crown_map_set(map, (const char*)key, val);
        free(key);
        s = skip_ws(s);
        if (*s == ',') { s++; continue; }
        if (*s == '}') { s++; break; }
    }
    *sp = s;
    return map;
}

static void* parse_value(const char** sp) {
    const char* s = skip_ws(*sp);
    void* val=NULL;
    if (*s == '"') { val = parse_string(&s); }
    else if (*s == '{') { val = parse_object(&s); }
    else if (*s == '[') { val = parse_array(&s); }
    else if (isdigit((unsigned char)*s) || *s=='-') {
        char* end;
        long v = strtol(s,(char**)&end,10);
        long* num=(long*)malloc(sizeof(long));
        *num=v; val=num; s=end;
    } else if (!strncmp(s,"true",4)) { int* b=(int*)malloc(sizeof(int));*b=1;val=b;s+=4;}
    else if (!strncmp(s,"false",5)){int* b=(int*)malloc(sizeof(int));*b=0;val=b;s+=5;}
    else if (!strncmp(s,"null",4)) { val=NULL; s+=4; }
    *sp = s;
    return val;
}

// --------- Public Entry Points ---------

void* crown_json_parse(const char* text) {
    const char* s=text;
    return parse_value(&s);
}

void* crown_json_read_file(const char* filename) {
    FILE* f=fopen(filename,"rb");
    if(!f) return NULL;
    fseek(f,0,SEEK_END); long sz=ftell(f);
    fseek(f,0,SEEK_SET);
    char* buf=(char*)malloc(sz+1);
    fread(buf,1,sz,f);
    buf[sz]=0;
    fclose(f);
    void* val = crown_json_parse(buf);
    free(buf);
    return val;
}

int crown_json_write_file(const char* filename, void* value) {
    FILE* f=fopen(filename,"wb");
    if(!f) return -1;
    char* str = crown_json_stringify(value);
    fputs(str,f);
    fclose(f);
    free(str);
    return 0;
}

