#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

#include "jdf.h"

extern const char *yyfilename;

static FILE *cfile;
static int   cfile_lineno;
static FILE *hfile;
static int   hfile_lineno;
static char *basename;

static int nblines(const char *p)
{
    int r = 0;
    for(; *p != '\0'; p++)
        if( *p == '\n' )
            r++;
    return r;
}

#if defined(__GNUC__)
static void coutput(const char *format, ...) __attribute__((format(printf,1,2)));
#endif
static void coutput(const char *format, ...)
{
    va_list ap;
    char *res;
    int len;

    va_start(ap, format);
    len = vasprintf(&res, format, ap);
    va_end(ap);

    if( len == -1 ) {
        fprintf(stderr, "Unable to ouptut a string: %s\n", strerror(errno));
    } else {
        fwrite(res, len, 1, cfile);
        cfile_lineno += nblines(res);
        free(res);
    }
}

#if defined(__GNUC__)
static void houtput(const char *format, ...) __attribute__((format(printf,1,2)));
#endif
static void houtput(const char *format, ...)
{
    va_list ap;
    char *res;
    int len;

    va_start(ap, format);
    len = vasprintf(&res, format, ap);
    va_end(ap);

    if( len == -1 ) {
        fprintf(stderr, "Unable to ouptut a string: %s\n", strerror(errno));
    } else {
        fwrite(res, len, 1, hfile);
        hfile_lineno += nblines(res);
        free(res);
    }
}

/** UTIL HELPERS **/

typedef char *(*dumper_function_t)(void *elt);

static char *dumpstring(void *elt)
{
    return (char*)elt;
}

typedef struct string_arena {
    char *ptr;
    int   pos;
    int   size;
} string_arena_t;

static string_arena_t *string_arena_new(int base_size)
{
    string_arena_t *sa;
    sa = (string_arena_t*)calloc(1, sizeof(string_arena_t));
    sa->ptr  = (char*)malloc(base_size);
    sa->pos  = 0;
    sa->size = base_size;
    return sa;
}

static void string_arena_free(string_arena_t *sa)
{
    free(sa->ptr);
    sa->pos  = -1;
    sa->size = -1;
    free(sa);
}

static void string_arena_ensure_space(string_arena_t *sa, int toadd)
{
    if( sa->pos + toadd > sa->size ) {
        sa->size = sa->pos + toadd;
        sa->ptr = (char*)realloc(sa->ptr, sa->size);
    }
}

static void string_arena_add_string(string_arena_t *sa, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    sa->pos += vsnprintf(sa->ptr + sa->pos, sa->size - sa->pos, format, ap);
    va_end(ap);
}

/**
 * util_dump_list:
 *  @param [IN] structure_ptr: pointer to a structure that implement any list
 *  @param [IN] nextfield:     the name of a field pointing to the next structure pointer
 *  @param [IN] eltfield:      the name of a field pointing to an element to print
 *  @param [IN] before:        string (of characters) representing what must appear before the list
 *  @param [IN] prefix:        string (of characters) representing what must appear before each element
 *  @param [IN] fct:           a function that transforms a pointer to an element to a string of characters
 *  @param [IN] separator:     string (of characters) that will be put between each element, but not at the end 
 *                             or before the first
 *  @param [IN] after:         string (of characters) that will be put at the end of the list, after the last
 *                             element
 *
 *  @return a string (of characters) with the list formed so. This string is useable until the next
 *          call to UTIL_DUMP_LIST
 *
 *  @example: to create the list of expressions that is a parameter call, use
 *    UTIL_DUMP_LIST(jdf->functions->predicates, next, expr, "(", "", dump_expr_inline, ", ", ")")
 *  @example: to create the list of declarations of globals, use
 *    UTIL_DUMP_LIST(jdf->globals, next, name, "", "  int ", dumpstring, ";\n", ";\n");
 */
#define UTIL_DUMP_LIST(arena, structure_ptr, nextfield, eltfield, before, prefix, fct, separator, after) \
    util_dump_list_fct( arena, structure_ptr,                           \
                        (char *)&(structure_ptr->nextfield)-(char *)structure_ptr, \
                        (char *)&(structure_ptr->eltfield)-(char *)structure_ptr, \
                        before, prefix, fct, separator, after)
static char *util_dump_list_fct( string_arena_t *sa, 
                                 void *firstelt, unsigned int next_offset, unsigned int elt_offset, 
                                 const char *before, const char *prefix, dumper_function_t fct, 
                                 const char *separator, const char *after)
{
    char *eltstr;
    void *elt;
    int pos = 0;
    
    string_arena_ensure_space(sa, strlen(before)+1);
    string_arena_add_string(sa, "%s", before);

    while(firstelt != NULL) {
        elt = *((void **)((char*)(firstelt) + elt_offset));
        eltstr = fct(elt);

        firstelt = *((void **)((char *)(firstelt) + next_offset));
        if( firstelt != NULL ) {
            string_arena_ensure_space(sa, strlen(eltstr) + strlen(separator) + strlen(prefix) + 1);
            string_arena_add_string(sa, "%s%s%s", prefix, eltstr, separator);
        } else {
            string_arena_ensure_space(sa, strlen(prefix) + strlen(prefix) + 1);
            string_arena_add_string(sa, "%s%s%s", prefix, eltstr);
        }
    }
    
    string_arena_ensure_space(sa, strlen(after) + 1);
    string_arena_add_string(sa, "%s", after);

    return sa->ptr;
}

static void typedef_structure(const jdf_t *jdf)
{
    jdf_global_entry_t *g;
    jdf_function_entry_t *f;
    int nbfunctions;
    string_arena_t *sa1, *sa2;

    nbfunctions = 0;
    for(f = jdf->functions; f != NULL; f = f->next)
        nbfunctions++;

    sa1 = string_arena_new(64);
    sa2 = string_arena_new(64);

    houtput("#include <dplasma.h>\n"
            "\n"
            "#define DPLASMA_%s_NB_FUNCTIONS %d\n", basename, nbfunctions);
    houtput("typedef struct dplasma_%s {\n", basename);
    houtput("  const dplasma_t *functions_array[DPLASMA_%s_NB_FUNCTIONS];\n", basename);
    houtput("%s", 
            UTIL_DUMP_LIST( sa1, jdf->globals, next, name, "", "  int ", dumpstring, ";\n", ";\n"));
    houtput("#  if defined(DPLASMA_PROFILING)\n");
    houtput("%s", 
            UTIL_DUMP_LIST( sa1, jdf->functions, next, fname, "", "  int ", dumpstring, "_start_key;\n", "_start_key;\n"));
    houtput("%s", 
            UTIL_DUMP_LIST( sa1, jdf->functions, next, fname, "", "  int ", dumpstring, "_end_key;\n", "_end_key;\n"));
    houtput("#  endif /* defined(DPLASMA_PROFILING) */\n");
    houtput("} dplasma_%s_t;\n"
            "\n", basename);
    houtput("dplasma_object_t *dplasma_%s_new(%s);\n", basename, basename,
            UTIL_DUMP_LIST( sa1, jdf->globals, next, name, "",  "int ", dumpstring, ", ", ""));

    string_arena_free(sa1);
    string_arena_free(sa2);
}

int jdf2c(char *_basename, const jdf_t *jdf)
{
    char filename[strlen(_basename)+4];
    int ret = 0;

    basename = _basename;
    cfile = NULL;
    hfile = NULL;

    sprintf(filename, "%s.c", basename);
    cfile = fopen(filename, "w");
    if( cfile == NULL ) {
        fprintf(stderr, "unable to create %s: %s\n", filename, strerror(errno));
        ret = -1;
        goto err;
    }

    sprintf(filename, "%s.h", basename);
    hfile = fopen(filename, "w");
    if( hfile == NULL ) {
        fprintf(stderr, "unable to create %s: %s\n", filename, strerror(errno));
        ret = -1;
        goto err;
    }

    cfile_lineno = 1;
    hfile_lineno = 1;
    
    houtput("#ifndef _%s_h_\n"
            "#define _%s_h_\n",
            basename, basename);
    typedef_structure(jdf);
    houtput("#endif /* _%s_h_ */ \n",
            basename);

 err:
    if( NULL != cfile ) 
        fclose(cfile);

    if( NULL != hfile )
        fclose(hfile);

    return ret;
}
