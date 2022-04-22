#ifndef PARSEC_CONFIG_H_HAS_BEEN_INCLUDED
#define PARSEC_CONFIG_H_HAS_BEEN_INCLUDED

/* Compiler dependent capabilities */
#define PARSEC_ATOMIC_USE_C11_ATOMICS
/* #undef PARSEC_ATOMIC_USE_GCC_32_BUILTINS */
/* #undef PARSEC_ATOMIC_USE_GCC_64_BUILTINS */
/* #undef PARSEC_ATOMIC_USE_GCC_128_BUILTINS */
/* #undef PARSEC_ATOMIC_USE_GCC_128_OTHER_BUILTINS */
/* #undef PARSEC_ATOMIC_USE_XLC_32_BUILTINS */
/* #undef PARSEC_ATOMIC_USE_XLC_64_BUILTINS */
/* #undef PARSEC_ATOMIC_USE_XLC_LLSC_32_BUILTINS */
/* #undef PARSEC_ATOMIC_USE_XLC_LLSC_64_BUILTINS */
/* #undef PARSEC_ATOMIC_USE_MIPOSPRO_32_BUILTINS */
/* #undef PARSEC_ATOMIC_USE_MIPOSPRO_64_BUILTINS */
/* #undef PARSEC_ATOMIC_USE_SUN_32 */
/* #undef PARSEC_ATOMIC_USE_SUN_64 */
/* #undef PARSEC_ARCH_X86 */
#define PARSEC_ARCH_X86_64
/* #undef PARSEC_ARCH_PPC */

#define PARSEC_HAVE_BUILTIN_EXPECT
#define PARSEC_HAVE_BUILTIN_CPU
#define PARSEC_HAVE_ATTRIBUTE_VISIBILITY
#define PARSEC_HAVE_ATTRIBUTE_ALWAYS_INLINE
#define PARSEC_HAVE_ATTRIBUTE_FORMAT_PRINTF
#define PARSEC_HAVE_ATTRIBUTE_DEPRECATED

#define PARSEC_HAVE_PTHREAD_BARRIER
/* #undef PARSEC_HAVE_PTHREAD_BARRIER_H */

#define PARSEC_HAVE_THREAD_LOCAL
#define PARSEC_HAVE_PTHREAD_GETSPECIFIC

/* Optional packages */
#define PARSEC_HAVE_HWLOC_BITMAP
#define PARSEC_HAVE_HWLOC_PARENT_MEMBER
#define PARSEC_HAVE_HWLOC_CACHE_ATTR
#define PARSEC_HAVE_HWLOC_OBJ_PU

/* #undef PARSEC_HAVE_RECENT_LEX */

#define PARSEC_PROFILING_USE_MMAP
#define PARSEC_PROFILING_USE_HELPER_THREAD

#define PARSEC_HAVE_VALGRIND_API

/* #undef PARSEC_HAVE_INDENT */
#define PARSEC_INDENT_PREFIX "INDENT_EXECUTABLE-NOTFOUND"
#define PARSEC_INDENT_OPTIONS "-nbad -bap -nbc -br -brs -ncdb -ce -cli0 -d0 -di1 -nfc1 -i4 -ip0 -lp -npcs -npsl -nsc -nsob -l120"

#define PARSEC_HAVE_AWK
#define PARSEC_AWK_PREFIX "/usr/bin/awk"

#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif  /* !defined(_GNU_SOURCE) */

#ifdef PARSEC_ARCH_PPC
#define inline __inline__
#define restrict
#endif

/* We undefined the PARSEC_CONFIG_H_HAS_BEEN_INCLUDED #define so that the parsec_options.h
 * can be loaded. This mechanism is only used durig the PaRSEC compilation, once installed
 * the parsec_options.h will become the new parsec_config.h.
 */
#undef PARSEC_CONFIG_H_HAS_BEEN_INCLUDED
#include "parsec/parsec_options.h"

#endif  /* PARSEC_CONFIG_H_HAS_BEEN_INCLUDED */
