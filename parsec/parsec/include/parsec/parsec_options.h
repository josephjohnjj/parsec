#ifndef PARSEC_CONFIG_H_HAS_BEEN_INCLUDED
#define PARSEC_CONFIG_H_HAS_BEEN_INCLUDED

/* This file contains the OS dependent capabilities, and should be generic for
 * all compilers o a particular architecture. It is used during the PaRSEC build to
 * import all OS dependent features, but once PaRSEC installed this file will
 * become the parsec_config.h and will hide all compiler dependent features used
 * during PaRSEC compilation.
 */
/** @brief Define the compilation date of the runtime */
#define PARSEC_COMPILE_DATE "2022-03-30T09:11:36"
/** @brief Define the PaRSEC major version number */
#define PARSEC_VERSION_MAJOR 4
/** @brief Define the PaRSEC minor version number */
#define PARSEC_VERSION_MINOR 0
/** @brief Define the PaRSEC patch version number */
#define PARSEC_VERSION_RELEASE 0
/** @brief Define the branch that was compiled */
#define PARSEC_GIT_BRANCH "gpu_migrate"
/** @brief Define the commit hash that was compiled */
#define PARSEC_GIT_HASH "804e58084"
/** @brief Define the changes to the commit hash that was compiled */
#define PARSEC_GIT_DIRTY "2 files changed, 4 insertions(+)"
/** @brief Define the commit date of the runtime */
#define PARSEC_GIT_DATE "2022-03-28T10:52:17-04:00"

/* OS dependent capabilities */
#define PARSEC_HAVE_PTHREAD
#define PARSEC_HAVE_SCHED_SETAFFINITY
#define PARSEC_HAVE_CLOCK_GETTIME
#define PARSEC_HAVE_ASPRINTF
#define PARSEC_HAVE_VASPRINTF
#define PARSEC_HAVE_RAND_R
#define PARSEC_HAVE_RANDOM
#define PARSEC_HAVE_ERAND48
#define PARSEC_HAVE_NRAND48
#define PARSEC_HAVE_LRAND48
#define PARSEC_HAVE_GETLINE
#define PARSEC_HAVE_SETENV
#define PARSEC_HAVE_STDARG_H
#define PARSEC_HAVE_UNISTD_H
#define PARSEC_HAVE_SYS_PARAM_H
#define PARSEC_HAVE_SYS_TYPES_H
#define PARSEC_HAVE_SYSLOG_H
#define PARSEC_HAVE_VA_COPY
/* #undef PARSEC_HAVE_UNDERSCORE_VA_COPY */
#define PARSEC_HAVE_GETOPT_LONG
#define PARSEC_HAVE_GETRUSAGE
#define PARSEC_HAVE_RUSAGE_THREAD
#define PARSEC_HAVE_GETOPT_H
#define PARSEC_HAVE_ERRNO_H
#define PARSEC_HAVE_STDDEF_H
#define PARSEC_HAVE_STDBOOL_H
#define PARSEC_HAVE_CTYPE_H
#define PARSEC_HAVE_LIMITS_H
#define PARSEC_HAVE_STRING_H
#define PARSEC_HAVE_GEN_H
#define PARSEC_HAVE_COMPLEX_H
#define PARSEC_HAVE_EXECINFO_H
#define PARSEC_HAVE_SYS_MMAN_H
#define PARSEC_HAVE_DLFCN_H
#define PARSEC_HAVE_SYSCONF
#define PARSEC_HAVE_ATTRIBUTE_DEPRECATED

/* Compiler Specific Options */
#define PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128
#define PARSEC_HAVE_INT128

/* Scheduling engine */
#define PARSEC_SCHED_DEPS_MASK

/* Communication engine */
#define PARSEC_DIST_WITH_MPI
#define PARSEC_DIST_THREAD
#define PARSEC_DIST_PRIORITIES
#define PARSEC_DIST_COLLECTIVES
#define PARSEC_DIST_SHORT_LIMIT 1

/* GPU Support */
#define PARSEC_GPU_WITH_CUDA
/* #undef PARSEC_GPU_CUDA_ALLOC_PER_TILE */
/* #undef PARSEC_GPU_WITH_OPENCL */
/* #undef PARSEC_HAVE_PEER_DEVICE_MEMORY_ACCESS */

/* debug */
/* #undef PARSEC_DEBUG */
/* #undef PARSEC_DEBUG_PARANOID */
/* #undef PARSEC_DEBUG_NOISIER */
/* #undef PARSEC_DEBUG_HISTORY */
/* #undef PARSEC_LIFO_USE_ATOMICS */

/* profiling */
/* #undef PARSEC_PROF_TRACE */
/* #undef PARSEC_PROF_TRACE_PTG_INTERNAL_INIT */
/* #undef PARSEC_PROF_RUSAGE_EU */
/* #undef PARSEC_PROF_TRACE_SCHEDULING_EVENTS */
/* #undef PARSEC_PROF_TRACE_ACTIVE_ARENA_SET */
/* #undef PARSEC_PROF_GRAPHER */
/* #undef PARSEC_PROF_DRY_RUN */
/* #undef PARSEC_PROF_DRY_BODY */
/* #undef PARSEC_PROF_DRY_DEP */

/* Software Defined Events through PAPI-SDE */
/* #undef PARSEC_PAPI_SDE */

/* Instrumenting (PINS) */
#define PARSEC_PROF_PINS

/* Simulating */
/* #undef PARSEC_SIM */

/* Configuration parameters */
#define PARSEC_WANT_HOME_CONFIG_FILES

/* Compiler and flags used to compile PaRSEC generated sources */
#define CMAKE_PARSEC_C_COMPILER   "/usr/bin/cc"
#define CMAKE_PARSEC_C_FLAGS      ""
#define CMAKE_PARSEC_C_INCLUDES   "/home/joseph/parsec/build/install/include;/usr/include;/usr/local/include"

#define PARSEC_HAVE_HWLOC
#define PARSEC_HAVE_PAPI
/* #undef PARSEC_HAVE_CUDA */
/* #undef PARSEC_HAVE_OPENCL */
#define PARSEC_HAVE_MPI
#define PARSEC_HAVE_MPI_20
#define PARSEC_HAVE_MPI_30
#define PARSEC_HAVE_MPI_OVERTAKE
/* #undef PARSEC_HAVE_AYUDAME */

#define PARSEC_INSTALL_PREFIX "/home/joseph/parsec/build/install"
/* Default PATH to look for the CUDA .so files */
#define PARSEC_LIB_CUDA_PREFIX "."
#define PARSEC_LIB_LEVE_ZERO_PREFIX "."

#define PARSEC_SIZEOF_VOID_P 8

#include "parsec/parsec_config_bottom.h"

#endif  /* PARSEC_CONFIG_H_HAS_BEEN_INCLUDED */

