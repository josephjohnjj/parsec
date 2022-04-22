set(PARSEC_VERSION 4.0.0)

# Required for check_language
include(CheckLanguage)


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was PaRSECConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set_and_check(PARSEC_DIR "${PACKAGE_PREFIX_DIR}")
set_and_check(PARSEC_INCLUDE_DIRS "/home/joseph/parsec")
set_and_check(PARSEC_CMAKE_DIRS "/home/joseph/parsec/cmake_modules")
set_and_check(PARSEC_LIBRARY_DIRS "${PACKAGE_PREFIX_DIR}/parsec")
set_and_check(PARSEC_BINARY_DIRS "${PACKAGE_PREFIX_DIR}")
set(PARSEC_LIBEXEC_DIRS "${PACKAGE_PREFIX_DIR}") # This is optional, may not exist in the installdir

# Pull the dependencies
list(APPEND CMAKE_PREFIX_PATH "${PARSEC_CMAKE_DIRS}")
list(APPEND CMAKE_MODULE_PATH "${PARSEC_CMAKE_DIRS}")

find_package(Threads)

if(TRUE)
  set_and_check(HWLOC_INCLUDE_DIR "/usr/include")
  set_and_check(HWLOC_LIBRARY "/usr/lib/x86_64-linux-gnu/libhwloc.so")
  find_package(HWLOC REQUIRED)
endif(TRUE)

if(FALSE)
  # Nothing exportable here, if this test succeed then PaRSEC supports OTF2 output.
  if( "" )
    set_and_check(OTF2_CONFIG_PATH "")
  elseif("")
    set_and_check(OTF2_DIR "")
  elseif( "" )
    cmake_path(GET "" PARENT_PATH OTF2_CONFIG_PATH_tmp)
    set_and_check(OTF2_CONFIG_PATH "${OTF2_CONFIG_PATH_tmp}")
    unset(OTF2_CONFIG_PATH_tmp)
  endif( "" )
  find_package(OTF2  REQUIRED)
endif(FALSE)

if(TRUE)
  set_and_check(PAPI_INCLUDE_DIR "/usr/local/include")
  set_and_check(PAPI_LIBRARY "/usr/local/lib/libpapi.so")
  find_package(PAPI REQUIRED)
endif(TRUE)

if(ON)
  # Try to find MPI::MPI_C
  if (NOT TARGET MPI::MPI_C)
    # ensure that language C is enabled
    check_language(C)
    if(CMAKE_C_COMPILER)
      enable_language(C)
    else()
      message(FATAL_ERROR "Cannot find package PaRSEC due to missing C language support; either enable_language(C) in your project or ensure that C compiler can be discovered")
    endif()
    find_package(MPI REQUIRED COMPONENTS C)
  endif(NOT TARGET MPI::MPI_C)
endif(ON)

if(FALSE)
  find_package(CUDAToolkit REQUIRED)
  SET(PARSEC_HAVE_CUDA TRUE)
endif(FALSE)

if(OFF)
  # Nothing exportable here, if this test succeed then PaRSEC supports tracing
endif(OFF)

# Pull the PaRSEC::<targets>
if(NOT TARGET PaRSEC::parsec)
  include(${CMAKE_CURRENT_LIST_DIR}/PaRSECTargets.cmake)
endif(NOT TARGET PaRSEC::parsec)

# Populate the variables

set(PARSEC_PTGFLAGS "$ENV{PTGFLAGS}" CACHE STRING "Flags to pass to the parsec-ptgpp executable")
set(PARSEC_PTGPP_EXECUTABLE ${PARSEC_BINARY_DIRS}/parsec-ptgpp CACHE STRING "Point to the parsec-ptgpp executable")
set(PARSEC_LIBRARIES PaRSEC::parsec CACHE STRING "List of libraries suitable for use in target_link_libraries") # for compatibility with older (non-target based) clients
