# For now assume that the upper level did the CUDA search and that all
# necessary dependencies on CUDA have been correctly setup

if( PARSEC_HAVE_CUDA )
  SET(MCA_${COMPONENT}_${MODULE} ON)
  FILE(GLOB MCA_${COMPONENT}_${MODULE}_SOURCES ${MCA_BASE_DIR}/${COMPONENT}/${MODULE}/[^\\.]*.c)
  SET(MCA_${COMPONENT}_${MODULE}_CONSTRUCTOR "${COMPONENT}_${MODULE}_static_component")
  set_property(TARGET parsec
               APPEND PROPERTY
                      PUBLIC_HEADER_H mca/device/cuda/device_cuda.h
                                      mca/device/cuda/device_cuda_internal.h)
else (PARSEC_HAVE_CUDA)
  MESSAGE(STATUS "Module ${MODULE} not selectable: does not have CUDA")
  SET(MCA_${COMPONENT}_${MODULE} OFF)
endif(PARSEC_HAVE_CUDA)
