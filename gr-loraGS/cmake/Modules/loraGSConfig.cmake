INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_LORAGS loraGS)

FIND_PATH(
    LORAGS_INCLUDE_DIRS
    NAMES loraGS/api.h
    HINTS $ENV{LORAGS_DIR}/include
        ${PC_LORAGS_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    LORAGS_LIBRARIES
    NAMES gnuradio-loraGS
    HINTS $ENV{LORAGS_DIR}/lib
        ${PC_LORAGS_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/loraGSTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LORAGS DEFAULT_MSG LORAGS_LIBRARIES LORAGS_INCLUDE_DIRS)
MARK_AS_ADVANCED(LORAGS_LIBRARIES LORAGS_INCLUDE_DIRS)
