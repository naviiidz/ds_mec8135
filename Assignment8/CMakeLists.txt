add_executable(so2_average so2_average.cpp)

add_executable(se2_average se2_average.cpp)
add_executable(se2_interpolation se2_interpolation.cpp)
add_executable(se2_decasteljau se2_DeCasteljau.cpp)
add_executable(se2_localization se2_localization.cpp)
add_executable(se2_localization_ukfm se2_localization_ukfm.cpp)
add_executable(se2_sam se2_sam.cpp)
add_executable(se2_sam_gps se2_sam_gps.cpp)

add_executable(se3_localization se3_localization.cpp)
add_executable(se3_sam se3_sam.cpp)
add_executable(se3_sam_selfcalib se3_sam_selfcalib.cpp)

add_executable(se_2_3_localization se_2_3_localization.cpp)

set(CXX_11_EXAMPLE_TARGETS

  # SO2
  so2_average

  # SE2
  se2_interpolation
  se2_decasteljau
  se2_average
  se2_localization
  se2_sam
  se2_sam_gps

  se2_localization_ukfm

  # SE3
  se3_localization
  se3_sam
  se3_sam_selfcalib

  # SE_2_3
  se_2_3_localization
)

if(NOT MSVC)
  add_executable(bundle_sam bundle_sam.cpp)

  set(CXX_11_EXAMPLE_TARGETS

    ${CXX_11_EXAMPLE_TARGETS}

    # Bundle
    bundle_sam
  )
endif()

# Link to manif
foreach(target ${CXX_11_EXAMPLE_TARGETS})
  target_link_libraries(${target} ${PROJECT_NAME})

  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # GCC is not strict enough by default, so enable most of the warnings.
    target_compile_options(${target} PRIVATE
      -Werror=all
      -Werror=extra
    )
  endif()

endforeach()

# Set required C++11 flag
set_property(TARGET ${CXX_11_EXAMPLE_TARGETS} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${CXX_11_EXAMPLE_TARGETS} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${CXX_11_EXAMPLE_TARGETS} PROPERTY CXX_EXTENSIONS OFF)
