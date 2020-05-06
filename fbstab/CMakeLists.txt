add_subdirectory(components)

# libraries
add_library(fbstab_dense fbstab_dense.cc fbstab_algorithm.h)
target_include_directories(fbstab_dense PUBLIC ${FBSTAB_ROOT})
target_link_libraries(fbstab_dense PUBLIC dense_components)
target_compile_features(fbstab_dense PUBLIC cxx_std_11)
set_target_properties(fbstab_dense PROPERTIES CXX_EXTENSIONS OFF)

add_library(fbstab_mpc fbstab_mpc.cc fbstab_algorithm.h)
target_include_directories(fbstab_mpc PUBLIC ${FBSTAB_ROOT})
target_link_libraries(fbstab_mpc PUBLIC mpc_components)
target_compile_features(fbstab_mpc PUBLIC cxx_std_11)
set_target_properties(fbstab_mpc PROPERTIES CXX_EXTENSIONS OFF)

add_library(ocp_generator test/ocp_generator.cc)
target_include_directories(ocp_generator PUBLIC ${FBSTAB_ROOT})
target_compile_features(ocp_generator PUBLIC cxx_std_11)
set_target_properties(ocp_generator PROPERTIES CXX_EXTENSIONS OFF)

# tests
# Testing is disabled by default if fbstab is being included in another project.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
	add_executable(dense_unit_tests test/fbstab_dense_unit_tests.cc)
	target_include_directories(dense_unit_tests PUBLIC ${FBSTAB_ROOT})
	target_link_libraries(dense_unit_tests PUBLIC fbstab_dense gtest_main)
	target_compile_features(dense_unit_tests PUBLIC cxx_std_11)
	set_target_properties(dense_unit_tests PROPERTIES CXX_EXTENSIONS OFF)

	add_executable(mpc_unit_tests test/fbstab_mpc_unit_tests.cc)
	target_include_directories(mpc_unit_tests PUBLIC ${FBSTAB_ROOT})
	target_link_libraries(mpc_unit_tests PUBLIC fbstab_mpc ocp_generator gtest_main)
	target_compile_features(mpc_unit_tests PUBLIC cxx_std_11)
	set_target_properties(mpc_unit_tests PROPERTIES CXX_EXTENSIONS OFF)


	add_test(dense_unit_tests dense_unit_tests)
	add_test(mpc_unit_tests mpc_unit_tests)
endif()

