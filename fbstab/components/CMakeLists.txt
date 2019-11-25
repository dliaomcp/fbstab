# libraries
add_library(dense_components dense_data.cc dense_variable.cc dense_residual.cc dense_feasibility.cc dense_linear_solver.cc)
target_include_directories(dense_components PUBLIC ${FBSTAB_ROOT})
target_compile_features(dense_components PUBLIC cxx_std_11)
set_target_properties(dense_components PROPERTIES CXX_EXTENSIONS OFF)

add_library(mpc_components mpc_data.cc mpc_variable.cc mpc_residual.cc mpc_feasibility.cc riccati_linear_solver.cc)
target_include_directories(mpc_components PUBLIC ${FBSTAB_ROOT})
target_compile_features(mpc_components PUBLIC cxx_std_11)
set_target_properties(mpc_components PROPERTIES CXX_EXTENSIONS OFF)