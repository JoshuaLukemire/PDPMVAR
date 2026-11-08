# Exercise the R reference engine through identical input preparation in validation tests.
# The public fitting function always calls the native engine outside this scoped test mock.
pdpmvar_reference_for_test <- function(...) {
  testthat::local_mocked_bindings(
    .pdpmvar_eigen_sampler = function(X, design, design_cross, state, control, callbacks) {
      control$backend <- "R"
      return(.pdpmvar_test_reference$.pdpmvar_sampler_reference(X, design, design_cross, state, control))
    },
    .package = "PDPMVAR"
  )

  return(pdpmvar(...))
}
