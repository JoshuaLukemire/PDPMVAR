test_that("compiled coefficient rows match R with gaps, unequal lengths and heterogeneous variances", {
  set.seed(111)
  for (dimensions in list(c(1, 1), c(3, 2))) {
    n_nodes <- dimensions[1]
    K <- dimensions[2]
    n_predictors <- n_nodes * K
    X <- lapply(c(8, 9, 11, 10), function(n) matrix(rnorm(n_nodes * n), n_nodes))
    X[[1]] <- matrix(seq_len(n_nodes * 8), n_nodes)
    design <- lapply(X, .pdpmvar_lag_design, K = K)
    contributions <- lapply(X, function(x) matrix(rnorm(length(x)), nrow(x)))
    variance <- matrix(seq(0.1, 0.8, length.out = n_nodes * n_predictors), n_nodes)
    residual_variances <- matrix(runif(n_nodes * 4, 0.4, 2), n_nodes)
    atoms <- lapply(seq_len(5), function(i) .pdpmvar_test_reference$.pdpmvar_new_coefficient(variance))
    original <- atoms
    set.seed(112)
    reference <- .pdpmvar_test_reference$.pdpmvar_update_coefficients(X, design, lapply(design, tcrossprod), contributions,
      c(4, 1, 4, 1), atoms, variance, residual_variances)
    state <- .Random.seed
    set.seed(112)
    compiled <- .pdpmvar_test_reference$.pdpmvar_update_coefficients(X, design, lapply(design, tcrossprod), contributions,
      c(4, 1, 4, 1), atoms, variance, residual_variances, backend = "RcppEigen")
    expect_equal(compiled, reference, tolerance = 1e-10)
    expect_identical(.Random.seed, state)
    expect_identical(atoms, original)
    expect_identical(compiled[c(2, 3, 5)], atoms[c(2, 3, 5)])
  }
  expect_error(.pdpmvar_test_reference$.pdpmvar_update_coefficients(X, design, lapply(design, tcrossprod), contributions,
    c(0.5, 1, 4, 1), atoms, variance, residual_variances, backend = "RcppEigen"), "integer atom")
  expect_error(.pdpmvar_test_reference$.pdpmvar_update_coefficients(X, design, lapply(design, tcrossprod), contributions,
    c(4, 1, 4, 1), atoms, variance * 0, residual_variances, backend = "RcppEigen"), "positive")
})

test_that("compiled rows use the analytic weighted and factor-adjusted conditional", {
  X <- list(matrix(c(1, 2, -1, 3), 1), matrix(c(-1, 0, 2, 1, -2), 1))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  contributions <- list(matrix(c(0.2, 0.1, 0.3, -0.2), 1), matrix(c(0.1, -0.2, 0, 0.3, 0.2), 1))
  variance <- matrix(c(0.5, 2), 1)
  residual_variances <- matrix(c(0.7, 1.5), 1)
  precision <- diag(c(2, 0.5)) + tcrossprod(design[[1]]) / 0.7 + tcrossprod(design[[2]]) / 1.5
  information <- design[[1]] %*% as.numeric(X[[1]] - contributions[[1]]) / 0.7 +
    design[[2]] %*% as.numeric(X[[2]] - contributions[[2]]) / 1.5
  set.seed(113)
  expected <- as.numeric(solve(precision, information) + backsolve(chol(precision), rnorm(2)))
  set.seed(113)
  compiled <- .pdpmvar_test_reference$.pdpmvar_update_coefficients(X, design, lapply(design, tcrossprod), contributions,
    c(1, 1), list(list(coefficient = matrix(0, 1, 2))), variance, residual_variances, backend = "RcppEigen")
  expect_equal(as.numeric(compiled[[1]]$coefficient), expected, tolerance = 1e-12)
})
