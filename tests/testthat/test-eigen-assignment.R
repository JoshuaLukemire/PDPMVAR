test_that("compiled likelihoods and scatter scores match R normalization", {
  set.seed(121)
  residuals <- list(matrix(rnorm(14), 2), matrix(rnorm(22), 2))
  covariance <- matrix(c(2, -0.4, -0.4, 1), 2)
  for (residual in residuals) {
    expect_equal(.pdpmvar_test_reference$.pdpmvar_log_likelihood(residual, covariance, backend = "RcppEigen"),
      .pdpmvar_test_reference$.pdpmvar_log_likelihood(residual, covariance), tolerance = 1e-11)
  }
  expect_equal(.pdpmvar_test_reference$.pdpmvar_covariance_scores(lapply(residuals, tcrossprod), c(7, 11),
    list(covariance = covariance), backend = "RcppEigen"),
    .pdpmvar_test_reference$.pdpmvar_covariance_scores(lapply(residuals, tcrossprod), c(7, 11), list(covariance = covariance)), tolerance = 1e-11)
  atoms <- list(list(covariance = covariance), list(covariance = diag(2)))
  expect_equal(.pdpmvar_test_reference$.pdpmvar_total_likelihood(residuals, c(1, 2), atoms, backend = "RcppEigen"),
    .pdpmvar_test_reference$.pdpmvar_total_likelihood(residuals, c(1, 2), atoms), tolerance = 1e-11)
  expect_error(.pdpmvar_test_reference$.pdpmvar_log_likelihood(residuals[[1]], -diag(2), backend = "RcppEigen"), "positive definite")
})

test_that("coefficient and covariance assignments preserve slices, ties, probabilities and RNG", {
  set.seed(122)
  X <- lapply(c(7, 9, 11, 8, 10), function(n) matrix(rnorm(2 * n), 2))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  coefficients <- lapply(seq_len(5), function(i) list(coefficient = matrix(rnorm(8, sd = 0.1), 2)))
  coefficients[[5]] <- coefficients[[2]]
  covariance_atoms <- lapply(seq_len(5), function(i) list(covariance = diag(c(1 + i / 5, 1 + i / 3))))
  covariance_atoms[[5]] <- covariance_atoms[[2]]
  covariance_labels <- c(1, 3, 1, 3, 1)
  slice <- list(weights = c(0.35, 0.25, 0.1, 0.15, 0.15), slices = c(0.3, 0.15, 0.099, 0.25, 0.01))
  for (seed in 123:127) {
    set.seed(seed)
    reference <- .pdpmvar_test_reference$.pdpmvar_assign_coefficients(X, design, coefficients, covariance_labels, covariance_atoms, slice)
    state <- .Random.seed
    set.seed(seed)
    compiled <- .pdpmvar_test_reference$.pdpmvar_assign_coefficients(X, design, coefficients, covariance_labels, covariance_atoms, slice, backend = "RcppEigen")
    expect_equal(compiled, reference, tolerance = 1e-11)
    expect_identical(.Random.seed, state)
    set.seed(seed)
    reference <- .pdpmvar_test_reference$.pdpmvar_assign_covariances(X, covariance_atoms, slice)
    state <- .Random.seed
    set.seed(seed)
    compiled <- .pdpmvar_test_reference$.pdpmvar_assign_covariances(X, covariance_atoms, slice, backend = "RcppEigen")
    expect_equal(compiled, reference, tolerance = 1e-11)
    expect_identical(.Random.seed, state)
  }
  expect_equal(compiled$probabilities[3, 2], compiled$probabilities[3, 5])
  expect_equal(compiled$labels[c(1, 4)], c(1, 1))
  expect_equal(compiled$probabilities[2, c(3, 4, 5)], c(0, 0, 0))
  bad <- slice
  bad$slices[1] <- 1
  expect_error(.pdpmvar_test_reference$.pdpmvar_assign_covariances(X, covariance_atoms, bad, backend = "RcppEigen"), "eligible")
})

test_that("large-prefix alias sampling retains the R RNG behavior", {
  X <- list(matrix(c(1, 0, -1), 1))
  atoms <- rep(list(list(covariance = matrix(1, 1))), 201)
  slice <- list(weights = rep(1 / 201, 201), slices = 0)
  set.seed(128)
  reference <- .pdpmvar_test_reference$.pdpmvar_assign_covariances(X, atoms, slice)
  state <- .Random.seed
  set.seed(128)
  compiled <- .pdpmvar_test_reference$.pdpmvar_assign_covariances(X, atoms, slice, backend = "RcppEigen")
  expect_equal(compiled, reference)
  expect_identical(.Random.seed, state)
})

test_that("compiled covariance likelihood scores preserve the full MH ratio", {
  set.seed(129)
  residuals <- list(matrix(rnorm(27), 3), matrix(rnorm(36), 3), matrix(rnorm(30), 3))
  center <- .pdpmvar_initialize_covariance(tcrossprod(residuals[[2]]) / 12, 2)
  old_atoms <- lapply(seq_len(3), function(i) .pdpmvar_test_reference$.pdpmvar_new_covariance(3, 2, 3, 2))
  new_atoms <- old_atoms
  new_atoms[[2]] <- .pdpmvar_test_reference$.pdpmvar_propose_covariance(center)
  old <- c(1, 1, 1)
  new <- c(1, 2, 2)
  weights <- log(c(0.6, 0.3, 0.05))
  reference <- .pdpmvar_test_reference$.pdpmvar_covariance_move_ratio(old, new, old_atoms, new_atoms, weights,
    lapply(residuals, tcrossprod), c(9, 12, 10), c(1, 2), center, 3, 3, 2)
  state <- .Random.seed
  compiled <- .pdpmvar_test_reference$.pdpmvar_covariance_move_ratio(old, new, old_atoms, new_atoms, weights,
    lapply(residuals, tcrossprod), c(9, 12, 10), c(1, 2), center, 3, 3, 2, backend = "RcppEigen")
  reverse <- .pdpmvar_test_reference$.pdpmvar_covariance_move_ratio(new, old, new_atoms, old_atoms, weights,
    lapply(residuals, tcrossprod), c(9, 12, 10), c(1, 2), center, 3, 3, 2, backend = "RcppEigen")
  expect_equal(compiled, reference, tolerance = 1e-10)
  expect_equal(compiled$log_ratio, -reverse$log_ratio, tolerance = 1e-10)
  expect_identical(.Random.seed, state)
})
