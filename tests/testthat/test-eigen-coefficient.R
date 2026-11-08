test_that("Eigen sufficient statistics preserve vectorization with unequal lengths", {
  set.seed(91)
  X <- list(matrix(rnorm(14), 2), matrix(rnorm(18), 2), matrix(rnorm(22), 2))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  cross <- lapply(design, tcrossprod)
  covariance <- rep(list(matrix(c(2, 0.4, 0.4, 1), 2)), 3)
  variance <- matrix(seq(0.1, 0.8, length.out = 8), 2)
  reference <- .pdpmvar_test_reference$.pdpmvar_coefficient_statistics(X, design, cross, covariance, variance)
  compiled <- .pdpmvar_test_reference$.pdpmvar_coefficient_statistics(X, design, cross, covariance, variance, backend = "RcppEigen")
  reference$backend <- "RcppEigen"
  expect_equal(compiled, reference, tolerance = 1e-11)
  expect_error(.pdpmvar_eigen_statistics(X, design[-1], cross, covariance, variance), "matching")
  expect_error(.pdpmvar_eigen_statistics(X, design, cross, covariance, variance * 0), "positive")
})

test_that("Eigen restricted proposals reproduce forced and sampled R evaluations", {
  set.seed(92)
  X <- lapply(seq_len(6), function(i) matrix(rnorm(16 + 2 * i), 2))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  covariance <- rep(list(matrix(c(2, 0.4, 0.4, 1), 2)), 6)
  statistics <- .pdpmvar_test_reference$.pdpmvar_coefficient_statistics(X, design, lapply(design, tcrossprod), covariance, matrix(0.2, 2, 4))
  compiled_statistics <- statistics
  compiled_statistics$backend <- "RcppEigen"
  subjects <- c(6, 2, 4, 1, 5)
  anchors <- c(6, 4)
  weights <- log(c(0.1, 0.4, 0.1, 0.1, 0.2))
  forced <- c(5, 2, 2, 5, 2, 2)
  reference <- .pdpmvar_test_reference$.pdpmvar_coefficient_restricted_split(statistics, subjects, anchors, weights, 2, 5, forced)
  set.seed(93)
  state <- .Random.seed
  compiled <- .pdpmvar_test_reference$.pdpmvar_coefficient_restricted_split(compiled_statistics, subjects, anchors, weights, 2, 5, forced)
  expect_equal(compiled, reference, tolerance = 1e-10)
  expect_identical(.Random.seed, state)
  for (seed in 94:98) {
    set.seed(seed)
    reference <- .pdpmvar_test_reference$.pdpmvar_coefficient_restricted_split(statistics, subjects, anchors, weights, 2, 5)
    state <- .Random.seed
    set.seed(seed)
    compiled <- .pdpmvar_test_reference$.pdpmvar_coefficient_restricted_split(compiled_statistics, subjects, anchors, weights, 2, 5)
    expect_equal(compiled, reference, tolerance = 1e-10)
    expect_identical(.Random.seed, state)
  }
  expect_error(.pdpmvar_eigen_restricted_split(compiled_statistics, as.integer(subjects),
    c(1L, 3L), weights, 2, 5, NULL, runif(3)), "anchors")
})

test_that("compiled split loops preserve the collapsed MH forward and reverse ratio", {
  set.seed(99)
  X <- lapply(seq_len(3), function(i) matrix(rnorm(20), 2))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  covariance <- rep(list(matrix(c(1, -0.3, -0.3, 2), 2)), 3)
  variance <- matrix(seq(0.1, 0.8, length.out = 8), 2)
  reference <- .pdpmvar_test_reference$.pdpmvar_coefficient_statistics(X, design, lapply(design, tcrossprod), covariance, variance)
  compiled <- .pdpmvar_test_reference$.pdpmvar_coefficient_statistics(X, design, lapply(design, tcrossprod), covariance, variance, backend = "RcppEigen")
  old <- c(1, 1, 1)
  new <- c(1, 3, 3)
  weights <- log(c(0.6, 0.1, 0.2, 0.05))
  expected <- .pdpmvar_test_reference$.pdpmvar_coefficient_collapsed_ratio(old, new, reference, weights, c(1, 2), 4)
  observed <- .pdpmvar_test_reference$.pdpmvar_coefficient_collapsed_ratio(old, new, compiled, weights, c(1, 2), 4)
  reverse <- .pdpmvar_test_reference$.pdpmvar_coefficient_collapsed_ratio(new, old, compiled, weights, c(1, 2), 4)
  expect_equal(observed, expected, tolerance = 1e-10)
  expect_equal(observed$log_ratio, -reverse$log_ratio, tolerance = 1e-10)
})
