test_that("native covariance centers reproduce the R factor rotation", {
  set.seed(1000)
  for (nodes in c(1, 3, 5)) {
    z <- matrix(rnorm(nodes * 12), nodes)
    covariance <- tcrossprod(z) / ncol(z)
    for (factors in unique(c(1, min(2, nodes), nodes))) {
      expect_equal(.pdpmvar_eigen_covariance_center(covariance, factors),
        .pdpmvar_initialize_covariance(covariance, factors), tolerance = 1e-10)
    }
    # Tied eigenvalues and a deficient covariance exercise the QR boundary.
    expect_equal(.pdpmvar_eigen_covariance_center(diag(nodes), nodes),
      .pdpmvar_initialize_covariance(diag(nodes), nodes), tolerance = 1e-10)
    expect_equal(.pdpmvar_eigen_covariance_center(matrix(1, nodes, nodes), nodes),
      .pdpmvar_initialize_covariance(matrix(1, nodes, nodes), nodes), tolerance = 1e-10)
  }
  expect_error(.pdpmvar_eigen_covariance_center(diag(2), 3), "dimensions")
})

test_that("native full controllers reproduce proposals, decisions, refreshes and RNG", {
  set.seed(1001)
  X <- lapply(1:4, function(i) matrix(rnorm(20 + 2 * i), 2))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  cross <- lapply(design, tcrossprod)
  covariances <- rep(list(matrix(c(2, 0.4, 0.4, 1), 2)), 4)
  variance <- matrix(seq(0.1, 0.8, length.out = 8), 2)
  coefficient_atoms <- lapply(1:5, function(i) .pdpmvar_test_reference$.pdpmvar_new_coefficient(variance))
  covariance_atoms <- lapply(1:5, function(i) .pdpmvar_test_reference$.pdpmvar_new_covariance(2, 1, 3, 2))
  snapshot <- serialize(list(coefficient_atoms, covariance_atoms), NULL)
  records <- list()
  for (labels in list(rep(1, 4), c(1, 3, 1, 3), 1:4, rep(5, 4), 1)) {
    subjects <- seq_along(labels)
    expected <- observed <- list()
    for (seed in 1002:1011) {
      set.seed(seed)
      a <- .pdpmvar_test_reference$.pdpmvar_coefficient_split_merge_reference(X[subjects], design[subjects], cross[subjects],
        covariances[subjects], labels, coefficient_atoms[seq_len(max(labels))], variance, 1, 4,
        backend = "RcppEigen")
      a_rng <- .Random.seed
      set.seed(seed)
      b <- .pdpmvar_test_reference$.pdpmvar_coefficient_split_merge(X[subjects], design[subjects], cross[subjects],
        covariances[subjects], labels, coefficient_atoms[seq_len(max(labels))], variance, 1, 4,
        backend = "RcppEigen")
      b_rng <- .Random.seed
      set.seed(seed)
      c <- .pdpmvar_test_reference$.pdpmvar_covariance_split_merge_reference(X[subjects], labels,
        covariance_atoms[seq_len(max(labels))], 1, 1, 3, 2, 4, backend = "RcppEigen")
      c_rng <- .Random.seed
      set.seed(seed)
      d <- .pdpmvar_test_reference$.pdpmvar_covariance_split_merge(X[subjects], labels,
        covariance_atoms[seq_len(max(labels))], 1, 1, 3, 2, 4, backend = "RcppEigen")
      expected[[as.character(seed)]] <- list(a, a_rng, c, c_rng)
      observed[[as.character(seed)]] <- list(b, b_rng, d, .Random.seed)
      records[[length(records) + 1]] <- list(a$record, c$record)
    }
    expect_equal(observed, expected, tolerance = 1e-8)
  }
  expect_identical(serialize(list(coefficient_atoms, covariance_atoms), NULL), snapshot)
  types <- unlist(lapply(records, function(pair) vapply(pair, `[[`, "", "type")))
  accepted <- unlist(lapply(records, function(pair) vapply(pair, `[[`, FALSE, "accepted")))
  expect_true(all(c("split", "merge", "skip") %in% types))
  expect_true(any(accepted))
  expect_true(any(!accepted & types != "skip"))
})

test_that("native acceptance ratios preserve forward/reverse detailed balance", {
  set.seed(1012)
  X <- lapply(1:4, function(i) matrix(rnorm(20 + 2 * i), 2))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  variance <- matrix(0.2, 2, 4)
  covariances <- rep(list(diag(2)), 4)
  statistics <- .pdpmvar_test_reference$.pdpmvar_coefficient_statistics(X, design, lapply(design, tcrossprod),
    covariances, variance, backend = "RcppEigen")
  old <- c(2, 2, 4, 4)
  new <- c(2, 1, 4, 4)
  weights <- log(c(0.05, 0.4, 0.05, 0.3, 0.05))
  anchors <- c(1L, 2L)
  reference <- .pdpmvar_test_reference$.pdpmvar_coefficient_collapsed_ratio(old, new, statistics, weights, anchors, 5)
  compiled <- .pdpmvar_eigen_coefficient_move_ratio(old, new, statistics, weights, anchors, 5)
  reverse <- .pdpmvar_eigen_coefficient_move_ratio(new, old, statistics, weights, anchors, 5)
  expect_equal(compiled, reference, tolerance = 1e-10)
  expect_equal(compiled$log_ratio, -reverse$log_ratio, tolerance = 1e-10)
  expect_equal(compiled$log_forward + min(0, compiled$log_ratio),
    compiled$target_change + reverse$log_forward + min(0, reverse$log_ratio))
  atoms <- lapply(1:5, function(i) .pdpmvar_test_reference$.pdpmvar_new_covariance(2, 1, 3, 2))
  center <- .pdpmvar_initialize_covariance(tcrossprod(X[[2]]) / ncol(X[[2]]), 1)
  proposed <- atoms
  proposed[[1]] <- .pdpmvar_test_reference$.pdpmvar_propose_covariance(center)
  scatters <- lapply(X, tcrossprod)
  times <- vapply(X, ncol, numeric(1))
  reference <- .pdpmvar_test_reference$.pdpmvar_covariance_move_ratio(old, new, atoms, proposed, weights,
    scatters, times, anchors, center, 5, 3, 2, backend = "RcppEigen")
  compiled <- .pdpmvar_eigen_covariance_move_ratio(old, new, atoms, proposed, weights,
    scatters, times, anchors, center, 5, 3, 2)
  reverse <- .pdpmvar_eigen_covariance_move_ratio(new, old, proposed, atoms, weights,
    scatters, times, anchors, center, 5, 3, 2)
  expect_equal(compiled, reference, tolerance = 1e-10)
  expect_equal(compiled$log_ratio, -reverse$log_ratio, tolerance = 1e-10)
  expect_equal(compiled$log_forward + min(0, compiled$log_ratio),
    compiled$target_change + reverse$log_forward + min(0, reverse$log_ratio))
  expect_error(.pdpmvar_eigen_coefficient_move_ratio(old, new, statistics, weights, c(0L, 2L), 5), "anchors")
  expect_error(.pdpmvar_eigen_covariance_move_ratio(old, new, atoms, proposed, weights,
    scatters, times, c(1L, 1L), center, 5, 3, 2), "anchors")
  expect_error(.pdpmvar_eigen_covariance_move(X, c(0, 1, 1, 1), atoms, 1, 1, 3, 2, 5), "positive integer")
})


test_that("controllers agree at the ten-node, two-lag model dimensions", {
  set.seed(1013)
  X <- lapply(1:6, function(i) matrix(rnorm(10 * (20 + i)), 10))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  cross <- lapply(design, tcrossprod)
  variance <- matrix(0.2, 10, 20)
  coefficients <- list(.pdpmvar_test_reference$.pdpmvar_new_coefficient(variance))
  covariance <- list(.pdpmvar_test_reference$.pdpmvar_new_covariance(10, 5, 3, 2))
  snapshot <- serialize(list(coefficients, covariance), NULL)
  labels <- rep(1, 6)
  for (seed in 1014:1016) {
    set.seed(seed)
    expected <- .pdpmvar_test_reference$.pdpmvar_coefficient_split_merge_reference(X, design, cross,
      rep(list(diag(10)), 6), labels, coefficients, variance, 1, 4, backend = "RcppEigen")
    expected_rng <- .Random.seed
    set.seed(seed)
    observed <- .pdpmvar_test_reference$.pdpmvar_coefficient_split_merge(X, design, cross,
      rep(list(diag(10)), 6), labels, coefficients, variance, 1, 4, backend = "RcppEigen")
    expect_equal(observed, expected, tolerance = 1e-8)
    expect_identical(.Random.seed, expected_rng)
    set.seed(seed)
    expected <- .pdpmvar_test_reference$.pdpmvar_covariance_split_merge_reference(X, labels, covariance,
      1, 5, 3, 2, 4, backend = "RcppEigen")
    expected_rng <- .Random.seed
    set.seed(seed)
    observed <- .pdpmvar_test_reference$.pdpmvar_covariance_split_merge(X, labels, covariance,
      1, 5, 3, 2, 4, backend = "RcppEigen")
    expect_equal(observed, expected, tolerance = 1e-8)
    expect_identical(.Random.seed, expected_rng)
  }
  expect_identical(serialize(list(coefficients, covariance), NULL), snapshot)
})
