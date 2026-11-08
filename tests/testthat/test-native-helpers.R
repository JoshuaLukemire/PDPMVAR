test_that("production helpers have no backend selection and preserve native results", {
  native <- asNamespace("PDPMVAR")
  original <- .pdpmvar_test_reference
  helpers <- intersect(ls(original, all.names = TRUE), ls(native, all.names = TRUE))
  for (name in helpers) {
    expect_false("backend" %in% names(formals(get(name, native))))
  }

  set.seed(9501)
  X <- lapply(1:4, function(i) matrix(rnorm(3 * (12 + i)), 3))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  cross  <- lapply(design, tcrossprod)
  tau    <- matrix(0.3, 3, 6)
  atoms  <- lapply(1:3, function(i) .pdpmvar_new_coefficient(tau))
  covariance <- .pdpmvar_new_covariance(3, 2, 3, 2)
  covariances <- rep(list(covariance$covariance), 4)
  variance <- matrix(rep(covariance$variance, 4), 3)

  # These are namespace calls, not the legacy reference aliases used elsewhere in tests.
  cases <- list(
    list(".pdpmvar_coefficient_statistics", list(X, design, cross, covariances, tau)),
    list(".pdpmvar_coefficient_row_statistics", list(X, design, cross, variance, tau)),
    list(".pdpmvar_update_coefficients", list(X, design, cross, lapply(X, function(x) x * 0),
      c(1, 2, 1, 2), atoms, tau, variance)),
    list(".pdpmvar_update_shrinkage", list(c(1, 2, 1, 2), atoms, tau, 1)),
    list(".pdpmvar_draw_factors", list(X[[1]], covariance)),
    list(".pdpmvar_update_covariance", list(X[[1]], covariance, 3, 2)),
    list(".pdpmvar_slice_weights", list(c(1, 2, 1, 2), 1)),
    list(".pdpmvar_draw_gig", list(-1, 2, 3)),
    list(".pdpmvar_coefficient_split_merge", list(X, design, cross, covariances,
      c(1, 1, 1, 1), atoms, tau, 1, 3)),
    list(".pdpmvar_covariance_split_merge", list(X, c(1, 1, 1, 1), rep(list(covariance), 3), 1, 2, 3, 2, 3)),
    list(".pdpmvar_coefficient_factor_move", list(X, design, cross, c(1, 1, 1, 1), atoms,
      c(1, 1, 1, 1), list(covariance), tau, 1, 3))
  )
  for (case in cases) {
    set.seed(9502)
    expected <- do.call(get(case[[1]], original), c(case[[2]], list(backend = "RcppEigen")))
    rng <- .Random.seed
    set.seed(9502)
    observed <- do.call(get(case[[1]], native), case[[2]])
    expect_equal(observed, expected, tolerance = 1e-12, info = case[[1]])
    expect_identical(.Random.seed, rng)
  }

  # Exercise both the direct kernel and the established large-prefix alias callback.
  for (slots in c(3, 201)) {
    slice <- list(weights = rep(1 / slots, slots), slices = rep(0, 4))
    cases <- list(
      list(".pdpmvar_assign_coefficients", list(X, design, rep(atoms[1], slots),
        rep(1, 4), list(covariance), slice)),
      list(".pdpmvar_assign_covariances", list(X, rep(list(covariance), slots), slice))
    )
    for (case in cases) {
      set.seed(9503)
      expected <- do.call(get(case[[1]], original), c(case[[2]], list(backend = "RcppEigen")))
      rng <- .Random.seed
      set.seed(9503)
      observed <- do.call(get(case[[1]], native), case[[2]])
      expect_equal(observed, expected, tolerance = 1e-12)
      expect_identical(.Random.seed, rng)
    }
  }
})
