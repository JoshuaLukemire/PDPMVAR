test_that("Eigen Gaussian integrals match R across block dimensions", {
  set.seed(81)
  for (dimension in c(1, 8, 200)) {
    m <- matrix(rnorm(dimension^2), dimension)
    precision <- crossprod(m) + diag(0.5, dimension)
    information <- rnorm(dimension)
    reference <- .pdpmvar_test_reference$.pdpmvar_coefficient_integral(precision, information, 0.7)
    compiled <- .pdpmvar_test_reference$.pdpmvar_coefficient_integral(precision, information, 0.7, backend = "RcppEigen")
    expect_equal(compiled, reference, tolerance = 1e-10)
  }
})

test_that("Eigen Gaussian draws preserve the R random-number stream", {
  precision <- matrix(c(3, 0.8, 0.8, 2), 2)
  information <- c(1, -2)
  set.seed(82)
  reference <- .pdpmvar_test_reference$.pdpmvar_draw_normal(precision, information)
  reference_state <- .Random.seed
  set.seed(82)
  compiled <- .pdpmvar_test_reference$.pdpmvar_draw_normal(precision, information, backend = "RcppEigen")
  expect_equal(compiled, reference, tolerance = 1e-12)
  expect_identical(.Random.seed, reference_state)
  expect_error(.pdpmvar_eigen_integral(matrix(c(1, 2, 2, 1), 2), c(0, 0), 0), "positive definite")
  expect_error(.pdpmvar_eigen_integral(matrix(c(1, 0, 2, 1), 2), c(0, 0), 0), "symmetric")
  expect_error(.pdpmvar_eigen_integral(diag(2), 0, 0), "match")
})

test_that("native public fits and the R reference agree in a seeded sampler with both partition moves", {
  set.seed(83)
  fixture <- simulate_pdpmvar_data(n_subjects = 6, n_nodes = 3, n_timepoints = 20)
  set.seed(84)
  reference <- pdpmvar_reference_for_test(fixture$data, initial_clusters = 1, n_factors = 2,
    n_burnin = 5, n_mcmc = 10, split_merge_every = 2, coefficient_split_merge_every = 2)
  reference_state <- .Random.seed
  set.seed(84)
  compiled <- pdpmvar(fixture$data, initial_clusters = 1, n_factors = 2,
    n_burnin = 5, n_mcmc = 10, split_merge_every = 2, coefficient_split_merge_every = 2)
  expect_equal(compiled$coefficient_labels, reference$coefficient_labels)
  expect_equal(compiled$covariance_labels, reference$covariance_labels)
  expect_equal(compiled$coefficient_mean, reference$coefficient_mean, tolerance = 1e-8)
  expect_equal(compiled$covariance_mean, reference$covariance_mean, tolerance = 1e-8)
  expect_equal(compiled$lambda, reference$lambda, tolerance = 1e-8)
  expect_identical(.Random.seed, reference_state)
})
