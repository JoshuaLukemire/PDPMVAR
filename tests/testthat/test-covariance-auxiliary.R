test_that("conjugate auxiliary draws, densities, and all temperature kernels match R", {
  set.seed(1840)
  for (dimensions in list(c(1, 1), c(3, 2), c(3, 3))) {
    D <- dimensions[1]
    B <- dimensions[2]
    residual <- matrix(rnorm(D * 40), D)
    scatter <- tcrossprod(residual)
    center <- .pdpmvar_initialize_covariance(scatter / 40, B)
    set.seed(1841)
    reference <- .pdpmvar_test_reference$.pdpmvar_bridge_propose(center)
    rng <- .Random.seed
    set.seed(1841)
    compiled <- .pdpmvar_eigen_bridge_propose(center)
    expect_equal(compiled, reference, tolerance = 1e-10)
    expect_identical(.Random.seed, rng)
    expect_equal(.pdpmvar_eigen_bridge_density(compiled, center), .pdpmvar_test_reference$.pdpmvar_bridge_density(reference, center), tolerance = 1e-10)
    # Independent density reconstruction checks all normalizing constants, with no scale Jacobian.
    expected <- sum(dnorm(reference$working_loadings[lower.tri(reference$working_loadings, diag = TRUE)], log = TRUE)) +
      sum(dgamma(reference$factor_precision, 2, rate = 0.5, log = TRUE)) +
      sum(dgamma(1 / reference$variance, 16, rate = 15 * mean(center$variance), log = TRUE) - 2 * log(reference$variance))
    expect_equal(.pdpmvar_test_reference$.pdpmvar_bridge_density(reference, center), expected, tolerance = 1e-10)
    for (beta in c(0, 0.001, 0.2, 1)) {
      set.seed(1842)
      updated <- .pdpmvar_test_reference$.pdpmvar_bridge_auxiliary(reference, scatter, 40, beta, center, B, 3, 2)
      rng <- .Random.seed
      set.seed(1842)
      compiled <- .pdpmvar_eigen_bridge_auxiliary(reference, scatter, 40, beta, center, B, 3, 2)
      expect_equal(compiled, updated, tolerance = 1e-8)
      expect_identical(.Random.seed, rng)
    }
    # At beta=1 the original model kernel is recovered exactly, including its random stream.
    set.seed(1843)
    endpoint <- .pdpmvar_test_reference$.pdpmvar_covariance_reversible(reference, scatter, 40, 3, 2)
    rng <- .Random.seed
    set.seed(1843)
    expect_equal(.pdpmvar_test_reference$.pdpmvar_bridge_auxiliary(reference, scatter, 40, 1, center, B, 3, 2), endpoint)
    expect_identical(.Random.seed, rng)
  }
})

test_that("intermediate conjugate prior equals H^beta q^(1-beta) up to an atom-independent constant", {
  set.seed(1844)
  center <- .pdpmvar_initialize_covariance(diag(c(2, 3, 4)), 2)
  atoms <- lapply(1:2, function(i) .pdpmvar_test_reference$.pdpmvar_new_covariance(3, 2, 3, 2))
  for (beta in c(0, 0.2, 0.8, 1)) {
    a <- beta * 3 + (1 - beta) * 16
    b <- beta * 2 + (1 - beta) * 15 * mean(center$variance)
    c <- beta * 0.5 + (1 - beta) * 2
    difference <- vapply(atoms, function(atom) {
      intermediate <- .pdpmvar_test_reference$.pdpmvar_covariance_log_base(atom, a, b) +
        sum(dgamma(atom$factor_precision, c, rate = 0.5, log = TRUE) -
          dgamma(atom$factor_precision, 0.5, rate = 0.5, log = TRUE))
      return(beta * .pdpmvar_test_reference$.pdpmvar_covariance_log_base(atom, 3, 2) +
        (1 - beta) * .pdpmvar_test_reference$.pdpmvar_bridge_density(atom, center) - intermediate)
    }, numeric(1))
    expect_equal(difference[1], difference[2], tolerance = 1e-10)
  }
})

test_that("fractional conjugate bridge preserves its intermediate prior predictive law", {
  set.seed(1845)
  center <- .pdpmvar_initialize_covariance(diag(c(2, 3)), 1)
  beta <- 0.3
  shape <- beta * 3 + (1 - beta) * 16
  rate <- beta * 2 + (1 - beta) * 15 * mean(center$variance)
  factor_shape <- beta * 0.5 + (1 - beta) * 2
  values <- matrix(0, 300, 4)
  for (i in seq_len(nrow(values))) {
    atom <- .pdpmvar_test_reference$.pdpmvar_bridge_prior(2, 1, shape, rate, factor_shape)
    scatter <- rWishart(1, 8.5, atom$covariance)[, , 1]
    updated <- .pdpmvar_eigen_bridge_auxiliary(atom, scatter / beta, 8.5 / beta, beta, center, 1, 3, 2)
    values[i, ] <- c(updated$factor_precision, mean(1 / updated$variance), mean(updated$working_loadings^2),
      atom$working_loadings[1, 1]^2 / updated$variance[1] - updated$working_loadings[1, 1]^2 / atom$variance[1])
  }
  expected <- c(2 * factor_shape, shape / rate, 1, 0)
  expect_true(all(abs(colMeans(values) - expected) < 5 * apply(values, 2, sd) / sqrt(nrow(values))))
  expect_error(.pdpmvar_eigen_bridge_auxiliary(atom, scatter, 10, -0.1, center, 1, 3, 2), "controls")
  expect_error(.pdpmvar_eigen_covariance_reversible(atom, scatter, 10, 3, 2, 0), "effective observations")
})


test_that("auxiliary native boundaries reject malformed shapes before indexing", {
  center <- .pdpmvar_initialize_covariance(diag(2), 1)
  atom <- .pdpmvar_test_reference$.pdpmvar_new_covariance(2, 1, 3, 2)
  malformed <- atom
  malformed$factor_precision <- numeric()
  expect_error(.pdpmvar_eigen_bridge_density(malformed, center), "dimensions")
  malformed$working_loadings <- matrix(0, 2, 3)
  expect_error(.pdpmvar_eigen_bridge_auxiliary(malformed, diag(2), 1, 0, center, 3, 3, 2), "dimensions")
})
