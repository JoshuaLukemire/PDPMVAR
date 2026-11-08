test_that("fractional-df reversible covariance updates match R and preserve inputs", {
  set.seed(1610)
  for (dimensions in list(c(1, 1), c(3, 2), c(3, 3))) {
    D <- dimensions[1]
    B <- dimensions[2]
    y <- matrix(rnorm(D * 40), D)
    scatter <- tcrossprod(y) * 40.25 / 40
    atom <- .pdpmvar_test_reference$.pdpmvar_new_covariance(D, B, 3, 2)
    snapshot <- serialize(atom, NULL)
    set.seed(1611)
    reference <- .pdpmvar_test_reference$.pdpmvar_covariance_reversible(atom, scatter, 40.25, 3, 2)
    rng <- .Random.seed
    set.seed(1611)
    compiled <- .pdpmvar_eigen_covariance_reversible(atom, scatter, 40.25, 3, 2)
    expect_equal(compiled, reference, tolerance = 1e-9)
    expect_identical(.Random.seed, rng)
    expect_identical(serialize(atom, NULL), snapshot)
    expect_true(all(compiled$working_loadings[upper.tri(compiled$working_loadings)] == 0))
    expect_equal(.pdpmvar_test_reference$.pdpmvar_refresh_covariance(compiled)$covariance, compiled$covariance)
    expect_error(.pdpmvar_eigen_covariance_reversible(atom, scatter, D + B, 3, 2), "effective observations")
  }
})

test_that("fractional augmentation preserves prior-predictive moments and exchangeability", {
  set.seed(1612)
  values <- matrix(0, 300, 4)
  for (replicate in seq_len(nrow(values))) {
    old <- .pdpmvar_test_reference$.pdpmvar_new_covariance(2, 1, 3, 2)
    scatter <- rWishart(1, 8.5, old$covariance)[, , 1]
    updated <- .pdpmvar_eigen_covariance_reversible(old, scatter, 8.5, 3, 2)
    values[replicate, ] <- c(updated$factor_precision, mean(1 / updated$variance),
      mean(updated$working_loadings^2),
      old$working_loadings[1, 1]^2 / updated$variance[1] -
        updated$working_loadings[1, 1]^2 / old$variance[1])
  }
  expected <- c(1, 1.5, 1, 0)
  expect_true(all(abs(colMeans(values) - expected) < 5 * apply(values, 2, sd) / sqrt(nrow(values))))
})

test_that("proxy predictive split probabilities normalize and match R", {
  set.seed(1615)
  residuals <- lapply(c(10, 12, 14), function(n) matrix(rnorm(2 * n), 2))
  scatters <- lapply(residuals, tcrossprod)
  times <- c(10, 12, 14)
  args <- list(scatters, times, 1:3, c(1L, 2L), 1, 3)
  first <- do.call(.pdpmvar_test_reference$.pdpmvar_covariance_restricted, c(args, list(forced = c(1, 3, 1))))
  second <- do.call(.pdpmvar_test_reference$.pdpmvar_covariance_restricted, c(args, list(forced = c(1, 3, 3))))
  expect_equal(exp(first$log_probability) + exp(second$log_probability), 1)
  expect_equal(do.call(.pdpmvar_eigen_covariance_restricted, c(args, list(forced = c(1, 3, 1)))), first)
  for (seed in 1616:1618) {
    set.seed(seed)
    reference <- do.call(.pdpmvar_test_reference$.pdpmvar_covariance_restricted, args)
    rng <- .Random.seed
    set.seed(seed)
    expect_equal(do.call(.pdpmvar_eigen_covariance_restricted, args), reference)
    expect_identical(.Random.seed, rng)
  }
})

test_that("annealing work equals the endpoint target and exact reverse path correction", {
  set.seed(1619)
  residuals <- lapply(c(10, 12, 14), function(n) matrix(rnorm(2 * n), 2))
  scatters <- lapply(residuals, tcrossprod)
  times <- c(10, 12, 14)
  atoms <- lapply(1:3, function(i) .pdpmvar_test_reference$.pdpmvar_new_covariance(2, 1, 3, 2))
  snapshot <- serialize(atoms, NULL)
  old <- c(1, 1, 1)
  new <- c(1, 2, 2)
  set.seed(1620)
  reference <- .pdpmvar_test_reference$.pdpmvar_covariance_bridge(old, new, atoms, scatters, times, 1, 2, 1, 3, 2,
    store_path = TRUE)
  rng <- .Random.seed
  set.seed(1620)
  compiled <- .pdpmvar_eigen_covariance_bridge(old, new, atoms, scatters, times, 1, 2, 1, 3, 2, TRUE)
  expect_equal(compiled, reference, tolerance = 1e-7)
  expect_identical(.Random.seed, rng)
  expect_identical(serialize(atoms, NULL), snapshot)
  # Compute each intermediate target independently from subject likelihoods and the original base.
  center <- .pdpmvar_initialize_covariance(Reduce(`+`, scatters[which(new == 2)]) / sum(times[new == 2]), 1)
  target <- function(pair, beta) {
    value <- .pdpmvar_test_reference$.pdpmvar_covariance_log_base(pair[[1]], 3, 2) +
      beta * .pdpmvar_test_reference$.pdpmvar_covariance_log_base(pair[[2]], 3, 2) +
      (1 - beta) * .pdpmvar_test_reference$.pdpmvar_bridge_density(pair[[2]], center)
    for (subject in 1:3) {
      value <- value + (1 - beta) * .pdpmvar_test_reference$.pdpmvar_log_likelihood(residuals[[subject]], pair[[old[subject]]]$covariance) +
        beta * .pdpmvar_test_reference$.pdpmvar_log_likelihood(residuals[[subject]], pair[[new[subject]]]$covariance)
    }
    return(value)
  }
  path <- reference$path
  temperatures <- reference$temperatures
  correction <- target(tail(path, 1)[[1]], 1) - target(path[[1]], 0)
  for (stage in 2:length(path)) {
    beta <- temperatures[stage]
    correction <- correction + target(path[[stage - 1]], beta) - target(path[[stage]], beta)
  }
  expect_equal(reference$log_work, correction, tolerance = 1e-9)
  reverse_work <- sum(diff(temperatures) * vapply(rev(path), function(pair)
    target(pair, 0) - target(pair, 1), numeric(1)))
  expect_equal(reverse_work, -reference$log_work, tolerance = 1e-9)
  expect_equal(reference$atoms[[3]], atoms[[3]])
})

test_that("collapsed labeled-stick probabilities retain the original DP prior and gaps", {
  expect_equal(exp(.pdpmvar_test_reference$.pdpmvar_covariance_label_prior(c(1, 1, 1), 1)), 1 / 4)
  expect_equal(exp(.pdpmvar_test_reference$.pdpmvar_covariance_label_prior(c(1, 1, 2), 1)), 1 / 24)
  for (concentration in c(0.1, 1, 5)) {
    labels <- c(1, 4, 4, 8)
    counts <- tabulate(labels)
    beyond <- length(labels) - cumsum(counts)
    expected <- sum(lbeta(1 + counts, concentration + beyond) - lbeta(1, concentration))
    expect_equal(.pdpmvar_test_reference$.pdpmvar_covariance_label_prior(labels, concentration), expected)
    expect_equal(.pdpmvar_eigen_covariance_label_prior(labels, concentration), expected)
    log_probability <- .pdpmvar_test_reference$.pdpmvar_covariance_empty_probabilities(c(2, 3, 7), concentration)
    expect_equal(sum(exp(log_probability)), 1)
    expect_true(all(diff(log_probability) < 0))
  }
  labels <- c(1000000, 1000000)
  expect_equal(.pdpmvar_test_reference$.pdpmvar_covariance_label_prior(labels, 1), .pdpmvar_eigen_covariance_label_prior(labels, 1))
  expect_error(.pdpmvar_eigen_covariance_label_prior(c(0, 1), 1), "positive integer")
})

test_that("proposal-only regularization handles singular scatters and invalid indices safely", {
  scatters <- rep(list(matrix(0, 2, 2)), 3)
  args <- list(scatters, c(10, 12, 14), 1:3, c(1L, 2L), 1, 2, c(1, 2, 1))
  expect_equal(do.call(.pdpmvar_eigen_covariance_restricted, args), do.call(.pdpmvar_test_reference$.pdpmvar_covariance_restricted, args))
  expect_error(.pdpmvar_eigen_covariance_restricted(scatters, c(10, 12, 14), c(1L, 1L, 2L), c(1L, 2L), 1, 2), "distinct")
  set.seed(1622)
  atoms <- lapply(1:3, function(i) .pdpmvar_test_reference$.pdpmvar_new_covariance(2, 1, 3, 2))
  bridge <- .pdpmvar_eigen_covariance_bridge(c(1, 1, 1), c(1, 2, 2), atoms, scatters,
    c(10, 12, 14), 1, 2, 1, 3, 2)
  expect_true(is.finite(bridge$log_work))
  expect_error(.pdpmvar_eigen_covariance_bridge(c(1, 1, 1), c(1, 2, 2), atoms, scatters,
    c(10, 12, 14), 0, 2, 1, 3, 2), "dimensions")
  for (concentration in c(1e-310, 1e300)) {
    labels <- c(1, 3, 3, 8)
    expect_true(is.finite(.pdpmvar_test_reference$.pdpmvar_covariance_label_prior(labels, concentration)))
    expect_equal(.pdpmvar_test_reference$.pdpmvar_covariance_label_prior(labels, concentration),
      .pdpmvar_eigen_covariance_label_prior(labels, concentration))
    expect_equal(sum(exp(.pdpmvar_test_reference$.pdpmvar_covariance_empty_probabilities(c(2, 3, 7), concentration))), 1)
  }
})
