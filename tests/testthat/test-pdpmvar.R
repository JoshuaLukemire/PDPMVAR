test_that("lag predictors preserve orientation and early-time observations", {
  x <- rbind(c(1, 2, 4, 8), c(3, 5, 7, 9))
  design <- .pdpmvar_lag_design(x, 2)
  expect_equal(design, rbind(c(0, 1, 2, 4), c(0, 3, 5, 7),
                             c(0, 0, 1, 2), c(0, 0, 3, 5)))
  coefficient <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8), 2, byrow = TRUE)
  expect_equal(as.numeric((coefficient %*% design)[, 3]),
               as.numeric(coefficient[, 1:2] %*% x[, 2] + coefficient[, 3:4] %*% x[, 1]))
})

test_that("Gaussian precision draws reproduce a correlated target", {
  set.seed(21)
  covariance <- matrix(c(2, 0.7, 0.7, 1), 2)
  target_mean <- c(1, -2)
  precision <- solve(covariance)
  draws <- replicate(5000, .pdpmvar_test_reference$.pdpmvar_draw_normal(precision, precision %*% target_mean))
  expect_equal(rowMeans(draws), target_mean, tolerance = 0.06)
  expect_equal(unname(cov(t(draws))), covariance, tolerance = 0.08)
})

test_that("PDPMVAR summaries match retained draws on the requested fixture", {
  set.seed(22)
  fixture <- simulate_pdpmvar_data()
  fit <- pdpmvar(fixture$data, n_burnin = 3, n_mcmc = 5, store_draws = TRUE)
  expect_s3_class(fit, "pdpmvar")
  expect_equal(dim(fit$coefficient_mean), c(10, 20, 25))
  expect_equal(fit$coefficient_mean, apply(fit$draws$coefficients, 1:3, mean))
  expect_equal(fit$covariance_mean, apply(fit$draws$covariance, 1:3, mean))
  expect_equal(fit$coefficient_sign_probability, apply(fit$draws$coefficients > 0, 1:3, mean))
  expect_true(all(is.finite(fit$coefficient_mean)))
  expect_true(all(fit$coefficient_sign_tail >= 0 & fit$coefficient_sign_tail <= 1))
  for (subject in seq_along(fixture$data)) {
    expect_gt(min(eigen(fit$covariance_mean[, , subject], symmetric = TRUE)$values), 0)
  }
  for (component in c("coefficient", "covariance")) {
    labels <- fit[[paste0(component, "_labels")]]
    similarity <- fit[[paste0(component, "_coclustering")]]
    expected <- matrix(0, 25, 25)
    for (iteration in seq_len(5)) {
      expected <- expected + outer(labels[iteration, ], labels[iteration, ], "==") / 5
    }
    expect_equal(similarity, expected)
    expect_equal(fit$cluster_counts[4:8, component],
                 apply(labels, 1, function(x) length(unique(x))), ignore_attr = TRUE)
  }
})

test_that("normal prior handles one node, unequal lengths, and one retained draw", {
  x <- list(matrix(seq(-1, 1, length.out = 8), 1), matrix(c(0, 1, -1, 0, 2), 1))
  set.seed(23)
  fit <- pdpmvar(x, K = 1, n_factors = 1, initial_clusters = 1,
                 n_burnin = 0, n_mcmc = 1, coefficient_prior = "normal")
  set.seed(23)
  expect_identical(pdpmvar(x, K = 1, n_factors = 1, initial_clusters = 1,
                          n_burnin = 0, n_mcmc = 1, coefficient_prior = "normal"), fit)
  expect_null(fit$draws)
  expect_equal(dim(fit$coefficient_mean), c(1, 1, 2))
  expect_true(all(is.finite(fit$covariance_mean)))
})

test_that("invalid data and controls fail before sampling", {
  x <- list(matrix(0, 2, 5), matrix(0, 2, 6))
  expect_error(pdpmvar(list(matrix(NA_real_, 2, 5))), "finite numeric")
  expect_error(pdpmvar(list(x[[1]], matrix(0, 3, 5))), "same number")
  expect_error(pdpmvar(x, K = 1.5), "whole number")
  expect_error(pdpmvar(x, K = 5), "series length")
  expect_error(pdpmvar(x, n_mcmc = 0), "whole number")
  expect_error(pdpmvar(x, initial_clusters = 1, lambda = -1), "positive finite")
  expect_error(pdpmvar(x, initial_clusters = 1, store_draws = NA), "TRUE or FALSE")
})

test_that("GIG draws match analytic moments for shared shrinkage", {
  set.seed(24)
  for (p in c(-3, 0, 0.5)) {
    chi <- 2
    psi <- 3
    draws <- replicate(6000, .pdpmvar_test_reference$.pdpmvar_draw_gig(p, chi, psi))
    argument <- sqrt(chi * psi)
    expected_mean <- sqrt(chi / psi) *
      besselK(argument, p + 1, expon.scaled = TRUE) / besselK(argument, p, expon.scaled = TRUE)
    expected_inverse <- sqrt(psi / chi) *
      besselK(argument, p - 1, expon.scaled = TRUE) / besselK(argument, p, expon.scaled = TRUE)
    expect_equal(mean(draws), expected_mean, tolerance = 0.03)
    expect_equal(mean(1 / draws), expected_inverse, tolerance = 0.03)
  }
})

test_that("factor draws use the current factor precisions", {
  set.seed(25)
  atom <- list(working_loadings = matrix(0, 2, 2), variance = c(1, 1),
               factor_precision = c(4, 0.25))
  factors <- .pdpmvar_test_reference$.pdpmvar_draw_factors(matrix(0, 2, 10000), atom)
  expect_equal(rowMeans(factors), c(0, 0), tolerance = 0.05)
  expect_equal(unname(cov(t(factors))), diag(c(0.25, 4)), tolerance = 0.08)
})

test_that("expanded covariance atoms and shared shrinkage remain consistent", {
  set.seed(26)
  fixture <- simulate_pdpmvar_data(n_subjects = 6, n_nodes = 3, n_timepoints = 20)
  fit <- pdpmvar(fixture$data, n_factors = 2, n_burnin = 2, n_mcmc = 5,
                 cov_shape = 4, cov_rate = 3)
  expect_gt(stats::sd(fit$lambda), 0)
  expect_true(all(fit$final_state$shared_variance > 0))
  expect_equal(fit$settings$cov_shape, 4)
  expect_equal(fit$settings$cov_rate, 3)
  for (atom in fit$final_state$covariance_atoms) {
    expect_true(all(atom$working_loadings[upper.tri(atom$working_loadings)] == 0))
    expect_true(all(diag(atom$loadings) > 0))
    expected <- atom$working_loadings %*% diag(1 / atom$factor_precision) %*%
      t(atom$working_loadings) + diag(atom$variance)
    expect_equal(atom$covariance, expected)
    expect_gt(min(eigen(atom$covariance, symmetric = TRUE)$values), 0)
  }
})

test_that("empirical calibration matches Gamma precision moments", {
  x <- list(rbind(c(-1, 0, 1), c(-2, 0, 2)))
  calibrated <- .pdpmvar_test_reference$.pdpmvar_calibrate_covariance(x)
  precisions <- c(1, 0.25)
  expect_equal(unname(calibrated["shape"] / calibrated["rate"]), mean(precisions))
  expect_equal(unname(calibrated["shape"] / calibrated["rate"]^2), var(precisions))
  expect_error(.pdpmvar_test_reference$.pdpmvar_calibrate_covariance(list(matrix(0, 1, 5))), "supply cov_shape")
})

test_that("slice extension covers every current slice and allows label gaps", {
  set.seed(27)
  labels <- c(1, 4, 4, 8)
  slice <- .pdpmvar_test_reference$.pdpmvar_slice_weights(labels, concentration = 10)
  expect_true(all(slice$slices < slice$weights[labels]))
  expect_lt(1 - sum(slice$weights), min(slice$slices))
  expect_true(all(slice$weights > 0))
})
