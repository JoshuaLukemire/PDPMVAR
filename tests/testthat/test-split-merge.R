test_that("scatter scores equal direct residual likelihoods", {
  set.seed(61)
  residuals <- list(matrix(rnorm(21), 3), matrix(rnorm(30), 3))
  atom <- .pdpmvar_test_reference$.pdpmvar_new_covariance(3, 2, 3, 2)
  expect_equal(.pdpmvar_test_reference$.pdpmvar_covariance_scores(lapply(residuals, tcrossprod), c(7, 10), atom),
    vapply(residuals, .pdpmvar_test_reference$.pdpmvar_log_likelihood, numeric(1), covariance = atom$covariance))
})

test_that("continuous proposal density includes loading scale Jacobians", {
  set.seed(62)
  center <- .pdpmvar_initialize_covariance(diag(c(1, 2, 3)), 2)
  atom <- .pdpmvar_test_reference$.pdpmvar_propose_covariance(center)
  free <- lower.tri(atom$working_loadings, diag = TRUE)
  scale <- matrix(rep(sqrt(atom$factor_precision), each = 3), 3)
  transformed <- atom$working_loadings / scale
  signs <- as.matrix(expand.grid(rep(list(c(-1, 1)), 2)))
  loading_log_density <- apply(signs, 1, function(sign) {
    means <- sweep(center$working_loadings, 2, sign, "*")
    return(sum(dnorm(transformed[free], means[free], 0.15, log = TRUE) - log(scale[free])))
  })
  maximum <- max(loading_log_density)
  expected <- maximum + log(mean(exp(loading_log_density - maximum))) +
    sum(dlnorm(atom$factor_precision, 0, 0.5, log = TRUE)) +
    sum(dlnorm(atom$variance, log(center$variance), 0.25, log = TRUE))
  expect_equal(.pdpmvar_test_reference$.pdpmvar_covariance_log_proposal(atom, center), expected)
  expect_true(all(is.finite(atom$covariance)))
})

test_that("continuous split and reverse merge obey the full MH ratio", {
  set.seed(63)
  residuals <- list(matrix(rnorm(24), 3), matrix(rnorm(30), 3), matrix(rnorm(18), 3))
  center <- .pdpmvar_initialize_covariance(tcrossprod(residuals[[2]]) / 10, 2)
  old_atoms <- lapply(seq_len(3), function(i) .pdpmvar_test_reference$.pdpmvar_new_covariance(3, 2, 3, 2))
  new_atoms <- old_atoms
  new_atoms[[2]] <- .pdpmvar_test_reference$.pdpmvar_propose_covariance(center)
  old_labels <- c(1, 1, 1)
  new_labels <- c(1, 2, 2)
  log_weights <- log(c(0.6, 0.3, 0.05))
  forward <- .pdpmvar_test_reference$.pdpmvar_covariance_move_ratio(old_labels, new_labels, old_atoms, new_atoms,
    log_weights, lapply(residuals, tcrossprod), c(8, 10, 6), c(1, 2), center, 3, 3, 2)
  reverse <- .pdpmvar_test_reference$.pdpmvar_covariance_move_ratio(new_labels, old_labels, new_atoms, old_atoms,
    log_weights, lapply(residuals, tcrossprod), c(8, 10, 6), c(1, 2), center, 3, 3, 2)
  old_target <- new_target <- 0
  for (subject in seq_len(3)) {
    old_target <- old_target + log_weights[old_labels[subject]] +
      .pdpmvar_test_reference$.pdpmvar_log_likelihood(residuals[[subject]], old_atoms[[old_labels[subject]]]$covariance)
    new_target <- new_target + log_weights[new_labels[subject]] +
      .pdpmvar_test_reference$.pdpmvar_log_likelihood(residuals[[subject]], new_atoms[[new_labels[subject]]]$covariance)
  }
  old_target <- old_target + sum(vapply(old_atoms, .pdpmvar_test_reference$.pdpmvar_covariance_log_base, numeric(1), cov_shape = 3, cov_rate = 2))
  new_target <- new_target + sum(vapply(new_atoms, .pdpmvar_test_reference$.pdpmvar_covariance_log_base, numeric(1), cov_shape = 3, cov_rate = 2))
  expect_equal(forward$target_change, new_target - old_target)
  expect_equal(forward$log_ratio, new_target - old_target + forward$log_reverse - forward$log_forward)
  expect_equal(forward$log_ratio, -reverse$log_ratio)
  expect_equal(forward$log_forward, reverse$log_reverse)
  expect_equal(forward$log_reverse, reverse$log_forward)
  expect_equal(old_target + forward$log_forward + min(0, forward$log_ratio),
               new_target + reverse$log_forward + min(0, reverse$log_ratio))
})

test_that("the optional move integrates into the complete sampler", {
  set.seed(64)
  fixture <- simulate_pdpmvar_data(n_subjects = 6, n_nodes = 3, n_timepoints = 15)
  fit <- pdpmvar(fixture$data, n_factors = 2, initial_clusters = 1,
    n_burnin = 2, n_mcmc = 8, split_merge_every = 1, split_merge_slots = 4)
  expect_equal(nrow(fit$split_merge), 10)
  expect_true(any(fit$split_merge$type == "split"))
  expect_true(all(is.finite(fit$coefficient_mean)))
  expect_true(all(is.finite(fit$covariance_mean)))
  expect_equal(fit$settings$split_merge_slots, 4)
  expect_error(pdpmvar(fixture$data, split_merge_every = -1), "whole number")
})

test_that("split/merge handles empty label gaps and unchanged occupied groups", {
  set.seed(66)
  residuals <- lapply(seq_len(4), function(i) matrix(rnorm(12), 2))
  old_atoms <- lapply(seq_len(5), function(i) .pdpmvar_test_reference$.pdpmvar_new_covariance(2, 1, 3, 2))
  center <- .pdpmvar_initialize_covariance(tcrossprod(residuals[[2]]) / 6, 1)
  new_atoms <- old_atoms
  new_atoms[[1]] <- .pdpmvar_test_reference$.pdpmvar_propose_covariance(center)
  old_labels <- c(2, 2, 4, 4)
  new_labels <- c(2, 1, 4, 4)
  log_weights <- log(c(0.05, 0.4, 0.05, 0.3, 0.05))
  forward <- .pdpmvar_test_reference$.pdpmvar_covariance_move_ratio(old_labels, new_labels, old_atoms, new_atoms,
    log_weights, lapply(residuals, tcrossprod), rep(6, 4), c(1, 2), center, 5, 3, 2)
  reverse <- .pdpmvar_test_reference$.pdpmvar_covariance_move_ratio(new_labels, old_labels, new_atoms, old_atoms,
    log_weights, lapply(residuals, tcrossprod), rep(6, 4), c(1, 2), center, 5, 3, 2)
  expect_equal(forward$log_ratio, -reverse$log_ratio)
  expect_true(is.finite(forward$log_ratio))
})
