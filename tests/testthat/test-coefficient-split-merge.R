test_that("joint coefficient proposal matches a direct vectorized Gaussian posterior", {
  set.seed(71)
  X <- list(matrix(rnorm(14), 2), matrix(rnorm(16), 2))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  covariances <- list(matrix(c(2, 0.5, 0.5, 1), 2), matrix(c(1, -0.2, -0.2, 3), 2))
  precisions <- lapply(covariances, solve)
  variance <- matrix(seq(0.1, 0.8, length.out = 8), 2)
  proposal <- .pdpmvar_test_reference$.pdpmvar_coefficient_gaussian(X, design, lapply(design, tcrossprod), precisions, variance, 1:2)
  precision <- diag(1 / as.numeric(variance))
  information <- numeric(8)
  for (subject in 1:2) {
    full_design <- kronecker(t(design[[subject]]), diag(2))
    residual_precision <- kronecker(diag(ncol(X[[subject]])), precisions[[subject]])
    precision <- precision + crossprod(full_design, residual_precision %*% full_design)
    information <- information + as.numeric(crossprod(full_design, residual_precision %*% as.numeric(X[[subject]])))
  }
  expect_equal(crossprod(proposal$upper), precision)
  expect_equal(proposal$mean, as.numeric(solve(precision, information)))
  statistics <- .pdpmvar_test_reference$.pdpmvar_coefficient_statistics(X, design, lapply(design, tcrossprod), covariances, variance)
  group <- .pdpmvar_test_reference$.pdpmvar_coefficient_group(statistics, 1:2)
  expect_equal(group$mean, proposal$mean)
  full_design <- do.call(rbind, lapply(design, function(z) kronecker(t(z), diag(2))))
  noise_covariance <- matrix(0, 30, 30)
  noise_covariance[1:14, 1:14] <- kronecker(diag(7), covariances[[1]])
  noise_covariance[15:30, 15:30] <- kronecker(diag(8), covariances[[2]])
  marginal_covariance <- noise_covariance + full_design %*% diag(as.numeric(variance)) %*% t(full_design)
  marginal_log_likelihood <- .pdpmvar_test_reference$.pdpmvar_log_likelihood(matrix(unlist(X), 30), marginal_covariance)
  zero_log_likelihood <- sum(vapply(seq_along(X), function(i)
    .pdpmvar_test_reference$.pdpmvar_log_likelihood(X[[i]], covariances[[i]]), numeric(1)))
  expect_equal(group$log_integral, marginal_log_likelihood - zero_log_likelihood)

})

test_that("coefficient split and reverse merge satisfy continuous detailed balance", {
  set.seed(72)
  X <- lapply(seq_len(3), function(i) matrix(rnorm(20), 2))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  covariances <- rep(list(matrix(c(2, 0.5, 0.5, 1), 2)), 3)
  variance <- matrix(seq(0.1, 0.8, length.out = 8), 2)
  statistics <- .pdpmvar_test_reference$.pdpmvar_coefficient_statistics(X, design, lapply(design, tcrossprod), covariances, variance)
  old_labels <- c(1, 1, 1)
  new_labels <- c(1, 2, 2)
  weights <- log(c(0.6, 0.3, 0.05))
  forward <- .pdpmvar_test_reference$.pdpmvar_coefficient_collapsed_ratio(old_labels, new_labels, statistics, weights, c(1, 2), 3)
  reverse <- .pdpmvar_test_reference$.pdpmvar_coefficient_collapsed_ratio(new_labels, old_labels, statistics, weights, c(1, 2), 3)
  expect_equal(forward$log_ratio, -reverse$log_ratio)
  expect_equal(forward$log_forward, reverse$log_reverse)
  expect_equal(forward$target_change + forward$log_reverse - forward$log_forward, forward$log_ratio)
  expect_equal(forward$log_forward + min(0, forward$log_ratio),
    forward$target_change + reverse$log_forward + min(0, reverse$log_ratio))
  first <- .pdpmvar_test_reference$.pdpmvar_coefficient_restricted_split(statistics, 1:3, c(1, 2), weights, 1, 2,
    forced_labels = c(1, 2, 1))
  second <- .pdpmvar_test_reference$.pdpmvar_coefficient_restricted_split(statistics, 1:3, c(1, 2), weights, 1, 2,
    forced_labels = c(1, 2, 2))
  expect_equal(exp(first$log_probability) + exp(second$log_probability), 1)

})

test_that("both partition moves integrate with shared shrinkage", {
  set.seed(73)
  fixture <- simulate_pdpmvar_data(n_subjects = 6, n_nodes = 2, n_timepoints = 12)
  fit <- pdpmvar(fixture$data, initial_clusters = 1, n_factors = 1, n_burnin = 2, n_mcmc = 8,
    split_merge_every = 1, coefficient_split_merge_every = 1, coefficient_split_merge_slots = 4)
  expect_true(any(fit$coefficient_split_merge$type == "split"))
  expect_true(all(is.finite(fit$coefficient_mean)))
  expect_true(all(is.finite(fit$covariance_mean)))
  expect_true(all(fit$final_state$shared_variance > 0))
  expect_equal(dim(fit$coefficient_mean), c(2, 4, 6))
})
