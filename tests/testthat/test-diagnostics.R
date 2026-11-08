test_that("data initialization is reproducible and handles duplicate subjects", {
  set.seed(41)
  fixture <- simulate_pdpmvar_data(n_subjects = 6, n_nodes = 3, n_timepoints = 15)
  design <- lapply(fixture$data, .pdpmvar_lag_design, K = 2)
  initial <- .pdpmvar_initialize_data(fixture$data, design, lapply(design, tcrossprod), 3, 2)
  expect_length(initial$covariance_labels, 6)
  expect_equal(length(unique(initial$coefficient_labels)), 3)
  for (subject in seq_len(6)) {
    expect_true(all(is.finite(initial$coefficient_atoms[[initial$coefficient_labels[subject]]]$coefficient)))
  }
  labels <- .pdpmvar_initial_labels(matrix(1, 4, 2), 3)
  expect_equal(labels, rep(1, 4))
  expect_equal(.pdpmvar_initial_labels(matrix(1:4, 4), 4), 1:4)
})

test_that("rank R-hat flags constants and shifted chains", {
  expect_true(is.na(.pdpmvar_rank_rhat(matrix(1, 20, 2))))
  expect_identical(.pdpmvar_rank_rhat(cbind(rep(1, 20), rep(2, 20))), Inf)
  set.seed(42)
  draws <- matrix(rnorm(4000), 1000, 4)
  expect_lt(.pdpmvar_rank_rhat(draws), 1.01)
  draws[, 1] <- draws[, 1] + 3
  expect_gt(.pdpmvar_rank_rhat(draws), 1.1)
  expect_equal(.pdpmvar_partition_changes(rbind(c(1, 1, 2), c(3, 3, 1), c(1, 2, 2))), 1)
})

test_that("multiple-chain diagnostics work on unequal-length one-node series", {
  x <- list(matrix(seq(-1, 1, length.out = 8), 1), matrix(c(0, 1, -1, 0, 2), 1))
  result <- pdpmvar_chains(x, seeds = c(43, 44), initial_clusters = 1,
    initialization = c("data", "random"), K = 1, n_factors = 1,
    n_burnin = 2, n_mcmc = 8, coefficient_prior = "normal")
  expect_length(result$fits, 2)
  expect_equal(result$diagnostics$chains$initialization, c("data", "random"))
  expect_true(all(c("log_likelihood", "covariance_clusters") %in% result$diagnostics$rhat$quantity))
  expect_equal(nrow(result$diagnostics$coclustering_differences), 2)
  expect_true(all(is.finite(result$fits[[1]]$parameter_traces)))
  expect_error(pdpmvar_chains(x, seeds = c(1, 1)), "distinct")
  expect_error(pdpmvar_chains(x, seeds = c(1, 2), initial_clusters = 1, n_mcmc = 4), "whole number")
  expect_error(pdpmvar_diagnostics(result$fits[1]), "at least two")
})

test_that("diagnostics detect frozen disagreement without comparing label numbers", {
  x <- list(matrix(seq(-1, 1, length.out = 8), 1), matrix(c(0, 1, -1, 0, 2), 1))
  set.seed(45)
  fit <- pdpmvar(x, K = 1, initial_clusters = 1, n_burnin = 0, n_mcmc = 8)
  second <- fit
  fit$covariance_labels[,] <- 1
  fit$covariance_coclustering[,] <- 1
  second$covariance_labels[,] <- matrix(rep(c(1, 2), each = 8), 8)
  second$covariance_coclustering <- diag(2)
  fit$cluster_counts[, "covariance"] <- 1
  second$cluster_counts[, "covariance"] <- 2
  diagnostic <- pdpmvar_diagnostics(list(fit, second))
  expect_identical(diagnostic$rhat$rhat[diagnostic$rhat$quantity == "covariance_clusters"], Inf)
  expect_equal(diagnostic$chains$covariance_partition_changes, c(0, 0))
  expect_equal(max(diagnostic$coclustering_differences$max_difference), 1)
  expect_true(any(grepl("never change", diagnostic$flags)))
})
