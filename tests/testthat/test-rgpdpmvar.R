test_that("the row sampler uses factor-adjusted slice scores and full-lag Gaussian draws", {
  set.seed(710)
  X <- lapply(c(5, 6, 7, 8), function(n) matrix(rnorm(2 * n), 2))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  cross <- lapply(design, tcrossprod)
  state <- .rgpdpmvar_initialize(X, design, cross, 2, 1, 3, 2, "random")
  state$coefficient_labels <- list(node_1 = c(1, 1, 2, 2), node_2 = c(1, 2, 1, 2))
  for (node in 1:2) {
    for (group in 1:2) {
      state$coefficient_atoms[[node]][[group]]$coefficient[] <- rnorm(4, sd = 0.2)
    }
  }
  state$shared_variance <- list(matrix(c(0.5, 0.7, 1, 1.4), 1), matrix(c(1, 1.5, 0.7, 2), 1))
  state$lambda_squared <- c(1, 2)
  control <- list(n_factors = 1, n_burnin = 0, n_mcmc = 1, split_merge_every = 0,
                  split_merge_slots = 10, coefficient_split_merge_every = 0,
                  coefficient_split_merge_slots = 10, coefficient_concentration = c(1, 1.5),
                  covariance_concentration = 1, cov_shape = 3, cov_rate = 2,
                  store_draws = TRUE, verbose = FALSE)

  set.seed(711)
  fit <- .rgpdpmvar_eigen_sampler(X, design, cross, state, control)

  # Reproduce the factors and slice variables, then compute the categorical
  # probabilities and Gaussian draws independently in R.
  set.seed(711)
  residual_variances <- vapply(state$covariance_labels, function(group) {
    state$covariance_atoms[[group]]$variance
  }, numeric(2))
  adjusted <- lapply(seq_along(X), function(subject) {
    coefficient <- do.call(rbind, lapply(1:2, function(node) {
      group <- state$coefficient_labels[[node]][subject]
      state$coefficient_atoms[[node]][[group]]$coefficient
    }))
    atom <- state$covariance_atoms[[state$covariance_labels[subject]]]
    residual <- X[[subject]] - coefficient %*% design[[subject]]
    factors <- .pdpmvar_eigen_factors(residual, atom, matrix(rnorm(ncol(residual)), 1))
    X[[subject]] - atom$working_loadings %*% factors
  })

  for (node in 1:2) {
    slice <- .pdpmvar_eigen_slice_weights(state$coefficient_labels[[node]], control$coefficient_concentration[node])
    variance <- state$shared_variance[[node]]
    atoms <- state$coefficient_atoms[[node]][seq_len(length(slice$weights))]
    for (group in seq_along(atoms)) {
      if (is.null(atoms[[group]])) atoms[[group]] <- list(coefficient = matrix(rnorm(4) * sqrt(variance), 1))
    }
    uniforms <- runif(length(X))
    labels <- vapply(seq_along(X), function(subject) {
      eligible <- which(slice$weights > slice$slices[subject])
      scores <- vapply(eligible, function(group) {
        error <- adjusted[[subject]][node, ] - atoms[[group]]$coefficient %*% design[[subject]]
        -sum(error^2) / (2 * residual_variances[node, subject])
      }, numeric(1))
      sorted <- order(scores, decreasing = TRUE)
      probability <- exp(scores[sorted] - max(scores))
      position <- which(cumsum(probability / sum(probability)) >= uniforms[subject])[1]
      as.numeric(eligible[sorted[position]])
    }, numeric(1))
    expect_equal(as.numeric(fit$coefficient_labels[[node]][1, ]), labels)

    for (group in sort(unique(labels))) {
      precision <- diag(1 / as.numeric(variance))
      information <- numeric(4)
      for (subject in which(labels == group)) {
        precision <- precision + cross[[subject]] / residual_variances[node, subject]
        information <- information + design[[subject]] %*% adjusted[[subject]][node, ] /
          residual_variances[node, subject]
      }
      draw <- solve(precision, information) + backsolve(chol(precision), rnorm(4))
      atoms[[group]]$coefficient <- matrix(draw, 1)
      expect_equal(fit$final_state$coefficient_atoms[[node]][[group]]$coefficient,
                   matrix(draw, 1), tolerance = 1e-12)
      subject <- which(labels == group)[1]
      expect_equal(fit$coefficient_mean[node, , subject], as.numeric(draw), tolerance = 1e-12)
    }
    shrinkage <- .pdpmvar_eigen_shrinkage(labels, atoms, variance, state$lambda_squared[node])
    expect_equal(fit$final_state$lambda_squared[node], shrinkage$lambda_squared, tolerance = 1e-12)
    for (group in setdiff(seq_along(atoms), labels)) rnorm(4)
  }

  expect_equal(dim(fit$coefficient_mean), c(2, 4, 4))
  expect_equal(fit$draws$coefficients[, , , 1], fit$coefficient_mean)

  # Saving selected entries preserves their requested order and the fit.
  indices <- c(32, 2, 17)
  control$store_draws <- FALSE
  control$selected_coefficient_indices <- indices
  set.seed(711)
  selected <- .rgpdpmvar_eigen_sampler(X, design, cross, state, control)
  expect_equal(selected$draws$coefficient_indices, indices)
  expect_identical(selected$draws$coefficients, matrix(fit$draws$coefficients[indices], ncol = 1))
  expect_identical(selected[setdiff(names(fit), "draws")], fit[setdiff(names(fit), "draws")])
  control$selected_coefficient_indices <- c(1, 33)
  expect_error(.rgpdpmvar_eigen_sampler(X, design, cross, state, control), "within the subject coefficient array")
})

test_that("rgpdpmvar saves row-specific outputs and runs scheduled group moves", {
  set.seed(712)
  X <- lapply(c(10, 11, 12, 13), function(n) matrix(rnorm(2 * n), 2))
  fit <- rgpdpmvar(X, K = 2, n_burnin = 1, n_mcmc = 3, n_factors = 1, initial_clusters = 1,
                  cov_shape = 3, cov_rate = 2, coefficient_split_merge_every = 1, split_merge_every = 1,
                  coefficient_relocation_every = 2, store_draws = TRUE)
  expect_s3_class(fit, "rgpdpmvar")
  expect_named(fit$coefficient_labels, c("node_1", "node_2"))
  expect_equal(dim(fit$coefficient_labels[[1]]), c(3, 4))
  expect_equal(dim(fit$lambda), c(3, 2))
  expect_equal(dim(fit$shared_variance_mean), c(2, 4))
  expect_equal(fit$node_means, do.call(cbind, lapply(X, rowMeans)))
  expect_true(all(fit$split_merge$type != "none"))
  expect_true(all(vapply(fit$coefficient_split_merge, function(x) all(x$type != "none"), logical(1))))
  expect_named(fit$coefficient_relocation, c("node_1", "node_2"))
  expect_equal(lapply(fit$coefficient_relocation, function(x) which(x$type != "none")),
               list(node_1 = c(2L, 4L), node_2 = c(2L, 4L)))
  expect_equal(fit$coefficient_mean, apply(fit$draws$coefficients, 1:3, mean))
  expect_equal(fit$covariance_mean, apply(fit$draws$covariance, 1:3, mean))
  expect_error(rgpdpmvar(X, coefficient_concentration = c(1, 2, 3)), "per response node")
})

test_that("row diagnostics compare every response separately from covariance", {
  set.seed(713)
  X <- lapply(12:17, function(n) matrix(rnorm(3 * n), 3))
  chains <- rgpdpmvar_chains(X, seeds = c(7, 9), initial_clusters = c(1, 3),
    n_burnin = 2, n_mcmc = 8, n_factors = 1, cov_shape = 3, cov_rate = 2,
    coefficient_split_merge_every = 2, split_merge_every = 2)
  fit <- chains$fits[[1]]
  result <- chains$diagnostics
  partitions <- c("node_1", "node_2", "node_3", "covariance")
  expect_identical(chains$seeds, c(7, 9))
  expect_identical(result, rgpdpmvar_diagnostics(chains$fits))
  expect_equal(result$chains$partition, rep(partitions, 2))
  expect_equal(result$coclustering_differences$partition, partitions)
  expect_true(all(c(paste0(partitions, "_clusters"), paste0(partitions[1:3], "_lambda")) %in%
                    result$rhat$quantity))
  expect_equal(result$chains$initial_clusters, rep(c(1, 3), each = 4))
  trace <- fit$coefficient_split_merge[[3]]
  expect_equal(result$chains$accepted_merges[3], sum(trace$type == "merge" & trace$accepted))

  # Relabel a response in every draw without changing its grouping.
  relabeled <- fit
  relabeled$coefficient_labels[[3]] <- relabeled$coefficient_labels[[3]] + 100
  unchanged <- rgpdpmvar_diagnostics(list(fit, relabeled))
  expect_equal(unchanged$chains$partition_changes[3], unchanged$chains$partition_changes[7])
  expect_equal(unchanged$coclustering_differences$max_difference, rep(0, 4))

  # Change one response's co-clustering probabilities and check that the
  # reported disagreement is confined to that response.
  changed <- fit
  original <- fit$coefficient_coclustering[[3]][1, 2]
  replacement <- if (original > 0.5) 0 else 1
  changed$coefficient_coclustering[[3]][1, 2] <- replacement
  changed$coefficient_coclustering[[3]][2, 1] <- replacement
  differences <- rgpdpmvar_diagnostics(list(fit, changed))$coclustering_differences
  gap <- c(0, 0, abs(replacement - original), 0)
  expect_equal(differences$max_difference, gap)
  expect_equal(differences$mean_difference, gap / choose(length(X), 2))

  wrong <- fit
  wrong$settings$coefficient_concentration[3] <- 2
  expect_error(rgpdpmvar_diagnostics(list(fit, wrong)), "matching data dimensions and model settings")
  expect_error(rgpdpmvar_chains(X, n_mcmc = 7), "n_mcmc")
})

test_that("guided row transfers use the factor-marginal response before fresh factors", {
  set.seed(714)
  X <- lapply(8:13, function(n) matrix(rnorm(2 * n), 2))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  cross <- lapply(design, tcrossprod)
  state <- .rgpdpmvar_initialize(X, design, cross, 2, 1, 3, 2, "random")
  state$coefficient_labels <- list(node_1 = c(1, 1, 1, 2, 2, 2), node_2 = c(1, 2, 1, 2, 1, 2))
  state$shared_variance <- list(matrix(0.2, 1, 4), matrix(0.3, 1, 4))
  state$lambda_squared <- c(1, 2)
  control <- list(n_factors = 1, n_burnin = 0, n_mcmc = 1, split_merge_every = 0,
                  split_merge_slots = 4, coefficient_split_merge_every = 0,
                  coefficient_split_merge_slots = 4, coefficient_concentration = c(1, 1.5),
                  coefficient_relocation_every = 1, covariance_concentration = 1,
                  cov_shape = 3, cov_rate = 2, store_draws = TRUE, verbose = FALSE)

  set.seed(715)
  fit <- .rgpdpmvar_eigen_sampler(X, design, cross, state, control)
  rng <- .Random.seed

  # Recalculate the conditional responses directly from covariance
  # submatrices. After each transfer, use the refreshed coefficient row.
  set.seed(715)
  for (node in 1:2) {
    other <- 3 - node
    conditional <- lapply(state$covariance_atoms, function(atom) {
      sigma <- atom$covariance
      adjustment <- sigma[node, other] / sigma[other, other]
      variance <- sigma[node, node] - adjustment * sigma[other, node]
      list(adjustment = adjustment, variance = variance)
    })
    response <- lapply(seq_along(X), function(subject) {
      covariance <- conditional[[state$covariance_labels[subject]]]
      group <- state$coefficient_labels[[other]][subject]
      row <- state$coefficient_atoms[[other]][[group]]$coefficient
      other_residual <- X[[subject]][other, ] - row %*% design[[subject]]
      matrix(X[[subject]][node, ] - covariance$adjustment * other_residual, nrow = 1)
    })
    covariances <- lapply(conditional, function(covariance) {
      list(covariance = matrix(covariance$variance, 1), variance = covariance$variance,
           working_loadings = matrix(numeric(), 1, 0), factor_precision = numeric())
    })
    move <- .lgpdpmvar_eigen_relocation_move(response, design, cross, covariances,
      state$covariance_labels, state$coefficient_labels[[node]], state$coefficient_atoms[[node]],
      state$shared_variance[[node]], control$coefficient_concentration[node], 4, method = "guided")
    fields <- c("type", "accepted", "log_ratio", "subject", "source", "destination")
    expect_equal(as.list(fit$coefficient_relocation[[node]][1, fields]), move$record[fields],
                 tolerance = 1e-10)
    state$coefficient_labels[[node]] <- move$labels
    state$coefficient_atoms[[node]] <- move$atoms
  }

  control$coefficient_relocation_every <- 0
  reference <- .rgpdpmvar_eigen_sampler(X, design, cross, state, control)
  fit$coefficient_relocation <- NULL
  expect_equal(fit, reference, tolerance = 1e-10)
  expect_identical(.Random.seed, rng)
})
