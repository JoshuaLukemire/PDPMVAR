test_that("lag coefficient draws preserve cross-lag information with different partitions", {
  D <- 2
  labels <- list(c(1, 1, 3), c(1, 2, 1))
  X <- list(matrix(c(2, -1, 1, 4, 3, 2, -2, 3), D),
            matrix(c(-1, 4, 3, 1, 2, 3, 1, 1, 3, -2), D),
            matrix(c(3, 1, -2, 1, 4, 3, 1, 2), D))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  prior <- list(matrix(c(0.2, 0.4, 0.8, 0.5), D), matrix(c(0.3, 0.8, 0.2, 0.9), D))
  noise_variance <- matrix(c(1, 3, 2, 4, 3, 1), D)
  empty_atom <- list(coefficient = matrix(-99, D, D))
  atoms <- list(rep(list(empty_atom), 3), rep(list(empty_atom), 2))

  # Build the equivalent regression by hand: lag 1 groups 1 and 3,
  # then lag 2 groups 1 and 2. Each group occupies two columns.
  columns <- list(c(1, 2, 5, 6), c(1, 2, 7, 8), c(3, 4, 5, 6))
  Z <- lapply(seq_along(X), function(i) {
    z <- matrix(0, ncol(X[[i]]), 8)
    z[, columns[[i]]] <- t(design[[i]])
    return(z)
  })
  Z_all <- do.call(rbind, Z)
  expect_gt(abs(crossprod(Z_all)[1, 5]), 0)

  set.seed(27)
  expected <- vapply(seq_len(D), function(row) {
    v <- rep(noise_variance[row, ], vapply(X, ncol, integer(1)))
    y <- unlist(lapply(X, function(x) x[row, ]))
    prior_row <- c(rep(prior[[1]][row, ], 2), rep(prior[[2]][row, ], 2))
    precision <- crossprod(Z_all, Z_all / v) + diag(1 / prior_row)
    rhs <- crossprod(Z_all, y / v)
    return(as.numeric(solve(precision, rhs) + backsolve(chol(precision), rnorm(8))))
  }, numeric(8))

  set.seed(27)
  actual <- .lgpdpmvar_eigen_coefficient_update(X, design, lapply(design, tcrossprod),
    labels, atoms, prior, noise_variance)
  stacked <- rbind(t(actual[[1]][[1]]$coefficient), t(actual[[1]][[3]]$coefficient),
                   t(actual[[2]][[1]]$coefficient), t(actual[[2]][[2]]$coefficient))
  expect_equal(stacked, expected, tolerance = 1e-12)
  expect_identical(actual[[1]][[2]], empty_atom)
  expect_identical(atoms[[1]][[1]], empty_atom)
})

test_that("the one-lag coefficient draw agrees with PDPMVAR", {
  X <- list(matrix(c(2, 3, 1, 5, 2, -1, -1, 3), 2), matrix(seq_len(10) / 7, 2))
  design <- lapply(X, .pdpmvar_lag_design, K = 1)
  cross <- lapply(design, tcrossprod)
  labels <- c(1, 1)
  atoms <- list(list(coefficient = matrix(0, 2, 2)))
  prior <- matrix(c(1, 2, 3, 4) / 4, 2)
  variances <- matrix(c(2, 3, 5, 4), 2)

  set.seed(68)
  full <- .pdpmvar_eigen_coefficient_update(X, design, cross,
    lapply(X, function(x) x * 0), labels, atoms, prior, variances)
  set.seed(68)
  lag <- .lgpdpmvar_eigen_coefficient_update(X, design, cross,
    list(labels), list(atoms), list(prior), variances)
  expect_equal(lag[[1]], full, tolerance = 1e-12)
})

test_that("three-lag fits retain subject draws and separate grouping summaries", {
  set.seed(581)
  X <- lapply(13:18, function(n) matrix(rnorm(2 * n), 2))
  fit <- lgpdpmvar(X, K = 3, n_factors = 1, initial_clusters = 2,
    n_burnin = 3, n_mcmc = 8, store_draws = TRUE, split_merge_every = 2,
    coefficient_concentration = c(0.8, 1, 1.2), cov_shape = 3, cov_rate = 2,
    coefficient_split_merge_every = 2, coefficient_split_merge = "collapsed")
  expect_s3_class(fit, "lgpdpmvar")
  expect_equal(dim(fit$coefficient_mean), c(2, 6, 6))
  expect_equal(dim(fit$lambda), c(8, 3))
  expect_equal(dim(fit$draws$coefficients), c(2, 6, 6, 8))
  expect_equal(colnames(fit$cluster_counts), c("lag_1", "lag_2", "lag_3", "covariance"))
  expect_equal(fit$node_means, do.call(cbind, lapply(X, rowMeans)))
  expect_equal(fit$coefficient_mean, apply(fit$draws$coefficients, 1:3, mean), tolerance = 1e-12)
  expect_equal(fit$covariance_mean, apply(fit$draws$covariance, 1:3, mean), tolerance = 1e-12)
  expect_equal(fit$coefficient_sign_probability, apply(fit$draws$coefficients > 0, 1:3, mean))
  for (lag in 1:3) {
    expect_equal(which(fit$coefficient_split_merge[[lag]]$type != "none"), seq(2, 10, by = 2))
    labels <- fit$coefficient_labels[[lag]]
    expect_equal(dim(labels), c(8, 6))
    expect_equal(fit$coefficient_coclustering[[lag]],
      Reduce("+", lapply(seq_len(nrow(labels)), function(i) outer(labels[i, ], labels[i, ], "=="))) / nrow(labels))
    expect_equal(unname(tail(fit$cluster_counts[, lag], 8)), apply(labels, 1, function(x) length(unique(x))))
    for (subject in seq_along(X)) {
      label <- fit$final_state$coefficient_labels[[lag]][subject]
      columns <- (lag - 1) * 2 + 1:2
      expect_equal(fit$draws$coefficients[, columns, subject, 8],
        fit$final_state$coefficient_atoms[[lag]][[label]]$coefficient)
    }
  }

  # The saved likelihood uses the sum of all three lag contributions.
  log_likelihood <- vapply(seq_along(X), function(i) {
    x <- sweep(X[[i]], 1, fit$node_means[, i])
    residual <- x - fit$draws$coefficients[, , i, 8] %*% .pdpmvar_lag_design(x, 3)
    sigma <- fit$draws$covariance[, , i, 8]
    return(-sum(forwardsolve(t(chol(sigma)), residual)^2) / 2 -
             ncol(x) * sum(log(diag(chol(sigma)))))
  }, numeric(1))
  expect_equal(unname(fit$parameter_traces[8, "log_likelihood"]), sum(log_likelihood))
  expect_true(any(fit$split_merge$type != "none"))
  expect_error(lgpdpmvar(X, K = 3, coefficient_concentration = c(1, 2)), "one per lag")
})

test_that("one-lag samplers agree from the same starting state", {
  set.seed(52)
  X <- lapply(12:15, function(n) matrix(rnorm(n), 1))
  design <- lapply(X, .pdpmvar_lag_design, K = 1)
  cross <- lapply(design, tcrossprod)
  state <- .lgpdpmvar_initialize(X, design, cross, 2, 1, 3, 2, "random")
  state$shared_variance <- list(matrix(0.25, 1, 1))
  state$lambda_squared <- 1
  control <- list(n_burnin = 0, n_mcmc = 8, n_factors = 1,
    coefficient_concentration = 1, covariance_concentration = 1,
    cov_shape = 3, cov_rate = 2, coefficient_prior = "laplace", laplace = TRUE,
    store_draws = TRUE, verbose = FALSE, split_merge_every = 0, split_merge_slots = 10,
    coefficient_split_merge_every = 0, coefficient_split_merge_slots = 10)
  full_state <- state
  full_state$coefficient_atoms <- state$coefficient_atoms[[1]]
  full_state$coefficient_labels <- state$coefficient_labels[[1]]
  full_state$shared_variance <- state$shared_variance[[1]]
  callbacks <- list(message = base::message, assign_coefficients = .pdpmvar_assign_coefficients,
                    assign_covariances = .pdpmvar_assign_covariances)

  set.seed(413)
  full <- .pdpmvar_eigen_sampler(X, design, cross, full_state, control, callbacks)
  set.seed(413)
  lag <- .lgpdpmvar_eigen_sampler(X, design, cross, state, control)
  expect_equal(lag$coefficient_labels[[1]], full$coefficient_labels)
  expect_equal(lag$covariance_labels, full$covariance_labels)
  expect_equal(lag$draws, full$draws, tolerance = 1e-10)
  expect_equal(as.numeric(lag$lambda), full$lambda, tolerance = 1e-10)
})

test_that("lag split/merge decisions match dense references using current partial residuals", {
  set.seed(7301)
  D <- 2
  N <- 5
  X <- lapply(9:13, function(n) matrix(rnorm(D * n), D))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  adjusted <- lapply(X, function(x) x - matrix(rnorm(length(x), sd = 0.2), D))
  design_cross <- lapply(design, tcrossprod)
  labels <- list(rep(1, N), c(1, 3, 1, 3, 3))
  variance <- list(matrix(seq(0.1, 0.4, by = 0.1), D), matrix(seq(0.2, 0.5, by = 0.1), D))
  v <- matrix(seq(0.6, 1.5, length.out = D * N), D)
  concentration <- c(0.8, 1.6)
  atoms <- lapply(1:2, function(lag) {
    replicate(4, list(coefficient = matrix(rnorm(D^2, sd = 0.2), D)), simplify = FALSE)
  })
  original <- serialize(list(labels, atoms), NULL)
  for (collapsed in c(FALSE, TRUE)) {
    response <- if (collapsed) X else adjusted
    reference_covariance <- lapply(seq_len(N), function(i) {
      sigma <- diag(v[, i])
      if (collapsed) sigma[1, 2] <- sigma[2, 1] <- (-1)^i * 0.3
      return(sigma)
    })
    covariances <- if (collapsed) reference_covariance else NULL
    saw <- character()
    for (seed in 7302:7309) {
      set.seed(seed)
      actual <- .lgpdpmvar_eigen_coefficient_moves(response, design, design_cross,
        labels, atoms, variance, v, concentration, 4, covariances)
      rng <- .Random.seed

      # Rebuild the partial residual directly. The second lag must see the
      # first lag's refreshed draws, including when the split is rejected.
      set.seed(seed)
      expected <- list(labels = labels, atoms = atoms, records = vector("list", 2))
      for (lag in 1:2) {
        other <- 3 - lag
        other_rows <- (other - 1) * D + seq_len(D)
        lag_rows <- (lag - 1) * D + seq_len(D)
        partial <- lapply(seq_len(N), function(i) {
          other_atom <- expected$atoms[[other]][[expected$labels[[other]][i]]]$coefficient
          return(response[[i]] - other_atom %*% design[[i]][other_rows, , drop = FALSE])
        })
        lag_design <- lapply(design, function(z) z[lag_rows, , drop = FALSE])
        move <- .pdpmvar_test_reference$.pdpmvar_coefficient_split_merge_reference(
          partial, lag_design, lapply(lag_design, tcrossprod),
          reference_covariance,
          labels[[lag]], atoms[[lag]], variance[[lag]], concentration[lag], 4)
        expected$atoms[[lag]] <- move$atoms
        expected$labels[[lag]] <- move$labels
        expected$records[[lag]] <- move$record
        saw <- c(saw, move$record$type)
      }
      expect_equal(actual, expected, tolerance = 1e-9)
      expect_identical(.Random.seed, rng)
    }
    expect_true(all(c("split", "merge") %in% saw))
  }
  expect_identical(serialize(list(labels, atoms), NULL), original)
})

test_that("lag diagnostics compare each partition and ignore group names", {
  set.seed(7431)
  X <- lapply(12:17, function(n) matrix(rnorm(2 * n), 2))
  chains <- lgpdpmvar_chains(X, seeds = c(7, 9), initial_clusters = c(1, 3),
    n_burnin = 2, n_mcmc = 8, n_factors = 1, cov_shape = 3, cov_rate = 2,
    coefficient_split_merge_every = 2, split_merge_every = 2)
  fit <- chains$fits[[1]]
  result <- chains$diagnostics
  expect_identical(result, lgpdpmvar_diagnostics(chains$fits))
  expect_equal(result$chains$partition, rep(c("lag_1", "lag_2", "covariance"), 2))
  expect_equal(result$coclustering_differences$partition, c("lag_1", "lag_2", "covariance"))
  expect_true(all(c("lag_1_clusters", "lag_2_clusters", "covariance_clusters",
                    "lag_1_lambda", "lag_2_lambda") %in% result$rhat$quantity))
  for (lag in 1:2) {
    trace <- fit$coefficient_split_merge[[lag]]
    expect_equal(which(trace$type != "none"), seq(2, 10, by = 2))
    expect_equal(result$chains$accepted_merges[lag], sum(trace$type == "merge" & trace$accepted))
  }

  relabeled <- fit
  relabeled$coefficient_labels[[2]] <- relabeled$coefficient_labels[[2]] + 100
  unchanged <- lgpdpmvar_diagnostics(list(fit, relabeled))
  expect_equal(unchanged$chains$partition_changes[2], unchanged$chains$partition_changes[5])
  expect_equal(unchanged$coclustering_differences$max_difference, rep(0, 3))

  wrong <- fit
  wrong$settings$coefficient_concentration[2] <- 2
  expect_error(lgpdpmvar_diagnostics(list(fit, wrong)), "matching data dimensions and model settings")
})

test_that("collapsed lag split and merge ratios reverse at fixed weights", {
  X <- list(matrix(c(2, 1, 3, 2, 4, 2, 6, 1), 2), matrix(seq_len(10) / 3, 2),
            matrix(c(1, 5, 2, 1, 2, 0, 4, 3), 2))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  other <- matrix(c(0.1, 0.3, -0.2, 0.4), 2)
  partial <- lapply(seq_along(X), function(i) X[[i]] - other %*% design[[i]][1:2, ])
  Z <- lapply(design, function(z) z[3:4, , drop = FALSE])
  cross <- lapply(Z, tcrossprod)
  covariance <- list(matrix(c(1, 0.4, 0.4, 2), 2), matrix(c(2, -0.4, -0.4, 3), 2),
                     matrix(c(3, 0.8, 0.8, 2), 2))
  variance <- matrix(c(0.2, 0.4, 0.1, 0.7), 2)
  merged <- c(1, 1, 1)
  split <- c(1, 3, 1)
  weights <- log(c(0.4, 0.1, 0.3, 0.1))
  anchors <- c(1L, 2L)

  # Express the same full covariance matrices as a rank-one factor plus
  # positive diagonal variances, with mixed covariance labels in the union.
  atoms <- lapply(covariance, function(sigma) {
    loadings <- matrix(c(sqrt(abs(sigma[1, 2])), sign(sigma[1, 2]) * sqrt(abs(sigma[1, 2]))), 2)
    list(working_loadings = loadings, factor_precision = 1,
         variance = diag(sigma - tcrossprod(loadings)))
  })
  cpp <- .pdpmvar_eigen_statistics(partial, Z, cross, covariance, variance)
  forward <- .lgpdpmvar_eigen_collapsed_ratio(partial, Z, cross, atoms, 1:3,
    variance, merged, split, weights, anchors, 4)
  reverse <- .lgpdpmvar_eigen_collapsed_ratio(partial, Z, cross, atoms, 1:3,
    variance, split, merged, weights, anchors, 4)
  expect_equal(forward,
    .pdpmvar_eigen_coefficient_move_ratio(merged, split, cpp, weights, anchors, 4), tolerance = 1e-10)
  expect_equal(reverse,
    .pdpmvar_eigen_coefficient_move_ratio(split, merged, cpp, weights, anchors, 4), tolerance = 1e-10)
  expect_equal(forward$log_ratio, -reverse$log_ratio)
  expect_equal(forward$log_forward, reverse$log_reverse)
})

test_that("low-rank atom refreshes have the dense Gaussian mean and covariance", {
  Y <- list(matrix(c(0.2, 0.5, 0.1, -0.2, -0.4, 0.7), 2),
            matrix(c(-0.1, 0.4, 0.2, 0.5, 0.8, 0.1), 2))
  Z <- list(matrix(c(1, 2, -2, 1, 0, 1), 2), matrix(c(2, 1, 1, -2, 1, 0), 2))
  cross <- lapply(Z, tcrossprod)
  loading <- matrix(c(1.2, -0.9), 2)
  covariance <- list(list(working_loadings = loading, factor_precision = 0.8,
    variance = c(0.4, 0.6)))
  sigma <- tcrossprod(loading) / 0.8 + diag(c(0.4, 0.6))
  variance <- matrix(c(0.2, 1, 0.7, 1.5), 2)
  atoms <- list(rep(list(list(coefficient = matrix(0, 2, 2))), 2))

  dense <- .pdpmvar_eigen_statistics(Y, Z, cross, list(sigma, sigma), variance)
  gaussian <- lapply(list(1, 1:2), function(members) {
    .pdpmvar_eigen_integral(dense$prior_precision + Reduce("+", dense$precision[members]),
      Reduce("+", dense$information[members]), dense$prior_log_determinant_half)
  })

  # The affected atom is refreshed after either outcome. Standardize the
  # draw under the dense conditional corresponding to the selected labels.
  set.seed(7321)
  standardized <- replicate(1000, {
    move <- .lgpdpmvar_eigen_coefficient_moves(Y, Z, cross, list(c(1, 2)),
      atoms, list(variance), matrix(numeric(), 0, 0), 1, 4, covariance, c(1, 1))
    labels <- move$labels[[1]]
    distribution <- gaussian[[if (labels[1] == labels[2]) 2 else 1]]
    coefficient <- move$atoms[[1]][[labels[1]]]$coefficient
    as.numeric(distribution$upper %*% (as.numeric(coefficient) - distribution$mean))
  })
  expect_lt(max(abs(rowMeans(standardized))), 0.15)
  expect_lt(max(abs(stats::cov(t(standardized)) - diag(4))), 0.16)
})

test_that("collapsed relocation ratios match dense integration in both directions", {
  D <- 2
  Y <- list(matrix(c(0.2, 0.5, 0.1, -0.2, -0.4, 0.7), D),
            matrix(c(-0.1, 0.4, 0.2, 0.5, 0.8, 0.1), D),
            matrix(c(0.3, 0.2, 0.1, 0.9, -0.4, 0.7), D))
  Z <- lapply(Y, .pdpmvar_lag_design, K = 1)
  cross <- lapply(Z, tcrossprod)
  loading <- matrix(c(1.2, -0.9), D)
  cov_atoms <- list(list(working_loadings = loading, factor_precision = 0.8,
    variance = c(0.4, 0.6)))
  sigma <- tcrossprod(loading) / 0.8 + diag(c(0.4, 0.6))
  cov_atoms[[1]]$covariance <- sigma
  tau <- matrix(c(0.2, 1, 0.7, 1.5), D)
  atoms <- rep(list(list(coefficient = matrix(0, D, D))), 4)
  weights <- log(c(0.2, 0.4, 0.15, 0.1))
  labels <- c(1, 2, 2)
  dense <- .pdpmvar_eigen_statistics(Y, Z, cross, rep(list(sigma), 3), tau)
  evidence <- function(groups) {
    sum(vapply(split(seq_along(groups), groups), function(indices) {
      .pdpmvar_eigen_integral(dense$prior_precision + Reduce("+", dense$precision[indices]),
        Reduce("+", dense$information[indices]), dense$prior_log_determinant_half)$log_integral
    }, numeric(1)))
  }

  set.seed(7514)
  rng <- .Random.seed
  # Merge a singleton into an occupied group, then split it back out.
  # Also move a member of a non-singleton directly to an empty slot.
  for (pair in list(c(1, 2), c(2, 4))) {
    subject <- pair[1]
    destination <- pair[2]
    proposed <- replace(labels, subject, destination)
    forward <- .lgpdpmvar_eigen_relocation_move(Y, Z, cross, cov_atoms, rep(1, 3),
      labels, atoms, tau, 1, 4, subject, destination, weights, TRUE)
    reverse <- .lgpdpmvar_eigen_relocation_move(Y, Z, cross, cov_atoms, rep(1, 3),
      proposed, atoms, tau, 1, 4, subject, labels[subject], weights, TRUE)
    expected <- evidence(proposed) - evidence(labels)
    expect_equal(forward$gaussian, expected, tolerance = 1e-10)
    expect_equal(forward$log_ratio, expected + weights[destination] - weights[labels[subject]])
    expect_equal(forward$log_ratio, -reverse$log_ratio)
    expect_equal(forward$log_forward, reverse$log_reverse)
    expect_equal(forward$log_forward, -log(3) - log(3))

    scores <- vapply(seq_len(4), function(group) {
      members <- setdiff(which(labels == group), subject)
      if (!length(members)) return(-Inf)
      ridge <- max(1e-8, 0.1 * mean(diag(cross[[subject]])))
      precision <- Reduce("+", cross[members]) + diag(ridge, D)
      information <- Reduce("+", lapply(members, function(i) Z[[i]] %*% t(Y[[i]])))
      residual <- Y[[subject]] - t(solve(precision, information)) %*% Z[[subject]]
      sum(forwardsolve(t(chol(sigma)), residual)^2) / (-0.1 * length(residual))
    }, numeric(1))
    proposal <- function(from, to) {
      eligible <- setdiff(seq_len(4), from)
      if (all(!is.finite(scores[eligible]))) return(-log(3) - log(3))
      mass <- exp(scores[eligible] - max(scores[eligible]))
      log((0.8 * mass[match(to, eligible)] / sum(mass) + 0.2 / 3) / 3)
    }
    subject_probability <- function(groups) {
      errors <- vapply(seq_along(groups), function(i) {
        occupied <- unique(groups)
        prediction_errors <- vapply(occupied, function(group) {
          members <- setdiff(which(groups == group), i)
          residual <- Y[[i]]
          if (length(members)) {
            ridge <- max(1e-8, 0.1 * mean(diag(cross[[i]])))
            precision <- Reduce("+", cross[members]) + diag(ridge, D)
            information <- Reduce("+", lapply(members, function(j) Z[[j]] %*% t(Y[[j]])))
            residual <- residual - t(solve(precision, information)) %*% Z[[i]]
          }
          sum(forwardsolve(t(chol(sigma)), residual)^2) / (0.1 * length(residual))
        }, numeric(1))
        return(prediction_errors[match(groups[i], occupied)] - min(prediction_errors))
      }, numeric(1))
      mass <- exp(errors - max(errors))
      return(0.8 * mass[subject] / sum(mass) + 0.2 / length(groups))
    }

    for (method in c("destination", "guided")) {
      guided <- .lgpdpmvar_eigen_relocation_move(Y, Z, cross, cov_atoms, rep(1, 3),
        labels, atoms, tau, 1, 4, subject, destination, weights, TRUE, method)
      guided_reverse <- .lgpdpmvar_eigen_relocation_move(Y, Z, cross, cov_atoms, rep(1, 3),
        proposed, atoms, tau, 1, 4, subject, labels[subject], weights, TRUE, method)
      subject_forward <- if (method == "guided") subject_probability(labels) else 1 / 3
      subject_reverse <- if (method == "guided") subject_probability(proposed) else 1 / 3
      expect_equal(guided$log_subject_forward, log(subject_forward))
      expect_equal(guided$log_subject_reverse, log(subject_reverse))
      expect_equal(guided$log_forward, proposal(labels[subject], destination) + log(3 * subject_forward))
      expect_equal(guided$log_reverse, proposal(destination, labels[subject]) + log(3 * subject_reverse))
      expect_equal(guided$log_forward, guided_reverse$log_reverse)
      expect_equal(guided$log_reverse, guided_reverse$log_forward)
      expect_equal(guided$log_ratio, -guided_reverse$log_ratio)
      expect_equal(guided$log_ratio, forward$log_ratio + guided$log_reverse - guided$log_forward)
    }
  }
  # A pooled state needs a uniform fallback, but the reverse has an occupied destination.
  pooled <- rep(1, 3)
  fallback <- .lgpdpmvar_eigen_relocation_move(Y, Z, cross, cov_atoms, rep(1, 3),
    pooled, atoms, tau, 1, 4, 1, 4, weights, TRUE, "destination")
  expect_equal(fallback$log_forward, -log(3) - log(3))
  expect_equal(fallback$log_reverse, log((0.8 + 0.2 / 3) / 3))

  expect_identical(.Random.seed, rng)
})

test_that("lgpdpmvar schedules and records one relocation per lag", {
  set.seed(7520)
  X <- lapply(12:17, function(n) matrix(rnorm(2 * n), 2))
  fit <- lgpdpmvar(X, n_factors = 1, initial_clusters = 2, n_burnin = 1, n_mcmc = 5,
    cov_shape = 3, cov_rate = 2, coefficient_relocation_every = 2,
    coefficient_split_merge_every = 2, split_merge_every = 2)
  for (trace in fit$coefficient_relocation) {
    expect_s3_class(trace, "data.frame")
    expect_equal(nrow(trace), 6)
    expect_equal(which(trace$type == "relocate"), c(2, 4, 6))
    expect_true(all(trace$subject[trace$type == "relocate"] %in% seq_along(X)))
    expect_true(all(with(trace, source[type == "relocate"] != destination[type == "relocate"])))
  }
})
