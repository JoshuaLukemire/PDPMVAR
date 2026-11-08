test_that("automatic centering matches manual centering with unchanged priors and draws", {
  set.seed(9601)
  X <- lapply(1:4, function(i) {
    x <- matrix(rnorm(2 * (14 + i), sd = 0.8 + i / 10), 2)
    rownames(x) <- c("node_a", "node_b")
    return(sweep(x, 1, c(4 * i, -3 * i), "+"))
  })
  names(X) <- paste0("subject_", seq_along(X))
  snapshot <- serialize(X, NULL)
  means    <- do.call(cbind, lapply(X, rowMeans))
  centered <- lapply(X, function(x) sweep(x, 1, rowMeans(x), "-"))

  for (method in c("collapsed", "factor_conditional")) {
    for (prior in c("normal", "laplace")) {
      arguments <- list(K = 2, n_factors = 1, initial_clusters = 2,
        initialization = if (prior == "normal") "random" else "data",
        coefficient_prior = prior, coefficient_split_merge = method,
        n_burnin = 1, n_mcmc = 4, store_draws = TRUE,
        coefficient_split_merge_every = 1, split_merge_every = 2)
      set.seed(9602)
      automatic <- do.call(pdpmvar, c(list(X = X), arguments))
      rng <- .Random.seed
      set.seed(9602)
      manual <- do.call(pdpmvar, c(list(X = centered, center = FALSE), arguments))

      expect_identical(.Random.seed, rng)
      expect_identical(automatic$node_means, means)
      expect_identical(manual$node_means, do.call(cbind, lapply(centered, rowMeans)))
      expect_true(automatic$settings$center)
      expect_false(manual$settings$center)

      # Only the recorded call, input means and centering setting should differ.
      automatic$call <- manual$call <- NULL
      automatic$node_means <- manual$node_means <- NULL
      automatic$settings$center <- manual$settings$center <- NULL
      expect_identical(automatic, manual)
    }
  }
  expect_identical(serialize(X, NULL), snapshot)
})

test_that("saved means retain matrix dimensions for one node or one subject", {
  for (X in list(
    list(first = matrix(seq_len(8), 1), second = matrix(seq_len(10) + 7, 1)),
    list(first = matrix(seq_len(18), 3))
  )) {
    set.seed(9603)
    fit <- pdpmvar(X, K = 1, n_factors = 1, initial_clusters = 1,
      n_burnin = 0, n_mcmc = 1, cov_shape = 3, cov_rate = 2)
    expect_identical(dim(fit$node_means), c(nrow(X[[1]]), length(X)))
    expect_identical(fit$node_means, do.call(cbind, lapply(X, rowMeans)))
    expect_identical(colnames(fit$node_means), names(X))
  }
})

test_that("centering requires a single TRUE or FALSE and consistent chain settings", {
  X <- list(matrix(seq(-1, 1, length.out = 10), 1))
  for (value in list(NA, NULL, 1, "TRUE", c(TRUE, FALSE))) {
    expect_error(pdpmvar(X, center = value), "TRUE or FALSE")
  }

  set.seed(9604)
  fit <- pdpmvar(X, K = 1, n_factors = 1, initial_clusters = 1,
    n_burnin = 0, n_mcmc = 8, cov_shape = 3, cov_rate = 2)
  other <- fit
  other$settings$center <- FALSE
  expect_error(pdpmvar_diagnostics(list(fit, other)), "model settings")
})
