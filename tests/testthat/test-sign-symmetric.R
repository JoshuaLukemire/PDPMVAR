test_that("factorized sign densities equal explicit mixtures and ignore column orientations", {
  set.seed(1400)
  center <- .pdpmvar_initialize_covariance(crossprod(matrix(rnorm(100), 10)) + diag(10), 5)
  atom <- .pdpmvar_test_reference$.pdpmvar_propose_covariance(center)
  signs <- as.matrix(expand.grid(rep(list(c(-1, 1)), 5)))
  free <- lower.tri(atom$working_loadings, diag = TRUE)
  scale <- matrix(rep(sqrt(atom$factor_precision), each = 10), 10)
  components <- apply(signs, 1, function(sign) {
    means <- sweep(center$working_loadings, 2, sign, "*") * scale
    return(sum(dnorm(atom$working_loadings[free], means[free], (0.15 * scale)[free], log = TRUE)))
  })
  maximum <- max(components)
  expected <- maximum + log(mean(exp(components - maximum))) +
    sum(dlnorm(atom$factor_precision, 0, 0.5, log = TRUE)) +
    sum(dlnorm(atom$variance, log(center$variance), 0.25, log = TRUE))
  expect_equal(.pdpmvar_test_reference$.pdpmvar_covariance_log_proposal(atom, center), expected)
  expect_equal(.pdpmvar_eigen_covariance_log_proposal(atom, center), expected, tolerance = 1e-11)
  transformed <- atom
  transformed$working_loadings <- sweep(atom$working_loadings, 2, c(-1, 1, -1, 1, -1), "*")
  transformed <- .pdpmvar_test_reference$.pdpmvar_refresh_covariance(transformed)
  transformed_center <- center
  transformed_center$working_loadings <- -center$working_loadings
  expect_equal(transformed$covariance, atom$covariance)
  expect_equal(.pdpmvar_test_reference$.pdpmvar_covariance_log_base(transformed, 3, 2), .pdpmvar_test_reference$.pdpmvar_covariance_log_base(atom, 3, 2))
  expect_equal(.pdpmvar_test_reference$.pdpmvar_covariance_log_proposal(transformed, transformed_center), expected)
  expect_equal(.pdpmvar_eigen_covariance_log_proposal(transformed, transformed_center), expected, tolerance = 1e-11)
})

test_that("sign-symmetric draws match R and have the specified shared-column moments", {
  center <- list(working_loadings = matrix(c(0.8, 0.4, -0.3, 0.2, 0, 0.6, 0.1, -0.4), 4),
    variance = c(1, 2, 1, 2))
  for (seed in 1401:1403) {
    set.seed(seed)
    reference <- .pdpmvar_test_reference$.pdpmvar_propose_covariance(center)
    rng <- .Random.seed
    set.seed(seed)
    compiled <- .pdpmvar_eigen_propose_covariance(center)
    expect_equal(compiled, reference, tolerance = 1e-12)
    expect_identical(.Random.seed, rng)
    reflected_center <- center
    reflected_center$working_loadings <- -center$working_loadings
    set.seed(seed)
    expect_equal(.pdpmvar_test_reference$.pdpmvar_propose_covariance(reflected_center), reference)
    set.seed(seed)
    expect_equal(.pdpmvar_eigen_propose_covariance(reflected_center), reference, tolerance = 1e-12)
  }
  expect_error(.pdpmvar_eigen_propose_covariance(list(working_loadings = matrix(0, 2), variance = -1)), "center")
  expect_error(.pdpmvar_eigen_covariance_log_proposal(list(working_loadings = matrix(0, 2),
    factor_precision = 1, variance = c(1, 1)), list(working_loadings = matrix(0, 3), variance = rep(1, 3))), "dimensions")
  set.seed(1404)
  draws <- replicate(800, {
    atom <- .pdpmvar_eigen_propose_covariance(center)
    atom$working_loadings[, 1] / sqrt(atom$factor_precision[1])
  })
  expect_true(all(abs(rowMeans(draws)) < 5 * apply(draws, 1, sd) / sqrt(ncol(draws))))
  products <- rbind(draws[1, ]^2, draws[1, ] * draws[2, ])
  expected <- c(0.8^2 + 0.15^2, 0.8 * 0.4)
  expect_true(all(abs(rowMeans(products) - expected) < 5 * apply(products, 1, sd) / sqrt(ncol(draws))))
})

test_that("covariance merge ratios are invariant to the removed atom's signs", {
  set.seed(1405)
  X <- lapply(1:3, function(i) matrix(rnorm(30), 3))
  atoms <- lapply(1:3, function(i) .pdpmvar_test_reference$.pdpmvar_new_covariance(3, 2, 3, 2))
  proposed <- atoms
  proposed[[2]] <- .pdpmvar_test_reference$.pdpmvar_new_covariance(3, 2, 3, 2)
  center <- .pdpmvar_initialize_covariance(tcrossprod(X[[2]]) / 10, 2)
  args <- list(c(1, 2, 2), c(1, 1, 1), atoms, proposed, log(c(0.5, 0.3, 0.1)),
    lapply(X, tcrossprod), rep(10, 3), c(1L, 2L), center, 3, 3, 2)
  reference <- do.call(.pdpmvar_test_reference$.pdpmvar_covariance_move_ratio, args)
  args[[3]][[2]]$working_loadings <- -atoms[[2]]$working_loadings
  args[[3]][[2]] <- .pdpmvar_test_reference$.pdpmvar_refresh_covariance(args[[3]][[2]])
  expect_equal(do.call(.pdpmvar_test_reference$.pdpmvar_covariance_move_ratio, args), reference)
  expect_equal(do.call(.pdpmvar_eigen_covariance_move_ratio, args), reference, tolerance = 1e-10)
})
