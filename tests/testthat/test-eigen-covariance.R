test_that("Eigen factor draws match R with current factor precisions", {
  atom <- list(working_loadings = rbind(c(0.7, 0), c(0.2, -0.5), c(0.3, 0.1)),
               variance = c(0.6, 1.2, 2), factor_precision = c(0.7, 2.5))
  set.seed(101)
  residual <- matrix(rnorm(51), 3)
  set.seed(102)
  reference <- .pdpmvar_test_reference$.pdpmvar_draw_factors(residual, atom)
  state <- .Random.seed
  set.seed(102)
  compiled <- .pdpmvar_test_reference$.pdpmvar_draw_factors(residual, atom, backend = "RcppEigen")
  expect_equal(compiled, reference, tolerance = 1e-12)
  expect_identical(.Random.seed, state)
  expect_equal(dim(compiled), c(2, 17))
  bad <- atom
  bad$variance[1] <- -1
  expect_error(.pdpmvar_eigen_factors(residual, bad, matrix(0, 2, 17)), "positive")
  expect_error(.pdpmvar_eigen_factors(residual, atom, matrix(0, 3, 17)), "normals")
})

test_that("compiled occupied covariance updates match each R Gibbs step", {
  set.seed(103)
  for (dimensions in list(c(1, 1), c(3, 2), c(4, 4))) {
    n_nodes <- dimensions[1]
    n_factors <- dimensions[2]
    atom <- .pdpmvar_test_reference$.pdpmvar_new_covariance(n_nodes, n_factors, 3.2, 1.7)
    original <- atom
    residual <- matrix(rnorm(n_nodes * 17), n_nodes)
    set.seed(104)
    reference <- .pdpmvar_test_reference$.pdpmvar_update_covariance(residual, atom, 3.2, 1.7)
    state <- .Random.seed
    set.seed(104)
    compiled <- .pdpmvar_test_reference$.pdpmvar_update_covariance(residual, atom, 3.2, 1.7, backend = "RcppEigen")
    expect_equal(compiled, reference, tolerance = 1e-10)
    expect_identical(.Random.seed, state)
    expect_identical(atom, original)
    expect_true(all(compiled$working_loadings[upper.tri(compiled$working_loadings)] == 0))
    expect_true(all(diag(compiled$loadings) > 0))
    expect_gt(min(eigen(compiled$covariance, symmetric = TRUE)$values), 0)
  }
  atom$working_loadings[1, 2] <- 1
  expect_error(.pdpmvar_test_reference$.pdpmvar_update_covariance(residual, atom, 3.2, 1.7, backend = "RcppEigen"), "triangular")
})

test_that("compiled factor draws reproduce an analytic correlated conditional covariance", {
  atom <- list(working_loadings = rbind(c(0.7, 0), c(0.2, -0.5), c(0.3, 0.1)),
               variance = c(0.6, 1.2, 2), factor_precision = c(0.7, 2.5))
  target <- solve(diag(atom$factor_precision) + crossprod(atom$working_loadings,
    atom$working_loadings / atom$variance))
  set.seed(105)
  factors <- .pdpmvar_test_reference$.pdpmvar_draw_factors(matrix(0, 3, 10000), atom, backend = "RcppEigen")
  expect_equal(rowMeans(factors), c(0, 0), tolerance = 0.025)
  expect_equal(unname(cov(t(factors))), target, tolerance = 0.04)
})
