test_that("the active API exposes only the retained sampler paths", {
  X <- list(matrix(seq(-1,1,length.out=12),2))
  for(method in c("split_refine","joint_refine","dual_refine")) {
    expect_error(pdpmvar(X,n_burnin=0,n_mcmc=1,cov_shape=3,cov_rate=2,
      coefficient_split_merge=method),"arg")
  }
  for(method in c("explicit","explicit_joint","split_refine")) {
    expect_error(pdpmvar(X,n_burnin=0,n_mcmc=1,cov_shape=3,cov_rate=2,
      covariance_split_merge=method),"arg")
  }
  expect_false(any(c("split_refine_sweeps","split_refine_step","covariance_split_shape",
    "covariance_joint_step","covariance_joint_proposal") %in% names(formals(pdpmvar))))
  expect_equal(eval(formals(pdpmvar)$coefficient_split_merge),c("auto","collapsed","factor_conditional"))
})

test_that("public fitting always uses the native sampler without a backend option", {
  X <- list(matrix(seq(-1, 1, length.out = 8), 1))
  expect_false("backend" %in% names(formals(pdpmvar)))
  expect_error(pdpmvar(X, backend = "R"), "unused argument")

  expect_false(exists(".pdpmvar_sampler_reference", asNamespace("PDPMVAR"), inherits = FALSE))
  set.seed(9401)
  fit <- pdpmvar(X, K = 1, n_factors = 1, initial_clusters = 1,
    n_burnin = 0, n_mcmc = 1, cov_shape = 3, cov_rate = 2)
  expect_s3_class(fit, "pdpmvar")
  expect_identical(fit$settings$backend, "RcppEigen")
})
