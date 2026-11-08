test_that("covariance starts distinguish correlations with identical marginal variances", {
  first <- rep(c(1,-1),4)
  second <- rep(c(1,1,-1,-1),2)
  positive <- rbind(first,0.8*first+0.6*second)
  negative <- rbind(first,-0.8*first+0.6*second)
  residuals <- list(positive,positive,negative,negative)
  features <- .pdpmvar_initial_covariance_features(residuals)
  expect_equal(dim(features),c(4,3))
  expect_equal(features[,1:2],matrix(0,4,2),tolerance=1e-12)
  set.seed(3010)
  labels <- .pdpmvar_initial_labels(features,4)
  expect_equal(labels[1],labels[2])
  expect_equal(labels[3],labels[4])
  expect_false(labels[1]==labels[3])
  scaled <- .pdpmvar_initial_covariance_features(lapply(residuals,function(x)3*x))
  expect_equal(scaled,features,tolerance=1e-12)
})

test_that("covariance feature blocks remain finite and have equal total weight", {
  set.seed(3011)
  residuals <- lapply(1:6,function(i)matrix(rnorm(30,sd=i),3))
  features <- .pdpmvar_initial_covariance_features(residuals)
  expect_true(all(is.finite(features)))
  expect_equal(sum(features[,1:3]^2),5,tolerance=1e-12)
  expect_equal(sum(features[,4:6]^2),5,tolerance=1e-12)
  expect_equal(.pdpmvar_initial_covariance_features(rep(list(matrix(0,2,5)),4)),matrix(0,4,3))
  expect_equal(dim(.pdpmvar_initial_covariance_features(list(matrix(0,1,5)))),c(1,1))
  expect_equal(.pdpmvar_initial_covariance_features(list(matrix(0,1,5))),matrix(0,1,1))
})

test_that("overclustered starts are reproducible and valid for both sampler backends", {
  set.seed(3012)
  X <- lapply(1:9,function(i)matrix(rnorm(40),2))
  design <- lapply(X,.pdpmvar_lag_design,K=1)
  set.seed(3013)
  first <- .pdpmvar_initialize_data(X,design,lapply(design,tcrossprod),8,1)
  set.seed(3013)
  second <- .pdpmvar_initialize_data(X,design,lapply(design,tcrossprod),8,1)
  expect_identical(first,second)
  expect_equal(length(unique(first$coefficient_labels)),8)
  expect_equal(length(unique(first$covariance_labels)),8)
  for(atom in first$covariance_atoms) {
    expect_true(all(is.finite(chol(atom$covariance))))
    expect_true(all(atom$variance>0))
  }
  for(fitter in list(pdpmvar_reference_for_test, pdpmvar)) {
    set.seed(3014)
    fit <- fitter(X,K=1,n_factors=1,n_burnin=0,n_mcmc=2,cov_shape=3,cov_rate=2)
    expect_equal(fit$settings$initial_clusters,8)
    expect_equal(length(unique(fit$initial_labels$coefficient)),8)
    expect_equal(length(unique(fit$initial_labels$covariance)),8)
    expect_equal(fit$settings[c("cov_shape","cov_rate","coefficient_concentration","covariance_concentration")],
      list(cov_shape=3,cov_rate=2,coefficient_concentration=1,covariance_concentration=1))
  }
})

test_that("multichain defaults vary initial counts and cap them for small datasets", {
  set.seed(3015)
  X <- lapply(1:9,function(i)matrix(rnorm(12),1))
  result <- pdpmvar_chains(X,seeds=c(11,12,13),K=1,n_factors=1,n_burnin=0,n_mcmc=8,
    cov_shape=3,cov_rate=2)
  expect_equal(vapply(result$fits,function(fit)fit$settings$initial_clusters,numeric(1)),c(4,6,8))
  expect_equal(vapply(result$fits,function(fit)length(unique(fit$initial_labels$covariance)),numeric(1)),c(4,6,8))
  small <- pdpmvar_chains(X[1:2],seeds=c(14,15),K=1,n_factors=1,n_burnin=0,n_mcmc=8,
    cov_shape=3,cov_rate=2)
  expect_equal(vapply(small$fits,function(fit)fit$settings$initial_clusters,numeric(1)),c(2,2))
})
