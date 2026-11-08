test_that("native driver preserves the reference loop, summaries and RNG at callbacks", {
  set.seed(901)
  X <- list(matrix(rnorm(24), 2), matrix(rnorm(34), 2), matrix(rnorm(42), 2))
  design <- lapply(X, .pdpmvar_lag_design, K = 2)
  design_cross <- lapply(design, tcrossprod)
  callbacks <- list(message = base::message,
    assign_coefficients = .pdpmvar_test_reference$.pdpmvar_assign_coefficients, assign_covariances = .pdpmvar_test_reference$.pdpmvar_assign_covariances)
  for (prior in c("laplace", "normal")) {
    initial <- pdpmvar(X, K = 2, n_factors = 1, initial_clusters = 1,
      n_burnin = 0, n_mcmc = 1, coefficient_prior = prior)
    state <- initial$final_state
    if (prior == "normal") {
      state$coefficient_atoms <- rep(state$coefficient_atoms[1], 3)
      state$covariance_atoms <- rep(state$covariance_atoms[1], 3)
      state$coefficient_labels <- c(1, 3, 3)
      state$covariance_labels <- c(3, 1, 3)
    }
    snapshot <- serialize(state, NULL)
    control <- c(initial$settings, list(store_draws = TRUE, verbose = FALSE))
    control$backend <- "RcppEigen"
    control$n_burnin <- 2
    control$n_mcmc <- 5
    control$split_merge_every <- 2
    control$coefficient_split_merge_every <- 2
    set.seed(902)
    reference <- .pdpmvar_test_reference$.pdpmvar_sampler_reference(X, design, design_cross, state, control)
    reference_rng <- .Random.seed
    set.seed(902)
    compiled <- .pdpmvar_eigen_sampler(X, design, design_cross, state, control, callbacks)
    expect_equal(compiled, reference, tolerance = 1e-8)
    expect_identical(.Random.seed, reference_rng)
    expect_identical(serialize(state, NULL), snapshot)
    expect_equal(compiled$coefficient_mean, apply(compiled$draws$coefficients, 1:3, mean))
    expect_equal(compiled$covariance_mean, apply(compiled$draws$covariance, 1:3, mean))
    expect_equal(compiled$coefficient_sign_probability,
      apply(compiled$draws$coefficients > 0, 1:3, mean))
    expect_equal(compiled$parameter_traces[, "coefficient_rms"],
      sqrt(apply(compiled$draws$coefficients^2, 4, mean)))
    expect_equal(compiled$coefficient_coclustering,
      Reduce(`+`, lapply(seq_len(control$n_mcmc), function(i)
        outer(compiled$coefficient_labels[i, ], compiled$coefficient_labels[i, ], "=="))) / control$n_mcmc)
  }
  # An R callback error must leave R's RNG in the state actually consumed.
  control$verbose <- FALSE
  set.seed(905)
  .pdpmvar_eigen_sampler(X, design, design_cross, state, control, callbacks)
  runif(1)
  expected_rng <- .Random.seed
  control$verbose <- TRUE
  callbacks$message <- function(...) {
    runif(1)
    stop("intentional callback failure")
  }
  set.seed(905)
  expect_error(.pdpmvar_eigen_sampler(X, design, design_cross, state, control, callbacks),
    "intentional callback failure")
  expect_identical(.Random.seed, expected_rng)
})

test_that("one-subject native runs handle zero burnin and optional outputs", {
  set.seed(903)
  X <- list(matrix(rnorm(12), 1))
  for (prior in c("normal", "laplace")) {
    set.seed(904)
    reference <- pdpmvar_reference_for_test(X, initial_clusters = 1, n_factors = 1, cov_shape = 3, cov_rate = 1,
      n_burnin = 0, n_mcmc = 1, coefficient_prior = prior)
    reference_rng <- .Random.seed
    set.seed(904)
    expect_message(compiled <- pdpmvar(X, initial_clusters = 1, n_factors = 1, cov_shape = 3, cov_rate = 1,
      n_burnin = 0, n_mcmc = 1,
      coefficient_prior = prior, verbose = TRUE),
      "Iteration 1: 1 coefficient clusters, 1 covariance clusters")
    expect_identical(.Random.seed, reference_rng)
    fields <- setdiff(names(reference), c("call", "settings"))
    expect_equal(compiled[fields], reference[fields], tolerance = 1e-10)
    expect_null(compiled$draws)
    expect_identical(is.null(compiled$lambda), prior == "normal")
  }
})
