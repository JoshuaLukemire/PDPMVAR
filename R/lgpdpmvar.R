# Functions for preparing lag-specific fits and choosing starting groups.

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fit the lag-specific PDPMVAR model ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#' Fit a lag-specific product of Dirichlet process mixtures VAR model
#'
#' Fit a Gaussian VAR with a separate partition of subjects for the full
#' coefficient matrix at each lag and another partition for residual covariance.
#'
#' @param X List of finite numeric matrices, with nodes in rows and times in
#'   columns. All subjects must have the same nodes in the same order.
#' @param K Number of lags.
#' @param n_burnin Number of initial iterations to discard.
#' @param n_mcmc Number of iterations to retain.
#' @param n_factors Number of residual covariance factors.
#' @param coefficient_concentration Positive Dirichlet process concentration,
#'   either one value for all lags or one value per lag.
#' @param covariance_concentration Positive Dirichlet process concentration for
#'   covariance groups.
#' @param coefficient_prior Use the published Laplace hierarchy or a fixed
#'   normal prior.
#' @param coefficient_sd Standard deviation for the fixed normal prior.
#' @param lambda Initial Laplace shrinkage rate, either one value for all lags
#'   or one value per lag.
#' @param cov_shape,cov_rate Shape and rate of the Gamma prior for residual
#'   precisions. Leave both NULL to calibrate them from the data.
#' @param initial_clusters Starting group count in each partition.
#' @param store_draws Save subject coefficient and covariance draws.
#' @param verbose Print iteration progress.
#' @param initialization Use separate k-means starts or random balanced labels.
#' @param split_merge_every Attempt a covariance split or merge every this
#'   many iterations. Zero disables these moves.
#' @param split_merge_slots Number of stick positions considered by the
#'   covariance split/merge move.
#' @param center Subtract each node's time average within each subject.
#' @param coefficient_split_merge_every Attempt one coefficient split or merge
#'   per lag every this many iterations. Zero disables these moves.
#' @param coefficient_split_merge_slots Number of stick positions considered
#'   separately at each lag. The slice sampler still uses the full mixture.
#' @param coefficient_split_merge Coefficient move method. `"factor_conditional"`
#'   holds the sampled factors fixed. `"collapsed"` integrates them out and
#'   uses the full residual covariance.
#' @param coefficient_relocation_every Attempt one single-subject transfer
#'   per lag every this many iterations. Zero disables these moves. Transfers
#'   use the same stick prefix as coefficient split/merge and integrate out
#'   the two affected coefficient matrices and the factors.
#' @param coefficient_relocation_method How to select the subject and
#'   destination. `"guided"` favors subjects whose prediction improves most
#'   in another occupied group, compared with a fit that leaves them out of
#'   their current group. It also favors occupied destinations that predict
#'   them well. A 20% uniform component includes empty positions, using a
#'   uniform choice alone if no occupied alternative is available.
#'   `"destination"` guides only the destination. `"uniform"` selects
#'   both uniformly. All methods preserve the same posterior.
#'
#' @details The coefficient mixtures are independent a priori. With the
#'   Laplace prior, corresponding entries share a prior variance across
#'   occupied matrices within a lag, and each lag has its own shrinkage rate.
#'   The sampler assigns groups one lag at a time and then draws the
#'   coefficient rows jointly across lags conditional on the sampled factors.
#'   Coefficient group assignments use slice sampling. Optional split/merge
#'   moves integrate the two affected matrices at one lag while holding the
#'   other lags fixed. The collapsed method draws fresh factors after the moves
#'   and uses the low-rank covariance structure to integrate the lag matrices.
#'   Both methods preserve the model and priors.
#'
#' @return An object of class `lgpdpmvar`. The subject coefficient means and
#'   sign summaries are nodes by lagged predictors by subjects,
#'   `D x (D * K) x N`, where D is nodes, K is lags and N is subjects.
#'   Lag blocks sit side by side. Covariance means are nodes by nodes by
#'   subjects, `D x D x N`. Coefficient labels and co-clustering probabilities
#'   are lists of length K, with a retained-iterations-by-subjects label
#'   matrix and a subjects-by-subjects co-clustering matrix, `N x N`,
#'   per lag. Covariance has one of each.
#'   Group counts include burn-in. The Laplace shrinkage trace has retained
#'   iterations in rows and lags in columns. `coefficient_split_merge` contains
#'   one move record per iteration for each lag. `coefficient_relocation`
#'   records the transfer decisions and the subject, source and destination.
#'   The result also includes input node means, starting labels, settings
#'   and the final sampling state.
#' @export
lgpdpmvar <- function(X,
                      K = 2,
                      n_burnin = 500,
                      n_mcmc = 1000,
                      n_factors = min(5, nrow(X[[1]])),
                      coefficient_concentration = 1,
                      covariance_concentration = 1,
                      coefficient_prior = c("laplace", "normal"),
                      coefficient_sd = 0.25,
                      lambda = 1,
                      cov_shape = NULL,
                      cov_rate = NULL,
                      initial_clusters = min(8, length(X)),
                      store_draws = FALSE,
                      verbose = FALSE,
                      initialization = c("data", "random"),
                      split_merge_every = 0,
                      split_merge_slots = 10,
                      center = TRUE,
                      coefficient_split_merge_every = 0,
                      coefficient_split_merge_slots = 10,
                      coefficient_split_merge = c("factor_conditional", "collapsed"),
                      coefficient_relocation_every = 0,
                      coefficient_relocation_method = c("guided", "destination", "uniform")) {
  # Save the call and check that each subject has a finite nodes-by-time matrix.
  call <- match.call()
  if (!is.list(X) || !length(X) ||
      !all(vapply(X, function(x) is.matrix(x) && is.numeric(x) &&
                    nrow(x) > 0 && ncol(x) > 0 && all(is.finite(x)), logical(1)))) {
    stop("X must be a nonempty list of finite numeric matrices (nodes by time points).")
  }

  n_nodes    <- nrow(X[[1]])
  n_subjects <- length(X)

  # Lags, factors, starting groups and move settings use whole numbers.
  .pdpmvar_check_count(K, "K")
  .pdpmvar_check_count(n_burnin, "n_burnin", 0)
  .pdpmvar_check_count(n_mcmc, "n_mcmc")
  .pdpmvar_check_count(n_factors, "n_factors")
  .pdpmvar_check_count(initial_clusters, "initial_clusters")
  .pdpmvar_check_count(split_merge_every, "split_merge_every", 0)
  .pdpmvar_check_count(split_merge_slots, "split_merge_slots", 2)
  .pdpmvar_check_count(coefficient_split_merge_every, "coefficient_split_merge_every", 0)
  .pdpmvar_check_count(coefficient_split_merge_slots, "coefficient_split_merge_slots", 2)
  .pdpmvar_check_count(coefficient_relocation_every, "coefficient_relocation_every", 0)

  # Check that the requested model and starting groups fit the data dimensions.
  if (any(vapply(X, nrow, integer(1)) != n_nodes)) {
    stop("All subject matrices must have the same number of nodes.")
  }
  if (n_factors > n_nodes || initial_clusters > n_subjects ||
      any(vapply(X, ncol, integer(1)) <= K)) {
    stop("Require n_factors <= nodes, initial_clusters <= subjects, and every series length > K.")
  }

  # Iteration counts and move controls must fit in the integers used by C++.
  if (as.double(n_burnin) + n_mcmc > .Machine$integer.max ||
      any(c(split_merge_every, split_merge_slots, coefficient_split_merge_every,
            coefficient_split_merge_slots, coefficient_relocation_every) > .Machine$integer.max)) {
    stop("Iteration and split/merge controls exceed the supported integer range.")
  }

  # Accept one concentration and starting shrinkage rate per lag, or repeat
  # a single value across all lags. Covariance has only one concentration.
  coefficient_concentration <- .lgpdpmvar_lag_control(coefficient_concentration, K, "coefficient_concentration")
  lambda                    <- .lgpdpmvar_lag_control(lambda, K, "lambda")
  .lgpdpmvar_lag_control(covariance_concentration, 1, "covariance_concentration")
  .lgpdpmvar_lag_control(coefficient_sd, 1, "coefficient_sd")
  if (any(!is.finite(lambda^2) | lambda^2 == 0) ||
      !is.finite(coefficient_sd^2) || coefficient_sd^2 == 0) {
    stop("Squared lambda and coefficient_sd must be positive and finite.")
  }

  # Check the switches and resolve the chosen prior, starts and move methods.
  flags <- list(store_draws, verbose, center)
  if (!all(vapply(flags, function(x) is.logical(x) && length(x) == 1 && !is.na(x), logical(1)))) {
    stop("store_draws, verbose and center must be TRUE or FALSE.")
  }

  coefficient_prior             <- match.arg(coefficient_prior)
  initialization                <- match.arg(initialization)
  coefficient_split_merge       <- match.arg(coefficient_split_merge)
  coefficient_relocation_method <- match.arg(coefficient_relocation_method)

  # Save the input means in a nodes-by-subjects matrix, D x N, and subtract
  # them within each subject when centering is requested.
  node_means <- do.call(cbind, lapply(X, rowMeans))
  if (center) {
    for (subject in seq_len(n_subjects)) {
      X[[subject]] <- sweep(X[[subject]], 1, node_means[, subject], "-")
    }
  }

  # If neither covariance prior parameter is supplied, estimate both from the data.
  if (is.null(cov_shape) && is.null(cov_rate)) {
    calibrated <- .pdpmvar_calibrate_covariance(X)
    cov_shape  <- unname(calibrated["shape"])
    cov_rate   <- unname(calibrated["rate"])
  }
  .lgpdpmvar_lag_control(cov_shape, 1, "cov_shape")
  .lgpdpmvar_lag_control(cov_rate, 1, "cov_rate")

  # Stack the lagged predictors once for each subject, (D * K) x T_i, with
  # all nodes at lag 1 first. T_i is the subject's number of time points.
  design       <- lapply(X, .pdpmvar_lag_design, K = K)
  design_cross <- lapply(design, tcrossprod)

  # Choose separate starting groups at each lag and for covariance.
  state <- .lgpdpmvar_initialize(
    X,
    design,
    design_cross,
    initial_clusters,
    n_factors,
    cov_shape,
    cov_rate,
    initialization
  )

  # At each lag, corresponding entries share a prior variance across groups.
  # Start with a D x D matrix of fixed variances or draws from the Laplace hierarchy.
  state$lambda_squared <- lambda^2
  state$shared_variance <- lapply(seq_len(K), function(lag) {
    variance <- matrix(coefficient_sd^2, n_nodes, n_nodes)
    if (coefficient_prior == "laplace") {
      variance[] <- rexp(length(variance), rate = state$lambda_squared[lag] / 2)
    }
    return(variance)
  })

  # Collect the settings for C++ and keep them with the returned fit.
  settings <- list(
    K                             = K,
    n_burnin                      = n_burnin,
    n_mcmc                        = n_mcmc,
    n_factors                     = n_factors,
    coefficient_concentration     = coefficient_concentration,
    covariance_concentration      = covariance_concentration,
    coefficient_prior             = coefficient_prior,
    coefficient_sd                = coefficient_sd,
    lambda                        = lambda,
    cov_shape                     = cov_shape,
    cov_rate                      = cov_rate,
    initial_clusters              = initial_clusters,
    initialization                = initialization,
    split_merge_every             = split_merge_every,
    split_merge_slots             = split_merge_slots,
    center                        = center,
    backend                       = "RcppEigen",
    coefficient_split_merge_every = coefficient_split_merge_every,
    coefficient_split_merge_slots = coefficient_split_merge_slots,
    coefficient_split_merge       = coefficient_split_merge,
    coefficient_relocation_every  = coefficient_relocation_every,
    coefficient_relocation_method = coefficient_relocation_method
  )
  control <- c(settings, list(laplace = coefficient_prior == "laplace",
                             store_draws = store_draws, verbose = verbose))

  # Run the sampler in C++ and add the starting labels and input means to its results.
  result                <- .lgpdpmvar_eigen_sampler(X, design, design_cross, state, control)
  result$call           <- call
  result$node_means     <- node_means
  result$initial_labels <- list(coefficient = state$coefficient_labels, covariance = state$covariance_labels)
  result$settings       <- settings
  class(result)         <- "lgpdpmvar"

  return(result)
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Choose starting groups and matrices ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Check positive prior controls before lgpdpmvar() prepares the sampler.
# Return K values, one per lag, repeating a scalar when needed. Use K = 1
# for controls that must be a single number.
.lgpdpmvar_lag_control <- function(value, K, name) {
  if (!is.numeric(value) || !(length(value) %in% c(1, K)) ||
      any(!is.finite(value) | value <= 0)) {
    stop(name, " must contain positive finite values, one or one per lag where applicable.")
  }
  return(rep_len(value, K))
}

# Choose the starting labels and matrices used by lgpdpmvar() before sampling.
# Return one vector of N subject labels per lag and one for covariance. Store
# a D x D matrix per starting group, separately for each of the K lags and
# for covariance. D is nodes, N is subjects and K is lags.
.lgpdpmvar_initialize <- function(X,
                                  design,
                                  design_cross,
                                  n_clusters,
                                  n_factors,
                                  cov_shape,
                                  cov_rate,
                                  initialization) {
  n_nodes    <- nrow(X[[1]])
  n_subjects <- length(X)
  K          <- nrow(design[[1]]) / n_nodes
  lag_names  <- paste0("lag_", seq_len(K))
  labels     <- atoms <- stats::setNames(vector("list", K), lag_names)

  if (initialization == "random") {
    # Start with balanced random labels, zero lag matrices and covariance draws from the prior.
    allocation <- rep(seq_len(n_clusters), length.out = n_subjects)
    for (lag in seq_len(K)) {
      labels[[lag]] <- sample(allocation, n_subjects)
      atoms[[lag]]  <- replicate(n_clusters, list(coefficient = matrix(0, n_nodes, n_nodes)), simplify = FALSE)
    }
    covariance_labels <- sample(allocation, n_subjects)
    covariance_atoms  <- replicate(n_clusters,
      .pdpmvar_new_covariance(n_nodes, n_factors, cov_shape, cov_rate), simplify = FALSE)
  } else {
    # Fit all lags together for each subject before clustering their lag blocks.
    ridge        <- diag(0.01, n_nodes * K)
    coefficients <- residuals <- vector("list", n_subjects)
    for (subject in seq_len(n_subjects)) {
      upper                   <- chol(design_cross[[subject]] + ridge)
      information             <- design[[subject]] %*% t(X[[subject]])
      coefficients[[subject]] <- t(backsolve(upper, forwardsolve(t(upper), information)))
      residuals[[subject]]    <- X[[subject]] - coefficients[[subject]] %*% design[[subject]]
    }

    # Cluster the subject estimates separately at each lag. The k-means input
    # has subjects in rows and coefficient entries in columns, N x D^2.
    for (lag in seq_len(K)) {
      columns  <- (lag - 1) * n_nodes + seq_len(n_nodes)
      blocks   <- lapply(coefficients, function(A) A[, columns, drop = FALSE])
      features <- matrix(unlist(blocks, use.names = FALSE), nrow = n_subjects, byrow = TRUE)

      # Use the entrywise median of the member matrices to start each group.
      labels[[lag]] <- .pdpmvar_initial_labels(features, n_clusters)
      atoms[[lag]]  <- lapply(seq_len(max(labels[[lag]])), function(group) {
        members <- which(labels[[lag]] == group)
        center  <- apply(matrix(unlist(blocks[members], use.names = FALSE), nrow = n_nodes^2), 1, stats::median)
        return(list(coefficient = matrix(center, n_nodes, n_nodes)))
      })
    }

    # Group the residual covariance summaries separately, then pool the
    # residual time points within each group to estimate its starting covariance.
    features          <- .pdpmvar_initial_covariance_features(residuals)
    covariance_labels <- .pdpmvar_initial_labels(features, n_clusters)
    covariance_atoms  <- lapply(seq_len(max(covariance_labels)), function(group) {
      pooled <- do.call(cbind, residuals[covariance_labels == group])
      return(.pdpmvar_initialize_covariance(tcrossprod(pooled) / ncol(pooled), n_factors))
    })
  }

  return(list(
    coefficient_labels = labels,
    coefficient_atoms  = atoms,
    covariance_labels  = covariance_labels,
    covariance_atoms   = covariance_atoms
  ))
}
