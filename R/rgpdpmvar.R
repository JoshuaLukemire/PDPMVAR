# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fit the row-specific PDPMVAR model ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#' Fit a row-specific product of Dirichlet process mixtures VAR model
#'
#' Fit a Gaussian VAR with a separate partition of subjects for each response
#' node and another partition for residual covariance. Subjects in the same
#' group for a node share all of that node's coefficients across all lags.
#'
#' @param X List of finite numeric matrices, with nodes in rows and times in columns.
#' @param K Number of lags.
#' @param n_burnin Number of initial iterations to discard.
#' @param n_mcmc Number of iterations to retain.
#' @param n_factors Number of residual covariance factors.
#' @param coefficient_concentration Positive Dirichlet process concentration,
#'   either one value for all response nodes or one per node.
#' @param covariance_concentration Positive Dirichlet process concentration for covariance.
#' @param lambda Starting Laplace shrinkage rate, either one value for all
#'   response nodes or one per node.
#' @param cov_shape,cov_rate Shape and rate of the Gamma prior for residual
#'   precisions. Leave both NULL to calibrate them from the data.
#' @param initial_clusters Starting group count in each partition.
#' @param store_draws Save subject coefficient and covariance draws.
#' @param verbose Print iteration progress.
#' @param initialization Use separate k-means starts or random balanced labels.
#' @param split_merge_every Attempt a covariance split or merge every this
#'   many iterations. Zero disables these moves.
#' @param split_merge_slots Number of stick positions considered by covariance split/merge.
#' @param center Subtract each node's time average within each subject.
#' @param coefficient_split_merge_every Attempt one coefficient split or merge
#'   per response node every this many iterations. Zero disables these moves.
#' @param coefficient_split_merge_slots Number of stick positions considered
#'   separately at each node. Slice sampling still uses the full mixture.
#' @param coefficient_relocation_every Attempt one single-subject transfer
#'   per response node every this many iterations. Zero disables these
#'   moves. Transfers use the coefficient split/merge stick positions.
#'
#' @details The coefficient mixtures are independent a priori. Each response
#'   has a Laplace shrinkage rate, and corresponding entries share a prior
#'   variance across the groups for that response. The sampler draws latent
#'   factors, then uses slice sampling for coefficient group assignments and
#'   draws each group's entire row in one regression. The optional coefficient
#'   split/merge move holds the factors fixed and integrates out the two
#'   affected rows. The optional transfer uses the data to favor subjects
#'   that predict better in another group and destinations that predict them
#'   well. It integrates out the factors and the two affected coefficient
#'   rows. Its acceptance probability accounts for the proposal in both
#'   directions. These moves preserve the model and priors.
#'
#' @return An object of class `rgpdpmvar`. Subject coefficient means and
#'   sign summaries are nodes by lagged predictors by subjects,
#'   `D x (D * K) x N`, with lag blocks side by side. Covariance means are
#'   `D x D x N`. Coefficient labels and co-clustering probabilities are
#'   lists by response node, with an iterations-by-subjects label matrix
#'   and a subjects-by-subjects co-clustering matrix per node. Covariance has
#'   one of each. Group counts and split/merge records include burn-in.
#'   The result also includes shrinkage traces by node, input node means,
#'   starting labels, settings and the final sampling state.
#'   When transfers are enabled, `coefficient_relocation` records the
#'   proposed subject, source and destination, log acceptance ratio and
#'   decision by response node and iteration, including burn-in.
#' @export
rgpdpmvar <- function(X,
                      K = 2,
                      n_burnin = 500,
                      n_mcmc = 1000,
                      n_factors = min(5, nrow(X[[1]])),
                      coefficient_concentration = 1,
                      covariance_concentration = 1,
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
                      coefficient_relocation_every = 0) {
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
    stop("Iteration and group move controls exceed the supported integer range.")
  }

  # Accept one concentration and starting shrinkage rate per response node, or repeat
  # a single value across all response nodes. Covariance has only one concentration.
  coefficient_concentration <- .rgpdpmvar_row_control(coefficient_concentration, n_nodes, "coefficient_concentration")
  lambda                    <- .rgpdpmvar_row_control(lambda, n_nodes, "lambda")
  .rgpdpmvar_row_control(covariance_concentration, 1, "covariance_concentration")
  if (any(!is.finite(lambda^2) | lambda^2 == 0)) {
    stop("Squared lambda must be positive and finite.")
  }

  # Check the switches and select the starting group method.
  flags <- list(store_draws, verbose, center)
  if (!all(vapply(flags, function(x) is.logical(x) && length(x) == 1 && !is.na(x), logical(1)))) {
    stop("store_draws, verbose and center must be TRUE or FALSE.")
  }

  initialization <- match.arg(initialization)

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
  .rgpdpmvar_row_control(cov_shape, 1, "cov_shape")
  .rgpdpmvar_row_control(cov_rate, 1, "cov_rate")

  # Stack the lagged predictors once for each subject, (D * K) x T_i, with
  # all nodes at lag 1 first. T_i is the subject's number of time points.
  design       <- lapply(X, .pdpmvar_lag_design, K = K)
  design_cross <- lapply(design, tcrossprod)

  # Choose separate starting groups at each response node and for covariance.
  state <- .rgpdpmvar_initialize(
    X,
    design,
    design_cross,
    initial_clusters,
    n_factors,
    cov_shape,
    cov_rate,
    initialization
  )

  # Each response node has one prior variance per lagged predictor,
  # shared across its groups. Each node gets a 1 x (D * K) row.
  state$lambda_squared <- lambda^2
  state$shared_variance <- lapply(state$lambda_squared, function(rate_squared) {
    return(matrix(rexp(n_nodes * K, rate = rate_squared / 2), nrow = 1))
  })

  # Collect the settings for C++ and keep them with the returned fit.
  settings <- list(
    K                             = K,
    n_burnin                      = n_burnin,
    n_mcmc                        = n_mcmc,
    n_factors                     = n_factors,
    coefficient_concentration     = coefficient_concentration,
    covariance_concentration      = covariance_concentration,
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
    coefficient_relocation_every  = coefficient_relocation_every
  )
  control <- c(settings, list(store_draws = store_draws, verbose = verbose))

  # Run the sampler in C++ and add the starting labels and input means to its results.
  result                <- .rgpdpmvar_eigen_sampler(X, design, design_cross, state, control)
  result$call           <- call
  result$node_means     <- node_means
  result$initial_labels <- list(coefficient = state$coefficient_labels, covariance = state$covariance_labels)
  result$settings       <- settings
  class(result)         <- "rgpdpmvar"

  return(result)
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Choose starting groups and matrices ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Check the positive prior controls used in rgpdpmvar(). Repeat a scalar
# across the D response nodes, or return the D supplied values.
.rgpdpmvar_row_control <- function(value, n_nodes, name) {
  if (!is.numeric(value) || !(length(value) %in% c(1, n_nodes)) ||
      any(!is.finite(value) | value <= 0)) {
    stop(name, " must contain one positive finite value or one per response node.")
  }
  return(rep_len(value, n_nodes))
}

# Choose starting groups for rgpdpmvar(). Return one vector of N subject
# labels per response node and one for covariance. Each coefficient group
# stores a 1 x (D * K) row, with all D predictors at each lag in order.
.rgpdpmvar_initialize <- function(X,
                                  design,
                                  design_cross,
                                  n_clusters,
                                  n_factors,
                                  cov_shape,
                                  cov_rate,
                                  initialization) {
  n_nodes      <- nrow(X[[1]])
  n_subjects   <- length(X)
  n_predictors <- nrow(design[[1]])
  node_names   <- paste0("node_", seq_len(n_nodes))
  labels       <- atoms <- stats::setNames(vector("list", n_nodes), node_names)

  if (initialization == "random") {
    # Start with balanced random labels, zero coefficient rows and covariance draws from the prior.
    allocation <- rep(seq_len(n_clusters), length.out = n_subjects)
    for (node in seq_len(n_nodes)) {
      labels[[node]] <- sample(allocation, n_subjects)
      atoms[[node]]  <- replicate(n_clusters, list(coefficient = matrix(0, 1, n_predictors)), simplify = FALSE)
    }
    covariance_labels <- sample(allocation, n_subjects)
    covariance_atoms  <- replicate(n_clusters,
      .pdpmvar_new_covariance(n_nodes, n_factors, cov_shape, cov_rate), simplify = FALSE)
  } else {
    # Fit all lags together for each subject before clustering each response row.
    ridge        <- diag(0.01, n_predictors)
    coefficients <- residuals <- vector("list", n_subjects)
    for (subject in seq_len(n_subjects)) {
      upper                   <- chol(design_cross[[subject]] + ridge)
      information             <- design[[subject]] %*% t(X[[subject]])
      coefficients[[subject]] <- t(backsolve(upper, forwardsolve(t(upper), information)))
      residuals[[subject]]    <- X[[subject]] - coefficients[[subject]] %*% design[[subject]]
    }

    # Each row of the N x (D * K) feature matrix holds one subject's
    # coefficients for this response node, with lag blocks side by side.
    for (node in seq_len(n_nodes)) {
      features <- do.call(rbind, lapply(coefficients, function(A) A[node, , drop = FALSE]))
      labels[[node]] <- .pdpmvar_initial_labels(features, n_clusters)
      atoms[[node]] <- lapply(seq_len(max(labels[[node]])), function(group) {
        members <- labels[[node]] == group
        median_row <- apply(features[members, , drop = FALSE], 2, stats::median)
        return(list(coefficient = matrix(median_row, nrow = 1)))
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
