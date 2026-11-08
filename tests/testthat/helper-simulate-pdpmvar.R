# Generate stable PDPMVAR data with independent coefficient and covariance clusters.
simulate_pdpmvar_data <- function(n_timepoints = 100, n_subjects = 25, n_nodes = 10,
                                 n_coefficient_clusters = 3, n_covariance_clusters = 3,
                                 n_lags = 2, sparsity = 0.5, n_warmup = 500,
                                 covariance_model = c("unrestricted", "factor"),
                                 n_factors = min(5, n_nodes)) {
  covariance_model <- match.arg(covariance_model)
  dimensions <- c(n_timepoints, n_subjects, n_nodes, n_coefficient_clusters,
                  n_covariance_clusters, n_lags)
  if (any(!is.finite(dimensions)) || any(dimensions < 1) ||
      any(dimensions != floor(dimensions))) {
    stop("Dimensions must be positive whole numbers.")
  }
  if (length(n_warmup) != 1 || !is.finite(n_warmup) || n_warmup < 0 || n_warmup != floor(n_warmup)) {
    stop("n_warmup must be a nonnegative whole number.")
  }
  if (length(n_factors) != 1 || !is.finite(n_factors) || n_factors < 1 ||
      n_factors > n_nodes || n_factors != floor(n_factors)) {
    stop("n_factors must be a positive whole number no greater than n_nodes.")
  }
  if (n_coefficient_clusters > n_subjects || n_covariance_clusters > n_subjects) {
    stop("Each cluster must have at least one subject.")
  }
  if (length(sparsity) != 1 || !is.finite(sparsity) || sparsity < 0 || sparsity > 1) {
    stop("sparsity must be a number between 0 and 1.")
  }

  # Rows index responses, columns index predictors, and the third axis indexes lags.
  coefficients <- array(0, c(n_nodes, n_nodes, n_lags, n_coefficient_clusters))
  covariance   <- array(0, c(n_nodes, n_nodes, n_covariance_clusters))
  n_coefficients <- n_nodes * n_nodes * n_lags
  n_nonzero      <- round((1 - sparsity) * n_coefficients)

  for (cluster in seq_len(n_coefficient_clusters)) {
    values <- numeric(n_coefficients)
    if (n_nonzero > 0) {
      nonzero_indices <- sample.int(n_coefficients, n_nonzero)
      values[nonzero_indices] <- runif(n_nonzero, 0.1, 0.5) *
        sample(c(-1, 1), n_nonzero, replace = TRUE)
    }
    lag_matrices <- array(values, c(n_nodes, n_nodes, n_lags))

    # Sum of lag-matrix spectral norms below one is sufficient for VAR stability.
    # A common scale preserves structural zeros and relative lag strengths.
    norm_sum <- 0
    for (lag in seq_len(n_lags)) {
      lag_matrix <- matrix(lag_matrices[, , lag], n_nodes, n_nodes)
      norm_sum <- norm_sum + norm(lag_matrix, type = "2")
    }
    if (norm_sum > 0) {
      lag_matrices <- lag_matrices * (0.8 / norm_sum)
    }
    coefficients[, , , cluster] <- lag_matrices
  }

  covariance_atoms <- if (covariance_model == "factor") vector("list", n_covariance_clusters) else NULL
  for (cluster in seq_len(n_covariance_clusters)) {
    if (covariance_model == "unrestricted") {
      # Preserve the original full-covariance stress fixture and its random stream.
      loadings       <- matrix(rnorm(n_nodes * n_nodes), n_nodes, n_nodes)
      raw_covariance <- tcrossprod(loadings) / n_nodes + diag(0.5, n_nodes)
      correlation    <- cov2cor(raw_covariance)
      standard_deviations <- runif(n_nodes, 0.7, 1.3) * sqrt(cluster)
      covariance[, , cluster] <- correlation * tcrossprod(standard_deviations)
    } else {
      # Fixed parameters inside the fitted factor family; these are not new priors.
      working_loadings <- matrix(rnorm(n_nodes * n_factors, sd = 0.5), n_nodes) * sqrt(cluster)
      working_loadings[upper.tri(working_loadings)] <- 0
      factor_precision <- rep(1, n_factors)
      variance <- runif(n_nodes, 0.7, 1.3) * cluster
      signs <- sign(diag(working_loadings))
      signs[signs == 0] <- 1
      loadings <- sweep(working_loadings, 2, signs, "*")
      covariance[, , cluster] <- tcrossprod(loadings) + diag(variance, n_nodes)
      covariance_atoms[[cluster]] <- list(working_loadings = working_loadings,
        factor_precision = factor_precision, variance = variance, loadings = loadings,
        covariance = matrix(covariance[, , cluster], n_nodes))
    }
  }

  # Shuffle each balanced allocation independently; every cluster is represented.
  coefficient_cluster <- sample(rep(seq_len(n_coefficient_clusters), length.out = n_subjects))
  covariance_cluster  <- sample(rep(seq_len(n_covariance_clusters), length.out = n_subjects))
  data                <- vector("list", n_subjects)
  innovations         <- vector("list", n_subjects)
  initial_history     <- vector("list", n_subjects)
  n_generated         <- n_warmup + n_timepoints
  retained_indices    <- n_lags + n_warmup + seq_len(n_timepoints)

  for (subject in seq_len(n_subjects)) {
    subject_covariance <- matrix(covariance[, , covariance_cluster[subject]], n_nodes, n_nodes)
    noise <- t(chol(subject_covariance)) %*%
      matrix(rnorm(n_nodes * n_generated), n_nodes, n_generated)
    time_series <- matrix(0, n_nodes, n_lags + n_generated)

    # With zero warm-up, presample history stays zero exactly as in the fitted likelihood.
    for (time in seq_len(n_generated)) {
      time_index <- n_lags + time
      conditional_mean <- numeric(n_nodes)
      for (lag in seq_len(n_lags)) {
        lag_matrix <- matrix(coefficients[, , lag, coefficient_cluster[subject]], n_nodes, n_nodes)
        conditional_mean <- conditional_mean + lag_matrix %*% time_series[, time_index - lag]
      }
      time_series[, time_index] <- conditional_mean + noise[, time]
    }
    data[[subject]] <- time_series[, retained_indices, drop = FALSE]
    innovations[[subject]] <- noise[, n_warmup + seq_len(n_timepoints), drop = FALSE]
    initial_history[[subject]] <- time_series[, n_warmup + seq_len(n_lags), drop = FALSE]
  }

  # History is oldest to newest. Keeping innovations permits exact recursion tests.
  # Call set.seed() externally to reproduce a fixture without resetting RNG here.
  return(list(data = data, coefficients = coefficients, covariance = covariance,
              coefficient_cluster = coefficient_cluster, covariance_cluster = covariance_cluster,
              innovations = innovations, initial_history = initial_history, covariance_atoms = covariance_atoms,
              settings = list(n_timepoints = n_timepoints, n_subjects = n_subjects,
                              n_nodes = n_nodes, n_lags = n_lags, sparsity = sparsity,
                              n_warmup = n_warmup, covariance_model = covariance_model,
                              n_factors = if (covariance_model == "factor") n_factors else NULL)))
}

# Match the fitted likelihood, including its factor dimension and zero presample history.
simulate_pdpmvar_matched_data <- function(n_timepoints = 100, n_subjects = 25, n_nodes = 10,
                                         n_coefficient_clusters = 3, n_covariance_clusters = 3,
                                         n_lags = 2, sparsity = 0.5, n_factors = min(5, n_nodes)) {
  return(simulate_pdpmvar_data(n_timepoints, n_subjects, n_nodes, n_coefficient_clusters,
    n_covariance_clusters, n_lags, sparsity, n_warmup = 0,
    covariance_model = "factor", n_factors = n_factors))
}
