# Functions for fitting PDPMVAR, choosing starting values and calling the C++ sampling steps.

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check inputs and arrange lagged observations ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Check that an argument is a single whole number at or above the allowed minimum.
# Used by pdpmvar() and pdpmvar_chains() to check arguments such as the number of iterations.
.pdpmvar_check_count <- function(value, name, minimum = 1) {
  if (!is.numeric(value) || length(value) != 1 || !is.finite(value) ||
      value < minimum || value != floor(value)) {
    stop(name, " must be a whole number >= ", minimum, ".")
  }

  return(invisible(NULL))
}

# Arrange lagged observations in a nodes-times-lags by time-points matrix, (D * K) x T,
# where D is the number of nodes, K is the number of lags and T is the number of time points.
# Put all nodes at lag 1 in the first D rows, then lag 2 in the next D rows, and so on.
# Fill entries with zero when the required earlier observation is before the start of the series.
# Used by pdpmvar() to prepare the predictors for each subject before fitting.
.pdpmvar_lag_design <- function(x, K) {
  n_nodes <- nrow(x)
  n_times <- ncol(x)
  design  <- matrix(0, n_nodes * K, n_times)

  # Copy earlier observations into the rows for this lag, leaving unavailable values at zero.
  for (lag in seq_len(K)) {
    if (lag >= n_times) {next}
    rows                            <- (lag - 1) * n_nodes + seq_len(n_nodes)
    design[rows, (lag + 1):n_times] <- x[, seq_len(n_times - lag), drop = FALSE]
  }

  return(design)
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fit the PDPMVAR model ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#' Fit a product of Dirichlet process mixtures VAR model
#'
#' Fit a Gaussian VAR(K), grouping subjects by their coefficient matrices and,
#' separately, their residual covariances. By default, center each node within
#' each subject before fitting.
#'
#' @param X A list of numeric matrices, one per subject, with nodes in rows and
#'   time points in columns. All entries must be finite. Subjects must have the
#'   same nodes in the same order, but may have different numbers of time points.
#' @param K Number of lags, a positive whole number.
#' @param center Whether to subtract each node's time average separately within
#'   each subject. Defaults to TRUE. The data are not scaled.
#' @param n_burnin Number of initial iterations to discard; may be zero.
#' @param n_mcmc Number of iterations to retain after burn-in.
#' @param n_factors Number of covariance factors; cannot exceed the number of nodes.
#' @param coefficient_concentration,covariance_concentration Positive, fixed DP
#'   concentration parameters. These influence the prior on the number of groups;
#'   they do not fix the number of groups.
#' @param coefficient_prior Either `"laplace"` or `"normal"`.
#' @param coefficient_sd Prior standard deviation for coefficients when using
#'   the fixed normal prior.
#' @param lambda Starting Laplace rate. The sampler updates its square under a
#'   Gamma(1,2) prior, with shape and rate as the two parameters.
#' @param cov_shape,cov_rate Shape and rate of the Gamma prior on node residual
#'   precisions (inverse variances). If both are NULL, estimate them by matching
#'   the mean and variance of the empirical subject/node precisions.
#' @param initial_clusters Starting number of groups for each of the two mixtures.
#'   One coefficient label covers the full matrix across all rows and lags.
#'   Coefficient and covariance groups are chosen separately and can change during
#'   sampling. The default starts with eight groups, or fewer if there are fewer subjects.
#' @param initialization Either `"data"` (default) or `"random"`. The data-based
#'   start fits crude subject VAR regressions, then uses separate k-means fits on
#'   the coefficient matrices and residual covariance summaries.
#'   Identical subject summaries may give fewer groups than requested. The random
#'   start assigns subjects to groups separately for each mixture, starts coefficient
#'   matrices at zero and draws covariances from the prior. Neither method changes
#'   the model priors or fixes the number of groups during sampling.
#' @param split_merge_every Attempt a covariance split/merge every this many
#'   iterations; zero (default) disables these additional moves.
#' @param split_merge_slots Number of initial stick positions available to the
#'   additional covariance split/merge moves. Ordinary slice sampling can still
#'   create groups beyond these positions, so this does not truncate the DP.
#' @param covariance_split_merge Covariance split/merge method. The supported
#'   method is `"bridge"`.
#' @param coefficient_split_merge_every Attempt a coefficient split/merge every
#'   this many iterations; zero (default) disables these additional moves.
#'   Each move concerns complete coefficient matrices, with all rows and lags together.
#' @param coefficient_split_merge_slots Number of initial stick positions
#'   available to coefficient split/merge moves, independently of the covariance
#'   setting. Ordinary slice sampling can still use positions beyond this number.
#' @param coefficient_split_merge Method for coefficient split/merge moves.
#'   `"collapsed"` integrates out both coefficient matrices and latent factors.
#'   `"factor_conditional"` draws factors first, then integrates out coefficients
#'   one response row at a time while holding those factors fixed. Both methods
#'   assign one label per full coefficient matrix and use the same model priors.
#'   The default `"auto"` uses the latter method above 1000 coefficient entries or
#'   when storing the full Gaussian statistics would require more than 128 MiB.
#'   Otherwise it uses `"collapsed"`. Full Gaussian statistics requiring more than
#'   512 MiB are rejected before allocating memory. The chosen method is recorded
#'   in the fit settings.
#' @param store_draws Whether to store the coefficient and covariance draws for each subject.
#' @param verbose Whether to print progress every 100 iterations and at the last iteration.
#'
#' @details The model has no intercept. With center = TRUE, fitting uses deviations
#'   from each subject's node means, saved in `node_means`. With center = FALSE,
#'   observations are used as supplied.
#'
#'   All observations enter the likelihood; unavailable lagged values before the
#'   start of a series are set to zero. Coefficient matrices are D x (D * K), with
#'   response nodes in rows and predictor columns ordered by lag, starting with lag 1.
#'   Here D is the number of nodes. Stationarity is not enforced.
#'
#'   MCMC sampling runs in C++ using RcppEigen. Separate Walker slice samplers
#'   assign coefficient and covariance groups. Coefficient rows are drawn
#'   conditional on latent factors.
#'
#'   For the Laplace prior, a_hj | tau_j ~ N(0,tau_j),
#'   tau_j | lambda^2 ~ Exp(lambda^2/2), and lambda^2 ~ Gamma(1,2), using shape-rate
#'   notation. Each tau_j is shared across occupied coefficient matrices, with
#'   full conditional GIG(1-C/2, sum_h a_hj^2, lambda^2), where C is the number
#'   of occupied coefficient groups. Unused matrices do not enter this update.
#'   The optional normal prior fixes the coefficient standard deviation at coefficient_sd.
#'
#'   Residual covariances are Sigma = Lstar Q^-1 Lstar' + diag(v). Free entries
#'   of lower-triangular Lstar have independent N(0,1) priors. Diagonal entries
#'   of Q have independent Gamma(1/2,1/2) priors, and entries of v^-1 have independent
#'   Gamma(cov_shape,cov_rate) priors. Latent factors have covariance Q^-1.
#'   Reported loadings have a positive diagonal without changing Sigma.
#'
#'   The additional coefficient and covariance split/merge moves extend the paper's
#'   sampling procedure while keeping the published model and priors unchanged.
#'
#'   Covariance moves hold coefficient matrices fixed and integrate out the previous
#'   factors, slice variables and stick weights. An inverse-Wishart predictive
#'   distribution guides proposed assignments only; it is not the covariance prior.
#'   An annealing bridge uses reversible factor-model updates to move between
#'   intermediate distributions for the current and proposed groups.
#'
#'   Coefficient moves hold shared shrinkage variances fixed and integrate out the
#'   affected coefficient matrices. The factor-conditional method sums contributions
#'   across response rows before assigning one label per subject. Split proposals
#'   keep two selected subjects in opposite groups; forward/reverse proposal
#'   probabilities enter the MH acceptance ratio. Both affected coefficient matrices
#'   are drawn after accepting or rejecting the move.
#'
#'   Use `set.seed()` before fitting for reproducibility. Numeric labels refer to
#'   stick positions, not fixed group identities across iterations. Co-clustering
#'   probabilities report how often each pair of subjects belongs to the same group.
#' @return An object of class `pdpmvar` containing subject-level coefficient and
#'   covariance posterior means, probabilities that coefficients are above zero,
#'   and `coefficient_sign_tail` (twice the smaller posterior sign probability).
#'   Also includes retained group labels, co-clustering probabilities, starting
#'   labels, fit settings, the last sampled parameter values and diagnostic traces.
#'   Cluster-count traces and the two split/merge attempt tables include burn-in;
#'   parameter traces and posterior summaries use retained iterations only.
#'   The split/merge tables record the move type, acceptance decision and log
#'   acceptance ratio. Results also include lambda draws under the Laplace prior,
#'   the mean shared coefficient variances and, if requested, subject-level draws
#'   with retained iteration as the last array dimension. Sign-tail summaries are
#'   posterior quantities, not frequentist p-values. `node_means` is a nodes-by-subjects
#'   matrix, D x N, containing the input data's means whether or not centering was
#'   requested. The `center` setting records whether these means were subtracted.
#' @references Kundu and Lukemire (2024), Flexible Bayesian Product Mixture
#'   Models for Vector Autoregressions, JMLR 25, 1-52.
#' @importFrom stats rnorm runif rbeta rgamma rexp
#' @references Karagiannis, G. and Andrieu, C. (2013). Annealed Importance Sampling
#'   Reversible Jump MCMC Algorithms. Journal of Computational and Graphical
#'   Statistics, 22(3), 623-648. <https://doi.org/10.1080/10618600.2013.805651>.
#' @export
pdpmvar <- function(X,
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
                    coefficient_split_merge_every = 0,
                    coefficient_split_merge_slots = 10,
                    covariance_split_merge = "bridge",
                    coefficient_split_merge = c("auto", "collapsed", "factor_conditional"),
                    center = TRUE) {
  # Check the data and arguments before preparing predictors or choosing starting values.
  call <- match.call()
  if (!is.list(X) || length(X) == 0 ||
      !all(vapply(X, function(x) is.matrix(x) && is.numeric(x) &&
                    nrow(x) > 0 && ncol(x) > 0 && all(is.finite(x)), logical(1)))) {
    stop("X must be a nonempty list of finite numeric matrices (nodes by time points).")
  }
  n_nodes    <- nrow(X[[1]])
  n_subjects <- length(X)
  if (any(vapply(X, nrow, numeric(1)) != n_nodes)) {
    stop("All subject matrices must have the same number of nodes.")
  }

  # Numbers of lags, factors, clusters and iterations must be whole numbers.
  .pdpmvar_check_count(K, "K")
  .pdpmvar_check_count(n_burnin, "n_burnin", 0)
  .pdpmvar_check_count(n_mcmc, "n_mcmc")
  .pdpmvar_check_count(n_factors, "n_factors")
  .pdpmvar_check_count(initial_clusters, "initial_clusters")
  .pdpmvar_check_count(split_merge_every, "split_merge_every", 0)
  .pdpmvar_check_count(split_merge_slots, "split_merge_slots", 2)
  .pdpmvar_check_count(coefficient_split_merge_every, "coefficient_split_merge_every", 0)
  .pdpmvar_check_count(coefficient_split_merge_slots, "coefficient_split_merge_slots", 2)
  if (n_factors > n_nodes || initial_clusters > n_subjects || any(vapply(X, ncol, numeric(1)) <= K)) {
    stop("Require n_factors <= nodes, initial_clusters <= subjects, and every series length > K.")
  }

  # Each concentration, coefficient standard deviation and Laplace rate must be one positive number.
  controls <- list(coefficient_concentration, covariance_concentration, coefficient_sd,
                   lambda)
  if (!all(vapply(controls, function(x) is.numeric(x) && length(x) == 1 &&
                   is.finite(x) && x > 0, logical(1)))) {
    stop("Concentrations, coefficient_sd and lambda must be positive finite scalars.")
  }
  if (!is.logical(store_draws) || length(store_draws) != 1 || is.na(store_draws) ||
      !is.logical(verbose) || length(verbose) != 1 || is.na(verbose) ||
      !is.logical(center) || length(center) != 1 || is.na(center)) {
    stop("store_draws, verbose and center must be TRUE or FALSE.")
  }

  # Save each node's mean within each subject, then subtract it if centering was requested.
  # The returned matrix has nodes in rows and subjects in columns: D x N.
  node_means <- do.call(cbind, lapply(X, rowMeans))
  if (center) {
    for (subject in seq_len(n_subjects)) {
      X[[subject]] <- sweep(X[[subject]], 1, node_means[, subject], "-")
    }
  }

  # If neither covariance-prior parameter is supplied, estimate both from the observed data.
  if (is.null(cov_shape) && is.null(cov_rate)) {
    calibrated <- .pdpmvar_calibrate_covariance(X)
    cov_shape  <- unname(calibrated["shape"])
    cov_rate   <- unname(calibrated["rate"])
  }
  if (is.null(cov_shape) || is.null(cov_rate)) {
    stop("Supply both cov_shape and cov_rate, or leave both NULL for calibration.")
  }
  covariance_controls <- list(cov_shape, cov_rate)
  if (!all(vapply(covariance_controls, function(x) is.numeric(x) && length(x) == 1 &&
                   is.finite(x) && x > 0, logical(1)))) {
    stop("cov_shape and cov_rate must be positive finite scalars.")
  }

  # Check the requested fitting methods and estimate the memory needed for the full
  # coefficient precision matrices. Use calculations by response row for large models.
  coefficient_prior       <- match.arg(coefficient_prior)
  initialization          <- match.arg(initialization)
  covariance_split_merge  <- match.arg(covariance_split_merge, "bridge")
  coefficient_split_merge <- match.arg(coefficient_split_merge)
  coefficient_dimension   <- n_nodes * n_nodes * K
  dense_cache_bytes       <- 8 * n_subjects * coefficient_dimension^2
  if (coefficient_split_merge == "auto") coefficient_split_merge <-
    if (coefficient_dimension > 1000 || dense_cache_bytes > 128 * 1024^2) "factor_conditional" else "collapsed"
  if (coefficient_split_merge == "collapsed" && coefficient_split_merge_every > 0 && dense_cache_bytes > 512 * 1024^2) {
    stop("Dense coefficient cache exceeds 512 MiB; use factor_conditional or auto.")
  }

  # Calculate each subject's lagged predictors and their cross-products once, then reuse them.
  n_predictors <- n_nodes * K
  design       <- lapply(X, .pdpmvar_lag_design, K = K)
  design_cross <- lapply(design, tcrossprod)

  # Assign starting coefficient and covariance groups separately. The starting number
  # of groups is the same for both, but a subject need not have the same label in each.
  allocation         <- rep(seq_len(initial_clusters), length.out = n_subjects)
  coefficient_labels <- allocation[sample.int(n_subjects)]
  covariance_labels  <- allocation[sample.int(n_subjects)]
  coefficient_atoms  <- vector("list", initial_clusters)
  covariance_atoms   <- vector("list", initial_clusters)

  # For the Laplace prior, corresponding entries of all cluster coefficient matrices
  # share a prior variance. Draw starting variances and set the starting value of lambda squared.
  shared_variance <- matrix(coefficient_sd^2, n_nodes, n_predictors)
  lambda_squared  <- lambda^2
  if (coefficient_prior == "laplace") {
    shared_variance[] <- rexp(length(shared_variance), rate = lambda_squared / 2)
  }

  # Set up the starting cluster parameters: zero coefficient matrices and covariances drawn from the prior.
  for (cluster in seq_len(initial_clusters)) {
    coefficient_atoms[[cluster]]               <- .pdpmvar_new_coefficient(shared_variance)
    coefficient_atoms[[cluster]]$coefficient[] <- 0
    covariance_atoms[[cluster]]                <- .pdpmvar_new_covariance(n_nodes, n_factors, cov_shape, cov_rate)
  }

  # For a data-based start, replace these values with estimates from subject regressions
  # and k-means groups. The sampler can change both labels and parameter values afterward.
  if (initialization == "data") {
    initial            <- .pdpmvar_initialize_data(X, design, design_cross, initial_clusters, n_factors)
    coefficient_atoms  <- initial$coefficient_atoms
    covariance_atoms   <- initial$covariance_atoms
    coefficient_labels <- initial$coefficient_labels
    covariance_labels  <- initial$covariance_labels
  }

  # Save the starting labels and collect the parameter values and settings needed by the C++ sampler.
  initial_labels <- list(coefficient = coefficient_labels, covariance = covariance_labels)
  state          <- list(coefficient_atoms = coefficient_atoms, covariance_atoms = covariance_atoms,
    coefficient_labels = coefficient_labels, covariance_labels = covariance_labels,
    shared_variance = shared_variance, lambda_squared = lambda_squared)
  settings <- list(K = K, n_burnin = n_burnin, n_mcmc = n_mcmc, n_factors = n_factors,
    coefficient_concentration = coefficient_concentration, covariance_concentration = covariance_concentration,
    coefficient_prior = coefficient_prior, coefficient_sd = coefficient_sd, lambda = lambda,
    cov_shape = cov_shape, cov_rate = cov_rate, initial_clusters = initial_clusters,
    initialization = initialization, split_merge_every = split_merge_every,
    split_merge_slots = split_merge_slots,
    coefficient_split_merge_every = coefficient_split_merge_every,
    coefficient_split_merge_slots = coefficient_split_merge_slots, backend = "RcppEigen",
    covariance_split_merge = covariance_split_merge, center = center)
  settings$coefficient_split_merge <- coefficient_split_merge
  control                          <- c(settings, list(store_draws = store_draws, verbose = verbose))

  # Run the sampler in C++. It calls R to print progress and, when there are more than
  # 200 available mixture components, to sample cluster labels with the same R sampling method as before.
  callbacks <- list(message = base::message,
    assign_coefficients = .pdpmvar_assign_coefficients, assign_covariances = .pdpmvar_assign_covariances)
  sampling <- .pdpmvar_eigen_sampler(X, design, design_cross, state, control, callbacks)

  # Collect the posterior summaries, diagnostic results and last sampled values to return to the user.
  result <- list(call = call, node_means = node_means, coefficient_mean = sampling$coefficient_mean,
    covariance_mean = sampling$covariance_mean,
    coefficient_sign_probability = sampling$coefficient_sign_probability,
    coefficient_sign_tail = sampling$coefficient_sign_tail,
    coefficient_labels = sampling$coefficient_labels, covariance_labels = sampling$covariance_labels,
    coefficient_coclustering = sampling$coefficient_coclustering,
    covariance_coclustering = sampling$covariance_coclustering, cluster_counts = sampling$cluster_counts,
    initial_labels = initial_labels, parameter_traces = sampling$parameter_traces,
    split_merge = sampling$split_merge, coefficient_split_merge = sampling$coefficient_split_merge,
    lambda = sampling$lambda, shared_variance_mean = sampling$shared_variance_mean,
    settings = settings, final_state = sampling$final_state, draws = sampling$draws)
  class(result) <- "pdpmvar"

  return(result)
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Choose starting groups from the data ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Use k-means to choose starting groups from a matrix with one row per subject.
# Use fewer groups if there are fewer distinct rows than the requested number of groups.
# Called separately for coefficient estimates and covariance summaries when choosing starting labels.
.pdpmvar_initial_labels <- function(features, n_clusters) {
  # Subjects with identical summaries cannot provide different starting k-means centers.
  n_distinct <- nrow(unique(as.data.frame(features)))
  n_clusters <- min(n_clusters, n_distinct)
  if (n_clusters == 1) {return(rep(1, nrow(features)))}
  if (n_clusters == nrow(features)) {return(seq_len(nrow(features)))}

  return(stats::kmeans(features, centers = n_clusters, nstart = 10, iter.max = 50)$cluster)
}

# Center each column and rescale the whole matrix so its column variances sum to one.
# Used to give residual variances and correlations equal influence when choosing starting covariance groups.
.pdpmvar_initial_scale_block <- function(features) {
  # Divide every column by the same number, rather than scaling each column separately.
  # Return zeros if there is essentially no variation between subjects.
  # This affects only the starting k-means groups; it does not rescale the fitted model.
  centered  <- sweep(features, 2, colMeans(features))
  magnitude <- sqrt(sum(centered^2) / max(1, nrow(features) - 1))
  tolerance <- sqrt(.Machine$double.eps) * max(1, abs(features))
  if (magnitude <= tolerance) return(centered * 0)

  return(centered / magnitude)
}

# Summarize residual variances and correlations in a subjects-by-summaries matrix,
# N x (D + D * (D - 1) / 2), where N is the number of subjects and D is the number of nodes.
# The first D columns are log variances; the remaining columns are Fisher-transformed
# correlations, with each node pair included once. Rescale the two sets of columns separately.
# Used by k-means to choose starting covariance groups from the subject regression residuals.
.pdpmvar_initial_covariance_features <- function(residuals) {
  n_subjects           <- length(residuals)
  n_nodes              <- nrow(residuals[[1]])
  pairs                <- lower.tri(matrix(0, n_nodes, n_nodes))
  variance_features    <- matrix(0, n_subjects, n_nodes)
  correlation_features <- matrix(0, n_subjects, sum(pairs))

  # Calculate log variances and Fisher-transformed correlations for each subject.
  # Keep variances above zero and correlations away from -1 and 1 so the transformations are finite.
  for (subject in seq_len(n_subjects)) {
    covariance                   <- tcrossprod(residuals[[subject]]) / ncol(residuals[[subject]])
    variance                     <- pmax(diag(covariance), .Machine$double.eps)
    variance_features[subject, ] <- log(variance)
    if (any(pairs)) {
      correlation                     <- covariance / sqrt(outer(variance, variance))
      correlation_features[subject, ] <- atanh(pmax(-1 + 1e-6, pmin(1 - 1e-6, correlation[pairs])))
    }
  }

  # With one node there are no node pairs, so return only the rescaled log variance.
  variance_features <- .pdpmvar_initial_scale_block(variance_features)
  if (!any(pairs)) return(variance_features)

  correlation_features <- .pdpmvar_initial_scale_block(correlation_features)

  return(cbind(variance_features, correlation_features))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Choose starting coefficient matrices and covariances ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Approximate an estimated node-by-node covariance matrix, D x D, using a lower-triangular
# node-by-factor loading matrix, D x B, and D residual variances. Here D is the number of
# nodes and B is n_factors. Return these values together with the resulting covariance matrix.
# Used for starting covariance parameters and as a center for covariance split/merge proposals.
.pdpmvar_initialize_covariance <- function(covariance, n_factors) {
  n_nodes <- nrow(covariance)

  # Use the largest eigenvalues and their eigenvectors to choose the starting factor loadings.
  decomposition     <- eigen(covariance, symmetric = TRUE)
  floor_variance    <- max(mean(diag(covariance)), 1) * 1e-6
  residual_variance <- if (n_factors < n_nodes) {
    mean(decomposition$values[(n_factors + 1):n_nodes])
  } else {
    min(decomposition$values) / 2
  }
  loadings <- sweep(decomposition$vectors[, seq_len(n_factors), drop = FALSE], 2,
    sqrt(pmax(decomposition$values[seq_len(n_factors)] - residual_variance, floor_variance)), "*")

  # Rotate the loadings to make them lower triangular without changing the covariance they describe.
  rotation                      <- qr.Q(qr(t(loadings[seq_len(n_factors), , drop = FALSE])))
  loadings                      <- loadings %*% rotation
  loadings[upper.tri(loadings)] <- 0

  # Give each node a positive residual variance, even if the loadings explain all of its estimated variance.
  variance <- pmax(diag(covariance - tcrossprod(loadings)), floor_variance)

  return(.pdpmvar_refresh_covariance(list(working_loadings = loadings,
    factor_precision = rep(1, n_factors), variance = variance)))
}

# Fit a VAR regression to each subject, then choose coefficient and covariance starting groups separately.
# Return one response-by-predictor coefficient matrix, D x P, per coefficient group and one
# node-by-node covariance matrix, D x D, per covariance group, together with subject labels.
# Here D is the number of nodes and P = D * K is the number of lagged predictors.
# Called by pdpmvar() when initialization = "data", before sampling begins.
.pdpmvar_initialize_data <- function(X,
                                     design,
                                     design_cross,
                                     n_clusters,
                                     n_factors) {
  n_subjects   <- length(X)
  n_nodes      <- nrow(X[[1]])
  n_predictors <- nrow(design[[1]])

  # Add 0.01 to the predictor cross-product diagonal so the starting regressions can be
  # solved even when predictors are linearly dependent. This is only for starting values.
  subject_coefficients <- residuals <- vector("list", n_subjects)
  for (subject in seq_len(n_subjects)) {
    precision                       <- design_cross[[subject]] + diag(0.01, n_predictors)
    cross_response                  <- design[[subject]] %*% t(X[[subject]])
    upper                           <- chol(precision)
    subject_coefficients[[subject]] <- t(backsolve(upper, forwardsolve(t(upper), cross_response)))
    residuals[[subject]]            <- X[[subject]] - subject_coefficients[[subject]] %*% design[[subject]]
  }

  # Use separate k-means fits to group the full coefficient estimates and the residual covariance summaries.
  coefficient_features <- matrix(unlist(subject_coefficients, use.names = FALSE),
    nrow = n_subjects, byrow = TRUE)
  covariance_features <- .pdpmvar_initial_covariance_features(residuals)
  coefficient_labels  <- .pdpmvar_initial_labels(coefficient_features, n_clusters)
  covariance_labels   <- .pdpmvar_initial_labels(covariance_features, n_clusters)
  coefficient_atoms   <- vector("list", max(coefficient_labels))
  covariance_atoms    <- vector("list", max(covariance_labels))

  # Estimate a starting coefficient matrix for each group using all subjects assigned to that group.
  for (cluster in seq_along(coefficient_atoms)) {
    members     <- which(coefficient_labels == cluster)
    precision   <- Reduce("+", design_cross[members]) + diag(0.01, n_predictors)
    information <- matrix(0, n_predictors, n_nodes)
    for (subject in members) {
      information <- information + design[[subject]] %*% t(X[[subject]])
    }
    upper                        <- chol(precision)
    coefficient_atoms[[cluster]] <- list(coefficient =
      t(backsolve(upper, forwardsolve(t(upper), information))))
  }

  # Estimate a starting covariance for each covariance group using the residuals of its assigned subjects.
  for (cluster in seq_along(covariance_atoms)) {
    residual                    <- do.call(cbind, residuals[which(covariance_labels == cluster)])
    covariance_atoms[[cluster]] <- .pdpmvar_initialize_covariance(
      tcrossprod(residual) / ncol(residual), n_factors)
  }

  return(list(coefficient_atoms = coefficient_atoms, covariance_atoms = covariance_atoms,
    coefficient_labels = coefficient_labels, covariance_labels = covariance_labels))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Coefficient and shrinkage updates ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Draw each response row of the coefficient matrices for groups that currently contain subjects,
# conditional on the sampled factors and shared prior variances.
# Used to check the C++ coefficient update separately in benchmark comparisons.
.pdpmvar_update_coefficients <- function(X,
                                         design,
                                         design_cross,
                                         factor_contributions,
                                         labels,
                                         atoms,
                                         shared_variance,
                                         residual_variances) {
  return(.pdpmvar_eigen_coefficient_update(X, design, design_cross, factor_contributions,
    as.numeric(labels), atoms, shared_variance, residual_variances))
}

# Draw the shared prior variance for each coefficient position using the matrices for groups
# that contain subjects, then draw lambda squared. These are the Laplace prior's shrinkage updates.
# Used to check the C++ shrinkage update separately in benchmark comparisons.
.pdpmvar_update_shrinkage <- function(labels, atoms, shared_variance, lambda_squared) {
  return(.pdpmvar_eigen_shrinkage(as.numeric(labels), atoms, shared_variance, lambda_squared))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Gaussian calculations for full coefficient matrices ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Store each subject's contribution to the Gaussian precision matrix and precision-times-mean vector.
# Their dimensions are coefficient-by-coefficient, (D * P) x (D * P), and length D * P.
# Here D is the number of response nodes and P = D * K is the number of predictors.
# Arrange the coefficient matrix as a vector by placing its columns one after another.
# Used to compare proposed coefficient groups and calculate their integrated likelihoods in benchmarks.
.pdpmvar_coefficient_statistics <- function(X,
                                            design,
                                            design_cross,
                                            covariances,
                                            variance) {
  return(.pdpmvar_eigen_statistics(X, design, design_cross, covariances, variance))
}

# Calculate the Gaussian conditional mean and integrate out one cluster's coefficient matrix.
# Leave out likelihood terms that are the same for every proposed clustering.
# Used to check the Gaussian calculations separately and compare coefficient groupings.
.pdpmvar_coefficient_integral <- function(precision, information, prior_log_determinant_half) {
  return(.pdpmvar_eigen_integral(precision, as.numeric(information), prior_log_determinant_half))
}

# Add the Gaussian statistics for subjects in a group, then integrate out their shared coefficient matrix.
# Used in benchmark comparisons of coefficient groupings. The stored statistics determine
# whether the calculation uses the full matrix or separate response rows.
.pdpmvar_coefficient_group <- function(statistics, subjects) {
  if (identical(statistics$representation, "factor_rows")) {
    return(.pdpmvar_coefficient_row_group(statistics, subjects))
  }

  # Add the stored subject contributions to the prior precision, then calculate the Gaussian integral in C++.
  precision   <- statistics$prior_precision
  information <- numeric(nrow(precision))
  for (subject in subjects) {
    precision   <- precision + statistics$precision[[subject]]
    information <- information + statistics$information[[subject]]
  }

  return(.pdpmvar_coefficient_integral(precision, information, statistics$prior_log_determinant_half))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Gaussian calculations conditional on sampled factors ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# After subtracting sampled factor contributions, store each subject's predictor cross-products
# and variance-weighted response-predictor cross-products. These are predictor-by-predictor,
# P x P, and response-by-predictor, D x P, where D is nodes and P = D * K is predictors.
# Store each distinct set of node residual variances once, in a nodes-by-variance-groups matrix,
# D x G. This lets us calculate one response row at a time, using much less memory.
# Used to check the coefficient calculations conditional on sampled factors.
.pdpmvar_coefficient_row_statistics <- function(adjusted,
                                                design,
                                                design_cross,
                                                residual_variances,
                                                variance) {
  return(.pdpmvar_eigen_row_statistics(adjusted, design, design_cross, residual_variances, variance))
}

# Calculate the Gaussian conditional distribution for each response row and add their
# integrated likelihood contributions to evaluate the full coefficient matrix.
# Return the means and precision-times-mean values as vectors of length D * P, with columns
# placed one after another. Here D is response nodes and P = D * K is lagged predictors.
# If normal draws are supplied in noise, also return a response-by-predictor coefficient draw, D x P.
# Used to compare coefficient groups and draw their matrices after a split/merge decision.
.pdpmvar_coefficient_row_group <- function(statistics, subjects, noise = NULL) {
  return(.pdpmvar_eigen_row_group(statistics, as.integer(subjects), noise))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Assign subjects to the two groups in a proposed split ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Keep two selected subjects in separate groups, then assign the other subjects to one of those groups.
# Calculate assignment probabilities by integrating out the two coefficient matrices.
# If forced_labels is supplied, evaluate the probability of those labels instead of drawing new ones.
# Used to propose a split and calculate the forward or reverse proposal probability for acceptance.
.pdpmvar_coefficient_restricted_split <- function(statistics,
                                                  subjects,
                                                  anchors,
                                                  log_weights,
                                                  h,
                                                  g,
                                                  forced_labels = NULL) {
  if (identical(statistics$representation, "factor_rows")) {
    return(.pdpmvar_coefficient_row_restricted(statistics, subjects, anchors, log_weights, h, g, forced_labels))
  }

  uniforms <- if (is.null(forced_labels)) runif(length(setdiff(subjects, anchors))) else numeric()
  forced   <- if (is.null(forced_labels)) NULL else as.numeric(forced_labels)

  return(.pdpmvar_eigen_restricted_split(statistics, as.integer(subjects), as.integer(anchors),
    log_weights, as.integer(h), as.integer(g), forced, uniforms))
}

# Assign subjects to two groups with two selected subjects held in opposite groups.
# Conditional on sampled factors, add the Gaussian contributions from every response row
# before assigning each subject one label for its entire coefficient matrix, across all lags.
# Used to propose a split or evaluate its probability when checking the factor-conditional move.
.pdpmvar_coefficient_row_restricted <- function(statistics,
                                                subjects,
                                                anchors,
                                                log_weights,
                                                h,
                                                g,
                                                forced_labels = NULL) {
  uniforms <- if (is.null(forced_labels)) runif(length(setdiff(subjects, anchors))) else numeric()

  return(.pdpmvar_eigen_row_restricted(statistics, as.integer(subjects), as.integer(anchors),
    log_weights, h, g, forced_labels, uniforms))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Coefficient split/merge moves ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Propose splitting one coefficient group or merging two groups, calculate the acceptance
# probability, and draw the affected coefficient matrices after accepting or rejecting the move.
# Used to check this C++ sampling step separately from the full fitting procedure.
.pdpmvar_coefficient_split_merge <- function(X,
                                             design,
                                             design_cross,
                                             covariances,
                                             labels,
                                             atoms,
                                             variance,
                                             concentration,
                                             slots) {
  return(.pdpmvar_eigen_coefficient_move(X, design, design_cross, covariances, as.numeric(labels),
    atoms, variance, concentration, slots))
}

# Draw latent factors, then propose a coefficient split/merge conditional on those draws.
# Keep the factors fixed while comparing the current and proposed groups, integrating out
# their coefficient matrices. Draw the factors again before the sampler next conditions on them.
# Used to check this sampling step separately; the full fitting procedure does the same work in C++.
.pdpmvar_coefficient_factor_move <- function(X,
                                             design,
                                             design_cross,
                                             labels,
                                             atoms,
                                             covariance_labels,
                                             covariance_atoms,
                                             variance,
                                             concentration,
                                             slots) {
  adjusted           <- vector("list", length(X))
  residual_variances <- matrix(0, nrow(variance), length(X))

  # Subtract the sampled factor contributions before proposing groups and drawing their coefficient matrices.
  for (i in seq_along(X)) {
    atom                      <- covariance_atoms[[covariance_labels[i]]]
    residual                  <- X[[i]] - atoms[[labels[i]]]$coefficient %*% design[[i]]
    factors                   <- .pdpmvar_draw_factors(residual, atom)
    adjusted[[i]]             <- X[[i]] - atom$working_loadings %*% factors
    residual_variances[, i]   <- atom$variance
  }

  return(.pdpmvar_eigen_row_move(adjusted, design, design_cross, residual_variances,
    as.numeric(labels), atoms, variance, concentration, slots))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Coefficient cluster assignments ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Draw one coefficient-group label per subject, using its likelihood under each group
# whose weight exceeds that subject's slice value. All rows and lags share this label.
# Called by the C++ sampler when more than 200 mixture components are available;
# also used to check cluster assignments separately.
.pdpmvar_assign_coefficients <- function(X,
                                         design,
                                         coefficient_atoms,
                                         covariance_labels,
                                         covariance_atoms,
                                         slice) {
  if (length(slice$weights) <= 200) {
    return(.pdpmvar_eigen_assign_coefficients(X, design, coefficient_atoms, as.numeric(covariance_labels),
      covariance_atoms, slice$weights, slice$slices, runif(length(X))))
  }

  # With more than 200 components, use R's weighted sampling method so a fixed seed
  # continues to give the same sequence of draws. Calculate the likelihoods in C++.
  labels        <- numeric(length(X))
  scores        <- matrix(-Inf, length(X), length(coefficient_atoms))
  probabilities <- matrix(0, length(X), length(coefficient_atoms))
  for (subject in seq_along(X)) {
    candidates <- which(slice$weights > slice$slices[subject])
    covariance <- covariance_atoms[[covariance_labels[subject]]]$covariance
    for (cluster in candidates) {
      residual <- X[[subject]] - coefficient_atoms[[cluster]]$coefficient %*% design[[subject]]
      scores[subject, cluster] <- .pdpmvar_log_likelihood(residual, covariance)
    }

    log_probability <- scores[subject, candidates]
    probability     <- exp(log_probability - max(log_probability))
    probabilities[subject, candidates] <- probability / sum(probability)
    labels[subject] <- .pdpmvar_sample_label(log_probability, candidates)
  }

  return(list(labels = labels, log_likelihood = scores, probabilities = probabilities))
}
