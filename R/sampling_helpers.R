# Shared C++ sampling functions and R functions for preparing data and starting values.

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Starting parameter values and covariance priors ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Draw a coefficient matrix from its normal prior, using the supplied prior variances.
# Return a response-by-predictor matrix, D x P, inside a list named coefficient, where
# D is the number of nodes, K is the number of lags and P = D * K is the number of predictors.
# Used while pdpmvar() sets up the starting coefficient groups.
.pdpmvar_new_coefficient <- function(variance) {
  coefficient <- matrix(rnorm(length(variance)), nrow(variance), ncol(variance)) * sqrt(variance)

  return(list(coefficient = coefficient))
}

# Calculate a covariance matrix from factor loadings, factor precisions and node variances.
# The node-by-factor loading matrices are D x B and the node-by-node covariance is D x D,
# where D is the number of nodes and B is the number of factors. Divide the working loadings
# by the square roots of the factor precisions and choose signs so the diagonal is positive.
# Used when constructing starting covariance parameters and checking simulated covariances.
.pdpmvar_refresh_covariance <- function(atom) {
  signs <- sign(diag(atom$working_loadings))
  signs[signs == 0] <- 1

  # Choose loading signs and scales, then add the node residual variances.
  atom$loadings   <- sweep(atom$working_loadings, 2, signs / sqrt(atom$factor_precision), "*")
  atom$covariance <- tcrossprod(atom$loadings) + diag(atom$variance, nrow(atom$loadings))

  return(atom)
}

# Draw factor loadings, factor precisions and node residual variances from the covariance prior.
# Return node-by-factor loadings, D x B; B factor precisions; D residual variances; and
# an implied node-by-node covariance matrix, D x D. Here D is n_nodes and B is n_factors.
# Used by pdpmvar() to set up the starting covariance groups.
.pdpmvar_new_covariance <- function(n_nodes, n_factors, cov_shape, cov_rate) {
  working_loadings <- matrix(rnorm(n_nodes * n_factors), n_nodes, n_factors)
  working_loadings[upper.tri(working_loadings)] <- 0

  factor_precision <- rgamma(n_factors, shape = 0.5, rate = 0.5)
  variance         <- 1 / rgamma(n_nodes, shape = cov_shape, rate = cov_rate)

  return(.pdpmvar_refresh_covariance(list(working_loadings = working_loadings,
    factor_precision = factor_precision, variance = variance)))
}

# Choose the Gamma prior's shape and rate by matching the mean and variance of
# inverse sample variances, calculated separately for each node within each subject.
# Used by pdpmvar() when neither cov_shape nor cov_rate is supplied. Return the two
# parameters as a named vector; an error is raised if the sample precisions cannot be used.
.pdpmvar_calibrate_covariance <- function(X) {
  precisions         <- unlist(lapply(X, function(x) 1 / apply(x, 1, stats::var)), use.names = FALSE)
  precision_mean     <- mean(precisions)
  precision_variance <- stats::var(precisions)
  if (any(!is.finite(precisions)) || !is.finite(precision_variance) || precision_variance <= 0) {
    stop("Covariance calibration needs positive, varying node variances; supply cov_shape and cov_rate explicitly.")
  }

  return(c(shape = precision_mean^2 / precision_variance,
           rate = precision_mean / precision_variance))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Gaussian and shrinkage draws ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Draw a Gaussian vector from a supplied precision matrix and precision-times-mean vector.
# For a vector with P entries, precision is P x P and information has length P.
# Provided for checking Gaussian coefficient draws separately; the full sampler draws them in C++.
.pdpmvar_draw_normal <- function(precision, information) {
  return(.pdpmvar_eigen_normal(precision, as.numeric(information), rnorm(nrow(precision))))
}

# Draw one positive value from the generalized inverse Gaussian distribution GIG(p, chi, psi).
# Provided for checking the distribution used to update shared coefficient prior variances;
# the full C++ sampler draws these values within the Laplace shrinkage update.
.pdpmvar_draw_gig <- function(p, chi, psi) {
  return(.pdpmvar_eigen_gig(p, chi, psi))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DP weights, slices and label sampling ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Draw mixture weights and one slice value per subject. Add further stick weights
# until every subject's slice value is covered; the number of components is not fixed.
# Return a weight vector of length C and slice values of length N, where C is the
# number of available components and N is the number of subjects.
# Provided for checking the stick-breaking step used by the C++ sampler.
.pdpmvar_slice_weights <- function(labels, concentration) {
  return(.pdpmvar_eigen_slice_weights(as.numeric(labels), concentration))
}

# Draw log weights for the first slots stick positions, returning a vector of length slots.
# Include subjects assigned to later positions when calculating these weights.
# Used to check the auxiliary weights for coefficient split/merge proposals; limiting
# this extra proposal to the first slots positions does not truncate the fitted DP mixture.
.pdpmvar_prefix_weights <- function(labels, concentration, slots) {
  return(.pdpmvar_eigen_prefix_weights(as.numeric(labels), concentration, slots))
}

# Draw one label from candidates, with probabilities proportional to exp(log_probability).
# The two input vectors have the same length. Subtract the largest log probability before
# exponentiating so very small likelihoods can be compared without numerical underflow.
# Used by the R coefficient and covariance assignment functions when more than 200
# mixture components are available.
.pdpmvar_sample_label <- function(log_probability, candidates) {
  if (!any(is.finite(log_probability))) {
    stop("No candidate cluster has a finite likelihood.")
  }

  probability <- exp(log_probability - max(log_probability))

  return(candidates[sample.int(length(candidates), 1, prob = probability)])
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Likelihoods from residual observations and cross-products ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Calculate the log likelihood of a nodes-by-observations residual matrix, D x T,
# under a node-by-node covariance matrix, D x D. Here D is nodes and T is observations.
# Omit the constant -D * T * log(2 * pi) / 2, which is the same across candidate groups.
# Used to compare group assignments and calculate predictive log likelihoods in simulation checks.
.pdpmvar_log_likelihood <- function(residual, covariance) {
  return(.pdpmvar_eigen_log_likelihood(residual, covariance))
}

# Add each subject's residual log likelihood under its currently assigned covariance.
# Residuals contains N nodes-by-observations matrices, D x T_i, and labels has length N,
# where N is subjects, D is nodes and T_i is the number of observations for subject i.
# Omit the same Gaussian constant as .pdpmvar_log_likelihood().
# Used to record log likelihoods when fitting simulated data with subject group labels held fixed.
.pdpmvar_total_likelihood <- function(residuals, labels, atoms) {
  return(.pdpmvar_eigen_total_likelihood(residuals, as.numeric(labels), atoms))
}

# Calculate each subject's residual log likelihood under one proposed covariance matrix.
# Supply N node-by-node residual cross-product matrices, D x D, and N observation counts.
# Each cross-product is Y %*% t(Y), not a sample covariance; D is nodes and N is subjects.
# Return a vector of N log likelihoods, omitting the common Gaussian constants.
# Provided for checking the likelihood calculations used in covariance proposals.
.pdpmvar_covariance_scores <- function(scatters, n_times, atom) {
  return(.pdpmvar_eigen_scatter_scores(scatters, n_times, atom$covariance))
}

# Calculate a covariance log likelihood from a residual cross-product matrix, D x D,
# and an observation count, where D is nodes. Both inputs may already have been
# multiplied by a fractional weight for an intermediate split/merge distribution.
# Return one log likelihood, omitting the common Gaussian constant.
# Provided for checking likelihood comparisons along the covariance annealing bridge.
.pdpmvar_covariance_weighted_score <- function(atom, scatter, n_times) {
  return(.pdpmvar_eigen_covariance_weighted_score(atom, scatter, n_times))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Factor and covariance updates ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Draw latent factors conditional on a nodes-by-observations residual matrix, D x T,
# and the assigned covariance parameters. Return a factors-by-observations matrix,
# B x T, where D is nodes, B is factors and T is observations.
# Used when checking coefficient split/merge moves and updating coefficients in simulation fits.
# Subtract the sampled factor contribution before drawing the coefficient rows.
.pdpmvar_draw_factors <- function(residual, atom) {
  n_factors <- ncol(atom$working_loadings)
  noise     <- matrix(rnorm(n_factors * ncol(residual)), n_factors)

  return(.pdpmvar_eigen_factors(residual, atom, noise))
}

# Draw new loadings, factor precisions and residual variances for one covariance group,
# first drawing factors conditional on that group's residuals. The input residual matrix
# is nodes by pooled observations, D x T. Return loadings of size D x B and a covariance
# matrix of size D x D, together with B factor precisions and D residual variances.
# Here D is nodes, B is factors and T includes all observations assigned to this group.
# Used to fit simulated data with subject group labels held fixed and compare R and C++ updates.
.pdpmvar_update_covariance <- function(residual, atom, cov_shape, cov_rate) {
  return(.pdpmvar_eigen_covariance_update(residual, atom, cov_shape, cov_rate))
}

# Update covariance parameters using a node-by-node residual cross-product matrix,
# D x D, rather than individual residual observations. Here D is nodes and B is factors.
# The effective observation count n_times may be fractional, but must exceed D + B + 1;
# the residual cross-product matrix must be positive definite.
# Used for reversible updates within the covariance annealing bridge. Loading and
# factor scales are also updated without changing the covariance they describe.
# factor_shape is 1/2 for the fitted model; other values apply only to intermediate proposals.
.pdpmvar_covariance_reversible <- function(atom,
                                           scatter,
                                           n_times,
                                           cov_shape,
                                           cov_rate,
                                           factor_shape = 0.5) {
  return(.pdpmvar_eigen_covariance_reversible(atom, scatter, n_times, cov_shape, cov_rate, factor_shape))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Covariance cluster assignments ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Draw one covariance-group label per subject from components whose weights exceed
# that subject's slice value. Slice eligibility already accounts for mixture weights,
# so the assignment probabilities use only the eligible component likelihoods.
# For N subjects and C available components, return N labels and two subjects-by-components
# matrices, N x C, containing log likelihoods and assignment probabilities.
# Called by the C++ sampler when more than 200 components are available; also used
# to check cluster assignments separately.
.pdpmvar_assign_covariances <- function(residuals, atoms, slice) {
  if (length(slice$weights) <= 200) {
    return(.pdpmvar_eigen_assign_covariances(residuals, atoms, slice$weights, slice$slices,
      runif(length(residuals))))
  }

  # With more than 200 components, draw labels using R's weighted sampling function.
  # The component likelihoods are still calculated in C++.
  labels        <- numeric(length(residuals))
  scores        <- matrix(-Inf, length(residuals), length(atoms))
  probabilities <- matrix(0, length(residuals), length(atoms))
  for (subject in seq_along(residuals)) {
    candidates <- which(slice$weights > slice$slices[subject])
    log_probability <- vapply(candidates, function(cluster)
      .pdpmvar_log_likelihood(residuals[[subject]], atoms[[cluster]]$covariance), numeric(1))

    scores[subject, candidates]        <- log_probability
    probability                       <- exp(log_probability - max(log_probability))
    probabilities[subject, candidates] <- probability / sum(probability)
    labels[subject]                    <- .pdpmvar_sample_label(log_probability, candidates)
  }

  return(list(labels = labels, log_likelihood = scores, probabilities = probabilities))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Covariance group priors and proposed subject assignments ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Calculate the log prior probability of the subject labels with stick weights integrated out.
# Keep the supplied stick positions, including any empty positions between occupied ones,
# rather than renumbering the groups. Return one log probability.
# Provided for checking the prior contribution to covariance split/merge acceptance probabilities.
.pdpmvar_covariance_label_prior <- function(labels, concentration) {
  return(.pdpmvar_eigen_covariance_label_prior(as.numeric(labels), concentration))
}

# Keep the two selected subjects in separate groups, then assign the remaining selected
# subjects between them using inverse-Wishart predictive probabilities. This distribution
# guides the proposal only; it does not replace the model's covariance prior.
# Supply node-by-node residual cross-products, D x D, for each of N subjects. Return a
# label vector of length N and the log probability for assigning the selected subjects.
# Here D is the number of nodes and N is the number of subjects.
# If forced is supplied, evaluate those assignments instead of drawing new ones.
# Used to check the assignment proposal within covariance split/merge moves.
.pdpmvar_covariance_restricted <- function(scatters,
                                           n_times,
                                           subjects,
                                           anchors,
                                           h,
                                           g,
                                           forced = NULL) {
  return(.pdpmvar_eigen_covariance_restricted(scatters, n_times, as.integer(subjects),
    as.integer(anchors), h, g, forced))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Propose covariance parameters and calculate their densities ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Propose covariance parameters near center, randomly choosing each loading column's sign.
# Return node-by-factor loadings, D x B; B factor precisions; D residual variances; and
# a node-by-node covariance matrix, D x D, where D is nodes and B is factors.
# Used in checks of direct covariance proposals; the fitted sampler instead uses the
# annealing bridge and its auxiliary proposal described below.
.pdpmvar_propose_covariance <- function(center) {
  return(.pdpmvar_eigen_propose_covariance(center))
}

# Calculate the log density of the direct covariance proposal around center, averaging
# over both possible signs of every loading column. The supplied loading matrices
# are nodes by factors, D x B, where D is nodes and B is factors.
# Used with .pdpmvar_propose_covariance() to check forward/reverse proposal probabilities.
.pdpmvar_covariance_log_proposal <- function(atom, center) {
  return(.pdpmvar_eigen_covariance_log_proposal(atom, center))
}

# Draw the auxiliary covariance parameters used at one end of an annealing bridge.
# Free working loadings are standard normal and factor precisions follow Gamma(2, 1/2).
# Residual variances have inverse-Gamma shape 16 and scale 15 * mean(center$variance),
# so their expected value matches the mean residual variance in center. These are proposal
# distributions, not changes to the fitted model's priors.
# Return node-by-factor loadings, D x B; B precisions; D residual variances; and a
# node-by-node covariance matrix, D x D, where D is nodes and B is factors.
# Provided for checking the auxiliary proposal used by the C++ covariance split/merge step.
.pdpmvar_bridge_propose <- function(center) {
  return(.pdpmvar_eigen_bridge_propose(center))
}

# Calculate the auxiliary proposal's log density for the supplied working loadings,
# factor precisions and residual variances, including all normalizing constants.
# Used with .pdpmvar_bridge_propose() to check the proposal-density contribution
# to the annealing bridge's acceptance calculation.
.pdpmvar_bridge_density <- function(atom, center) {
  return(.pdpmvar_eigen_bridge_density(atom, center))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Move between current and proposed covariance groups ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Update covariance parameters at an intermediate step between the auxiliary proposal
# (beta = 0) and the model prior with the full group likelihood (beta = 1).
# Weight the residual cross-product matrix and observation count by beta, and combine
# the proposal and model prior parameters accordingly. scatter is nodes by nodes,
# D x D, and B is the number of factors. Return the updated covariance parameters.
# Provided for checking the intermediate updates used by the C++ covariance annealing bridge.
.pdpmvar_bridge_auxiliary <- function(atom,
                                      scatter,
                                      n_times,
                                      beta,
                                      center,
                                      B,
                                      shape,
                                      rate) {
  return(.pdpmvar_eigen_bridge_auxiliary(atom, scatter, n_times, beta, center, B, shape, rate))
}

# Move covariance parameters through intermediate distributions for the current and
# proposed subject groups. The returned log_work contributes to the split/merge
# acceptance ratio; this function does not itself accept or reject the full move.
# Supply N node-by-node residual cross-products, D x D, and N observation counts,
# where N is subjects and D is nodes. Only the parameters at group positions h and g
# are changed; B is the number of factors. Return the parameter list, log_work and
# intermediate levels, with the two groups' parameter values at each saved level
# if store_path = TRUE. Used to check the annealing calculation separately.
# See Karagiannis and Andrieu (2013), JCGS 22:623-648, doi:10.1080/10618600.2013.805651.
.pdpmvar_covariance_bridge <- function(old_labels,
                                       new_labels,
                                       atoms,
                                       scatters,
                                       n_times,
                                       h,
                                       g,
                                       B,
                                       shape,
                                       rate,
                                       store_path = FALSE) {
  return(.pdpmvar_eigen_covariance_bridge(as.numeric(old_labels), as.numeric(new_labels), atoms, scatters,
    n_times, h, g, B, shape, rate, store_path))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Covariance split/merge moves and acceptance ratios ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Calculate the target-density change and forward/reverse proposal probabilities for
# a direct covariance split/merge at fixed stick positions. Subject label vectors
# have length N and each residual cross-product is nodes by nodes, D x D, where
# N is subjects and D is nodes. Return the log acceptance ratio and its components.
# Used in direct-proposal probability checks; the fitted sampler uses the annealing
# bridge's acceptance calculation instead.
.pdpmvar_covariance_move_ratio <- function(old_labels,
                                           new_labels,
                                           old_atoms,
                                           new_atoms,
                                           log_weights,
                                           scatters,
                                           n_times,
                                           anchors,
                                           center,
                                           slots,
                                           cov_shape,
                                           cov_rate) {
  return(.pdpmvar_eigen_covariance_move_ratio(as.numeric(old_labels), as.numeric(new_labels), old_atoms,
    new_atoms, log_weights, scatters, n_times, as.integer(anchors), center, slots, cov_shape, cov_rate))
}

# Propose splitting one covariance group or merging two, use the annealing bridge
# to propose their parameters, then accept or reject the complete move.
# For N subjects and D nodes, supply residual matrices of size D x T_i, where T_i is
# subject i's number of observations. Return N group labels, the covariance parameter
# list and a record of the proposed move and acceptance decision.
# Used to check the C++ covariance split/merge step separately from the full sampler.
.pdpmvar_covariance_split_merge <- function(residuals,
                                            labels,
                                            atoms,
                                            concentration,
                                            n_factors,
                                            cov_shape,
                                            cov_rate,
                                            slots) {
  return(.pdpmvar_eigen_covariance_move(residuals, as.numeric(labels), atoms, concentration,
    n_factors, cov_shape, cov_rate, slots))
}
