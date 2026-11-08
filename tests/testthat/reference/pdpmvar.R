# R coefficient and iteration references, loaded only for validation.

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Coefficient-row updates
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Update occupied coefficient rows conditional on factors and shared shrinkage variances.
# Used in ordinary R sampler iterations after factors are drawn and before shrinkage updates.
.pdpmvar_update_coefficients <- function(X, design, design_cross, factor_contributions,
                                         labels, atoms, shared_variance, residual_variances,
                                         backend = "R") {
  if (backend == "RcppEigen") {
    return(.pdpmvar_eigen_coefficient_update(X, design, design_cross, factor_contributions,
      as.numeric(labels), atoms, shared_variance, residual_variances))
  }
  n_nodes      <- nrow(shared_variance)
  n_predictors <- ncol(shared_variance)

  # Conditional on factors, response rows are Gaussian regressions with diagonal noise.
  for (cluster in sort(unique(labels))) {
    members <- which(labels == cluster)
    atom    <- atoms[[cluster]]
    for (response in seq_len(n_nodes)) {
      precision   <- diag(1 / shared_variance[response, ], n_predictors)
      information <- numeric(n_predictors)

      # Add each member's likelihood precision and precision-times-mean contribution.
      for (subject in members) {
        residual_precision <- 1 / residual_variances[response, subject]
        adjusted           <- X[[subject]][response, ] - factor_contributions[[subject]][response, ]
        precision          <- precision + residual_precision * design_cross[[subject]]
        information        <- information + residual_precision * as.numeric(design[[subject]] %*% adjusted)
      }

      atom$coefficient[response, ] <- .pdpmvar_draw_normal(precision, information)
    }
    atoms[[cluster]] <- atom
  }

  return(atoms)
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Shared coefficient shrinkage
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Update shared coefficient variances across occupied atoms, then update lambda squared.
# Used after coefficient updates when the Laplace prior is selected; the normal prior skips it.
.pdpmvar_update_shrinkage <- function(labels, atoms, shared_variance, lambda_squared, backend = "R") {
  if (backend == "RcppEigen") {
    return(.pdpmvar_eigen_shrinkage(as.numeric(labels), atoms, shared_variance, lambda_squared))
  }

  # Only occupied atoms enter the shared variance conditional.
  occupied    <- sort(unique(labels))
  sum_squared <- shared_variance * 0
  for (cluster in occupied) {
    sum_squared <- sum_squared + atoms[[cluster]]$coefficient^2
  }

  # GIG shape depends on the number of occupied matrices, not the number of subjects.
  for (index in seq_along(shared_variance)) {
    shared_variance[index] <- .pdpmvar_draw_gig(1 - length(occupied) / 2,
      sum_squared[index], lambda_squared)
  }

  # Update the global rate under the original Gamma(1, 2) shape/rate hyperprior.
  lambda_squared <- rgamma(1, shape = length(shared_variance) + 1,
                           rate = 2 + sum(shared_variance) / 2)

  return(list(shared_variance = shared_variance, lambda_squared = lambda_squared))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Full-matrix coefficient assignments
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Assign complete coefficient blocks conditional on slice eligibility and covariances.
# Used for ordinary coefficient slice allocations in R and as the C++ large-prefix fallback.
.pdpmvar_assign_coefficients <- function(X, design, coefficient_atoms, covariance_labels,
                                         covariance_atoms, slice, backend = "R") {
  # Very large prefixes retain R's alias-sampling RNG behavior.
  if (backend == "RcppEigen" && length(slice$weights) <= 200) {
    return(.pdpmvar_eigen_assign_coefficients(X, design, coefficient_atoms, as.numeric(covariance_labels),
      covariance_atoms, slice$weights, slice$slices, runif(length(X))))
  }

  # Slice eligibility already accounts for DP weights; scores contain likelihoods only.
  labels        <- numeric(length(X))
  scores        <- matrix(-Inf, length(X), length(coefficient_atoms))
  probabilities <- matrix(0, length(X), length(coefficient_atoms))
  for (subject in seq_along(X)) {
    candidates <- which(slice$weights > slice$slices[subject])
    covariance <- covariance_atoms[[covariance_labels[subject]]]$covariance
    for (cluster in candidates) {
      residual                 <- X[[subject]] - coefficient_atoms[[cluster]]$coefficient %*% design[[subject]]
      scores[subject, cluster] <- .pdpmvar_log_likelihood(residual, covariance, backend = backend)
    }

    # Subtract the maximum before exponentiating to avoid numerical underflow.
    log_probability                    <- scores[subject, candidates]
    probability                        <- exp(log_probability - max(log_probability))
    probabilities[subject, candidates] <- probability / sum(probability)
    labels[subject]                    <- .pdpmvar_sample_label(log_probability, candidates)
  }

  return(list(labels = labels, log_likelihood = scores, probabilities = probabilities))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dense coefficient split/merge reference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Construct a joint Gaussian proposal for the entire coefficient block.
# Retained as a direct Gaussian reference for tests; the sampler uses cached group calculations.
.pdpmvar_coefficient_gaussian <- function(X, design, design_cross, precisions, variance, subjects) {
  if (8 * length(variance)^2 > 512 * 1024^2) stop("Dense coefficient precision exceeds 512 MiB; use factor_conditional moves.")
  n_nodes      <- nrow(variance)
  n_predictors <- ncol(variance)
  precision    <- diag(1 / as.numeric(variance), length(variance))
  information  <- matrix(0, n_nodes, n_predictors)
  for (subject in subjects) {
    precision   <- precision + kronecker(design_cross[[subject]], precisions[[subject]])
    information <- information + precisions[[subject]] %*% X[[subject]] %*% t(design[[subject]])
  }
  upper <- chol(precision)
  mean  <- backsolve(upper, forwardsolve(t(upper), as.numeric(information)))

  return(list(mean = as.numeric(mean), upper = upper))
}

# Cache one coefficient-vector precision and information vector per subject:
# (D * P) x (D * P) and length D * P, respectively, where D is response nodes
# and P is predictors (nodes times lags). Each coefficient matrix is stacked columnwise.
# Used at the start of a covariance-marginal coefficient split/merge attempt to cache subject terms.
.pdpmvar_coefficient_statistics <- function(X, design, design_cross, covariances, variance, backend = "R") {
  if (8 * length(X) * length(variance)^2 > 512 * 1024^2) stop("Dense coefficient cache exceeds 512 MiB; use factor_conditional moves.")
  if (backend == "RcppEigen") {
    return(.pdpmvar_eigen_statistics(X, design, design_cross, covariances, variance))
  }

  # Cache per-subject likelihood terms so proposals can assemble groups by addition.
  precision <- information <- vector("list", length(X))
  for (subject in seq_along(X)) {
    omega                  <- chol2inv(chol(covariances[[subject]]))
    precision[[subject]]   <- kronecker(design_cross[[subject]], omega)
    information[[subject]] <- as.numeric(omega %*% X[[subject]] %*% t(design[[subject]]))
  }

  return(list(precision = precision, information = information,
    prior_precision = diag(1 / as.numeric(variance), length(variance)),
    prior_log_determinant_half = -sum(log(variance)) / 2, backend = backend))
}

# Integrate a normal coefficient block, omitting likelihood terms shared by all partitions.
# Used by dense group calculations and restricted allocations to score integrated coefficient atoms.
.pdpmvar_coefficient_integral <- function(precision, information, prior_log_determinant_half, backend = "R") {
  if (backend == "RcppEigen") {
    return(.pdpmvar_eigen_integral(precision, as.numeric(information), prior_log_determinant_half))
  }

  # Complete the square; the Cholesky diagonal supplies the integration determinant.
  upper        <- chol(precision)
  mean         <- as.numeric(backsolve(upper, forwardsolve(t(upper), information)))
  log_integral <- sum(information * mean) / 2 + prior_log_determinant_half - sum(log(diag(upper)))

  return(list(precision = precision, information = information, upper = upper,
              mean = mean, log_integral = log_integral))
}

# Build a cluster's integrated Gaussian contribution from cached subject statistics.
# Used to score old/proposed clusters in split/merge ratios and refresh affected coefficient atoms.
.pdpmvar_coefficient_group <- function(statistics, subjects) {
  if (identical(statistics$representation, "factor_rows")) return(.pdpmvar_coefficient_row_group(statistics,
    subjects))
  precision   <- statistics$prior_precision
  information <- numeric(nrow(precision))
  for (subject in subjects) {
    precision   <- precision + statistics$precision[[subject]]
    information <- information + statistics$information[[subject]]
  }

  return(.pdpmvar_coefficient_integral(precision, information, statistics$prior_log_determinant_half,
    backend = statistics$backend))
}

# Sample or evaluate a restricted sequential split using exact Gaussian predictive scores.
# Used to generate a split allocation or evaluate its forward/reverse probability for MH correction.
.pdpmvar_coefficient_restricted_split <- function(statistics, subjects, anchors, log_weights,
                                                  h, g, forced_labels = NULL) {
  if (identical(statistics$representation, "factor_rows")) return(.pdpmvar_coefficient_row_restricted(statistics,
    subjects, anchors, log_weights, h, g, forced_labels))
  if (statistics$backend == "RcppEigen") {
    uniforms <- if (is.null(forced_labels)) runif(length(setdiff(subjects, anchors))) else numeric()
    forced   <- if (is.null(forced_labels)) NULL else as.numeric(forced_labels)

    return(.pdpmvar_eigen_restricted_split(statistics, as.integer(subjects), as.integer(anchors),
      log_weights, as.integer(h), as.integer(g), forced, uniforms))
  }

  # Anchors stay in opposite groups; remaining subjects are visited in a fixed order.
  groups <- list(.pdpmvar_coefficient_group(statistics, anchors[1]),
                 .pdpmvar_coefficient_group(statistics, anchors[2]))
  labels             <- rep(h, length(statistics$precision))
  labels[anchors[2]] <- g
  log_probability    <- 0
  for (subject in setdiff(subjects, anchors)) {
    candidates <- lapply(groups, function(group) .pdpmvar_coefficient_integral(
      group$precision + statistics$precision[[subject]],
      group$information + statistics$information[[subject]], statistics$prior_log_determinant_half,
      backend = statistics$backend))

    # The change in integrated mass is the predictive score for adding this subject.
    scores <- log_weights[c(h, g)] + vapply(candidates, function(group) group$log_integral, numeric(1)) -
      vapply(groups, function(group) group$log_integral, numeric(1))
    maximum    <- max(scores)
    normalizer <- maximum + log(sum(exp(scores - maximum)))

    # Forced labels evaluate the same proposal path for the reverse MH probability.
    selected <- if (is.null(forced_labels)) {
      .pdpmvar_sample_label(scores, 1:2)
    } else {
      if (forced_labels[subject] == h) 1 else 2
    }
    log_probability    <- log_probability + scores[selected] - normalizer
    labels[subject]    <- c(h, g)[selected]
    groups[[selected]] <- candidates[[selected]]
  }

  return(list(labels = labels, log_probability = log_probability, groups = groups))
}

# Evaluate forward/reverse probabilities under the analytically marginalized target.
# Used by both R coefficient split/merge methods to calculate the MH acceptance ratio.
.pdpmvar_coefficient_collapsed_ratio <- function(old_labels, new_labels, statistics,
                                                 log_weights, anchors, slots) {
  h        <- old_labels[anchors[1]]
  split    <- h == old_labels[anchors[2]]
  g        <- if (split) new_labels[anchors[2]] else old_labels[anchors[2]]
  subjects <- which(old_labels %in% c(h, g))
  old_h    <- .pdpmvar_coefficient_group(statistics, which(old_labels == h))
  old_g    <- .pdpmvar_coefficient_group(statistics, which(old_labels == g))
  new_h    <- .pdpmvar_coefficient_group(statistics, which(new_labels == h))
  new_g    <- .pdpmvar_coefficient_group(statistics, which(new_labels == g))

  # Compare both integrated atoms plus their fixed auxiliary stick-weight contributions.
  target_change <- new_h$log_integral + new_g$log_integral - old_h$log_integral - old_g$log_integral +
    sum(log_weights[new_labels[subjects]] - log_weights[old_labels[subjects]])

  # Splits choose an empty stick index; the reverse merge has no allocation choice.
  if (split) {
    empty      <- setdiff(seq_len(slots), unique(old_labels))
    allocation <- .pdpmvar_coefficient_restricted_split(statistics, subjects, anchors,
      log_weights, h, g, forced_labels = new_labels)
    forward <- -log(length(empty)) + allocation$log_probability
    reverse <- 0
  } else {
    empty      <- setdiff(seq_len(slots), unique(new_labels))
    allocation <- .pdpmvar_coefficient_restricted_split(statistics, subjects, anchors,
      log_weights, h, g, forced_labels = old_labels)
    forward <- 0
    reverse <- -log(length(empty)) + allocation$log_probability
  }

  return(list(log_ratio = target_change + reverse - forward, target_change = target_change,
              log_forward = forward, log_reverse = reverse))
}

# Move the full coefficient partition with coefficients integrated out, then refresh blocks.
# Used by the R sampler when a covariance-marginal coefficient move is scheduled.
.pdpmvar_coefficient_split_merge_reference <- function(X, design, design_cross, covariances,
                                                       labels, atoms, variance, concentration, slots, backend = "R") {
  record <- list(type = "skip", accepted = FALSE, log_ratio = NA_real_)
  if (length(labels) < 2) {return(list(labels = labels, atoms = atoms, record = record))}
  anchors <- sample.int(length(labels), 2)
  h       <- labels[anchors[1]]
  g       <- labels[anchors[2]]
  if (h > slots || g > slots) {return(list(labels = labels, atoms = atoms, record = record))}
  split <- h == g
  empty <- setdiff(seq_len(slots), unique(labels))
  if (split && !length(empty)) {return(list(labels = labels, atoms = atoms, record = record))}

  # Keep original stick indices and counts beyond the proposal prefix.
  log_weights <- .pdpmvar_prefix_weights(labels, concentration, slots, backend = backend)
  statistics  <- .pdpmvar_coefficient_statistics(X, design, design_cross, covariances, variance,
    backend = backend)
  proposed_labels <- labels
  if (split) {
    g                         <- empty[sample.int(length(empty), 1)]
    subjects                  <- which(labels == h)
    allocation                <- .pdpmvar_coefficient_restricted_split(statistics, subjects, anchors, log_weights, h, g)
    proposed_labels[subjects] <- allocation$labels[subjects]
  } else {
    proposed_labels[labels == g] <- h
  }

  # Evaluate the forward/reverse allocation correction before the MH decision.
  ratio <- .pdpmvar_coefficient_collapsed_ratio(labels, proposed_labels, statistics, log_weights, anchors, slots)
  if (is.na(ratio$log_ratio)) {stop("Non-finite coefficient split/merge acceptance calculation.")}
  accepted <- log(runif(1)) < min(0, ratio$log_ratio)
  if (accepted) {labels <- proposed_labels}

  # Refresh after acceptance or rejection: the decision marginalized both blocks.
  while (length(atoms) < max(h, g)) {
    atoms[[length(atoms) + 1]] <- .pdpmvar_new_coefficient(variance)
  }
  for (cluster in c(h, g)) {
    group            <- .pdpmvar_coefficient_group(statistics, which(labels == cluster))
    coefficient      <- group$mean + backsolve(group$upper, rnorm(length(variance)))
    atoms[[cluster]] <- list(coefficient = matrix(coefficient, nrow(variance), ncol(variance)))
  }
  record <- list(type = if (split) "split" else "merge", accepted = accepted, log_ratio = ratio$log_ratio)

  return(list(labels = labels, atoms = atoms, record = record))
}

# Dispatch the complete controller while retaining the original R implementation.
# Provides the standalone R/C++ move interface for checks; full sampler engines call their controllers
# directly.
.pdpmvar_coefficient_split_merge <- function(X, design, design_cross, covariances, labels, atoms, variance,
                                             concentration, slots, backend = "R") {
  if (backend == "RcppEigen") {
    return(.pdpmvar_eigen_coefficient_move(X, design, design_cross, covariances, as.numeric(labels), atoms, variance,
      concentration, slots))
  }

  return(.pdpmvar_coefficient_split_merge_reference(X, design, design_cross, covariances, labels, atoms, variance,
    concentration, slots, backend = backend))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compact factor-conditional coefficient split/merge
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Cache predictor-by-predictor cross-products, P x P, and response-by-predictor
# information, D x P, per subject after conditioning on factors. Here D is nodes
# and P is predictors (nodes times lags). Noise columns share a D x G variance
# matrix, where G counts distinct noise columns; full-vector precisions are avoided.
# Used at the start of a factor-conditional coefficient move, after subtracting sampled factor effects.
.pdpmvar_coefficient_row_statistics <- function(adjusted, design, design_cross, residual_variances, variance,
                                                backend = "R") {
  if (backend == "RcppEigen") return(.pdpmvar_eigen_row_statistics(adjusted, design, design_cross,
    residual_variances, variance))
  N <- length(adjusted)
  D <- nrow(variance)
  P <- ncol(variance)

  if (N < 1 || length(design) != N || length(design_cross) != N || !all(dim(residual_variances) == c(D, N)) ||
     any(!is.finite(variance)) || any(variance <= 0) || any(!is.finite(residual_variances)) || any(residual_variances <= 0)) {
    stop("Row statistics require matching dimensions and positive finite variances.")
  }
  information    <- vector("list", N)
  group_variance <- matrix(0, D, 0)
  variance_group <- numeric(N)

  # Subjects with identical diagonal noise share a variance column in the compact cache.
  for (i in seq_len(N)) {
    if (!all(dim(adjusted[[i]]) == c(D, ncol(design[[i]]))) || nrow(design[[i]]) != P ||
       !all(dim(design_cross[[i]]) == c(P, P)) || any(!is.finite(adjusted[[i]])) ||
       any(!is.finite(design[[i]])) || any(!is.finite(design_cross[[i]]))) stop("Invalid row-statistic subject matrices.")
    information[[i]] <- (adjusted[[i]] %*% t(design[[i]])) / residual_variances[, i]
    matches          <- which(vapply(seq_len(ncol(group_variance)),
      function(g) all(group_variance[, g] == residual_variances[, i]), logical(1)))
    if (!length(matches)) {
      group_variance <- cbind(group_variance, residual_variances[, i])
      matches        <- ncol(group_variance)
    }
    variance_group[i] <- matches[1]
  }

  return(list(design_cross = design_cross, information = information, variance = variance,
    group_variance = group_variance,
    variance_group = variance_group, n_subjects = N, representation = "factor_rows", backend = backend))
}

# Integrate each coefficient row and return column-stacked mean/information vectors
# of length D * P, where D is response nodes and P is predictors (nodes times lags).
# The optional draw is D x P; its supplied noise uses the same column-major order.
# The original coefficient prior enters once per full atom.
# Used for compact group scores, restricted allocations and the post-decision coefficient draws.
.pdpmvar_coefficient_row_group <- function(statistics, subjects, noise = NULL) {
  if (statistics$backend == "RcppEigen") return(.pdpmvar_eigen_row_group(statistics, as.integer(subjects), noise))
  variance <- statistics$variance
  D        <- nrow(variance)
  P        <- ncol(variance)
  mean     <- information <- matrix(0, D, P)
  upper    <- vector("list", D)
  integral <- 0

  # Integrate rows separately, then sum their evidence for one full-matrix atom.
  for (row in seq_len(D)) {
    precision <- diag(1 / variance[row, ], P)
    for (i in subjects) {
      precision          <- precision + statistics$design_cross[[i]] / statistics$group_variance[row, statistics$variance_group[i]]
      information[row, ] <- information[row, ] + statistics$information[[i]][row, ]
    }
    upper[[row]] <- chol(precision)
    mean[row, ]  <- backsolve(upper[[row]], forwardsolve(t(upper[[row]]), information[row, ]))
    if (length(subjects)) integral <- integral + sum(information[row, ] * mean[row, ]) / 2 -
      sum(log(variance[row, ])) / 2 - sum(log(diag(upper[[row]])))
  }

  result <- list(mean = as.numeric(mean), information = as.numeric(information), log_integral = integral,
    upper = upper)

  # Reuse column-major full-matrix noise when reconstructing a coefficient draw.
  if (!is.null(noise)) {
    if (!all(dim(noise) == c(D, P))) stop("Coefficient noise must match the variance matrix.")
    coefficient <- mean
    for (row in seq_len(D)) coefficient[row, ] <- mean[row, ] + backsolve(upper[[row]], noise[row, ])
    result$coefficient <- coefficient
  }

  return(result)
}

# One anchored partition of full matrices; row predictive integrals are summed
# before assigning each subject. Rows/lags never receive separate memberships.
# Used to propose or evaluate anchored subject allocations within a factor-conditional coefficient move.
.pdpmvar_coefficient_row_restricted <- function(statistics, subjects, anchors, log_weights, h, g,
                                                forced_labels = NULL) {
  if (statistics$backend == "RcppEigen") {
    uniforms <- if (is.null(forced_labels)) runif(length(setdiff(subjects, anchors))) else numeric()

    return(.pdpmvar_eigen_row_restricted(statistics, as.integer(subjects), as.integer(anchors), log_weights, h, g,
      forced_labels, uniforms))
  }

  # These are two groups of subjects; all rows and lags share the resulting label.
  members            <- list(anchors[1], anchors[2])
  groups             <- lapply(members, function(i) .pdpmvar_coefficient_row_group(statistics, i))
  labels             <- rep(h, statistics$n_subjects)
  labels[anchors[2]] <- g
  probability        <- 0
  for (i in setdiff(subjects, anchors)) {
    candidates <- lapply(1:2, function(j) .pdpmvar_coefficient_row_group(statistics, c(members[[j]], i)))

    # Row integrals have already been summed before comparing the two memberships.
    scores <- log_weights[c(h, g)] + vapply(candidates, `[[`, numeric(1), "log_integral") -
      vapply(groups, `[[`, numeric(1), "log_integral")
    maximum    <- max(scores)
    normalizer <- maximum + log(sum(exp(scores - maximum)))
    chosen     <- if (is.null(forced_labels)) .pdpmvar_sample_label(scores, 1:2) else
      if (forced_labels[i] == h) 1 else 2
    labels[i]         <- c(h, g)[chosen]
    probability       <- probability + scores[chosen] - normalizer
    members[[chosen]] <- c(members[[chosen]], i)
    groups[[chosen]]  <- candidates[[chosen]]
  }

  return(list(labels = labels, log_probability = probability, groups = groups))
}

# Preserve the original prefix-weight/anchor MH construction on the conditional
# row target, and refresh both integrated atoms after acceptance or rejection.
# Called after the factor draw to perform the compact R coefficient split/merge and refresh both atoms.
.pdpmvar_coefficient_row_move_reference <- function(adjusted, design, design_cross, residual_variances,
                                                    labels, atoms, variance, concentration, slots, backend = "R") {
  record <- list(type = "skip", accepted = FALSE, log_ratio = NA_real_)
  if (length(labels) < 2) return(list(labels = labels, atoms = atoms, record = record))
  anchors <- sample.int(length(labels), 2)
  h       <- labels[anchors[1]]
  g       <- labels[anchors[2]]
  if (h > slots || g > slots) return(list(labels = labels, atoms = atoms, record = record))
  split <- h == g
  empty <- setdiff(seq_len(slots), unique(labels))
  if (split && !length(empty)) return(list(labels = labels, atoms = atoms, record = record))

  # Draw auxiliary weights and construct the exact factor-conditional Gaussian target.
  weights    <- .pdpmvar_prefix_weights(labels, concentration, slots, backend = backend)
  statistics <- .pdpmvar_coefficient_row_statistics(adjusted, design, design_cross, residual_variances, variance,
    backend)
  proposed   <- labels
  if (split) {
    g                  <- empty[sample.int(length(empty), 1)]
    subjects           <- which(labels == h)
    allocation         <- .pdpmvar_coefficient_row_restricted(statistics, subjects, anchors, weights, h, g)
    proposed[subjects] <- allocation$labels[subjects]
  } else proposed[labels == g] <- h

  # The same target/forward/reverse calculation applies to the compact row representation.
  ratio <- .pdpmvar_coefficient_collapsed_ratio(labels, proposed, statistics, weights, anchors, slots)
  if (is.na(ratio$log_ratio)) stop("Invalid conditional coefficient split/merge ratio.")
  accepted <- log(runif(1)) < min(0, ratio$log_ratio)
  if (accepted) labels <- proposed

  # Both atoms were integrated out of the decision and must be redrawn either way.
  while (length(atoms) < max(h, g)) atoms[[length(atoms) + 1]] <- .pdpmvar_new_coefficient(variance)
  for (cluster in c(h, g)) {
    noise            <- matrix(rnorm(length(variance)), nrow(variance), ncol(variance))
    group            <- .pdpmvar_coefficient_row_group(statistics, which(labels == cluster), noise)
    atoms[[cluster]] <- list(coefficient = group$coefficient)
  }

  return(list(labels = labels, atoms = atoms,
    record = list(type = if (split) "split" else "merge", accepted = accepted, log_ratio = ratio$log_ratio)))
}

# A Gibbs factor draw precedes the row-collapsed move. Factors are held common
# through MH and discarded afterward; the ordinary sampler refreshes them before
# any later factor-conditioned update. The published joint posterior is unchanged.
# Used by the R sampler when a factor-conditional coefficient move is scheduled; dispatches its
# controller.
.pdpmvar_coefficient_factor_move <- function(X, design, design_cross, labels, atoms, covariance_labels,
                                             covariance_atoms, variance, concentration, slots, backend = "R") {
  adjusted           <- vector("list", length(X))
  residual_variances <- matrix(0, nrow(variance), length(X))

  # Draw factors under the current state and remove their contribution from responses.
  for (i in seq_along(X)) {
    atom                    <- covariance_atoms[[covariance_labels[i]]]
    residual                <- X[[i]] - atoms[[labels[i]]]$coefficient %*% design[[i]]
    factors                 <- .pdpmvar_draw_factors(residual, atom, backend = backend)
    adjusted[[i]]           <- X[[i]] - atom$working_loadings %*% factors
    residual_variances[, i] <- atom$variance
  }

  # Keep these factors fixed through the proposal; subsequent ordinary updates redraw them.
  if (backend == "RcppEigen") return(.pdpmvar_eigen_row_move(adjusted, design, design_cross, residual_variances,
    as.numeric(labels), atoms, variance, concentration, slots))

  return(.pdpmvar_coefficient_row_move_reference(adjusted, design, design_cross, residual_variances,
    labels, atoms, variance, concentration, slots, backend))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Reference sampler driver
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Run the reference sampler, returning response-by-predictor-by-subject coefficient
# summaries, D x P x N, and node-by-node-by-subject covariance summaries, D x D x N.
# Here D is nodes, P is predictors (nodes times lags) and N is subjects. Optional
# draw arrays append retained iteration as the last dimension.
# Used internally for validation of native update order and posterior bookkeeping.
.pdpmvar_sampler_reference <- function(X, design, design_cross, state, control) {
  n_nodes      <- nrow(state$shared_variance)
  n_predictors <- ncol(state$shared_variance)
  n_subjects   <- length(X)

  # Unpack the state while preserving the independent partitions and stick indices.
  coefficient_atoms  <- state$coefficient_atoms
  covariance_atoms   <- state$covariance_atoms
  coefficient_labels <- state$coefficient_labels
  covariance_labels  <- state$covariance_labels
  shared_variance    <- state$shared_variance
  lambda_squared     <- state$lambda_squared

  # Read iteration, prior and proposal controls used by the reference engine.
  n_burnin                      <- control$n_burnin
  n_mcmc                        <- control$n_mcmc
  n_factors                     <- control$n_factors
  coefficient_concentration     <- control$coefficient_concentration
  covariance_concentration      <- control$covariance_concentration
  coefficient_prior             <- control$coefficient_prior
  cov_shape                     <- control$cov_shape
  cov_rate                      <- control$cov_rate
  split_merge_every             <- control$split_merge_every
  split_merge_slots             <- control$split_merge_slots
  coefficient_split_merge_every <- control$coefficient_split_merge_every
  coefficient_split_merge_slots <- control$coefficient_split_merge_slots
  coefficient_split_merge       <- if (is.null(control$coefficient_split_merge)) "collapsed" else control$coefficient_split_merge
  backend                       <- control$backend
  store_draws                   <- control$store_draws
  verbose                       <- control$verbose

  # Accumulate posterior summaries only after burn-in; move/count traces include burn-in.
  coefficient_sum        <- array(0, c(n_nodes, n_predictors, n_subjects))
  positive_sum           <- coefficient_sum
  covariance_sum         <- array(0, c(n_nodes, n_nodes, n_subjects))
  coefficient_membership <- matrix(0, n_mcmc, n_subjects)
  covariance_membership  <- matrix(0, n_mcmc, n_subjects)
  coefficient_similarity <- matrix(0, n_subjects, n_subjects)
  covariance_similarity  <- coefficient_similarity
  cluster_counts         <- matrix(0, n_burnin + n_mcmc, 2,
    dimnames = list(NULL, c("coefficient", "covariance")))
  parameter_traces <- matrix(0, n_mcmc, 3,
    dimnames = list(NULL, c("log_likelihood", "mean_node_variance", "coefficient_rms")))
  split_merge_trace <- data.frame(iteration = seq_len(n_burnin + n_mcmc),
    type = rep("none", n_burnin + n_mcmc), accepted = FALSE, log_ratio = NA_real_)
  coefficient_split_merge_trace <- split_merge_trace
  lambda_trace                  <- numeric(n_mcmc)
  shared_variance_sum           <- shared_variance * 0

  # Full subject-level arrays are optional because they can dominate memory use.
  draws <- NULL
  if (store_draws) {
    draws <- list(coefficients = array(0, c(dim(coefficient_sum), n_mcmc)),
                  covariance = array(0, c(dim(covariance_sum), n_mcmc)))
  }

  for (iteration in seq_len(n_burnin + n_mcmc)) {
    # Propose a full coefficient-partition move before the ordinary slice allocation.
    if (coefficient_split_merge_every > 0 && iteration %% coefficient_split_merge_every == 0) {
      if (coefficient_split_merge == "factor_conditional") {
        move <- .pdpmvar_coefficient_factor_move(X, design, design_cross, coefficient_labels, coefficient_atoms,
          covariance_labels, covariance_atoms, shared_variance, coefficient_concentration,
          coefficient_split_merge_slots, backend)
      } else {
        subject_covariances <- lapply(covariance_labels, function(h) covariance_atoms[[h]]$covariance)
        move                <- .pdpmvar_coefficient_split_merge_reference(X, design, design_cross, subject_covariances,
          coefficient_labels, coefficient_atoms, shared_variance, coefficient_concentration,
          coefficient_split_merge_slots, backend = backend)
      }
      coefficient_labels                                 <- move$labels
      coefficient_atoms                                  <- move$atoms
      coefficient_split_merge_trace$type[iteration]      <- move$record$type
      coefficient_split_merge_trace$accepted[iteration]  <- move$record$accepted
      coefficient_split_merge_trace$log_ratio[iteration] <- move$record$log_ratio
    }

    # Sample coefficient allocation conditional on the marginal residual covariance.
    slice   <- .pdpmvar_slice_weights(coefficient_labels, coefficient_concentration, backend = backend)
    n_atoms <- length(slice$weights)
    while (length(coefficient_atoms) < n_atoms) {
      coefficient_atoms[[length(coefficient_atoms) + 1]] <- .pdpmvar_new_coefficient(shared_variance)
    }
    coefficient_atoms  <- coefficient_atoms[seq_len(n_atoms)]
    coefficient_labels <- .pdpmvar_assign_coefficients(X, design, coefficient_atoms,
      covariance_labels, covariance_atoms, slice, backend = backend)$labels

    # Refresh factors after marginalized allocation before conditioning on them.
    factor_contributions <- vector("list", n_subjects)
    for (subject in seq_len(n_subjects)) {
      residual <- X[[subject]] -
        coefficient_atoms[[coefficient_labels[subject]]]$coefficient %*% design[[subject]]
      atom                            <- covariance_atoms[[covariance_labels[subject]]]
      factors                         <- .pdpmvar_draw_factors(residual, atom, backend = backend)
      factor_contributions[[subject]] <- atom$working_loadings %*% factors
    }
    residual_variances <- matrix(unlist(lapply(covariance_labels,
      function(cluster) covariance_atoms[[cluster]]$variance), use.names = FALSE), n_nodes, n_subjects)
    coefficient_atoms <- .pdpmvar_update_coefficients(X, design, design_cross, factor_contributions,
      coefficient_labels, coefficient_atoms, shared_variance, residual_variances, backend = backend)

    # Shared shrinkage uses occupied matrices; refresh unused atoms under the new base.
    occupied <- sort(unique(coefficient_labels))
    if (coefficient_prior == "laplace") {
      shrinkage <- .pdpmvar_update_shrinkage(coefficient_labels, coefficient_atoms,
        shared_variance, lambda_squared, backend = backend)
      shared_variance <- shrinkage$shared_variance
      lambda_squared  <- shrinkage$lambda_squared
    }

    # Redraw unused atoms after updating the shared variances.
    for (cluster in setdiff(seq_along(coefficient_atoms), occupied)) {
      coefficient_atoms[[cluster]] <- .pdpmvar_new_coefficient(shared_variance)
    }

    # Covariance proposals condition on residuals from the newly updated coefficients.
    residuals <- vector("list", n_subjects)
    for (subject in seq_len(n_subjects)) {
      residuals[[subject]] <- X[[subject]] -
        coefficient_atoms[[coefficient_labels[subject]]]$coefficient %*% design[[subject]]
    }

    # Discard old slices/factors; each move handles its own augmentation, then refresh below.
    if (split_merge_every > 0 && iteration %% split_merge_every == 0) {
      move <- .pdpmvar_covariance_split_merge_reference(residuals, covariance_labels, covariance_atoms,
        covariance_concentration, n_factors, cov_shape, cov_rate, split_merge_slots, backend = backend)
      covariance_labels                      <- move$labels
      covariance_atoms                       <- move$atoms
      split_merge_trace$type[iteration]      <- move$record$type
      split_merge_trace$accepted[iteration]  <- move$record$accepted
      split_merge_trace$log_ratio[iteration] <- move$record$log_ratio
    }

    # Extend the covariance stick until all slices are covered, retaining its unbounded tail.
    slice   <- .pdpmvar_slice_weights(covariance_labels, covariance_concentration, backend = backend)
    n_atoms <- length(slice$weights)
    while (length(covariance_atoms) < n_atoms) {
      covariance_atoms[[length(covariance_atoms) + 1]] <-
        .pdpmvar_new_covariance(n_nodes, n_factors, cov_shape, cov_rate)
    }
    covariance_atoms  <- covariance_atoms[seq_len(n_atoms)]
    covariance_labels <- .pdpmvar_assign_covariances(residuals, covariance_atoms,
      slice, backend = backend)$labels

    # Refresh factors for the selected covariance, then sample loadings and variances.
    for (cluster in seq_along(covariance_atoms)) {
      members <- which(covariance_labels == cluster)
      if (length(members) == 0) {
        covariance_atoms[[cluster]] <- .pdpmvar_new_covariance(n_nodes, n_factors, cov_shape, cov_rate)
        next
      }
      residual                    <- do.call(cbind, residuals[members])
      covariance_atoms[[cluster]] <- .pdpmvar_update_covariance(residual,
        covariance_atoms[[cluster]], cov_shape, cov_rate, backend = backend)
    }

    # Count every iteration; posterior sums below include only retained iterations.
    cluster_counts[iteration, ] <- c(length(unique(coefficient_labels)), length(unique(covariance_labels)))
    if (iteration > n_burnin) {
      retained                           <- iteration - n_burnin
      lambda_trace[retained]             <- sqrt(lambda_squared)
      shared_variance_sum                <- shared_variance_sum + shared_variance
      coefficient_membership[retained, ] <- coefficient_labels
      covariance_membership[retained, ]  <- covariance_labels

      # Co-clustering comparisons avoid aligning arbitrary stick labels across iterations.
      coefficient_similarity                       <- coefficient_similarity + outer(coefficient_labels, coefficient_labels, "==")
      covariance_similarity                        <- covariance_similarity + outer(covariance_labels, covariance_labels, "==")
      parameter_traces[retained, "log_likelihood"] <- .pdpmvar_total_likelihood(
        residuals, covariance_labels, covariance_atoms, backend = backend)

      # Map cluster atoms back to subjects before accumulating means and sign probabilities.
      for (subject in seq_len(n_subjects)) {
        coefficient                                      <- coefficient_atoms[[coefficient_labels[subject]]]$coefficient
        covariance                                       <- covariance_atoms[[covariance_labels[subject]]]$covariance
        parameter_traces[retained, "mean_node_variance"] <- parameter_traces[retained, "mean_node_variance"] +
          mean(diag(covariance)) / n_subjects
        parameter_traces[retained, "coefficient_rms"] <- parameter_traces[retained, "coefficient_rms"] +
          mean(coefficient^2) / n_subjects
        coefficient_sum[, , subject] <- coefficient_sum[, , subject] + coefficient
        positive_sum[, , subject]    <- positive_sum[, , subject] + (coefficient > 0)
        covariance_sum[, , subject]  <- covariance_sum[, , subject] + covariance
        if (store_draws) {
          draws$coefficients[, , subject, retained] <- coefficient
          draws$covariance[, , subject, retained]   <- covariance
        }
      }
    }

    if (verbose && (iteration %% 100 == 0 || iteration == n_burnin + n_mcmc)) {
      message("Iteration ", iteration, ": ", cluster_counts[iteration, 1],
              " coefficient clusters, ", cluster_counts[iteration, 2], " covariance clusters")
    }
  }

  # Normalize retained sums and convert second-moment traces to coefficient RMS.
  parameter_traces[, "coefficient_rms"] <- sqrt(parameter_traces[, "coefficient_rms"])
  sign_probability                      <- positive_sum / n_mcmc
  result                                <- list(coefficient_mean = coefficient_sum / n_mcmc,
    covariance_mean = covariance_sum / n_mcmc, coefficient_sign_probability = sign_probability,
    coefficient_sign_tail = 2 * pmin(sign_probability, 1 - sign_probability),
    coefficient_labels = coefficient_membership, covariance_labels = covariance_membership,
    coefficient_coclustering = coefficient_similarity / n_mcmc,
    covariance_coclustering = covariance_similarity / n_mcmc, cluster_counts = cluster_counts,
    parameter_traces = parameter_traces,
    split_merge = split_merge_trace, coefficient_split_merge = coefficient_split_merge_trace,
    lambda = if (coefficient_prior == "laplace") lambda_trace else NULL,
    shared_variance_mean = shared_variance_sum / n_mcmc,
    final_state = list(coefficient_atoms = coefficient_atoms, covariance_atoms = covariance_atoms,
      coefficient_labels = coefficient_labels, covariance_labels = covariance_labels,
      shared_variance = shared_variance, lambda_squared = lambda_squared), draws = draws)

  return(result)
}
