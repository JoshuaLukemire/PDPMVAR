# Shared sampling primitives and covariance-mixture machinery.

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Gaussian draws, likelihoods, DP slices and covariance primitives
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Draw a Gaussian vector from its precision and precision-times-mean vector.
.pdpmvar_draw_normal <- function(precision, information, backend = "R") {
  if (backend == "RcppEigen") {
    return(.pdpmvar_eigen_normal(precision, as.numeric(information), rnorm(nrow(precision))))
  }
  upper <- chol(precision)
  mean  <- backsolve(upper, forwardsolve(t(upper), information))
  return(as.numeric(mean + backsolve(upper, rnorm(nrow(precision)))))
}

# Evaluate the Gaussian residual likelihood through a covariance Cholesky factor.
.pdpmvar_log_likelihood <- function(residual, covariance, backend = "R") {
  if (backend == "RcppEigen") {return(.pdpmvar_eigen_log_likelihood(residual, covariance))}
  upper    <- chol(covariance)
  whitened <- forwardsolve(t(upper), residual)
  return(-ncol(residual) * sum(log(diag(upper))) - sum(whitened^2) / 2)
}

# Normalize log probabilities and sample a candidate without underflow.
.pdpmvar_sample_label <- function(log_probability, candidates) {
  if (!any(is.finite(log_probability))) {
    stop("No candidate cluster has a finite likelihood.")
  }
  probability <- exp(log_probability - max(log_probability))
  return(candidates[sample.int(length(candidates), 1, prob = probability)])
}

# Sample DP weights and slices, extending the stick until every slice is covered.
.pdpmvar_slice_weights <- function(labels, concentration, backend = "R") {
  if (backend == "RcppEigen") {return(.pdpmvar_eigen_slice_weights(as.numeric(labels), concentration))}
  counts    <- tabulate(labels, nbins = max(labels))
  weights   <- numeric(length(counts))
  remaining <- 1
  for (cluster in seq_along(counts)) {
    beyond <- sum(counts) - sum(counts[seq_len(cluster)])
    stick  <- rbeta(1, 1 + counts[cluster], concentration + beyond)
    weights[cluster] <- remaining * stick
    remaining <- remaining * (1 - stick)
  }
  slices <- runif(length(labels)) * weights[labels]
  if (any(slices <= 0)) {
    stop("Slice weights underflowed; rescale the data or adjust the DP concentration.")
  }
  while (remaining >= min(slices)) {
    stick     <- rbeta(1, 1, concentration)
    weights   <- c(weights, remaining * stick)
    remaining <- remaining * (1 - stick)
  }
  return(list(weights = weights, slices = slices))
}

# Draw a coefficient atom conditional on the shared base-measure variances.
.pdpmvar_new_coefficient <- function(variance) {
  coefficient <- matrix(rnorm(length(variance)), nrow(variance), ncol(variance)) * sqrt(variance)
  return(list(coefficient = coefficient))
}

# Convert expanded loadings and factor precisions to the marginal covariance.
.pdpmvar_refresh_covariance <- function(atom) {
  signs <- sign(diag(atom$working_loadings))
  signs[signs == 0] <- 1
  atom$loadings <- sweep(atom$working_loadings, 2, signs / sqrt(atom$factor_precision), "*")
  atom$covariance <- tcrossprod(atom$loadings) + diag(atom$variance, nrow(atom$loadings))
  return(atom)
}

# Draw a complete triangular factor covariance atom from its base measure.
.pdpmvar_new_covariance <- function(n_nodes, n_factors, cov_shape, cov_rate) {
  working_loadings <- matrix(rnorm(n_nodes * n_factors), n_nodes, n_factors)
  working_loadings[upper.tri(working_loadings)] <- 0
  factor_precision <- rgamma(n_factors, shape = 0.5, rate = 0.5)
  variance <- 1 / rgamma(n_nodes, shape = cov_shape, rate = cov_rate)
  return(.pdpmvar_refresh_covariance(list(working_loadings = working_loadings,
    factor_precision = factor_precision, variance = variance)))
}

# Draw working factors conditional on residuals and the selected covariance atom.
.pdpmvar_draw_factors <- function(residual, atom, backend = "R") {
  if (backend == "RcppEigen") {
    n_factors <- ncol(atom$working_loadings)
    noise <- matrix(rnorm(n_factors * ncol(residual)), n_factors)
    return(.pdpmvar_eigen_factors(residual, atom, noise))
  }
  n_factors <- ncol(atom$working_loadings)
  weighted_loadings <- atom$working_loadings / atom$variance
  precision <- diag(atom$factor_precision, n_factors) +
    crossprod(atom$working_loadings, weighted_loadings)
  information <- crossprod(weighted_loadings, residual)
  upper <- chol(precision)
  factors <- backsolve(upper, forwardsolve(t(upper), information)) +
    backsolve(upper, matrix(rnorm(n_factors * ncol(residual)), n_factors))
  return(factors)
}

# Draw GIG(p, chi, psi) using a rejection envelope on the log-variance scale.
.pdpmvar_draw_gig <- function(p, chi, psi, backend = "R") {
  if (backend == "RcppEigen") {return(.pdpmvar_eigen_gig(p, chi, psi))}
  # The log-scale density is p*y - (chi*exp(-y) + psi*exp(y))/2.
  # Its concavity gives exponential tangent bounds on either side of the mode.
  if (chi == 0 && p > 0) {
    return(rgamma(1, shape = p, rate = psi / 2))
  }
  if (!is.finite(chi) || chi <= 0 || !is.finite(psi) || psi <= 0) {
    stop("GIG chi and psi must be positive and finite.")
  }
  curvature <- sqrt(p^2 + chi * psi)
  if (p >= 0) {
    mode <- log((p + curvature) / psi)
  } else {
    mode <- log(chi / (curvature - p))
  }
  width <- min(1, 1 / sqrt(curvature))
  # At p = 0 the tails are exactly symmetric; avoid rounding-dependent region ordering.
  left_term  <- if (p == 0) curvature else chi * exp(-mode)
  right_term <- if (p == 0) curvature else psi * exp(mode)
  left  <- -1
  right <- 1
  left_slope <- width * (p + (left_term * exp(-width * left) - right_term * exp(width * left)) / 2)
  while (left_slope < 0.5) {
    left <- left - 1
    left_slope <- width * (p + (left_term * exp(-width * left) - right_term * exp(width * left)) / 2)
  }
  if (p == 0) right <- -left
  right_slope <- if (p == 0) -left_slope else
    width * (p + (left_term * exp(-width * right) - right_term * exp(width * right)) / 2)
  while (right_slope > -0.5) {
    right <- right + 1
    right_slope <- width * (p + (left_term * exp(-width * right) - right_term * exp(width * right)) / 2)
  }
  left_height <- p * width * left -
    (left_term * expm1(-width * left) + right_term * expm1(width * left)) / 2
  right_height <- if (p == 0) left_height else p * width * right -
    (left_term * expm1(-width * right) + right_term * expm1(width * right)) / 2
  log_mass <- c(left_height - log(left_slope), log(right - left),
                right_height - log(-right_slope))
  repeat {
    region <- sample.int(3, 1, prob = exp(log_mass - max(log_mass)))
    if (region == 1) {
      candidate <- left - rexp(1, rate = left_slope)
      envelope  <- left_height + left_slope * (candidate - left)
    } else if (region == 2) {
      candidate <- runif(1, left, right)
      envelope  <- 0
    } else {
      candidate <- right + rexp(1, rate = -right_slope)
      envelope  <- right_height + right_slope * (candidate - right)
    }
    height <- p * width * candidate -
      (left_term * expm1(-width * candidate) + right_term * expm1(width * candidate)) / 2
    if (log(runif(1)) <= height - envelope) {
      return(exp(mode + width * candidate))
    }
  }
}

# Calibrate a Gamma precision prior using shape-rate moment matching.
.pdpmvar_calibrate_covariance <- function(X) {
  precisions <- unlist(lapply(X, function(x) 1 / apply(x, 1, stats::var)), use.names = FALSE)
  precision_mean <- mean(precisions)
  precision_variance <- stats::var(precisions)
  if (any(!is.finite(precisions)) || !is.finite(precision_variance) || precision_variance <= 0) {
    stop("Covariance calibration needs positive, varying node variances; supply cov_shape and cov_rate explicitly.")
  }
  return(c(shape = precision_mean^2 / precision_variance,
           rate = precision_mean / precision_variance))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Auxiliary stick weights
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Draw auxiliary split/merge prefix weights without discarding counts beyond the prefix.
.pdpmvar_prefix_weights <- function(labels, concentration, slots, backend = "R") {
  if (backend == "RcppEigen") {
    return(.pdpmvar_eigen_prefix_weights(as.numeric(labels), concentration, slots))
  }
  counts <- tabulate(labels, nbins = max(slots, labels))
  log_weights <- numeric(slots)
  log_remaining <- 0
  for (cluster in seq_len(slots)) {
    beyond <- sum(counts) - sum(counts[seq_len(cluster)])
    stick <- rbeta(1, 1 + counts[cluster], concentration + beyond)
    log_weights[cluster] <- log_remaining + log(stick)
    log_remaining <- log_remaining + log1p(-stick)
  }
  return(log_weights)
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Covariance assignments and total likelihood
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Assign residual covariance atoms without multiplying eligible likelihoods by DP weights again.
.pdpmvar_assign_covariances <- function(residuals, atoms, slice, backend = "R") {
  if (backend == "RcppEigen" && length(slice$weights) <= 200) {
    return(.pdpmvar_eigen_assign_covariances(residuals, atoms, slice$weights, slice$slices, runif(length(residuals))))
  }
  labels <- numeric(length(residuals))
  scores <- matrix(-Inf, length(residuals), length(atoms))
  probabilities <- matrix(0, length(residuals), length(atoms))
  for (subject in seq_along(residuals)) {
    candidates <- which(slice$weights > slice$slices[subject])
    log_probability <- vapply(candidates, function(cluster)
      .pdpmvar_log_likelihood(residuals[[subject]], atoms[[cluster]]$covariance, backend = backend), numeric(1))
    scores[subject, candidates] <- log_probability
    probability <- exp(log_probability - max(log_probability))
    probabilities[subject, candidates] <- probability / sum(probability)
    labels[subject] <- .pdpmvar_sample_label(log_probability, candidates)
  }
  return(list(labels = labels, log_likelihood = scores, probabilities = probabilities))
}

# Evaluate the retained total likelihood without recalculating coefficient predictions.
.pdpmvar_total_likelihood <- function(residuals, labels, atoms, backend = "R") {
  if (backend == "RcppEigen") {
    return(.pdpmvar_eigen_total_likelihood(residuals, as.numeric(labels), atoms))
  }
  total <- 0
  for (subject in seq_along(residuals)) {
    total <- total + .pdpmvar_log_likelihood(residuals[[subject]], atoms[[labels[subject]]]$covariance)
  }
  return(total)
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Factor covariance updates
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Refresh one occupied covariance atom using the triangular expanded-factor hierarchy.
.pdpmvar_update_covariance <- function(residual, atom, cov_shape, cov_rate, backend = "R") {
  if (backend == "RcppEigen") {
    return(.pdpmvar_eigen_covariance_update(residual, atom, cov_shape, cov_rate))
  }
  n_nodes   <- nrow(residual)
  n_factors <- ncol(atom$working_loadings)
  factors <- .pdpmvar_draw_factors(residual, atom)
  for (response in seq_len(n_nodes)) {
    free <- seq_len(min(response, n_factors))
    row_factors <- factors[free, , drop = FALSE]
    precision <- diag(1, length(free)) + tcrossprod(row_factors) / atom$variance[response]
    information <- as.numeric(row_factors %*% residual[response, ]) / atom$variance[response]
    atom$working_loadings[response, free] <- .pdpmvar_draw_normal(precision, information)
  }
  atom$factor_precision <- rgamma(n_factors, shape = (1 + ncol(residual)) / 2,
                                  rate = (1 + rowSums(factors^2)) / 2)
  errors <- residual - atom$working_loadings %*% factors
  atom$variance <- 1 / rgamma(n_nodes, shape = cov_shape + ncol(residual) / 2,
                              rate = cov_rate + rowSums(errors^2) / 2)
  return(.pdpmvar_refresh_covariance(atom))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Covariance proposal utilities and controller
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Score cached residual scatter matrices under one marginal covariance atom.
.pdpmvar_covariance_scores <- function(scatters, n_times, atom, backend = "R") {
  if (backend == "RcppEigen") {
    return(.pdpmvar_eigen_scatter_scores(scatters, n_times, atom$covariance))
  }
  upper <- chol(atom$covariance)
  precision <- chol2inv(upper)
  log_determinant <- 2 * sum(log(diag(upper)))
  quadratic <- vapply(scatters, function(scatter) sum(precision * scatter), numeric(1))
  return(-(n_times * log_determinant + quadratic) / 2)
}

# Evaluate the base density on free working loadings, factor precisions and variances.
.pdpmvar_covariance_log_base <- function(atom, cov_shape, cov_rate) {
  free <- lower.tri(atom$working_loadings, diag = TRUE)
  return(sum(stats::dnorm(atom$working_loadings[free], log = TRUE)) +
    sum(stats::dgamma(atom$factor_precision, shape = 0.5, rate = 0.5, log = TRUE)) +
    sum(stats::dgamma(1 / atom$variance, shape = cov_shape, rate = cov_rate, log = TRUE) -
          2 * log(atom$variance)))
}

# Propose an anchor-centered atom, mixing uniformly over independent loading-column signs.
.pdpmvar_propose_covariance <- function(center) {
  n_nodes   <- nrow(center$working_loadings)
  n_factors <- ncol(center$working_loadings)
  factor_precision <- stats::rlnorm(n_factors, meanlog = 0, sdlog = 0.5)
  scale <- sqrt(factor_precision)
  center_loadings <- center$working_loadings
  # Orient by the largest entry to couple draws despite arbitrary eigen/QR signs.
  for (factor in seq_len(n_factors)) {
    pivot <- which.max(abs(center_loadings[, factor]))
    if (center_loadings[pivot, factor] < 0) center_loadings[, factor] <- -center_loadings[, factor]
  }
  working_loadings <- sweep(center_loadings +
    matrix(rnorm(n_nodes * n_factors, sd = 0.15), n_nodes), 2, scale, "*")
  working_loadings[upper.tri(working_loadings)] <- 0
  variance <- stats::rlnorm(n_nodes, meanlog = log(center$variance), sdlog = 0.25)
  signs <- ifelse(runif(n_factors) < 0.5, -1, 1)
  working_loadings <- sweep(working_loadings, 2, signs, "*")
  return(.pdpmvar_refresh_covariance(list(working_loadings = working_loadings,
    factor_precision = factor_precision, variance = variance)))
}

# Evaluate the sign mixture by columns, retaining precision-dependent loading scales.
.pdpmvar_covariance_log_proposal <- function(atom, center) {
  n_factors <- ncol(atom$working_loadings)
  loading_density <- numeric(n_factors)
  for (factor in seq_len(n_factors)) {
    free <- factor:nrow(atom$working_loadings)
    scale <- sqrt(atom$factor_precision[factor])
    values <- atom$working_loadings[free, factor]
    means  <- center$working_loadings[free, factor] * scale
    positive <- sum(stats::dnorm(values, means, 0.15 * scale, log = TRUE))
    negative <- sum(stats::dnorm(values, -means, 0.15 * scale, log = TRUE))
    maximum <- max(positive, negative)
    loading_density[factor] <- if (maximum == -Inf) -Inf else
      maximum + log(exp(positive - maximum) + exp(negative - maximum)) - log(2)
  }
  # Independent signs make the full 2^B-component mixture factorize by column.
  return(sum(loading_density) +
    sum(stats::dlnorm(atom$factor_precision, meanlog = 0, sdlog = 0.5, log = TRUE)) +
    sum(stats::dlnorm(atom$variance, meanlog = log(center$variance), sdlog = 0.25, log = TRUE)))
}

# Evaluate restricted allocation probabilities, holding the two anchor labels fixed.
.pdpmvar_allocation_log_probability <- function(labels, subjects, scores, h, g) {
  if (!length(subjects)) {return(0)}
  selected <- ifelse(labels[subjects] == h, 1, 2)
  maximum <- apply(scores, 1, max)
  normalizer <- maximum + log(rowSums(exp(scores - maximum)))
  return(sum(scores[cbind(seq_along(subjects), selected)] - normalizer))
}

# Evaluate forward and reverse densities of a fixed-index covariance split/merge.
.pdpmvar_covariance_move_ratio <- function(old_labels, new_labels, old_atoms, new_atoms,
                                          log_weights, scatters, n_times, anchors,
                                          center, slots, cov_shape, cov_rate, backend = "R") {
  h <- old_labels[anchors[1]]
  split <- old_labels[anchors[1]] == old_labels[anchors[2]]
  g <- if (split) new_labels[anchors[2]] else old_labels[anchors[2]]
  subjects <- which(old_labels %in% c(h, g))
  other <- setdiff(subjects, anchors)
  h_scores <- log_weights[h] + .pdpmvar_covariance_scores(scatters[subjects], n_times[subjects], old_atoms[[h]], backend = backend)
  old_g_scores <- log_weights[g] + .pdpmvar_covariance_scores(scatters[subjects], n_times[subjects], old_atoms[[g]], backend = backend)
  new_g_scores <- log_weights[g] + .pdpmvar_covariance_scores(scatters[subjects], n_times[subjects], new_atoms[[g]], backend = backend)
  old_score <- ifelse(old_labels[subjects] == h, h_scores, old_g_scores)
  new_score <- ifelse(new_labels[subjects] == h, h_scores, new_g_scores)
  old_base <- .pdpmvar_covariance_log_base(old_atoms[[g]], cov_shape, cov_rate)
  new_base <- .pdpmvar_covariance_log_base(new_atoms[[g]], cov_shape, cov_rate)
  target_change <- sum(new_score - old_score) + new_base - old_base
  if (split) {
    empty <- setdiff(seq_len(slots), unique(old_labels))
    scores <- cbind(h_scores, new_g_scores)[match(other, subjects), , drop = FALSE]
    allocation <- .pdpmvar_allocation_log_probability(new_labels, other, scores, h, g)
    forward <- -log(length(empty)) + .pdpmvar_covariance_log_proposal(new_atoms[[g]], center) + allocation
    reverse <- old_base
  } else {
    empty <- setdiff(seq_len(slots), unique(new_labels))
    scores <- cbind(h_scores, old_g_scores)[match(other, subjects), , drop = FALSE]
    allocation <- .pdpmvar_allocation_log_probability(old_labels, other, scores, h, g)
    forward <- new_base
    reverse <- -log(length(empty)) + .pdpmvar_covariance_log_proposal(old_atoms[[g]], center) + allocation
  }
  return(list(log_ratio = target_change + reverse - forward, target_change = target_change,
              log_forward = forward, log_reverse = reverse))
}

# Attempt a covariance split/merge with auxiliary weights and inactive prefix atoms.
.pdpmvar_covariance_split_merge_legacy <- function(residuals, labels, atoms, concentration,
                                           n_factors, cov_shape, cov_rate, slots, backend = "R") {
  n_subjects <- length(labels)
  record <- list(type = "skip", accepted = FALSE, log_ratio = NA_real_)
  if (n_subjects < 2) {return(list(labels = labels, atoms = atoms, record = record))}
  anchors <- sample.int(n_subjects, 2)
  h <- labels[anchors[1]]
  g <- labels[anchors[2]]
  if (h > slots || g > slots) {return(list(labels = labels, atoms = atoms, record = record))}
  split <- h == g
  empty <- setdiff(seq_len(slots), unique(labels))
  if (split && !length(empty)) {return(list(labels = labels, atoms = atoms, record = record))}

  # Only a fixed prefix receives extra moves; the ordinary DP tail is unchanged.
  log_weights <- .pdpmvar_prefix_weights(labels, concentration, slots, backend = backend)
  while (length(atoms) < slots) {
    atoms[[length(atoms) + 1]] <- .pdpmvar_new_covariance(nrow(residuals[[1]]),
      n_factors, cov_shape, cov_rate)
  }
  scatters <- lapply(residuals, tcrossprod)
  n_times  <- vapply(residuals, ncol, numeric(1))
  anchor_residual <- residuals[[anchors[2]]]
  center <- .pdpmvar_initialize_covariance(tcrossprod(anchor_residual) / ncol(anchor_residual), n_factors)
  proposed_labels <- labels
  proposed_atoms  <- atoms
  if (split) {
    g <- empty[sample.int(length(empty), 1)]
    proposed_atoms[[g]] <- .pdpmvar_propose_covariance(center)
    proposed_labels[anchors[2]] <- g
    other <- setdiff(which(labels == h), anchors)
    if (length(other)) {
      scores <- cbind(log_weights[h] + .pdpmvar_covariance_scores(scatters[other], n_times[other], atoms[[h]], backend = backend),
        log_weights[g] + .pdpmvar_covariance_scores(scatters[other], n_times[other], proposed_atoms[[g]], backend = backend))
      for (index in seq_along(other)) {
        proposed_labels[other[index]] <- .pdpmvar_sample_label(scores[index, ], c(h, g))
      }
    }
  } else {
    proposed_labels[labels == g] <- h
    proposed_atoms[[g]] <- .pdpmvar_new_covariance(nrow(residuals[[1]]), n_factors, cov_shape, cov_rate)
  }
  ratio <- .pdpmvar_covariance_move_ratio(labels, proposed_labels, atoms, proposed_atoms,
    log_weights, scatters, n_times, anchors, center, slots, cov_shape, cov_rate, backend = backend)
  if (is.na(ratio$log_ratio)) {stop("Non-finite split/merge acceptance calculation.")}
  accepted <- log(runif(1)) < min(0, ratio$log_ratio)
  if (accepted) {
    labels <- proposed_labels
    atoms  <- proposed_atoms
  }
  record <- list(type = if (split) "split" else "merge", accepted = accepted, log_ratio = ratio$log_ratio)
  return(list(labels = labels, atoms = atoms, record = record))
}

# Dispatch the complete controller while retaining the original R implementation.
.pdpmvar_covariance_split_merge <- function(residuals, labels, atoms, concentration, n_factors, cov_shape, cov_rate, slots, backend = "R") {
  if (backend == "RcppEigen") {
    return(.pdpmvar_eigen_covariance_move(residuals, as.numeric(labels), atoms, concentration, n_factors, cov_shape, cov_rate, slots))
  }
  return(.pdpmvar_covariance_split_merge_reference(residuals, labels, atoms, concentration, n_factors, cov_shape, cov_rate, slots, backend = backend))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Reversible covariance updates
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Update a powered Gaussian covariance likelihood via a real-df Wishart augmentation.
# For effective count nu, augment S_Y with C_FY ~ MN(A S_Y, C, S_Y) and
# S_F = C_FY S_Y^-1 C_FY' + Wishart(C, nu-D). This avoids sampling nu factors.
# L-v-L is a reversible conditional sweep, and Q commutes with those blocks.
# Marginalizing the augmented statistics gives a reversible atom kernel. Exact Q|Lstar/sqrt(Q)
# scale refreshes sandwich that kernel, preserving reversibility and reducing expansion dependence.
# factor_shape is 1/2 for the fitted model; other values are used only inside auxiliary bridges.
.pdpmvar_covariance_reversible <- function(atom, scatter, n_times, cov_shape, cov_rate, backend = "R", factor_shape = 0.5) {
  if (backend == "RcppEigen") return(.pdpmvar_eigen_covariance_reversible(atom, scatter, n_times, cov_shape, cov_rate, factor_shape))
  if (length(factor_shape) != 1 || !is.finite(factor_shape) || factor_shape <= 0) stop("Factor shape must be positive and finite.")
  D <- nrow(atom$working_loadings)
  B <- ncol(atom$working_loadings)
  if (n_times <= D + B + 1) stop("Reversible covariance augmentation needs more than nodes + factors + 1 effective observations.")
  normalized <- sweep(atom$working_loadings, 2, sqrt(atom$factor_precision), "/")
  sizes <- D - seq_len(B) + 1
  initial_precision <- rgamma(B, factor_shape + sizes / 2, rate = (1 + colSums(normalized^2)) / 2)
  working <- sweep(normalized, 2, sqrt(initial_precision), "*")
  variance <- atom$variance
  precision <- crossprod(working, working / variance) + diag(initial_precision, B)
  factor_covariance <- chol2inv(chol(precision))
  mean_map <- factor_covariance %*% t(working / variance)
  upper_factor <- chol(factor_covariance)
  factor_data <- mean_map %*% scatter + t(upper_factor) %*% matrix(rnorm(B * D), B) %*% chol(scatter)
  bartlett <- matrix(0, B, B)
  for (column in seq_len(B)) {
    bartlett[column, column] <- sqrt(rgamma(1, (n_times - D - column + 1) / 2, rate = 0.5))
    if (column < B) bartlett[(column + 1):B, column] <- rnorm(B - column)
  }
  noise <- t(upper_factor) %*% bartlett
  factor_cross <- factor_data %*% solve(scatter, t(factor_data)) + tcrossprod(noise)
  # L-v-L is palindromic; Q is conditionally independent of those blocks given latent statistics.
  for (sweep in 1:2) {
    for (row in seq_len(D)) {
      free <- seq_len(min(row, B))
      upper <- chol(diag(length(free)) + factor_cross[free, free, drop = FALSE] / variance[row])
      information <- factor_data[free, row] / variance[row]
      working[row, free] <- backsolve(upper, forwardsolve(t(upper), information)) + backsolve(upper, rnorm(length(free)))
    }
    if (sweep == 1) {
      residual <- diag(scatter) - 2 * rowSums(working * t(factor_data)) + rowSums((working %*% factor_cross) * working)
      variance <- 1 / rgamma(D, cov_shape + n_times / 2, rate = cov_rate + pmax(residual, 0) / 2)
    }
  }
  factor_precision <- rgamma(B, factor_shape + n_times / 2, rate = (1 + diag(factor_cross)) / 2)
  normalized <- sweep(working, 2, sqrt(factor_precision), "/")
  factor_precision <- rgamma(B, factor_shape + sizes / 2, rate = (1 + colSums(normalized^2)) / 2)
  working <- sweep(normalized, 2, sqrt(factor_precision), "*")
  return(.pdpmvar_refresh_covariance(list(working_loadings = working, factor_precision = factor_precision, variance = variance)))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Auxiliary covariance proposals
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Draw a conjugate bridge prior without changing the fitted model's covariance base.
# This helper is used only at an intermediate/auxiliary endpoint, never for occupied model atoms.
.pdpmvar_bridge_prior <- function(D, B, shape, rate, factor_shape) {
  working <- matrix(rnorm(D * B), D, B)
  working[upper.tri(working)] <- 0
  precision <- rgamma(B, factor_shape, rate = 0.5)
  variance <- 1 / rgamma(D, shape, rate = rate)
  return(.pdpmvar_refresh_covariance(list(working_loadings = working, factor_precision = precision, variance = variance)))
}

# q keeps standard normal working loadings, avoids near-zero Q, and centers E(v) on the data proxy.
# The inverse-Gamma(16, 15*mean(center$v)) law is a proposal only; all constants enter bridge work.
.pdpmvar_bridge_propose <- function(center) {
  return(.pdpmvar_bridge_prior(nrow(center$working_loadings), ncol(center$working_loadings),
    16, 15 * mean(center$variance), 2))
}

# Evaluate the normalized auxiliary density, using the same working-coordinate measure as H.
.pdpmvar_bridge_density <- function(atom, center) {
  value <- .pdpmvar_covariance_log_base(atom, 16, 15 * mean(center$variance)) +
    sum(dgamma(atom$factor_precision, 2, rate = 0.5, log = TRUE) -
      dgamma(atom$factor_precision, 0.5, rate = 0.5, log = TRUE))
  return(value)
}

# H^beta q^(1-beta) stays in the original conjugate family at every fixed temperature.
# Reversible fractional-data Gibbs moves therefore need no outer MH correction.
.pdpmvar_bridge_auxiliary <- function(atom, scatter, n_times, beta, center, B, shape, rate, backend = "R") {
  if (backend == "RcppEigen") return(.pdpmvar_eigen_bridge_auxiliary(atom, scatter, n_times, beta, center, B, shape, rate))
  if (length(beta) != 1 || !is.finite(beta) || beta < 0 || beta > 1) stop("Bridge temperature must lie between zero and one.")
  intermediate_shape <- beta * shape + (1 - beta) * 16
  intermediate_rate <- beta * rate + (1 - beta) * 15 * mean(center$variance)
  factor_shape <- beta * 0.5 + (1 - beta) * 2
  weighted_scatter <- beta * scatter
  weighted_count <- beta * n_times
  positive <- !is.null(tryCatch(chol(weighted_scatter), error = function(e) NULL))
  if (weighted_count > nrow(scatter) + B + 1 && positive) {
    return(.pdpmvar_covariance_reversible(atom, weighted_scatter, weighted_count,
      intermediate_shape, intermediate_rate, factor_shape = factor_shape))
  }
  proposed <- .pdpmvar_bridge_prior(nrow(scatter), B, intermediate_shape, intermediate_rate, factor_shape)
  ratio <- .pdpmvar_covariance_weighted_score(proposed, weighted_scatter, weighted_count) -
    .pdpmvar_covariance_weighted_score(atom, weighted_scatter, weighted_count)
  if (log(runif(1)) < min(0, ratio)) return(proposed)
  return(atom)
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Annealed covariance bridge
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Integrate the original stick-breaking weights while retaining label gaps and tail indices.
.pdpmvar_covariance_label_prior <- function(labels, concentration) {
  if (!is.numeric(labels) || !length(labels) || any(!is.finite(labels)) || any(labels < 1 | labels != floor(labels)) ||
      length(concentration) != 1 || !is.finite(concentration) || concentration <= 0) stop("Stick labels and concentration must be positive and finite.")
  counts <- table(labels)
  positions <- as.numeric(names(counts))
  remaining <- length(labels)
  previous <- 0
  terms <- numeric(length(counts))
  for (index in seq_along(counts)) {
    gap <- positions[index] - previous - 1
    ratio <- remaining / concentration
    log_gap_mass <- if (is.finite(ratio)) -log1p(ratio) else log(concentration) - log(concentration + remaining)
    terms[index] <- if (gap == 0) 0 else gap * log_gap_mass
    remaining <- remaining - counts[index]
    terms[index] <- terms[index] + lbeta(1 + counts[index], concentration + remaining) + log(concentration)
    previous <- positions[index]
  }
  return(sum(terms))
}

# Prefer early empty stick positions using prior expected masses; this affects only proposals.
.pdpmvar_covariance_empty_probabilities <- function(empty, concentration) {
  decay <- if (concentration < 1) log(concentration) - log1p(concentration) else -log1p(1 / concentration)
  log_mass <- (empty - 1) * decay
  maximum <- max(log_mass)
  return(log_mass - maximum - log(sum(exp(log_mass - maximum))))
}

# Use inverse-Wishart predictive scores only as an allocation proposal, never as a model prior.
.pdpmvar_covariance_restricted <- function(scatters, n_times, subjects, anchors, h, g, forced = NULL) {
  D <- nrow(scatters[[1]])
  pooled <- Reduce(`+`, scatters[subjects]) / sum(n_times[subjects])
  base <- pooled * (D + 1)
  if (is.null(tryCatch(chol(base), error = function(e) NULL))) {
    base <- base + diag(max(mean(diag(pooled)), 1) * (D + 1) * 1e-6, D)
  }
  degrees <- 2 * D + 2
  scales <- list(base + scatters[[anchors[1]]], base + scatters[[anchors[2]]])
  counts <- n_times[anchors]
  labels <- rep(h, length(scatters))
  labels[anchors[2]] <- g
  probability <- 0
  for (subject in setdiff(subjects, anchors)) {
    scores <- numeric(2)
    for (group in 1:2) {
      nu <- degrees + counts[group]
      scores[group] <- sum(lgamma((nu + n_times[subject] + 1 - seq_len(D)) / 2) -
        lgamma((nu + 1 - seq_len(D)) / 2)) + nu * sum(log(diag(chol(scales[[group]])))) -
        (nu + n_times[subject]) * sum(log(diag(chol(scales[[group]] + scatters[[subject]]))))
    }
    maximum <- max(scores)
    normalizer <- maximum + log(sum(exp(scores - maximum)))
    selected <- if (is.null(forced)) .pdpmvar_sample_label(scores, 1:2) else if (forced[subject] == h) 1 else 2
    labels[subject] <- c(h, g)[selected]
    probability <- probability + scores[selected] - normalizer
    scales[[selected]] <- scales[[selected]] + scatters[[subject]]
    counts[selected] <- counts[selected] + n_times[subject]
  }
  return(list(labels = labels, log_probability = probability))
}

# Move a covariance partition with collapsed stick weights and an exact annealed path correction.
.pdpmvar_covariance_split_merge_reference <- function(residuals, labels, atoms, concentration,
                                                     n_factors, cov_shape, cov_rate, slots, backend = "R") {
  record <- list(type = "skip", accepted = FALSE, log_ratio = NA_real_)
  if (length(labels) < 2) return(list(labels = labels, atoms = atoms, record = record))
  anchors <- sample.int(length(labels), 2)
  h <- labels[anchors[1]]
  g <- labels[anchors[2]]
  split <- h == g
  empty <- setdiff(seq_len(slots), unique(labels))
  if (h > slots || g > slots || (split && !length(empty))) return(list(labels = labels, atoms = atoms, record = record))
  while (length(atoms) < slots) atoms[[length(atoms) + 1]] <- .pdpmvar_new_covariance(nrow(residuals[[1]]), n_factors, cov_shape, cov_rate)
  scatters <- lapply(residuals, tcrossprod)
  n_times <- vapply(residuals, ncol, numeric(1))
  proposed_labels <- labels
  if (split) {
    log_empty_probability <- .pdpmvar_covariance_empty_probabilities(empty, concentration)
    g <- .pdpmvar_sample_label(log_empty_probability, empty)
    subjects <- which(labels == h)
    allocation <- .pdpmvar_covariance_restricted(scatters, n_times, subjects, anchors, h, g)
    proposed_labels[subjects] <- allocation$labels[subjects]
  } else proposed_labels[labels == g] <- h
  subjects <- which(labels %in% c(h, g))
  reverse_allocation <- if (split) allocation$log_probability else
    .pdpmvar_covariance_restricted(scatters, n_times, subjects, anchors, h, g, labels)$log_probability
  proposal_empty <- setdiff(seq_len(slots), unique(if (split) labels else proposed_labels))
  log_empty_probability <- .pdpmvar_covariance_empty_probabilities(proposal_empty, concentration)[match(g, proposal_empty)]
  allocation_ratio <- if (split) -log_empty_probability - reverse_allocation else log_empty_probability + reverse_allocation
  bridge <- .pdpmvar_covariance_bridge(labels, proposed_labels, atoms, scatters, n_times,
    h, g, n_factors, cov_shape, cov_rate, backend)
  prior_change <- .pdpmvar_covariance_label_prior(proposed_labels, concentration) -
    .pdpmvar_covariance_label_prior(labels, concentration)
  log_ratio <- bridge$log_work + prior_change + allocation_ratio
  if (is.na(log_ratio)) stop("Non-finite annealed covariance acceptance calculation.")
  accepted <- log(runif(1)) < min(0, log_ratio)
  if (accepted) {
    labels <- proposed_labels
    atoms <- bridge$atoms
    if (!split) atoms[[g]] <- .pdpmvar_new_covariance(nrow(residuals[[1]]), n_factors, cov_shape, cov_rate)
  }
  record <- list(type = if (split) "split" else "merge", accepted = accepted, log_ratio = log_ratio)
  return(list(labels = labels, atoms = atoms, record = record))
}

# Score an atom under a fractional Gaussian likelihood; the omitted constant cancels in MH.
.pdpmvar_covariance_weighted_score <- function(atom, scatter, n_times, backend = "R") {
  if (backend == "RcppEigen") return(.pdpmvar_eigen_covariance_weighted_score(atom, scatter, n_times))
  return(.pdpmvar_covariance_scores(list(scatter), n_times, atom, backend)[1])
}

# Use a reversible fractional-data Gibbs kernel, or prior independence MH near empty/singular endpoints.
.pdpmvar_covariance_bridge_step <- function(atom, scatter, n_times, B, shape, rate, backend = "R") {
  old_score <- .pdpmvar_covariance_weighted_score(atom, scatter, n_times, backend)
  positive <- !is.null(tryCatch(chol(scatter), error = function(e) NULL))
  if (n_times <= nrow(scatter) + B + 1 || !positive) {
    proposed <- .pdpmvar_new_covariance(nrow(scatter), B, shape, rate)
    ratio <- .pdpmvar_covariance_weighted_score(proposed, scatter, n_times, backend) - old_score
  } else {
    return(.pdpmvar_covariance_reversible(atom, scatter, n_times, shape, rate, backend))
  }
  if (log(runif(1)) < min(0, ratio)) return(proposed)
  return(atom)
}

# Retain the old proposal-to-posterior kernel for saved diagnostic comparisons only.
.pdpmvar_covariance_auxiliary_step <- function(atom, scatter, n_times, beta, center, B, shape, rate, backend) {
  density <- .pdpmvar_covariance_log_proposal(atom, center)
  base <- .pdpmvar_covariance_log_base(atom, shape, rate)
  if (runif(1) < 0.5) {
    proposed <- .pdpmvar_propose_covariance(center)
    proposed_density <- .pdpmvar_covariance_log_proposal(proposed, center)
    proposed_base <- .pdpmvar_covariance_log_base(proposed, shape, rate)
    ratio <- beta * (proposed_base - base - proposed_density + density +
      .pdpmvar_covariance_weighted_score(proposed, scatter, n_times, backend) -
      .pdpmvar_covariance_weighted_score(atom, scatter, n_times, backend))
  } else {
    proposed <- .pdpmvar_covariance_bridge_step(atom, scatter * beta, n_times * beta, B, shape, rate, backend)
    proposed_density <- .pdpmvar_covariance_log_proposal(proposed, center)
    proposed_base <- .pdpmvar_covariance_log_base(proposed, shape, rate)
    ratio <- (1 - beta) * (proposed_density - density - proposed_base + base)
  }
  if (log(runif(1)) < min(0, ratio)) return(proposed)
  return(atom)
}

# Anneal between fixed partitions using reversible intermediate kernels (Karagiannis and Andrieu,
# JCGS 22:623-648, 2013, doi:10.1080/10618600.2013.805651).
# At the merged end the unused atom is integrated out and an auxiliary q_g replaces it;
# at the split end its density is the original base times its group likelihood.
# q_g has the original loading prior, Gamma(2, 1/2) factor precisions, and an informed
# inverse-Gamma residual variance. Intermediate priors remain conjugate; only the endpoint
# has the model priors. Work includes log(base/q_g); an accepted merge refreshes from base.
.pdpmvar_covariance_bridge <- function(old_labels, new_labels, atoms, scatters, n_times,
                                      h, g, B, shape, rate, backend = "R", store_path = FALSE) {
  if (backend == "RcppEigen") return(.pdpmvar_eigen_covariance_bridge(old_labels, new_labels, atoms,
    scatters, n_times, h, g, B, shape, rate, store_path))
  lower <- plogis(seq(-12, 0, length.out = 51))
  schedule <- c(0, lower, 1 - rev(lower[-length(lower)]), 1)
  D <- nrow(scatters[[1]])
  old_scatter <- new_scatter <- list(matrix(0, D, D), matrix(0, D, D))
  old_count <- new_count <- numeric(2)
  groups <- c(h, g)
  for (index in 1:2) {
    for (subject in seq_along(scatters)) {
      if (old_labels[subject] == groups[index]) {
        old_scatter[[index]] <- old_scatter[[index]] + scatters[[subject]]
        old_count[index] <- old_count[index] + n_times[subject]
      }
      if (new_labels[subject] == groups[index]) {
        new_scatter[[index]] <- new_scatter[[index]] + scatters[[subject]]
        new_count[index] <- new_count[index] + n_times[subject]
      }
    }
  }
  split <- old_count[2] == 0
  active_scatter <- if (split) new_scatter[[2]] else old_scatter[[2]]
  active_count <- if (split) new_count[2] else old_count[2]
  center <- .pdpmvar_initialize_covariance(active_scatter / active_count, B)
  if (split) atoms[[g]] <- .pdpmvar_bridge_propose(center)
  work <- 0
  path <- if (store_path) vector("list", length(schedule) - 1) else NULL
  for (stage in 2:length(schedule)) {
    if (store_path) path[[stage - 1]] <- atoms[groups]
    difference <- 0
    for (index in 1:2) {
      difference <- difference + .pdpmvar_covariance_weighted_score(atoms[[groups[index]]],
        new_scatter[[index]] - old_scatter[[index]], new_count[index] - old_count[index], backend)
    }
    difference <- difference + (if (split) 1 else -1) *
      (.pdpmvar_covariance_log_base(atoms[[g]], shape, rate) - .pdpmvar_bridge_density(atoms[[g]], center))
    work <- work + (schedule[stage] - schedule[stage - 1]) * difference
    # Endpoint kernels are excluded so the reversed bridge uses the exact reversed path.
    if (stage == length(schedule)) break
    beta <- schedule[stage]
    for (repeat_step in 1:16) {
      for (index in 1:2) {
        scatter <- (1 - beta) * old_scatter[[index]] + beta * new_scatter[[index]]
        count <- (1 - beta) * old_count[index] + beta * new_count[index]
        if (index == 1) atoms[[h]] <- .pdpmvar_covariance_bridge_step(atoms[[h]], scatter, count, B, shape, rate, backend)
        else atoms[[g]] <- .pdpmvar_bridge_auxiliary(atoms[[g]], active_scatter, active_count,
          if (split) beta else 1 - beta, center, B, shape, rate, backend)
      }
    }
  }
  return(list(atoms = atoms, log_work = work, path = path, temperatures = schedule))
}
