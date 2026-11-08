# Functions for checking variation and group membership across PDPMVAR chains.

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compare variation within and between chains ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Compare variation between chains with variation within chains and return one R-hat value.
# draws has retained iterations in rows and chains in columns: M x C, where M is
# the number of iterations and C is the number of chains being compared.
# Return Inf when chains are constant at different values, and NA when all are constant
# at the same value. Used by .pdpmvar_rank_rhat() after transforming the draws.
.pdpmvar_variance_rhat <- function(draws) {
  # The ratio compares the average within-chain variance with variation in chain means.
  n       <- nrow(draws)
  within  <- mean(apply(draws, 2, stats::var))
  between <- stats::var(colMeans(draws))
  if (within == 0) {
    if (between > 0) {return(Inf)}
    return(NA_real_)
  }

  return(sqrt(((n - 1) * within / n + between) / within))
}

# Calculate R-hat after splitting each chain in half and transforming pooled ranks to normal scores.
# Repeat the calculation on absolute deviations from the pooled median to detect differences in spread.
# draws is retained iterations by chains, M x C. Splitting gives H x (2 * C), where
# H = floor(M / 2); an odd middle iteration is left out. Return the larger of the two R-hat values.
# Used by pdpmvar_diagnostics() to compare each saved quantity across chains.
.pdpmvar_rank_rhat <- function(draws) {
  # Compare the first and last half of every chain as separate chains.
  half  <- floor(nrow(draws) / 2)
  split <- cbind(draws[seq_len(half), , drop = FALSE],
                  draws[nrow(draws) - half + seq_len(half), , drop = FALSE])

  # Rank all split-chain draws together, then transform the ranks to normal scores.
  ranks      <- rank(split, ties.method = "average")
  normalized <- matrix(stats::qnorm((ranks - 3 / 8) / (length(ranks) + 1 / 4)), nrow(split))

  # Repeat for distance from the pooled median to check differences in spread.
  folded            <- abs(split - stats::median(split))
  ranks             <- rank(folded, ties.method = "average")
  normalized_folded <- matrix(stats::qnorm((ranks - 3 / 8) / (length(ranks) + 1 / 4)), nrow(split))

  values <- c(.pdpmvar_variance_rhat(normalized), .pdpmvar_variance_rhat(normalized_folded))
  if (all(is.na(values))) {return(NA_real_)}

  return(max(values, na.rm = TRUE))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Count changes in subject groups ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Count consecutive retained iterations in which the subject grouping changes.
# labels is retained iterations by subjects, M x N, where M is iterations and N is subjects.
# Compare N x N matrices showing which subject pairs share a group. Renaming groups
# alone does not count as a change. Return the number of changed groupings, not the
# number of subjects that moved. Used by pdpmvar_diagnostics() for both sets of group labels.
.pdpmvar_partition_changes <- function(labels) {
  # A pair stays together regardless of the numeric name given to its group.
  previous <- outer(labels[1, ], labels[1, ], "==")
  changes  <- 0
  for (iteration in 2:nrow(labels)) {
    current  <- outer(labels[iteration, ], labels[iteration, ], "==")
    changes  <- changes + !identical(current, previous)
    previous <- current
  }

  return(changes)
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compare completed fits ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#' Compare PDPMVAR chains
#'
#' @param fits A list of at least two pdpmvar fits to the same observations and
#'   model. Each must have the same number of retained iterations, at least eight.
#' @return A list containing `rhat`, `chains`, `coclustering_differences` and `flags`.
#'   For C chains and Q quantities, `rhat` is a Q-row table and `chains` has C rows.
#'   The co-clustering table has two rows per pair of chains, one for coefficient
#'   groups and one for covariance groups. `flags` describes possible mixing problems.
#' @details Compares the numbers of groups, log likelihood, average node variance,
#'   root mean square coefficient and lambda when sampled. Log likelihood omits
#'   the Gaussian constant shared by all fits. For each quantity, arrange M retained
#'   iterations in rows and C chains in columns, M x C, then calculate rank-normalized
#'   split R-hat for both the draws and their absolute deviations from the pooled median.
#'   Report the larger value. Identical constant traces return NA; different constant
#'   values across chains return Inf.
#'
#'   Group numbers need not agree between chains. Compare how often each pair of
#'   subjects shares a group instead, using subjects-by-subjects matrices, N x N.
#'   Also count iterations where group membership changes, ignoring simple renaming.
#'   Chains that agree but never change groups have not demonstrated mixing, and
#'   low R-hat for these summaries does not establish convergence of every parameter.
#'
#'   The function checks data dimensions and model settings. The user must ensure
#'   that the fits use the same observations with subjects and nodes in the same order.
#'   Called by pdpmvar_chains(), or used directly with a list of completed fits.
#' @export
pdpmvar_diagnostics <- function(fits) {
  if (!is.list(fits) || length(fits) < 2 ||
      !all(vapply(fits, inherits, logical(1), what = "pdpmvar"))) {
    stop("fits must contain at least two pdpmvar fits.")
  }

  # Require enough retained draws and matching model settings before comparing chains.
  n_draws <- vapply(fits, function(fit) nrow(fit$covariance_labels), numeric(1))
  if (length(unique(n_draws)) != 1 || n_draws[1] < 8) {
    stop("All chains need the same number of retained draws, at least eight.")
  }
  model_controls <- c("center", "K", "n_factors", "coefficient_prior", "coefficient_concentration",
                       "covariance_concentration", "cov_shape", "cov_rate")
  if (fits[[1]]$settings$coefficient_prior == "normal") {
    model_controls <- c(model_controls, "coefficient_sd")
  }
  reference <- fits[[1]]$settings[model_controls]
  for (fit in fits) {
    if (!identical(dim(fit$coefficient_mean), dim(fits[[1]]$coefficient_mean)) ||
        !isTRUE(all.equal(fit$settings[model_controls], reference))) {
      stop("Chains must have matching data dimensions and model settings.")
    }
    if (is.null(fit$parameter_traces)) {
      stop("Refit chains with the current pdpmvar version to record diagnostic traces.")
    }
  }

  # Choose summaries that can be compared without matching numeric group labels.
  quantities <- c("coefficient_clusters", "covariance_clusters", "log_likelihood",
                   "mean_node_variance", "coefficient_rms")
  if (reference$coefficient_prior == "laplace") {quantities <- c(quantities, "lambda")}
  traces        <- vector("list", length(fits))
  chain_summary <- vector("list", length(fits))

  # For each chain, store retained iterations in rows and saved quantities in columns, M x Q.
  for (chain in seq_along(fits)) {
    fit             <- fits[[chain]]
    retained        <- fit$settings$n_burnin + seq_len(n_draws[chain])
    traces[[chain]] <- cbind(coefficient_clusters = fit$cluster_counts[retained, "coefficient"],
      covariance_clusters = fit$cluster_counts[retained, "covariance"], fit$parameter_traces)
    if ("lambda" %in% quantities) {traces[[chain]] <- cbind(traces[[chain]], lambda = fit$lambda)}

    # Some saved fits record a move affecting both groupings; include it in the covariance counts.
    joint_splits <- joint_merges <- 0
    if ("partitions" %in% names(fit$coefficient_split_merge)) {
      joint        <- fit$coefficient_split_merge
      joint_splits <- sum(joint$partitions == "both" & joint$type == "split" & joint$accepted)
      joint_merges <- sum(joint$partitions == "both" & joint$type == "merge" & joint$accepted)
    }

    # Summarize the starting groups, retained group counts, changes and accepted moves.
    chain_summary[[chain]] <- data.frame(chain = chain,
      initialization = fit$settings$initialization,
      initial_coefficient_clusters = length(unique(fit$initial_labels$coefficient)),
      initial_covariance_clusters = length(unique(fit$initial_labels$covariance)),
      mean_coefficient_clusters = mean(traces[[chain]][, "coefficient_clusters"]),
      mean_covariance_clusters = mean(traces[[chain]][, "covariance_clusters"]),
      coefficient_partition_changes = .pdpmvar_partition_changes(fit$coefficient_labels),
      covariance_partition_changes = .pdpmvar_partition_changes(fit$covariance_labels),
      mean_log_likelihood = mean(fit$parameter_traces[, "log_likelihood"]),
      accepted_splits = sum(fit$split_merge$type == "split" & fit$split_merge$accepted) + joint_splits,
      accepted_merges = sum(fit$split_merge$type == "merge" & fit$split_merge$accepted) + joint_merges,
      accepted_coefficient_splits = sum(fit$coefficient_split_merge$type == "split" & fit$coefficient_split_merge$accepted),
      accepted_coefficient_merges = sum(fit$coefficient_split_merge$type == "merge" & fit$coefficient_split_merge$accepted))
  }

  # For each quantity, put retained iterations in rows and chains in columns, M x C.
  rhat <- data.frame(quantity = quantities, rhat = NA_real_)
  for (index in seq_along(quantities)) {
    draws            <- vapply(traces, function(x) x[, quantities[index]], numeric(n_draws[1]))
    rhat$rhat[index] <- .pdpmvar_rank_rhat(draws)
  }

  # Compare subject-pair co-clustering probabilities for every pair of chains.
  differences <- list()
  for (first in seq_len(length(fits) - 1)) {
    for (second in (first + 1):length(fits)) {
      for (component in c("coefficient", "covariance")) {
        difference <- abs(fits[[first]][[paste0(component, "_coclustering")]] -
                            fits[[second]][[paste0(component, "_coclustering")]])

        # Use each pair once and omit the diagonal, where every subject is paired with itself.
        off_diagonal                           <- difference[upper.tri(difference)]
        differences[[length(differences) + 1]] <- data.frame(first = first, second = second,
          component = component, mean_difference = if (length(off_diagonal)) mean(off_diagonal) else 0,
          max_difference = if (length(off_diagonal)) max(off_diagonal) else 0)
      }
    }
  }

  # Combine the per-chain and per-pair rows, then flag disagreement or a lack of movement.
  chain_summary <- do.call(rbind, chain_summary)
  differences   <- do.call(rbind, differences)
  flags         <- character()
  if (any(rhat$rhat > 1.01, na.rm = TRUE)) {
    flags <- c(flags, "Some rank-normalized split R-hat values exceed 1.01.")
  }
  if (anyNA(rhat$rhat)) {
    flags <- c(flags, "Some traces are constant across all chains; their R-hat is undefined.")
  }
  if (any(chain_summary$coefficient_partition_changes == 0 |
          chain_summary$covariance_partition_changes == 0)) {
    flags <- c(flags, "Some retained partitions never change; cross-chain agreement does not demonstrate partition mixing.")
  }
  if (any(differences$max_difference > 0.1)) {
    flags <- c(flags, "Some subject-pair co-clustering probabilities differ between chains by more than 0.1.")
  }

  return(list(rhat = rhat, chains = chain_summary, coclustering_differences = differences, flags = flags))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fit several chains and compare them ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#' Fit and compare PDPMVAR chains with separate seeds
#'
#' @param X A list of subject matrices passed to pdpmvar. Each matrix has nodes in
#'   rows and time points in columns, D x T_i, where D is the number of nodes and
#'   T_i is the number of time points for subject i.
#' @param seeds Distinct nonnegative whole-number seeds, one per chain; supply at least two.
#' @param initial_clusters Starting number of groups, either one value for every
#'   chain or one per chain. Each value applies separately to the coefficient and
#'   covariance groups; it does not give separate counts for different lags.
#'   For example, c(2,3,4) starts three chains with two, three and four groups
#'   in each mixture. Each coefficient label covers the full matrix across all lags.
#'   NULL (default) spaces the starting counts from four to eight, capped at the
#'   number of subjects: c(4,6,8) for three chains with at least eight subjects.
#' @param initialization One starting method for all chains or one per chain,
#'   each `"data"` or `"random"`.
#' @param ... Other arguments passed to pdpmvar. Use n_mcmc >= 8.
#' @return A list with C `fits`, their C `seeds` and the result of pdpmvar_diagnostics(),
#'   where C is the number of chains. Each fit stores label matrices of size retained
#'   iterations by subjects, M x N, for M retained iterations and N subjects.
#' @details Fit the chains one at a time, calling set.seed() before each fit, then
#'   compare the completed fits with pdpmvar_diagnostics(). Priors and other model
#'   settings must agree; starting groups and starting methods may differ.
#'   Each chain fits the model separately, so computation increases with the number of chains.
#' @export
pdpmvar_chains <- function(X,
                           seeds = c(1, 2, 3),
                           initial_clusters = NULL,
                           initialization = "data",
                           ...) {
  starts <- .pdpmvar_chain_starts(X, seeds, initial_clusters, initialization)
  n_chains <- length(seeds)

  # The diagnostics require at least eight retained iterations per chain.
  arguments <- list(...)
  if (!is.null(arguments$n_mcmc)) {.pdpmvar_check_count(arguments$n_mcmc, "n_mcmc", 8)}

  # Set each supplied seed immediately before fitting that chain.
  fits <- vector("list", n_chains)
  for (chain in seq_len(n_chains)) {
    set.seed(seeds[chain])
    fits[[chain]] <- pdpmvar(X, initial_clusters = starts$counts[chain],
      initialization = starts$methods[chain], ...)
  }

  return(list(fits = fits, seeds = seeds, diagnostics = pdpmvar_diagnostics(fits)))
}

# Check the seeds and starting choices used by the multi-chain fitting functions.
.pdpmvar_chain_starts <- function(X, seeds, initial_clusters, initialization) {
  if (!is.numeric(seeds) || length(seeds) < 2 || any(!is.finite(seeds)) ||
      any(seeds < 0 | seeds > .Machine$integer.max | seeds != floor(seeds)) || anyDuplicated(seeds)) {
    stop("seeds must contain at least two distinct nonnegative integer seeds.")
  }

  # Choose one starting group count per chain, capped at the number of subjects.
  n_chains <- length(seeds)
  if (is.null(initial_clusters)) {
    initial_clusters <- pmin(length(X), round(seq(4, 8, length.out = n_chains)))
  }
  if (!length(initial_clusters) %in% c(1, n_chains) ||
      !length(initialization) %in% c(1, n_chains) ||
      anyNA(initialization) || any(!initialization %in% c("data", "random"))) {
    stop("Supply one initial_clusters/initialization value or one per chain; initialization must be data or random.")
  }

  # Repeat shared starting choices so every chain has its own count and method.
  initial_clusters <- rep(initial_clusters, length.out = n_chains)
  initialization   <- rep(initialization, length.out = n_chains)
  for (value in initial_clusters) {.pdpmvar_check_count(value, "initial_clusters")}
  if (any(initial_clusters > length(X))) {stop("initial_clusters cannot exceed the subject count.")}

  return(list(counts = initial_clusters, methods = initialization))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compare lag and response partitions across chains ----
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#' Compare lgPDPMVAR chains
#'
#' @param fits List of at least two lgpdpmvar fits to the same data in the
#'   same order, with matching model settings and at least eight retained
#'   iterations per chain.
#' @return A list with `rhat`, `chains`, `coclustering_differences` and `flags`.
#'   The chain table has one row per chain and partition. The co-clustering
#'   table has one row per pair of chains and partition.
#' @details Compare rank-normalized split R-hat for group counts, the saved
#'   continuous summaries, and Laplace shrinkage rates when sampled. The
#'   partition summaries count changes during retained iterations, ignoring
#'   simple renaming of labels. Split and merge counts include burn-in.
#'   Co-clustering differences compare how often each pair of subjects
#'   shares a group. Identical constant traces have undefined R-hat. Agreement
#'   across chains with no partition changes does not establish mixing.
#' @export
lgpdpmvar_diagnostics <- function(fits) {
  return(.pdpmvar_partition_diagnostics(fits, "lgpdpmvar"))
}

#' Compare rgPDPMVAR chains
#'
#' @param fits List of at least two rgpdpmvar fits to the same data in the
#'   same order, with matching model settings and at least eight retained
#'   iterations per chain.
#' @return A list with `rhat`, `chains`, `coclustering_differences` and `flags`.
#'   The chain table has one row per chain and partition. The co-clustering
#'   table has one row per pair of chains and partition. Each response node
#'   has its own partition, named `node_1`, `node_2`, and so on.
#'   Residual covariance has a separate partition named `covariance`.
#' @details Compare rank-normalized split R-hat for group counts, the saved
#'   continuous summaries, and each response node's shrinkage rate. Count
#'   changes in subject groupings during retained iterations, ignoring
#'   renaming of labels. Split and merge counts include burn-in.
#'   Co-clustering differences compare how often each pair of subjects shares
#'   a group. Identical constant traces have undefined R-hat. Agreement
#'   across chains with no partition changes does not establish mixing.
#' @export
rgpdpmvar_diagnostics <- function(fits) {
  return(.pdpmvar_partition_diagnostics(fits, "rgpdpmvar"))
}

# Compare the fitted partitions for lgpdpmvar_diagnostics() or
# rgpdpmvar_diagnostics(). Use one M x N label matrix per partition,
# with M retained iterations and N subjects.
.pdpmvar_partition_diagnostics <- function(fits, model) {
  if (!is.list(fits) || length(fits) < 2 ||
      !all(vapply(fits, inherits, logical(1), what = model))) {
    stop("fits must contain at least two ", model, " fits.")
  }
  n_draws <- vapply(fits, function(fit) nrow(fit$covariance_labels), integer(1))
  if (length(unique(n_draws)) != 1 || n_draws[1] < 8) {
    stop("All chains need the same number of retained draws, at least eight.")
  }
  controls <- c("center", "K", "n_factors", "coefficient_concentration",
                "covariance_concentration", "cov_shape", "cov_rate")
  if (model == "lgpdpmvar") {
    controls <- c(controls, "coefficient_prior")
    if (fits[[1]]$settings$coefficient_prior == "normal") {
      controls <- c(controls, "coefficient_sd")
    }
  }
  reference <- fits[[1]]$settings[controls]
  for (fit in fits) {
    if (!identical(dim(fit$coefficient_mean), dim(fits[[1]]$coefficient_mean)) ||
        !isTRUE(all.equal(fit$settings[controls], reference))) {
      stop("Chains must have matching data dimensions and model settings.")
    }
  }

  coefficient_partitions <- if (model == "lgpdpmvar") {
    paste0("lag_", seq_len(reference$K))
  } else {
    paste0("node_", seq_len(dim(fits[[1]]$coefficient_mean)[1]))
  }
  partitions <- c(coefficient_partitions, "covariance")
  traces     <- lapply(fits, function(fit) {
    retained <- fit$settings$n_burnin + seq_len(nrow(fit$covariance_labels))
    counts   <- fit$cluster_counts[retained, partitions, drop = FALSE]
    colnames(counts) <- paste0(partitions, "_clusters")
    result <- cbind(counts, fit$parameter_traces)
    if (!is.null(fit$lambda)) {
      lambda           <- fit$lambda
      colnames(lambda) <- paste0(coefficient_partitions, "_lambda")
      result <- cbind(result, lambda)
    }
    return(result)
  })
  quantities <- colnames(traces[[1]])
  rhat <- data.frame(quantity = quantities, rhat = vapply(quantities, function(quantity) {
    draws <- vapply(traces, function(trace) trace[, quantity], numeric(n_draws[1]))
    return(.pdpmvar_rank_rhat(draws))
  }, numeric(1)), row.names = NULL)

  chain_rows <- lapply(seq_along(fits), function(chain) {
    fit    <- fits[[chain]]
    labels <- c(fit$coefficient_labels, list(covariance = fit$covariance_labels))
    starts <- c(fit$initial_labels$coefficient, list(covariance = fit$initial_labels$covariance))
    moves  <- c(fit$coefficient_split_merge, list(covariance = fit$split_merge))
    rows   <- lapply(partitions, function(partition) {
      trace <- moves[[partition]]
      return(data.frame(chain = chain, partition = partition,
        initialization = fit$settings$initialization,
        initial_clusters = length(unique(starts[[partition]])),
        mean_clusters = mean(traces[[chain]][, paste0(partition, "_clusters")]),
        partition_changes = .pdpmvar_partition_changes(labels[[partition]]),
        attempted_splits = sum(trace$type == "split"),
        attempted_merges = sum(trace$type == "merge"),
        accepted_splits = sum(trace$type == "split" & trace$accepted),
        accepted_merges = sum(trace$type == "merge" & trace$accepted),
        skipped_moves = sum(trace$type == "skip")))
    })
    return(do.call(rbind, rows))
  })
  chains <- do.call(rbind, chain_rows)

  similarities <- lapply(fits, function(fit) {
    return(c(fit$coefficient_coclustering, list(covariance = fit$covariance_coclustering)))
  })
  differences <- list()
  for (first in seq_len(length(fits) - 1)) {
    for (second in (first + 1):length(fits)) {
      for (partition in partitions) {
        gap <- abs(similarities[[first]][[partition]] - similarities[[second]][[partition]])
        pairs <- gap[upper.tri(gap)]
        differences[[length(differences) + 1]] <- data.frame(
          first = first, second = second, partition = partition,
          mean_difference = if (length(pairs)) mean(pairs) else 0,
          max_difference = if (length(pairs)) max(pairs) else 0)
      }
    }
  }
  differences <- do.call(rbind, differences)

  flags <- character()
  for (partition in partitions) {
    count <- rhat$rhat[rhat$quantity == paste0(partition, "_clusters")]
    if (is.na(count)) {
      flags <- c(flags, paste0(partition, ": group-count traces are constant across chains."))
    }
    if (any(chains$partition == partition & chains$partition_changes == 0)) {
      flags <- c(flags, paste0(partition, ": at least one retained partition never changes."))
    }
    if (any(differences$partition == partition & differences$max_difference > 0.1)) {
      flags <- c(flags, paste0(partition, ": some co-clustering probabilities differ by more than 0.1."))
    }
  }
  high <- rhat$quantity[!is.na(rhat$rhat) & rhat$rhat > 1.01]
  if (length(high)) {
    flags <- c(flags, paste0("Rank-normalized split R-hat exceeds 1.01 for: ", paste(high, collapse = ", "), "."))
  }
  return(list(rhat = rhat, chains = chains, coclustering_differences = differences, flags = flags))
}

#' Fit and compare lgPDPMVAR chains
#'
#' @inheritParams pdpmvar_chains
#' @param initial_clusters Starting group count, one for all chains or one per
#'   chain. Each count applies separately to every lag and to covariance.
#'   NULL spaces the counts from four to eight, capped at the subject count.
#' @param ... Other arguments passed to lgpdpmvar. Use n_mcmc >= 8.
#' @return A list with the fitted chains, their seeds and lgpdpmvar diagnostics.
#' @export
lgpdpmvar_chains <- function(X,
                             seeds = c(1, 2, 3),
                             initial_clusters = NULL,
                             initialization = "data",
                             ...) {
  starts <- .pdpmvar_chain_starts(X, seeds, initial_clusters, initialization)
  arguments <- list(...)
  if (!is.null(arguments$n_mcmc)) {.pdpmvar_check_count(arguments$n_mcmc, "n_mcmc", 8)}

  fits <- vector("list", length(seeds))
  for (chain in seq_along(seeds)) {
    set.seed(seeds[chain])
    fits[[chain]] <- lgpdpmvar(X, initial_clusters = starts$counts[chain],
      initialization = starts$methods[chain], ...)
  }
  return(list(fits = fits, seeds = seeds, diagnostics = lgpdpmvar_diagnostics(fits)))
}

#' Fit and compare rgPDPMVAR chains
#'
#' @inheritParams pdpmvar_chains
#' @param X A list of subject matrices passed to rgpdpmvar. Each matrix
#'   has response nodes in rows and time points in columns.
#' @param initial_clusters Starting group count, one for all chains or one per
#'   chain. Each count applies separately to every response node and to
#'   covariance. NULL spaces the counts from four to eight, capped at the
#'   subject count.
#' @param ... Other arguments passed to rgpdpmvar. Use n_mcmc >= 8.
#' @return A list with the fitted chains, their seeds and rgpdpmvar diagnostics.
#' @export
rgpdpmvar_chains <- function(X,
                             seeds = c(1, 2, 3),
                             initial_clusters = NULL,
                             initialization = "data",
                             ...) {
  starts    <- .pdpmvar_chain_starts(X, seeds, initial_clusters, initialization)
  arguments <- list(...)
  if (!is.null(arguments$n_mcmc)) {.pdpmvar_check_count(arguments$n_mcmc, "n_mcmc", 8)}

  # Fit each chain after setting its seed, then compare the completed fits.
  fits <- vector("list", length(seeds))
  for (chain in seq_along(seeds)) {
    set.seed(seeds[chain])
    fits[[chain]] <- rgpdpmvar(X, initial_clusters = starts$counts[chain],
      initialization = starts$methods[chain], ...)
  }
  return(list(fits = fits, seeds = seeds, diagnostics = rgpdpmvar_diagnostics(fits)))
}
