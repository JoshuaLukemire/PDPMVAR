test_that("compiled GIG draws match R values and RNG state", {
  cases <- list(c(-3, 2, 3), c(0, 2, 3), c(0.5, 2, 3), c(0.5, 0, 3), c(-0.5, 1e-6, 10))
  for (index in seq_along(cases)) {
    parameters <- cases[[index]]
    set.seed(130 + index)
    reference <- .pdpmvar_test_reference$.pdpmvar_draw_gig(parameters[1], parameters[2], parameters[3])
    state <- .Random.seed
    set.seed(130 + index)
    compiled <- .pdpmvar_test_reference$.pdpmvar_draw_gig(parameters[1], parameters[2], parameters[3], backend = "RcppEigen")
    expect_equal(compiled, reference, tolerance = 1e-12)
    expect_identical(.Random.seed, state)
  }
  expect_error(.pdpmvar_test_reference$.pdpmvar_draw_gig(0, 0, 3, backend = "RcppEigen"), "positive")
})

test_that("compiled shared shrinkage excludes inactive atoms and preserves the Gamma update", {
  set.seed(140)
  variance <- matrix(seq(0.1, 0.8, length.out = 8), 2)
  atoms <- lapply(seq_len(5), function(i) .pdpmvar_test_reference$.pdpmvar_new_coefficient(variance))
  original <- atoms
  labels <- c(4, 1, 4, 1)
  set.seed(141)
  reference <- .pdpmvar_test_reference$.pdpmvar_update_shrinkage(labels, atoms, variance, 3.3)
  state <- .Random.seed
  set.seed(141)
  compiled <- .pdpmvar_test_reference$.pdpmvar_update_shrinkage(labels, atoms, variance, 3.3, backend = "RcppEigen")
  expect_equal(compiled, reference, tolerance = 1e-12)
  expect_identical(.Random.seed, state)
  expect_identical(atoms, original)
  for (index in c(2, 3, 5)) atoms[[index]]$coefficient[] <- 1e6
  set.seed(141)
  expect_identical(.pdpmvar_test_reference$.pdpmvar_update_shrinkage(labels, atoms, variance, 3.3, backend = "RcppEigen"), compiled)
})

test_that("compiled stick and slice updates match R without truncation", {
  labels <- c(1, 4, 4, 8)
  for (concentration in c(0.5, 1, 10)) {
    set.seed(142)
    reference <- .pdpmvar_test_reference$.pdpmvar_slice_weights(labels, concentration)
    state <- .Random.seed
    set.seed(142)
    compiled <- .pdpmvar_test_reference$.pdpmvar_slice_weights(labels, concentration, backend = "RcppEigen")
    expect_equal(compiled, reference, tolerance = 1e-13)
    expect_identical(.Random.seed, state)
    expect_true(all(compiled$slices < compiled$weights[labels]))
    expect_lt(1 - sum(compiled$weights), min(compiled$slices))
  }
  set.seed(143)
  reference <- .pdpmvar_test_reference$.pdpmvar_prefix_weights(labels, 2.5, 3)
  state <- .Random.seed
  set.seed(143)
  compiled <- .pdpmvar_test_reference$.pdpmvar_prefix_weights(labels, 2.5, 3, backend = "RcppEigen")
  expect_equal(compiled, reference, tolerance = 1e-13)
  expect_identical(.Random.seed, state)
  expect_error(.pdpmvar_test_reference$.pdpmvar_slice_weights(c(1, 1.2), 1, backend = "RcppEigen"), "integer")
})

test_that("compiled GIG conditionals reproduce analytic moments", {
  set.seed(144)
  for (p in c(-3, 0, 0.5)) {
    draws <- replicate(4000, .pdpmvar_test_reference$.pdpmvar_draw_gig(p, 2, 3, backend = "RcppEigen"))
    argument <- sqrt(6)
    mean_target <- sqrt(2 / 3) * besselK(argument, p + 1, expon.scaled = TRUE) /
      besselK(argument, p, expon.scaled = TRUE)
    inverse_target <- sqrt(3 / 2) * besselK(argument, p - 1, expon.scaled = TRUE) /
      besselK(argument, p, expon.scaled = TRUE)
    expect_lt(abs(mean(draws) - mean_target), 5 * sd(draws) / sqrt(length(draws)))
    expect_lt(abs(mean(1 / draws) - inverse_target), 5 * sd(1 / draws) / sqrt(length(draws)))
  }
})


test_that("the symmetric GIG envelope is stable under rounding-scale perturbations", {
  for (parameters in list(c(0.2, 2), c(0.0171148207684813, 2.11556370293336))) {
    for (seed in 1410:1414) {
      set.seed(seed)
      reference <- .pdpmvar_test_reference$.pdpmvar_draw_gig(0, parameters[1], parameters[2])
      rng <- .Random.seed
      set.seed(seed)
      perturbed <- .pdpmvar_test_reference$.pdpmvar_draw_gig(0, parameters[1] * (1 + 1e-15), parameters[2])
      expect_equal(perturbed, reference, tolerance = 1e-12)
      expect_identical(.Random.seed, rng)
      set.seed(seed)
      expect_equal(.pdpmvar_eigen_gig(0, parameters[1], parameters[2]), reference)
      expect_identical(.Random.seed, rng)
    }
  }
})
