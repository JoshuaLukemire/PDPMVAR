test_that("the default PDPMVAR fixture has the requested truth and stable dynamics", {
  set.seed(104)
  fixture <- simulate_pdpmvar_data()

  expect_length(fixture$data, 25)
  expect_true(all(vapply(fixture$data, function(x) all(dim(x) == c(10, 100)), logical(1))))
  expect_equal(dim(fixture$coefficients), c(10, 10, 2, 3))
  expect_equal(sort(as.numeric(table(fixture$coefficient_cluster))), c(8, 8, 9))
  expect_equal(sort(as.numeric(table(fixture$covariance_cluster))), c(8, 8, 9))

  for (cluster in seq_len(3)) {
    expect_equal(mean(fixture$coefficients[, , , cluster] == 0), 0.5)
    companion <- rbind(cbind(fixture$coefficients[, , 1, cluster],
                             fixture$coefficients[, , 2, cluster]),
                       cbind(diag(10), matrix(0, 10, 10)))
    expect_lt(max(Mod(eigen(companion, only.values = TRUE)$values)), 1)
    expect_true(isSymmetric(fixture$covariance[, , cluster]))
    expect_gt(min(eigen(fixture$covariance[, , cluster], symmetric = TRUE,
                        only.values = TRUE)$values), 0)
  }
})

test_that("simulated observations satisfy the true VAR recursion", {
  set.seed(105)
  fixture <- simulate_pdpmvar_data()
  for (subject in seq_along(fixture$data)) {
    history <- cbind(fixture$initial_history[[subject]], fixture$data[[subject]])
    cluster <- fixture$coefficient_cluster[subject]
    conditional_mean <- fixture$coefficients[, , 1, cluster] %*% history[, 2:101] +
      fixture$coefficients[, , 2, cluster] %*% history[, 1:100]
    expect_equal(fixture$data[[subject]] - conditional_mean, fixture$innovations[[subject]])
  }
})

test_that("fixtures are reproducible from an external seed", {
  set.seed(106)
  first <- simulate_pdpmvar_data()
  set.seed(106)
  expect_identical(simulate_pdpmvar_data(), first)
})

test_that("the matched fixture has the exact fitted covariance and zero-history likelihood", {
  set.seed(1406)
  fixture <- simulate_pdpmvar_matched_data()
  expect_equal(fixture$settings$n_factors, 5)
  expect_equal(fixture$settings$n_warmup, 0)
  expect_equal(sort(as.numeric(table(fixture$coefficient_cluster))), c(8, 8, 9))
  expect_equal(sort(as.numeric(table(fixture$covariance_cluster))), c(8, 8, 9))
  for (cluster in 1:3) {
    atom <- fixture$covariance_atoms[[cluster]]
    expect_equal(dim(atom$working_loadings), c(10, 5))
    expect_true(all(atom$working_loadings[upper.tri(atom$working_loadings)] == 0))
    expect_equal(.pdpmvar_test_reference$.pdpmvar_refresh_covariance(atom)$covariance, fixture$covariance[, , cluster])
    expect_equal(mean(fixture$coefficients[, , , cluster] == 0), 0.5)
  }
  expect_true(all(vapply(fixture$initial_history, function(x) all(x == 0), logical(1))))
  residuals <- lapply(seq_along(fixture$data), function(subject) {
    coefficient <- matrix(fixture$coefficients[, , , fixture$coefficient_cluster[subject]], 10, 20)
    return(fixture$data[[subject]] - coefficient %*% .pdpmvar_lag_design(fixture$data[[subject]], 2))
  })
  expect_equal(residuals, fixture$innovations)
  set.seed(1406)
  expect_identical(simulate_pdpmvar_matched_data(), fixture)
  scalar <- simulate_pdpmvar_matched_data(n_nodes = 1, n_subjects = 3, n_lags = 1)
  expect_equal(dim(scalar$data[[1]]), c(1, 100))
  expect_equal(dim(scalar$covariance_atoms[[1]]$working_loadings), c(1, 1))
  expect_error(simulate_pdpmvar_matched_data(n_factors = 11), "n_factors")
  expect_error(simulate_pdpmvar_data(n_warmup = -1), "n_warmup")
})
