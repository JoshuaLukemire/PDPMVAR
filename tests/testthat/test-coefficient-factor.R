test_that("conditional row integrals and draws equal the current dense Gaussian calculation", {
  set.seed(8101)
  D <- 3; P <- 5; N <- 4
  Y <- lapply(1:N,function(i)matrix(rnorm(D*(8+i)),D))
  design <- lapply(Y,function(y)matrix(rnorm(P*ncol(y)),P))
  cross <- lapply(design,tcrossprod)
  v <- matrix(c(1,1.5,0.7),D,N)
  v[,3:4] <- v[,3:4]*1.3
  tau <- matrix(seq(0.08,0.8,length.out=D*P),D)
  covariances <- lapply(1:N,function(i)diag(v[,i]))
  dense <- .pdpmvar_test_reference$.pdpmvar_coefficient_statistics(Y,design,cross,covariances,tau)
  row_R <- .pdpmvar_test_reference$.pdpmvar_coefficient_row_statistics(Y,design,cross,v,tau)
  row_C <- .pdpmvar_test_reference$.pdpmvar_coefficient_row_statistics(Y,design,cross,v,tau,"RcppEigen")
  expect_equal(row_C[names(row_R)!="backend"],row_R[names(row_R)!="backend"],tolerance=1e-12)
  for(subjects in list(integer(),1,1:3,c(2,4))) {
    old <- .pdpmvar_test_reference$.pdpmvar_coefficient_group(dense,subjects)
    noise <- matrix(rnorm(D*P),D)
    for(statistics in list(row_R,row_C)) {
      group <- .pdpmvar_test_reference$.pdpmvar_coefficient_row_group(statistics,subjects,noise)
      expect_equal(group$mean,old$mean,tolerance=1e-10)
      expect_equal(group$log_integral,old$log_integral,tolerance=1e-10)
      expect_equal(as.numeric(group$coefficient),old$mean+backsolve(old$upper,as.numeric(noise)),tolerance=1e-10)
    }
  }
  anchors <- c(1,3)
  weights <- log(c(0.4,0.1,0.3,0.1,0.1))
  for(seed in 8102:8104) {
    set.seed(seed); a <- .pdpmvar_test_reference$.pdpmvar_coefficient_restricted_split(dense,1:N,anchors,weights,1,3); rng <- .Random.seed
    set.seed(seed); b <- .pdpmvar_test_reference$.pdpmvar_coefficient_row_restricted(row_R,1:N,anchors,weights,1,3)
    expect_equal(b$labels,a$labels); expect_equal(b$log_probability,a$log_probability,tolerance=1e-10)
    expect_identical(.Random.seed,rng)
    set.seed(seed); c <- .pdpmvar_test_reference$.pdpmvar_coefficient_row_restricted(row_C,1:N,anchors,weights,1,3)
    expect_equal(c$labels,a$labels); expect_equal(c$log_probability,a$log_probability,tolerance=1e-10)
    expect_identical(.Random.seed,rng)
  }
  old <- c(1,1,4,4); new <- c(1,3,4,4); anchors <- c(1,2)
  expected <- .pdpmvar_test_reference$.pdpmvar_coefficient_collapsed_ratio(old,new,dense,weights,anchors,5)
  for(statistics in list(row_R,row_C)) {
    forward <- .pdpmvar_test_reference$.pdpmvar_coefficient_collapsed_ratio(old,new,statistics,weights,anchors,5)
    reverse <- .pdpmvar_test_reference$.pdpmvar_coefficient_collapsed_ratio(new,old,statistics,weights,anchors,5)
    expect_equal(forward,expected,tolerance=1e-10)
    expect_equal(forward$log_ratio,-reverse$log_ratio,tolerance=1e-10)
  }
  # Independent likelihood/prior/posterior normalization of the conditional target.
  evidence <- numeric(2)
  for(j in 1:2) {
    labels <- if(j==1)old else new
    for(h in unique(labels)) {
      subjects <- which(labels==h)
      group <- .pdpmvar_test_reference$.pdpmvar_coefficient_row_group(row_R,subjects)
      atom <- matrix(seq(-0.1,0.2,length.out=D*P),D)
      mean <- matrix(group$mean,D)
      posterior <- 0
      for(row in 1:D) posterior <- posterior+sum(log(diag(group$upper[[row]])))-P*log(2*pi)/2-
        sum((group$upper[[row]]%*%(atom[row,]-mean[row,]))^2)/2
      likelihood <- sum(vapply(subjects,function(i).pdpmvar_test_reference$.pdpmvar_log_likelihood(Y[[i]]-atom%*%design[[i]],diag(v[,i])),numeric(1)))
      evidence[j] <- evidence[j]+likelihood+sum(dnorm(atom,sd=sqrt(tau),log=TRUE))-posterior
    }
  }
  expect_equal(diff(evidence)+sum(weights[new]-weights[old]),expected$target_change,tolerance=1e-10)
})

test_that("row controllers match the dense diagonal controller, including refresh and RNG", {
  set.seed(8110)
  D <- 2; P <- 3; N <- 4
  Y <- lapply(1:N,function(i)matrix(rnorm(D*(8+i)),D))
  design <- lapply(Y,function(y)matrix(rnorm(P*ncol(y)),P))
  cross <- lapply(design,tcrossprod)
  v <- matrix(seq(0.7,1.5,length.out=D*N),D)
  tau <- matrix(seq(0.1,0.6,length.out=D*P),D)
  atoms <- lapply(1:5,function(i).pdpmvar_test_reference$.pdpmvar_new_coefficient(tau))
  snapshot <- serialize(atoms,NULL)
  covariances <- lapply(1:N,function(i)diag(v[,i]))
  for(labels in list(rep(1,4),c(1,3,1,3),1:4,rep(5,4))) for(seed in 8111:8115) {
    set.seed(seed)
    dense <- .pdpmvar_test_reference$.pdpmvar_coefficient_split_merge_reference(Y,design,cross,covariances,labels,atoms,tau,1,4)
    rng <- .Random.seed
    set.seed(seed)
    R <- .pdpmvar_test_reference$.pdpmvar_coefficient_row_move_reference(Y,design,cross,v,labels,atoms,tau,1,4)
    expect_equal(R,dense,tolerance=1e-8); expect_identical(.Random.seed,rng)
    set.seed(seed)
    C <- .pdpmvar_eigen_row_move(Y,design,cross,v,as.numeric(labels),atoms,tau,1,4)
    expect_equal(C,dense,tolerance=1e-8); expect_identical(.Random.seed,rng)
  }
  expect_identical(serialize(atoms,NULL),snapshot)
})

test_that("compact statistics avoid dense blocks and reject unsafe dense allocations", {
  set.seed(8120)
  Y <- lapply(1:5,function(i)matrix(rnorm(8*12),8))
  Z <- lapply(Y,function(y)matrix(rnorm(16*ncol(y)),16))
  cross <- lapply(Z,tcrossprod)
  v <- matrix(1,8,5); tau <- matrix(0.2,8,16)
  compact <- .pdpmvar_test_reference$.pdpmvar_coefficient_row_statistics(Y,Z,cross,v,tau,"RcppEigen")
  dense <- .pdpmvar_test_reference$.pdpmvar_coefficient_statistics(Y,Z,cross,rep(list(diag(8)),5),tau,"RcppEigen")
  expect_lt(as.numeric(object.size(compact)),as.numeric(object.size(dense))/5)
  expect_equal(ncol(compact$group_variance),1)
  expect_false("precision" %in% names(compact))
  bigY <- list(matrix(0,100,3)); bigZ <- list(matrix(0,200,3)); bigCross <- lapply(bigZ,tcrossprod)
  expect_error(.pdpmvar_test_reference$.pdpmvar_coefficient_statistics(bigY,bigZ,bigCross,list(diag(100)),matrix(1,100,200)),"512 MiB")
  expect_error(.pdpmvar_eigen_statistics(bigY,bigZ,bigCross,list(diag(100)),matrix(1,100,200)),"512 MiB")
  expect_error(.pdpmvar_test_reference$.pdpmvar_coefficient_row_statistics(Y,Z,cross,-v,tau,"RcppEigen"),"positive")
  expect_error(.pdpmvar_eigen_row_group(compact,c(1L,1L)),"distinct")
  expect_error(pdpmvar(list(matrix(rnorm(400),100)),n_burnin=0,n_mcmc=1,cov_shape=3,cov_rate=2,
    coefficient_split_merge="collapsed",coefficient_split_merge_every=1),"512 MiB")
  fit <- pdpmvar(lapply(1:2,function(i)matrix(rnorm(25*5),25)),n_burnin=0,n_mcmc=1,initial_clusters=1,
    cov_shape=3,cov_rate=2,coefficient_split_merge_every=1)
  expect_identical(fit$settings$coefficient_split_merge,"factor_conditional")
  expect_true(all(is.finite(fit$coefficient_mean)))
})

test_that("factor-conditional native and R drivers agree with full published-model updates", {
  set.seed(8130)
  X <- lapply(1:4,function(i)matrix(rnorm(20+2*i),2))
  Z <- lapply(X,.pdpmvar_lag_design,K=1)
  cross <- lapply(Z,tcrossprod)
  callbacks <- list(message=base::message,assign_coefficients=.pdpmvar_test_reference$.pdpmvar_assign_coefficients,assign_covariances=.pdpmvar_test_reference$.pdpmvar_assign_covariances)
  for(prior in c("normal","laplace")) {
    initial <- pdpmvar(X,K=1,n_factors=1,initial_clusters=2,n_burnin=0,n_mcmc=1,cov_shape=3,cov_rate=2,coefficient_prior=prior)
    control <- c(initial$settings,list(store_draws=TRUE,verbose=FALSE))
    control$n_burnin <- 2; control$n_mcmc <- 6; control$backend <- "RcppEigen"
    control$coefficient_split_merge <- "factor_conditional"; control$coefficient_split_merge_every <- 1
    control$split_merge_every <- 2
    snapshot <- serialize(initial$final_state,NULL)
    set.seed(8131)
    R <- .pdpmvar_test_reference$.pdpmvar_sampler_reference(X,Z,cross,initial$final_state,control); rng <- .Random.seed
    set.seed(8131)
    C <- .pdpmvar_eigen_sampler(X,Z,cross,initial$final_state,control,callbacks)
    expect_equal(C,R,tolerance=1e-7)
    expect_identical(.Random.seed,rng)
    expect_identical(serialize(initial$final_state,NULL),snapshot)
  }
})
