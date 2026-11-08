// Shared functions for random draws, likelihoods, group assignments and covariance sampling.
// Coefficient groups and the full PDPMVAR sampling sequence are handled in PDPMVAR.cpp.
#include <RcppEigen.h>
#include "sampling_helpers.h"
#include <R_ext/Applic.h>
#include <R_ext/Lapack.h>
#include <R_ext/Random.h>
#include <R_ext/Utils.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Gaussian integrals and coefficient draws ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Check that the Gaussian precision is a finite, symmetric, square matrix and that
// its size matches the precision-times-mean vector. For M coefficients, these inputs
// are coefficient by coefficient, M x M, and length M, respectively.
// Used by pdpmvar_eigen_integral() and pdpmvar_eigen_normal() before their Cholesky calculations.
void validate_gaussian_inputs(const Rcpp::NumericMatrix& precision,
                              const Rcpp::NumericVector& information) {
    const int dimension = precision.nrow();
    if (dimension == 0 || precision.ncol() != dimension || information.size() != dimension) {
        Rcpp::stop("Precision must be square and match the information vector.");
    }
    const Eigen::Map<const Eigen::MatrixXd> matrix(precision.begin(), dimension, dimension);
    const Eigen::Map<const Eigen::VectorXd> vector(information.begin(), dimension);
    if (!matrix.allFinite() || !vector.allFinite()) {
        Rcpp::stop("Gaussian precision and information must be finite.");
    }
    const double scale = std::max(1.0, matrix.cwiseAbs().maxCoeff());
    if ((matrix - matrix.transpose()).cwiseAbs().maxCoeff() > 1e-12 * scale) {
        Rcpp::stop("Gaussian precision must be symmetric.");
    }
}

// Calculate a Gaussian conditional mean and integrate out the coefficient vector.
// For M coefficients, return precision and Cholesky matrices of size M x M, vectors
// of length M and the integrated log contribution. Terms common to competing groups
// are omitted. Solve with the Cholesky matrix rather than explicitly inverting precision.
// Used by full-matrix coefficient group calculations and their R checks.
// [[Rcpp::export(name = ".pdpmvar_eigen_integral")]]
Rcpp::List pdpmvar_eigen_integral(const Rcpp::NumericMatrix& precision,
                                  const Rcpp::NumericVector& information,
                                  double prior_log_determinant_half) {
    validate_gaussian_inputs(precision, information);
    if (!std::isfinite(prior_log_determinant_half)) {
        Rcpp::stop("Prior log determinant must be finite.");
    }
    const int dimension = precision.nrow();
    const Eigen::Map<const Eigen::MatrixXd> matrix(precision.begin(), dimension, dimension);
    const Eigen::Map<const Eigen::VectorXd> vector(information.begin(), dimension);
    const Eigen::LLT<Eigen::MatrixXd> factor(matrix);
    if (factor.info() != Eigen::Success) {
        Rcpp::stop("Gaussian precision must be positive definite.");
    }

    // Solve for the conditional mean, then use the Cholesky diagonal in the Gaussian integral.
    const Eigen::VectorXd mean  = factor.solve(vector);
    const Eigen::MatrixXd upper = factor.matrixU();
    const double log_integral = vector.dot(mean) / 2.0 + prior_log_determinant_half -
                                upper.diagonal().array().log().sum();
    Rcpp::NumericVector mean_result(dimension);
    std::copy(mean.data(), mean.data() + dimension, mean_result.begin());

    return Rcpp::List::create(Rcpp::Named("precision") = precision,
                              Rcpp::Named("information") = information,
                              Rcpp::Named("upper") = upper,
                              Rcpp::Named("mean") = mean_result,
                              Rcpp::Named("log_integral") = log_integral);
}

// Turn supplied standard normal draws into one draw from a Gaussian conditional distribution.
// For a vector with M entries, precision is M x M; information and standard_normal
// have length M. Return a vector of length M.
// Called by the R Gaussian-draw helper for separate coefficient-update checks.
// [[Rcpp::export(name = ".pdpmvar_eigen_normal")]]
Rcpp::NumericVector pdpmvar_eigen_normal(const Rcpp::NumericMatrix& precision,
                                         const Rcpp::NumericVector& information,
                                         const Rcpp::NumericVector& standard_normal) {
    validate_gaussian_inputs(precision, information);
    const int dimension = precision.nrow();
    if (standard_normal.size() != dimension) {
        Rcpp::stop("Standard normal vector must match the precision dimension.");
    }
    const Eigen::Map<const Eigen::MatrixXd> matrix(precision.begin(), dimension, dimension);
    const Eigen::Map<const Eigen::VectorXd> vector(information.begin(), dimension);
    const Eigen::Map<const Eigen::VectorXd> noise(standard_normal.begin(), dimension);
    if (!noise.allFinite()) {
        Rcpp::stop("Standard normal vector must be finite.");
    }
    const Eigen::LLT<Eigen::MatrixXd> factor(matrix);
    if (factor.info() != Eigen::Success) {
        Rcpp::stop("Gaussian precision must be positive definite.");
    }

    // Add a Gaussian deviation to the conditional mean without forming the inverse precision.
    const Eigen::VectorXd draw = factor.solve(vector) + factor.matrixU().solve(noise);
    Rcpp::NumericVector result(dimension);
    std::copy(draw.data(), draw.data() + dimension, result.begin());

    return result;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Draw shared coefficient prior variances ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Choose one of three regions in the proposal used to draw a generalized inverse Gaussian value.
// log_mass contains three log weights. Consider the regions in decreasing probability
// order, matching R's weighted sampling rule. Return a region number counted from zero.
// Used inside draw_shared_gig() when choosing a rejection-sampling proposal.
int draw_gig_region(const double log_mass[3]) {
    const double maximum = std::max(log_mass[0], std::max(log_mass[1], log_mass[2]));
    double probabilities[3];
    int order[3] = {0, 1, 2};
    double total = 0.0;
    for (int index = 0; index < 3; ++index) {
        probabilities[index] = std::exp(log_mass[index] - maximum);
        total += probabilities[index];
    }
    for (int index = 0; index < 3; ++index) probabilities[index] /= total;
    Rf_revsort(probabilities, order, 3);
    const double uniform = R::runif(0.0, 1.0);
    double cumulative    = 0.0;
    for (int index = 0; index < 2; ++index) {
        cumulative += probabilities[index];
        if (uniform <= cumulative) return order[index];
    }

    return order[2];
}

// Draw one positive value from GIG(p, chi, psi), using rejection sampling on the log scale.
// The log density is p*y - (chi*exp(-y) + psi*exp(y))/2. Bounds on either side of its
// mode give three regions from which proposals can be drawn. If chi is zero and p
// is positive, draw from the corresponding Gamma distribution instead.
// Called by the Laplace shrinkage update for each shared coefficient prior variance.
double draw_shared_gig(double p, double chi, double psi) {
    if (!std::isfinite(p) || !std::isfinite(chi) || !std::isfinite(psi) || psi <= 0.0 || chi < 0.0) {
        Rcpp::stop("GIG shape must be finite; chi must be nonnegative and psi positive and finite.");
    }
    if (chi == 0.0 && p > 0.0) return R::rgamma(p, 2.0 / psi);
    if (chi <= 0.0) Rcpp::stop("GIG chi and psi must be positive and finite.");

    // Find the log-density mode and the scale used to build the rejection-sampling bound.
    const double curvature = std::sqrt(p * p + chi * psi);
    if (!std::isfinite(curvature) || curvature == 0.0)
        Rcpp::stop("GIG parameters exceed the supported numerical range.");
    const double mode  = p >= 0.0 ? std::log((p + curvature) / psi) : std::log(chi / (curvature - p));
    const double width = std::min(1.0, 1.0 / std::sqrt(curvature));
    // At p = 0 the two tails are symmetric; keep their probabilities exactly equal when sorting.
    const double left_term  = p == 0.0 ? curvature : chi * std::exp(-mode);
    const double right_term = p == 0.0 ? curvature : psi * std::exp(mode);
    double left             = -1.0;
    double right            = 1.0;
    double left_slope       = width * (p + (left_term * std::exp(-width * left) - right_term * std::exp(width * left)) / 2.0);
    while (left_slope < 0.5) {
        Rcpp::checkUserInterrupt();
        left -= 1.0;
        left_slope = width * (p + (left_term * std::exp(-width * left) - right_term * std::exp(width * left)) / 2.0);
    }
    if (p == 0.0) right = -left;
    double right_slope = p == 0.0 ? -left_slope :
        width * (p + (left_term * std::exp(-width * right) - right_term * std::exp(width * right)) / 2.0);
    while (right_slope > -0.5) {
        Rcpp::checkUserInterrupt();
        right += 1.0;
        right_slope = width * (p + (left_term * std::exp(-width * right) - right_term * std::exp(width * right)) / 2.0);
    }

    // Calculate the three proposal regions: left tail, middle interval and right tail.
    const double left_height = p * width * left -
        (left_term * std::expm1(-width * left) + right_term * std::expm1(width * left)) / 2.0;
    const double right_height = p == 0.0 ? left_height : p * width * right -
        (left_term * std::expm1(-width * right) + right_term * std::expm1(width * right)) / 2.0;
    const double log_mass[3] = {left_height - std::log(left_slope), std::log(right - left),
                                right_height - std::log(-right_slope)};

    // Keep proposing until the draw passes the exact density-versus-bound comparison.
    while (true) {
        Rcpp::checkUserInterrupt();
        const int region = draw_gig_region(log_mass);
        double candidate, envelope;
        if (region == 0) {
            candidate = left - R::rexp(1.0 / left_slope);
            envelope  = left_height + left_slope * (candidate - left);
        } else if (region == 1) {
            candidate = R::runif(left, right);
            envelope  = 0.0;
        } else {
            candidate = right + R::rexp(1.0 / -right_slope);
            envelope  = right_height + right_slope * (candidate - right);
        }
        const double height = p * width * candidate -
            (left_term * std::expm1(-width * candidate) + right_term * std::expm1(width * candidate)) / 2.0;
        if (std::log(R::runif(0.0, 1.0)) <= height - envelope) return std::exp(mode + width * candidate);
    }
}

// Return one GIG(p, chi, psi) draw through an R-callable function.
// Used to check the distribution and compare random draws with the separate R calculation;
// the full sampler calls draw_shared_gig() directly during the shrinkage update.
// [[Rcpp::export(name = ".pdpmvar_eigen_gig")]]
double pdpmvar_eigen_gig(double p, double chi, double psi) {

    return draw_shared_gig(p, chi, psi);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Draw DP weights and slice variables ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Count subjects at each supplied stick position, including zero counts at empty positions.
// For N subjects, labels has length N. Return counts through the largest label or
// minimum_size, whichever is greater. Do not renumber labels or discard later positions.
// Used when drawing slice weights, extra split/merge weights and shared shrinkage variances.
std::vector<int> count_stick_labels(const Rcpp::NumericVector& labels, int minimum_size) {
    if (labels.size() == 0) Rcpp::stop("Stick labels must be nonempty.");
    int maximum = minimum_size;
    for (double label : labels) {
        if (!std::isfinite(label) || label < 1.0 || std::floor(label) != label || label > std::numeric_limits<int>::max()) {
            Rcpp::stop("Stick labels must be positive integer indices.");
        }
        maximum = std::max(maximum, static_cast<int>(label));
    }
    std::vector<int> counts(maximum, 0);
    for (double label : labels) ++counts[static_cast<int>(label) - 1];

    return counts;
}

// Draw mixture weights and a slice value for each subject, extending the stick until
// the remaining weight is smaller than every slice value. The number of groups is not fixed.
// For N subjects, return N slice values and C available component weights, where C is
// the number needed in this iteration. Empty stick positions keep their original labels.
// Called for both the coefficient and covariance assignments during each MCMC iteration.
// [[Rcpp::export(name = ".pdpmvar_eigen_slice_weights")]]
Rcpp::List pdpmvar_eigen_slice_weights(const Rcpp::NumericVector& labels, double concentration) {
    if (!std::isfinite(concentration) || concentration <= 0.0)
        Rcpp::stop("DP concentration must be positive and finite.");
    const std::vector<int> counts = count_stick_labels(labels);
    std::vector<double> weights;
    weights.reserve(counts.size() + 16);
    double remaining = 1.0;
    int beyond       = labels.size();
    for (int count : counts) {
        Rcpp::checkUserInterrupt();
        beyond -= count;
        const double stick = R::rbeta(1.0 + count, concentration + beyond);
        weights.push_back(remaining * stick);
        remaining *= 1.0 - stick;
    }

    // Draw each slice below its current group weight.
    Rcpp::NumericVector slices(labels.size());
    double minimum_slice = 1.0;
    for (int subject = 0; subject < labels.size(); ++subject) {
        slices[subject] = R::runif(0.0, 1.0) * weights[static_cast<int>(labels[subject]) - 1];
        if (slices[subject] <= 0.0) {
            Rcpp::stop("Slice weights underflowed; rescale the data or adjust the DP concentration.");
        }
        minimum_slice = std::min(minimum_slice, slices[subject]);
    }

    // Continue the stick so even the smallest slice is covered.
    while (remaining >= minimum_slice) {
        Rcpp::checkUserInterrupt();
        const double stick = R::rbeta(1.0, concentration);
        weights.push_back(remaining * stick);
        remaining *= 1.0 - stick;
    }

    return Rcpp::List::create(Rcpp::Named("weights") = Rcpp::wrap(weights), Rcpp::Named("slices") = slices);
}

// Draw log weights for the first slots stick positions, returning a vector of length slots.
// Subjects assigned to later positions still enter the conditional stick-weight calculations.
// Used by coefficient split/merge moves and the direct covariance proposal checks.
// The limit on this additional proposal does not truncate the DP used by ordinary slice sampling.
// [[Rcpp::export(name = ".pdpmvar_eigen_prefix_weights")]]
Rcpp::NumericVector pdpmvar_eigen_prefix_weights(const Rcpp::NumericVector& labels,
                                                 double concentration, int slots) {
    if (!std::isfinite(concentration) || concentration <= 0.0 || slots < 1) {
        Rcpp::stop("Prefix weights need a positive concentration and prefix size.");
    }
    const std::vector<int> counts = count_stick_labels(labels, slots);

    // Later-position counts remain in beyond even though only the first slots weights are returned.
    Rcpp::NumericVector log_weights(slots);
    double log_remaining = 0.0;
    int beyond           = labels.size();
    for (int cluster = 0; cluster < slots; ++cluster) {
        Rcpp::checkUserInterrupt();
        beyond -= counts[cluster];
        const double stick   = R::rbeta(1.0 + counts[cluster], concentration + beyond);
        log_weights[cluster] = log_remaining + std::log(stick);
        log_remaining += std::log1p(-stick);
    }

    return log_weights;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Residual likelihoods and covariance group assignments ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Check a node-by-node covariance matrix, D x D, and return its lower Cholesky matrix,
// also D x D, where D is n_nodes. Require finite entries, symmetry and positive definiteness.
// Used by residual likelihood calculations and when storing Cholesky matrices for repeated use.
Eigen::MatrixXd likelihood_covariance_lower(const Eigen::MatrixXd& covariance, int n_nodes) {
    if (n_nodes == 0 || covariance.rows() != n_nodes || covariance.cols() != n_nodes || !covariance.allFinite()) {
        Rcpp::stop("Covariance must be finite, square and match the residual dimension.");
    }
    const double scale = std::max(1.0, covariance.cwiseAbs().maxCoeff());
    if ((covariance - covariance.transpose()).cwiseAbs().maxCoeff() > 1e-12 * scale) {
        Rcpp::stop("Covariance must be symmetric.");
    }
    Eigen::LLT<Eigen::MatrixXd> factor(covariance);
    if (factor.info() != Eigen::Success) Rcpp::stop("Covariance must be positive definite.");

    return Eigen::MatrixXd(factor.matrixL());
}

// Calculate a residual log likelihood using an already calculated covariance Cholesky matrix.
// Residuals are nodes by observations, D x T, and lower is nodes by nodes, D x D.
// half_log_determinant is the sum of log diagonal entries of lower. Return one log likelihood,
// omitting the common constant -D * T * log(2 * pi) / 2.
// Used for coefficient/covariance assignments and total likelihood traces.
double residual_log_likelihood(const Eigen::MatrixXd& residual, const Eigen::MatrixXd& lower,
                               double half_log_determinant) {
    const Eigen::MatrixXd whitened = lower.triangularView<Eigen::Lower>().solve(residual);

    return -residual.cols() * half_log_determinant - whitened.squaredNorm() / 2.0;
}

// Calculate a group's covariance Cholesky matrix only the first time it is needed.
// For D nodes and C covariance groups, lowers stores C node-by-node matrices, D x D;
// determinants and ready have length C. index counts from zero within C++.
// Used by coefficient and covariance assignments and total likelihood calculations to
// reuse each group's Cholesky matrix across subjects.
void cache_likelihood_covariance(const Rcpp::List& atoms,
                                 int index,
                                 int n_nodes,
                                 std::vector<Eigen::MatrixXd>& lowers,
                                 std::vector<double>& determinants,
                                 std::vector<bool>& ready) {
    if (index < 0 || index >= atoms.size()) Rcpp::stop("Covariance label is outside the atom list.");
    if (ready[index]) return;
    Rcpp::List atom                  = atoms[index];
    const Eigen::MatrixXd covariance = Rcpp::as<Eigen::MatrixXd>(atom["covariance"]);
    lowers[index]                    = likelihood_covariance_lower(covariance, n_nodes);
    determinants[index]              = lowers[index].diagonal().array().log().sum();
    ready[index]                     = true;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Draw eligible group labels and calculate likelihoods ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Draw one label from eligible groups using their log likelihoods and a supplied uniform draw.
// Record the normalized probabilities in one row of a subjects-by-components matrix, N x C,
// where N is subjects and C is available groups. candidates contains C++ positions counted
// from zero; return an R group label counted from one. Consider higher probabilities first
// to match R's weighted sampling rule when there are at most 200 groups.
// Used by the ordinary coefficient and covariance assignment functions.
int sample_slice_candidate(const std::vector<int>& candidates,
                           const std::vector<double>& scores,
                           double uniform,
                           Rcpp::NumericMatrix& probabilities,
                           int subject) {
    if (candidates.empty()) Rcpp::stop("No slice-eligible cluster is available.");
    if (!std::isfinite(uniform) || uniform < 0.0 || uniform > 1.0) {
        Rcpp::stop("Assignment uniforms must be between zero and one.");
    }
    const double maximum = *std::max_element(scores.begin(), scores.end());
    if (!std::isfinite(maximum)) Rcpp::stop("No candidate cluster has a finite likelihood.");
    const int count = candidates.size();
    std::vector<double> probability(count);
    std::vector<int> order(count);
    double total = 0.0;
    for (int index = 0; index < count; ++index) {
        probability[index] = std::exp(scores[index] - maximum);
        total += probability[index];
        order[index] = index;
    }
    for (int index = 0; index < count; ++index) {
        probability[index] /= total;
        probabilities(subject, candidates[index]) = probability[index];
    }

    // Draw from normalized probabilities in the same decreasing-probability order used by R.
    Rf_revsort(probability.data(), order.data(), count);
    double cumulative = 0.0;
    for (int index = 0; index < count - 1; ++index) {
        cumulative += probability[index];
        if (uniform <= cumulative) return candidates[order[index]] + 1;
    }

    return candidates[order[count - 1]] + 1;
}

// Calculate the Gaussian log likelihood for residuals of size nodes by observations, D x T,
// under a node-by-node covariance matrix, D x D. Omit -D * T * log(2 * pi) / 2.
// Called from R when checking likelihoods, evaluating predictions and assigning labels
// with more than 200 available components.
// [[Rcpp::export(name = ".pdpmvar_eigen_log_likelihood")]]
double pdpmvar_eigen_log_likelihood(const Eigen::MatrixXd& residual, const Eigen::MatrixXd& covariance) {
    if (!residual.allFinite()) Rcpp::stop("Residuals must be finite.");
    const Eigen::MatrixXd lower = likelihood_covariance_lower(covariance, residual.rows());

    return residual_log_likelihood(residual, lower, lower.diagonal().array().log().sum());
}

// Calculate each subject's log likelihood under one covariance using residual cross-products.
// Supply N node-by-node matrices, D x D, and N observation counts, where D is nodes and
// N is subjects. Each matrix is Y * Y', the sum of residual cross-products, not a sample
// covariance. Return N log likelihoods, omitting the common Gaussian constants.
// Used to compare direct covariance proposals and to check likelihood calculations from R.
// [[Rcpp::export(name = ".pdpmvar_eigen_scatter_scores")]]
Rcpp::NumericVector pdpmvar_eigen_scatter_scores(const Rcpp::List& scatters,
                                                 const Rcpp::NumericVector& n_times,
                                                 const Eigen::MatrixXd& covariance) {
    const int n_nodes = covariance.rows();
    if (n_times.size() != scatters.size()) Rcpp::stop("Scatter matrices and time lengths must agree.");
    const Eigen::MatrixXd lower = likelihood_covariance_lower(covariance, n_nodes);

    // Reuse one covariance precision for all subject cross-products.
    const Eigen::MatrixXd inverse_lower = lower.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(n_nodes, n_nodes));
    const Eigen::MatrixXd precision     = inverse_lower.transpose() * inverse_lower;
    const double half_log_determinant   = lower.diagonal().array().log().sum();
    Rcpp::NumericVector scores(scatters.size());
    for (int subject = 0; subject < scatters.size(); ++subject) {
        const Eigen::MatrixXd scatter = Rcpp::as<Eigen::MatrixXd>(scatters[subject]);
        if (scatter.rows() != n_nodes || scatter.cols() != n_nodes || !scatter.allFinite() ||
            !std::isfinite(n_times[subject]) || n_times[subject] < 0.0) {
            Rcpp::stop("Scatter dimensions or time lengths are invalid.");
        }
        scores[subject] = -n_times[subject] * half_log_determinant -
                          (precision.array() * scatter.array()).sum() / 2.0;
    }

    return scores;
}

// Draw covariance-group labels from components whose weights exceed each subject's slice value.
// Slice eligibility already accounts for mixture weights; do not multiply the likelihoods
// by those weights again. For N subjects and C available groups, return N labels and
// subjects-by-components log likelihood and probability matrices, N x C.
// Called during each sampler iteration when at most 200 groups are available; the R
// assignment function handles larger numbers of groups.
// [[Rcpp::export(name = ".pdpmvar_eigen_assign_covariances")]]
Rcpp::List pdpmvar_eigen_assign_covariances(const Rcpp::List& residuals,
                                            const Rcpp::List& atoms,
                                            const Rcpp::NumericVector& weights,
                                            const Rcpp::NumericVector& slices,
                                            const Rcpp::NumericVector& uniforms) {
    const int n_subjects = residuals.size();
    const int n_atoms    = atoms.size();
    if (n_subjects == 0 || n_atoms == 0 || weights.size() != n_atoms || slices.size() != n_subjects ||
        uniforms.size() != n_subjects) Rcpp::stop("Covariance assignment inputs must have matching nonempty dimensions.");

    // Store likelihoods for eligible components only; leave the others at minus infinity.
    Rcpp::NumericVector labels(n_subjects);
    Rcpp::NumericMatrix scores(n_subjects, n_atoms), probabilities(n_subjects, n_atoms);
    std::fill(scores.begin(), scores.end(), R_NegInf);
    std::vector<Eigen::MatrixXd> lowers(n_atoms);
    std::vector<double> determinants(n_atoms);
    std::vector<bool> ready(n_atoms, false);
    for (int index = 0; index < weights.size(); ++index) {
        if (!std::isfinite(weights[index]) || weights[index] < 0.0)
            Rcpp::stop("Slice weights must be nonnegative and finite.");
    }
    for (int subject = 0; subject < n_subjects; ++subject) {
        Rcpp::checkUserInterrupt();
        const Eigen::MatrixXd residual = Rcpp::as<Eigen::MatrixXd>(residuals[subject]);
        if (residual.rows() == 0 || !residual.allFinite() || !std::isfinite(slices[subject]) || slices[subject] < 0.0) {
            Rcpp::stop("Residuals and slices must be finite with valid dimensions.");
        }
        std::vector<int> candidates;
        std::vector<double> log_scores;
        for (int cluster = 0; cluster < n_atoms; ++cluster) {
            if (weights[cluster] <= slices[subject]) continue;
            cache_likelihood_covariance(atoms, cluster, residual.rows(), lowers, determinants, ready);
            if (lowers[cluster].rows() != residual.rows())
                Rcpp::stop("All subjects must share the same node dimension.");
            const double score       = residual_log_likelihood(residual, lowers[cluster], determinants[cluster]);
            scores(subject, cluster) = score;
            candidates.push_back(cluster);
            log_scores.push_back(score);
        }
        labels[subject] = sample_slice_candidate(candidates, log_scores, uniforms[subject], probabilities, subject);
    }

    return Rcpp::List::create(Rcpp::Named("labels") = labels, Rcpp::Named("log_likelihood") = scores,
                              Rcpp::Named("probabilities") = probabilities);
}

// Add the subjects' log likelihoods under their currently assigned covariance matrices.
// Supply N residual matrices of size nodes by observations, D x T_i, N group labels and
// the covariance parameter list. T_i may differ between subjects. Omit the common Gaussian constants.
// Called after burn-in to record each retained iteration's total log likelihood.
// [[Rcpp::export(name = ".pdpmvar_eigen_total_likelihood")]]
double pdpmvar_eigen_total_likelihood(const Rcpp::List& residuals,
                                      const Rcpp::NumericVector& labels,
                                      const Rcpp::List& atoms) {
    if (residuals.size() == 0 || labels.size() != residuals.size()) Rcpp::stop("Residuals and labels must agree.");
    std::vector<Eigen::MatrixXd> lowers(atoms.size());
    std::vector<double> determinants(atoms.size());
    std::vector<bool> ready(atoms.size(), false);
    double total = 0.0;
    for (int subject = 0; subject < residuals.size(); ++subject) {
        const double label = labels[subject];
        if (!std::isfinite(label) || std::floor(label) != label || label < 1 || label > atoms.size()) {
            Rcpp::stop("Covariance labels must be valid atom indices.");
        }
        const Eigen::MatrixXd residual = Rcpp::as<Eigen::MatrixXd>(residuals[subject]);
        if (!residual.allFinite()) Rcpp::stop("Residuals must be finite.");
        const int index = static_cast<int>(label) - 1;
        cache_likelihood_covariance(atoms, index, residual.rows(), lowers, determinants, ready);
        if (lowers[index].rows() != residual.rows()) Rcpp::stop("Residual node dimensions must agree.");
        total += residual_log_likelihood(residual, lowers[index], determinants[index]);
    }

    return total;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Draw new group parameters and collect sampling results ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Draw one response-by-predictor coefficient matrix, D x P, from the normal prior,
// where D is nodes and P = D * K is lagged predictors. variance has the same dimensions.
// Draw entries column by column and return the matrix in a list named coefficient.
// Used to create newly needed groups and redraw unused matrices after shrinkage updates.
Rcpp::List sampler_new_coefficient(const Rcpp::NumericMatrix& variance) {
    Rcpp::NumericMatrix coefficient(variance.nrow(), variance.ncol());
    for (int index = 0; index < variance.size(); ++index) {
        coefficient[index] = R::rnorm(0.0, 1.0) * std::sqrt(variance[index]);
    }

    return Rcpp::List::create(Rcpp::Named("coefficient") = coefficient);
}

// Draw one group's covariance parameters from the model prior.
// For D = n_nodes and B = n_factors, return node-by-factor loading matrices, D x B,
// B factor precisions, D residual variances and a node-by-node covariance matrix, D x D.
// Draw all working loading entries before setting those above the diagonal to zero.
// Used to create new covariance groups, redraw unused groups and propose prior draws near
// empty groups during the annealing bridge.
Rcpp::List sampler_new_covariance(int n_nodes, int n_factors, double cov_shape, double cov_rate) {
    Eigen::MatrixXd working(n_nodes, n_factors);
    for (int column = 0; column < n_factors; ++column) {
        for (int row = 0; row < n_nodes; ++row) working(row, column) = R::rnorm(0.0, 1.0);
    }
    for (int column = 0; column < n_factors; ++column) {
        for (int row = 0; row < column; ++row) working(row, column) = 0.0;
    }

    // Gamma draws take scale rather than rate; residual variances are inverse precision draws.
    Rcpp::NumericVector factor_precision(n_factors), variance(n_nodes);
    for (int factor = 0; factor < n_factors; ++factor) factor_precision[factor] = R::rgamma(0.5, 2.0);
    for (int response = 0; response < n_nodes; ++response) variance[response] = 1.0 / R::rgamma(cov_shape, 1.0 / cov_rate);

    // Convert working loadings to reported loadings before calculating the covariance.
    Eigen::MatrixXd loadings = working;
    for (int column = 0; column < n_factors; ++column) {
        const double sign = working(column, column) < 0.0 ? -1.0 : 1.0;
        loadings.col(column) *= sign / std::sqrt(factor_precision[column]);
    }
    Eigen::MatrixXd covariance = loadings * loadings.transpose();
    for (int response = 0; response < n_nodes; ++response) covariance(response, response) += variance[response];

    return Rcpp::List::create(Rcpp::Named("working_loadings") = working,
        Rcpp::Named("factor_precision") = factor_precision, Rcpp::Named("variance") = variance,
        Rcpp::Named("loadings") = loadings, Rcpp::Named("covariance") = covariance);
}

// Count subjects in each existing group without removing empty stick positions.
// For N labels and n_atoms available groups, return n_atoms counts, including zeros.
// Used to check starting labels, identify occupied groups and decide which parameter
// matrices should be redrawn from the prior during the full sampler.
std::vector<int> sampler_counts(const Rcpp::NumericVector& labels, int n_atoms) {
    std::vector<int> counts(n_atoms, 0);
    for (double label : labels) {
        if (!std::isfinite(label) || label < 1.0 || label > n_atoms || std::floor(label) != label) {
            Rcpp::stop("Sampler labels must index existing atoms.");
        }
        ++counts[static_cast<int>(label) - 1];
    }

    return counts;
}

// Collect iteration numbers, move types, acceptance decisions and log acceptance ratios
// into a four-column data frame, with one row for every iteration whether a move was attempted or not.
// Used at the end of the full sampler to return coefficient and covariance move records,
// including the burn-in iterations.
Rcpp::DataFrame sampler_move_trace(const Rcpp::IntegerVector& iterations,
                                   const Rcpp::CharacterVector& types,
                                   const Rcpp::LogicalVector& accepted,
                                   const Rcpp::NumericVector& ratios) {

    return Rcpp::DataFrame::create(Rcpp::Named("iteration") = iterations,
        Rcpp::Named("type") = types, Rcpp::Named("accepted") = accepted, Rcpp::Named("log_ratio") = ratios);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Select subjects and group positions for split/merge moves ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Choose two distinct subjects from n subjects without replacement and return their
// numbers counted from one, matching R's sample.int(n, 2). Call only when n is at least two.
// Used to decide whether coefficient or covariance split/merge steps propose a split or a merge.
Rcpp::IntegerVector move_anchors(int n) {
    const int first        = static_cast<int>(R_unif_index(n));
    const int second_index = static_cast<int>(R_unif_index(n - 1));
    const int second       = second_index == first ? n - 1 : second_index;

    return Rcpp::IntegerVector::create(first + 1, second + 1);
}

// Find empty stick positions among positions 1 through slots, returning their numbers
// counted from one. Keep subject labels as supplied, including labels beyond slots.
// Used when deciding whether a split is possible and selecting its new group position.
std::vector<int> move_empty_slots(const Rcpp::NumericVector& labels, int slots) {
    if (slots < 2) Rcpp::stop("Split/merge prefix must contain at least two positions.");
    std::vector<bool> occupied(slots, false);
    for (double label : labels) {
        if (!std::isfinite(label) || label < 1.0 || label != std::floor(label)) {
            Rcpp::stop("Split/merge labels must be positive integer indices.");
        }
        if (label <= slots) occupied[static_cast<int>(label) - 1] = true;
    }

    // Return the original position numbers, rather than relabeling the available groups.
    std::vector<int> empty;
    for (int cluster = 0; cluster < slots; ++cluster) if (!occupied[cluster]) empty.push_back(cluster + 1);

    return empty;
}

// Check label lengths, the two selected subjects and the stick positions used by a proposed move.
// Each label vector has length N, where N is subjects; anchors contains two R subject numbers.
// Check that selected positions lie within slots and that subjects in the affected groups
// remain assigned to one of the two group positions.
// Used before calculating full-matrix coefficient and direct covariance acceptance ratios.
void move_validate_ratio(const Rcpp::NumericVector& old_labels,
                         const Rcpp::NumericVector& new_labels,
                         const Rcpp::NumericVector& weights,
                         const Rcpp::IntegerVector& anchors,
                         int slots) {
    move_empty_slots(old_labels, slots);
    move_empty_slots(new_labels, slots);
    if (old_labels.size() != new_labels.size() || anchors.size() != 2 || anchors[0] < 1 ||
        anchors[1] < 1 || anchors[0] > old_labels.size() || anchors[1] > old_labels.size() ||
        anchors[0] == anchors[1] || weights.size() < slots) {
        Rcpp::stop("Invalid split/merge labels, anchors or prefix weights.");
    }
    const int h      = old_labels[anchors[0] - 1];
    const bool split = h == old_labels[anchors[1] - 1];
    const int g      = split ? new_labels[anchors[1] - 1] : old_labels[anchors[1] - 1];
    if (h > slots || g > slots || h == g) Rcpp::stop("Move indices must be distinct positions in the prefix.");
    for (int subject = 0; subject < old_labels.size(); ++subject) {
        if (old_labels[subject] != h && old_labels[subject] != g) continue;
        if (new_labels[subject] != h && new_labels[subject] != g)
            Rcpp::stop("Move labels must stay within the affected pair.");
    }
}

// Return subject labels, group parameters and a record of the move in one list.
// The label vector has length N for N subjects; the record gives type, accepted and log_ratio.
// Used by coefficient and covariance split/merge functions for accepted, rejected and skipped moves.
// This helper itself draws no random values and does not change the supplied parameters.
Rcpp::List move_result(const Rcpp::NumericVector& labels,
                       const Rcpp::List& atoms,
                       const char* type,
                       bool accepted,
                       double ratio) {

    return Rcpp::List::create(Rcpp::Named("labels") = labels, Rcpp::Named("atoms") = atoms,
        Rcpp::Named("record") = Rcpp::List::create(Rcpp::Named("type") = type,
            Rcpp::Named("accepted") = accepted, Rcpp::Named("log_ratio") = ratio));
}

// Return subject numbers belonging to group h or g, preserving their original order.
// Numbers returned to callers start at one, as in R. With g = -1, select group h only.
// Used to collect the subjects affected by a coefficient or covariance split/merge move.
Rcpp::IntegerVector move_subjects(const Rcpp::NumericVector& labels, int h, int g) {
    std::vector<int> subjects;
    for (int subject = 0; subject < labels.size(); ++subject) {
        if (labels[subject] == h || labels[subject] == g) subjects.push_back(subject + 1);
    }

    return Rcpp::wrap(subjects);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Construct covariance matrices and proposal centers ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Calculate reported loadings and covariance from working loadings, factor precisions and residual variances.
// For D nodes and B factors, working and reported loadings are D x B, precision has length B,
// variance has length D and the covariance is D x D. Rescale loading columns by the factor
// precisions and choose signs giving a positive diagonal without changing the implied covariance.
// Used after covariance proposals and intermediate covariance updates.
Rcpp::List move_refresh_covariance(const Eigen::MatrixXd& working, const Rcpp::NumericVector& precision,
                                   const Rcpp::NumericVector& variance) {

    // Convert working loadings to reported loadings before calculating the covariance.
    Eigen::MatrixXd loadings = working;
    for (int factor = 0; factor < working.cols(); ++factor) {
        loadings.col(factor) *= (working(factor, factor) < 0.0 ? -1.0 : 1.0) / std::sqrt(precision[factor]);
    }
    Eigen::MatrixXd covariance = loadings * loadings.transpose();
    for (int node = 0; node < working.rows(); ++node) covariance(node, node) += variance[node];

    return Rcpp::List::create(Rcpp::Named("working_loadings") = working,
        Rcpp::Named("factor_precision") = precision, Rcpp::Named("variance") = variance,
        Rcpp::Named("loadings") = loadings, Rcpp::Named("covariance") = covariance);
}

// Approximate a node-by-node covariance estimate, D x D, with node-by-factor loadings,
// D x B, and D residual variances, where B is n_factors. Start factor precisions at one.
// Use the same eigenvalue and QR calculations as R's starting-value function so the
// factor loading orientation agrees in the two calculations.
// Used to choose data-based centers for direct covariance proposals and annealing bridges.
// [[Rcpp::export(name = ".pdpmvar_eigen_covariance_center")]]
Rcpp::List pdpmvar_eigen_covariance_center(const Eigen::MatrixXd& covariance, int n_factors) {
    int n = covariance.rows();
    if (n < 1 || covariance.cols() != n || n_factors < 1 || n_factors > n || !covariance.allFinite()) {
        Rcpp::stop("Invalid covariance center dimensions or values.");
    }
    Eigen::MatrixXd matrix = covariance, vectors(n, n);
    Eigen::VectorXd values(n);
    std::vector<int> support(2 * n);
    int found    = 0, info = 0, lwork = -1, liwork = -1, integer_query = 0;
    double query = 0.0, zero = 0.0;
    int unused   = 0;
    // Ask R's eigenvalue routine how much temporary storage it needs.
    F77_CALL(dsyevr)("V", "A", "L", &n, matrix.data(), &n, &zero, &zero, &unused, &unused,
        &zero, &found, values.data(), vectors.data(), &n, support.data(), &query, &lwork,
        &integer_query, &liwork, &info FCONE FCONE FCONE);
    if (info != 0) Rcpp::stop("Covariance center eigendecomposition workspace query failed.");
    lwork  = static_cast<int>(query);
    liwork = integer_query;
    std::vector<double> work(lwork);
    std::vector<int> iwork(liwork);
    F77_CALL(dsyevr)("V", "A", "L", &n, matrix.data(), &n, &zero, &zero, &unused, &unused,
        &zero, &found, values.data(), vectors.data(), &n, support.data(), work.data(), &lwork,
        iwork.data(), &liwork, &info FCONE FCONE FCONE);
    if (info != 0 || found != n) Rcpp::stop("Covariance center eigendecomposition failed.");

    // Use the largest covariance eigenvalues for the factor loadings and retain positive node variances.
    const double floor_variance    = std::max(covariance.diagonal().mean(), 1.0) * 1e-6;
    const double residual_variance = n_factors < n ? values.head(n - n_factors).mean() : values[0] / 2.0;
    Eigen::MatrixXd loadings(n, n_factors);
    for (int factor = 0; factor < n_factors; ++factor) {
        loadings.col(factor) = vectors.col(n - factor - 1) *
            std::sqrt(std::max(values[n - factor - 1] - residual_variance, floor_variance));
    }

    // Rotate the leading loading rows to lower-triangular form, using the same QR calculation as R.
    Eigen::MatrixXd qr = loadings.topRows(n_factors).transpose();
    Eigen::VectorXd qraux(n_factors), qr_work(2 * n_factors);
    std::vector<int> pivot(n_factors);
    for (int factor = 0; factor < n_factors; ++factor) pivot[factor] = factor + 1;
    double tolerance = 1e-7;
    int rank         = 0;
    F77_CALL(dqrdc2)(qr.data(), &n_factors, &n_factors, &n_factors, &tolerance, &rank,
        qraux.data(), pivot.data(), qr_work.data());
    Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n_factors, n_factors), rotation(n_factors, n_factors);
    F77_CALL(dqrqy)(qr.data(), &n_factors, &rank, qraux.data(), identity.data(), &n_factors, rotation.data());
    loadings = (loadings * rotation).eval();
    for (int column = 0; column < n_factors; ++column) for (int row = 0; row < column; ++row) loadings(row, column) = 0.0;
    Rcpp::NumericVector variance(n), precision(n_factors, 1.0);
    for (int node = 0; node < n; ++node) variance[node] = std::max(covariance(node, node) - loadings.row(node).squaredNorm(), floor_variance);

    return move_refresh_covariance(loadings, precision, variance);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Direct covariance proposals for separate checks ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Propose a covariance near center, drawing factor precisions and residual variances
// and randomly reversing the signs of loading columns. For D nodes and B factors,
// return D x B loadings, B precisions, D residual variances and a D x D covariance.
// Used by the direct covariance proposal retained for separate checks; the current
// fitting sampler uses the annealing bridge's auxiliary proposal instead.
// [[Rcpp::export(name = ".pdpmvar_eigen_propose_covariance")]]
Rcpp::List move_propose_covariance(const Rcpp::List& center) {
    Eigen::MatrixXd working             = Rcpp::as<Eigen::MatrixXd>(center["working_loadings"]);
    Rcpp::NumericVector center_variance = center["variance"];
    if (working.rows() < 1 || working.cols() < 1 || working.cols() > working.rows() ||
        center_variance.size() != working.rows() || !working.allFinite()) Rcpp::stop("Invalid covariance proposal center dimensions or values.");
    for (double value : center_variance) if (!std::isfinite(value) || value <= 0.0) Rcpp::stop("Proposal center variances must be positive and finite.");
    Rcpp::NumericVector precision(working.cols()), variance(working.rows());
    for (int column = 0; column < working.cols(); ++column) {
        Eigen::Index pivot;
        working.col(column).cwiseAbs().maxCoeff(&pivot);
        if (working(pivot, column) < 0.0) working.col(column) *= -1.0;
    }
    for (double& value : precision) value = R::rlnorm(0.0, 0.5);
    for (int column = 0; column < working.cols(); ++column) {
        for (int row = 0; row < working.rows(); ++row) working(row, column) =
            (working(row, column) + R::rnorm(0.0, 0.15)) * std::sqrt(precision[column]);
    }
    for (int column = 0; column < working.cols(); ++column) for (int row = 0; row < column; ++row) working(row, column) = 0.0;
    for (int node = 0; node < working.rows(); ++node) variance[node] = R::rlnorm(std::log(center_variance[node]), 0.25);
    for (int column = 0; column < working.cols(); ++column) {
        if (R::runif(0.0, 1.0) < 0.5) working.col(column) *= -1.0;
    }

    return move_refresh_covariance(working, precision, variance);
}

// Calculate the model prior's log density for the free working loadings, factor precisions
// and residual variances. Return one log density, including normalizing constants.
// The Gamma prior is on inverse residual variance, so include the change-of-variable
// term -2 * log(v) when evaluating density in terms of variance v.
// Used in covariance proposal acceptance calculations and the annealing bridge's prior correction.
double move_covariance_base(const Rcpp::List& atom, double shape, double rate) {
    const Eigen::MatrixXd working       = Rcpp::as<Eigen::MatrixXd>(atom["working_loadings"]);
    const Rcpp::NumericVector precision = atom["factor_precision"], variance = atom["variance"];
    long double loadings_sum            = 0.0, precision_sum = 0.0, variance_sum = 0.0;
    for (int column = 0; column < working.cols(); ++column) for (int row = column; row < working.rows(); ++row) {
        loadings_sum += R::dnorm(working(row, column), 0.0, 1.0, true);
    }
    for (double value : precision) precision_sum += R::dgamma(value, 0.5, 2.0, true);
    for (double value : variance) variance_sum += R::dgamma(1.0 / value, shape, 1.0 / rate, true) - 2.0 * std::log(value);

    return static_cast<double>(loadings_sum) + static_cast<double>(precision_sum) + static_cast<double>(variance_sum);
}

// Calculate the log density of the direct covariance proposal around center.
// For B loading columns, average over both signs separately for each column rather
// than explicitly enumerating all 2^B sign combinations. Loadings are nodes by factors,
// D x B; D is nodes and B is factors. Include precision and residual variance densities.
// Used with move_propose_covariance() to check direct-proposal acceptance probabilities.
// [[Rcpp::export(name = ".pdpmvar_eigen_covariance_log_proposal")]]
double move_covariance_proposal(const Rcpp::List& atom, const Rcpp::List& center) {
    const Eigen::MatrixXd working        = Rcpp::as<Eigen::MatrixXd>(atom["working_loadings"]);
    const Eigen::MatrixXd center_working = Rcpp::as<Eigen::MatrixXd>(center["working_loadings"]);
    const Rcpp::NumericVector precision  = atom["factor_precision"], variance = atom["variance"], center_variance = center["variance"];
    if (working.rows() < 1 || working.cols() < 1 || working.cols() > working.rows() ||
        center_working.rows() != working.rows() || center_working.cols() != working.cols() ||
        precision.size() != working.cols() || variance.size() != working.rows() ||
        center_variance.size() != working.rows() || !working.allFinite() || !center_working.allFinite()) {
        Rcpp::stop("Invalid covariance proposal atom or center dimensions or values.");
    }
    for (double value : precision) if (!std::isfinite(value) || value <= 0.0) Rcpp::stop("Proposal factor precisions must be positive and finite.");
    for (int node = 0; node < variance.size(); ++node) {
        if (!std::isfinite(variance[node]) || variance[node] <= 0.0 || !std::isfinite(center_variance[node]) ||
            center_variance[node] <= 0.0) Rcpp::stop("Proposal variances must be positive and finite.");
    }
    long double loadings_sum = 0.0, precision_sum = 0.0, variance_sum = 0.0;

    // Average positive and negative loading-column densities on the log scale.
    for (int column = 0; column < working.cols(); ++column) {
        const double scale       = std::sqrt(precision[column]);
        long double positive_sum = 0.0, negative_sum = 0.0;
        for (int row = column; row < working.rows(); ++row) {
            const double mean = center_working(row, column) * scale;
            positive_sum += R::dnorm(working(row, column), mean, 0.15 * scale, true);
            negative_sum += R::dnorm(working(row, column), -mean, 0.15 * scale, true);
        }
        const double positive = static_cast<double>(positive_sum);
        const double negative = static_cast<double>(negative_sum);
        const double maximum  = std::max(positive, negative);
        loadings_sum += maximum == R_NegInf ? R_NegInf : maximum +
            std::log(std::exp(positive - maximum) + std::exp(negative - maximum)) - std::log(2.0);
    }
    for (double value : precision) precision_sum += R::dlnorm(value, 0.0, 0.5, true);
    for (int node = 0; node < variance.size(); ++node) variance_sum += R::dlnorm(variance[node], std::log(center_variance[node]), 0.25, true);

    return static_cast<double>(loadings_sum) + static_cast<double>(precision_sum) + static_cast<double>(variance_sum);
}

// Calculate the log probability of specified subject labels between two proposed groups.
// For M selected subjects, first and second contain M log assignment scores; subjects
// contains their R subject numbers. Normalize each pair of scores before adding its
// chosen label's log probability. Draw no new labels.
// Used by the direct covariance proposal's forward/reverse probability calculation.
double move_allocation_probability(const Rcpp::NumericVector& labels,
                                   const Rcpp::IntegerVector& subjects,
                                   const Rcpp::NumericVector& first,
                                   const Rcpp::NumericVector& second,
                                   int h) {
    long double probability = 0.0;
    for (int index = 0; index < subjects.size(); ++index) {
        const double maximum = std::max(first[index], second[index]);
        probability += (labels[subjects[index] - 1] == h ? first[index] : second[index]) -
            (maximum + std::log(std::exp(first[index] - maximum) + std::exp(second[index] - maximum)));
    }

    return static_cast<double>(probability);
}

// Calculate selected subjects' log likelihoods under one covariance and add its log stick weight.
// Supply residual cross-products of size nodes by nodes, D x D, and observation counts.
// Return one score per selected subject, in the supplied subject order; D is nodes.
// The parameter named weight is already a log weight.
// Used by the direct covariance proposal and its acceptance-ratio checks.
Rcpp::NumericVector move_covariance_scores(const Rcpp::List& scatters,
                                           const Rcpp::NumericVector& times,
                                           const Rcpp::IntegerVector& subjects,
                                           const Rcpp::List& atom,
                                           double weight) {
    Rcpp::List selected(subjects.size());
    Rcpp::NumericVector selected_times(subjects.size());
    for (int index = 0; index < subjects.size(); ++index) {
        selected[index]       = scatters[subjects[index] - 1];
        selected_times[index] = times[subjects[index] - 1];
    }
    Rcpp::NumericVector scores = pdpmvar_eigen_scatter_scores(selected, selected_times, Rcpp::as<Eigen::MatrixXd>(atom["covariance"]));
    for (double& score : scores) score += weight;

    return scores;
}

// Calculate the log acceptance ratio for a direct covariance split/merge at fixed stick positions.
// Include subject likelihoods, covariance prior densities, proposal densities, assignment
// probabilities and the chance of choosing an empty stick position. Label vectors have
// length N and residual cross-products are D x D, for N subjects and D nodes.
// Return the log acceptance ratio and its target, forward-proposal and reverse-proposal terms.
// Used for direct-proposal checks; the fitted sampler uses the annealing bridge calculation.
// [[Rcpp::export(name = ".pdpmvar_eigen_covariance_move_ratio")]]
Rcpp::List pdpmvar_eigen_covariance_move_ratio(const Rcpp::NumericVector& old_labels,
                                               const Rcpp::NumericVector& new_labels,
                                               const Rcpp::List& old_atoms,
                                               const Rcpp::List& new_atoms,
                                               const Rcpp::NumericVector& log_weights,
                                               const Rcpp::List& scatters,
                                               const Rcpp::NumericVector& n_times,
                                               const Rcpp::IntegerVector& anchors,
                                               const Rcpp::List& center,
                                               int slots,
                                               double cov_shape,
                                               double cov_rate) {
    move_validate_ratio(old_labels, new_labels, log_weights, anchors, slots);
    if (scatters.size() != old_labels.size() || n_times.size() != old_labels.size() ||
        old_atoms.size() < slots || new_atoms.size() < slots) {
        Rcpp::stop("Covariance scatters, atoms and move labels must agree.");
    }
    const int h                        = old_labels[anchors[0] - 1];
    const bool split                   = h == old_labels[anchors[1] - 1];
    const int g                        = split ? new_labels[anchors[1] - 1] : old_labels[anchors[1] - 1];
    const Rcpp::IntegerVector subjects = move_subjects(old_labels, h, g);
    const Rcpp::NumericVector h_scores = move_covariance_scores(scatters, n_times, subjects, old_atoms[h - 1], log_weights[h - 1]);
    const Rcpp::NumericVector old_g    = move_covariance_scores(scatters, n_times, subjects, old_atoms[g - 1], log_weights[g - 1]);
    const Rcpp::NumericVector new_g    = move_covariance_scores(scatters, n_times, subjects, new_atoms[g - 1], log_weights[g - 1]);
    const double old_base              = move_covariance_base(old_atoms[g - 1], cov_shape, cov_rate);
    const double new_base              = move_covariance_base(new_atoms[g - 1], cov_shape, cov_rate);
    long double score_change           = 0.0;
    std::vector<int> others;
    std::vector<double> first_scores, second_scores;
    for (int index = 0; index < subjects.size(); ++index) {
        const int subject = subjects[index];
        score_change += (new_labels[subject - 1] == h ? h_scores[index] : new_g[index]) -
            (old_labels[subject - 1] == h ? h_scores[index] : old_g[index]);
        if (subject == anchors[0] || subject == anchors[1]) continue;
        others.push_back(subject);
        first_scores.push_back(h_scores[index]);
        second_scores.push_back(split ? new_g[index] : old_g[index]);
    }

    // Include both the model prior change and the forward/reverse proposal densities.
    const double target          = static_cast<double>(score_change) + new_base - old_base;
    const std::vector<int> empty = move_empty_slots(split ? old_labels : new_labels, slots);
    const double allocation = move_allocation_probability(split ? new_labels : old_labels,
        Rcpp::wrap(others), Rcpp::wrap(first_scores), Rcpp::wrap(second_scores), h);
    const double forward = split ? -std::log(static_cast<double>(empty.size())) +
        move_covariance_proposal(new_atoms[g - 1], center) + allocation : new_base;
    const double reverse = split ? old_base : -std::log(static_cast<double>(empty.size())) +
        move_covariance_proposal(old_atoms[g - 1], center) + allocation;

    return Rcpp::List::create(Rcpp::Named("log_ratio") = target + reverse - forward,
        Rcpp::Named("target_change") = target, Rcpp::Named("log_forward") = forward, Rcpp::Named("log_reverse") = reverse);
}

// Propose and accept or reject a direct covariance split/merge without an annealing bridge.
// For N subjects and D nodes, input residuals are D x T_i, where T_i is subject i's
// observation count. Return N labels, updated covariance parameters and a move record.
// Retained for separate probability and R/C++ checks; not called by the current fitting sampler.
// [[Rcpp::export(name = ".pdpmvar_eigen_covariance_move_legacy")]]
Rcpp::List pdpmvar_eigen_covariance_move_legacy(const Rcpp::List& residuals,
                                                const Rcpp::NumericVector& labels,
                                                const Rcpp::List& atoms,
                                                double concentration,
                                                int n_factors,
                                                double cov_shape,
                                                double cov_rate,
                                                int slots) {
    const std::vector<int> empty = move_empty_slots(labels, slots);
    for (double label : labels) if (label > atoms.size()) Rcpp::stop("Move labels must index existing atoms.");
    if (labels.size() < 2) return move_result(labels, atoms);
    if (residuals.size() != labels.size()) Rcpp::stop("Residuals and split/merge labels must agree.");
    const Rcpp::IntegerVector anchors = move_anchors(labels.size());
    const int h                       = labels[anchors[0] - 1];
    int g                             = labels[anchors[1] - 1];
    if (h > slots || g > slots) return move_result(labels, atoms);
    const bool split = h == g;
    if (split && empty.empty()) return move_result(labels, atoms);
    const Rcpp::NumericVector log_weights = pdpmvar_eigen_prefix_weights(labels, concentration, slots);
    const Eigen::MatrixXd first_residual  = Rcpp::as<Eigen::MatrixXd>(residuals[0]);
    const int n_nodes                     = first_residual.rows();
    Rcpp::List expanded_atoms             = Rcpp::clone(atoms);
    while (expanded_atoms.size() < slots) expanded_atoms.push_back(sampler_new_covariance(n_nodes, n_factors, cov_shape, cov_rate));
    Rcpp::List scatters(residuals.size());
    Rcpp::NumericVector times(residuals.size());
    for (int subject = 0; subject < residuals.size(); ++subject) {
        const Eigen::MatrixXd residual = Rcpp::as<Eigen::MatrixXd>(residuals[subject]);
        scatters[subject]              = Eigen::MatrixXd(residual * residual.transpose());
        times[subject]                 = residual.cols();
    }
    const Eigen::MatrixXd anchor_scatter = Rcpp::as<Eigen::MatrixXd>(scatters[anchors[1] - 1]);
    const Rcpp::List center              = pdpmvar_eigen_covariance_center(anchor_scatter / times[anchors[1] - 1], n_factors);
    Rcpp::NumericVector proposed         = Rcpp::clone(labels);
    Rcpp::List proposed_atoms            = Rcpp::clone(expanded_atoms);
    if (split) {
        g                                  = empty[static_cast<int>(R_unif_index(empty.size()))];
        proposed_atoms[g - 1]              = move_propose_covariance(center);
        proposed[anchors[1] - 1]           = g;
        const Rcpp::IntegerVector subjects = move_subjects(labels, h);
        const Rcpp::NumericVector first    = move_covariance_scores(scatters, times, subjects, expanded_atoms[h - 1], log_weights[h - 1]);
        const Rcpp::NumericVector second   = move_covariance_scores(scatters, times, subjects, proposed_atoms[g - 1], log_weights[g - 1]);
        for (int index = 0; index < subjects.size(); ++index) {
            const int subject = subjects[index];
            if (subject == anchors[0] || subject == anchors[1]) continue;
            const double maximum    = std::max(first[index], second[index]);
            const double mass_first = std::exp(first[index] - maximum), mass_second = std::exp(second[index] - maximum);
            const double uniform    = R::runif(0.0, 1.0);
            const bool choose_first = mass_first >= mass_second ? uniform <= mass_first / (mass_first + mass_second) :
                uniform > mass_second / (mass_first + mass_second);
            proposed[subject - 1] = choose_first ? h : g;
        }
    } else {
        for (int subject = 0; subject < labels.size(); ++subject) if (labels[subject] == g) proposed[subject] = h;
        proposed_atoms[g - 1] = sampler_new_covariance(n_nodes, n_factors, cov_shape, cov_rate);
    }
    const Rcpp::List ratio = pdpmvar_eigen_covariance_move_ratio(labels, proposed, expanded_atoms, proposed_atoms,
        log_weights, scatters, times, anchors, center, slots, cov_shape, cov_rate);
    const double log_ratio = Rcpp::as<double>(ratio["log_ratio"]);
    if (std::isnan(log_ratio)) Rcpp::stop("Non-finite split/merge acceptance calculation.");
    const bool accepted = std::log(R::runif(0.0, 1.0)) < std::min(0.0, log_ratio);

    return move_result(accepted ? proposed : labels, accepted ? proposed_atoms : expanded_atoms,
        split ? "split" : "merge", accepted, log_ratio);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Draw latent factors and update covariance parameters ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Read working loadings, residual variances and factor precisions from a group's parameter list.
// For D nodes, B factors and T observations, residuals are D x T and loadings are D x B;
// variances have length D and factor precisions have length B. Check matching dimensions,
// finite entries and positive variances and precisions before drawing Gaussian values.
// Used by factor draws and ordinary covariance updates.
void read_covariance_state(const Rcpp::List& atom,
                           const Eigen::MatrixXd& residual,
                           Eigen::MatrixXd& working,
                           Eigen::VectorXd& variance,
                           Eigen::VectorXd& factor_precision) {
    working          = Rcpp::as<Eigen::MatrixXd>(atom["working_loadings"]);
    variance         = Rcpp::as<Eigen::VectorXd>(atom["variance"]);
    factor_precision = Rcpp::as<Eigen::VectorXd>(atom["factor_precision"]);
    if (residual.rows() == 0 || residual.cols() == 0 || working.rows() != residual.rows() ||
        working.cols() == 0 || working.cols() > working.rows() ||
        variance.size() != working.rows() || factor_precision.size() != working.cols()) {
        Rcpp::stop("Residuals and covariance atom dimensions must agree.");
    }
    if (!residual.allFinite() || !working.allFinite() || !variance.allFinite() ||
        !factor_precision.allFinite() || (variance.array() <= 0.0).any() ||
        (factor_precision.array() <= 0.0).any()) {
        Rcpp::stop("Covariance state must be finite with positive residual variances and factor precisions.");
    }
}

// Draw all latent factors conditional on residuals and the group's covariance parameters.
// For D nodes, B factors and T observations, residuals are D x T, working loadings are
// D x B and supplied standard normals are B x T. Return a factors-by-observations matrix,
// B x T. Its Gaussian precision is diag(factor_precision) plus the loading cross-product
// weighted by inverse node variances.
// Used by ordinary covariance updates and pdpmvar_eigen_factors().
Eigen::MatrixXd draw_working_factors(const Eigen::MatrixXd& residual,
                                     const Eigen::MatrixXd& working,
                                     const Eigen::VectorXd& variance,
                                     const Eigen::VectorXd& factor_precision,
                                     const Eigen::MatrixXd& standard_normal) {
    if (standard_normal.rows() != working.cols() || standard_normal.cols() != residual.cols() ||
        !standard_normal.allFinite()) {
        Rcpp::stop("Factor standard normals must match factors by time points and be finite.");
    }

    // Conditional factor precision is Q plus the variance-weighted loading cross-product.
    Eigen::MatrixXd weighted = working;
    for (int row = 0; row < working.rows(); ++row) weighted.row(row) /= variance[row];
    Eigen::MatrixXd precision = working.transpose() * weighted;
    precision.diagonal() += factor_precision;
    Eigen::LLT<Eigen::MatrixXd> factor(precision);
    if (factor.info() != Eigen::Success) Rcpp::stop("Factor precision must be positive definite.");
    Eigen::MatrixXd information = weighted.transpose() * residual;
    Eigen::MatrixXd mean        = factor.solve(information);

    return mean + factor.matrixU().solve(standard_normal);
}

// Draw latent factors using the group's parameter list and supplied standard normal values.
// For D nodes, B factors and T observations, residuals are D x T, normal values are B x T
// and the returned factors are B x T. Check the parameter list before calculating the draw.
// Called before coefficient updates and factor-conditional coefficient split/merge proposals.
// [[Rcpp::export(name = ".pdpmvar_eigen_factors")]]
Eigen::MatrixXd pdpmvar_eigen_factors(const Eigen::MatrixXd& residual,
                                      const Rcpp::List& atom,
                                      const Eigen::MatrixXd& standard_normal) {
    Eigen::MatrixXd working;
    Eigen::VectorXd variance, factor_precision;
    read_covariance_state(atom, residual, working, variance, factor_precision);

    return draw_working_factors(residual, working, variance, factor_precision, standard_normal);
}

// Draw one covariance group's loadings, factor precisions and residual variances conditional on its residuals.
// For D nodes, B factors and T pooled observations, residuals are D x T; return D x B
// loadings, B factor precisions, D residual variances and a D x D covariance matrix.
// Draw factors first, then the free loading entries, factor precisions and residual variances.
// Called for each occupied covariance group during every MCMC iteration.
// [[Rcpp::export(name = ".pdpmvar_eigen_covariance_update")]]
Rcpp::List pdpmvar_eigen_covariance_update(const Eigen::MatrixXd& residual,
                                           const Rcpp::List& atom,
                                           double cov_shape, double cov_rate) {
    if (!std::isfinite(cov_shape) || !std::isfinite(cov_rate) || cov_shape <= 0.0 || cov_rate <= 0.0) {
        Rcpp::stop("Covariance prior shape and rate must be positive and finite.");
    }
    Eigen::MatrixXd working;
    Eigen::VectorXd variance, factor_precision;
    read_covariance_state(atom, residual, working, variance, factor_precision);
    const int n_nodes   = working.rows();
    const int n_factors = working.cols();
    const int n_times   = residual.cols();
    for (int column = 0; column < n_factors; ++column) {
        for (int row = 0; row < column; ++row) {
            if (working(row, column) != 0.0) Rcpp::stop("Working loadings must be lower triangular.");
        }
    }

    // Draw factors for the pooled residuals before updating the free loading entries.
    Eigen::MatrixXd standard_normal(n_factors, n_times);
    for (int column = 0; column < n_times; ++column) {
        for (int row = 0; row < n_factors; ++row) standard_normal(row, column) = R::rnorm(0.0, 1.0);
    }
    Eigen::MatrixXd factors            = draw_working_factors(residual, working, variance, factor_precision, standard_normal);
    const Eigen::MatrixXd factor_cross = factors * factors.transpose();
    for (int response = 0; response < n_nodes; ++response) {
        Rcpp::checkUserInterrupt();
        const int n_free          = std::min(response + 1, n_factors);
        Eigen::MatrixXd precision = factor_cross.topLeftCorner(n_free, n_free) / variance[response];
        precision.diagonal().array() += 1.0;
        const Eigen::VectorXd information = factors.topRows(n_free) * residual.row(response).transpose() / variance[response];
        Eigen::LLT<Eigen::MatrixXd> factor(precision);
        if (factor.info() != Eigen::Success) Rcpp::stop("Loading precision must be positive definite.");
        Eigen::VectorXd noise(n_free);
        for (int index = 0; index < n_free; ++index) noise[index] = R::rnorm(0.0, 1.0);
        const Eigen::VectorXd draw         = factor.solve(information) + factor.matrixU().solve(noise);
        working.row(response).head(n_free) = draw.transpose();
    }
    // R::rgamma takes a scale, so pass 1 / rate for these Gamma conditional draws.
    for (int factor = 0; factor < n_factors; ++factor) {
        const double rate        = (1.0 + factors.row(factor).squaredNorm()) / 2.0;
        factor_precision[factor] = R::rgamma((1.0 + n_times) / 2.0, 1.0 / rate);
    }

    // Draw node residual variances from their inverse-Gamma conditional distributions.
    const Eigen::MatrixXd errors = residual - working * factors;
    for (int response = 0; response < n_nodes; ++response) {
        const double rate  = cov_rate + errors.row(response).squaredNorm() / 2.0;
        variance[response] = 1.0 / R::rgamma(cov_shape + n_times / 2.0, 1.0 / rate);
    }
    if (!factor_precision.allFinite() || !variance.allFinite() ||
        (factor_precision.array() <= 0.0).any() || (variance.array() <= 0.0).any()) {
        Rcpp::stop("Covariance Gamma draws overflowed or underflowed.");
    }

    // Convert working loadings to reported loadings before calculating the covariance.
    Eigen::MatrixXd loadings = working;
    for (int column = 0; column < n_factors; ++column) {
        const double sign = working(column, column) < 0.0 ? -1.0 : 1.0;
        loadings.col(column) *= sign / std::sqrt(factor_precision[column]);
    }
    Eigen::MatrixXd covariance = loadings * loadings.transpose();
    covariance.diagonal() += variance;
    Rcpp::NumericVector variance_result(n_nodes), precision_result(n_factors);
    std::copy(variance.data(), variance.data() + n_nodes, variance_result.begin());
    std::copy(factor_precision.data(), factor_precision.data() + n_factors, precision_result.begin());
    Rcpp::List result          = Rcpp::clone(atom);
    result["working_loadings"] = working;
    result["factor_precision"] = precision_result;
    result["variance"]         = variance_result;
    result["loadings"]         = loadings;
    result["covariance"]       = covariance;

    return result;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Covariance updates for fractionally weighted observations ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Rcpp::List move_refresh_covariance(const Eigen::MatrixXd&, const Rcpp::NumericVector&, const Rcpp::NumericVector&);

// Update covariance parameters from a node-by-node residual cross-product matrix, D x D,
// using an effective observation count that may be fractional. Draw factor cross-products
// rather than a factor vector for each observation, then update loadings and residual
// variances in a reversible order needed by the annealing bridge.
// With B factors, require a positive-definite cross-product matrix and count > D + B + 1.
// Also update factor/loading scales while holding their implied covariance fixed.
// Used within intermediate bridge distributions; factor_shape = 1/2 is the model prior,
// while other values apply only to the bridge's intermediate proposals.
// [[Rcpp::export(name = ".pdpmvar_eigen_covariance_reversible")]]
Rcpp::List pdpmvar_eigen_covariance_reversible(const Rcpp::List& atom,
                                               const Eigen::MatrixXd& scatter,
                                               double n_times,
                                               double cov_shape,
                                               double cov_rate,
                                               double factor_shape = 0.5) {
    Eigen::MatrixXd working       = Rcpp::as<Eigen::MatrixXd>(atom["working_loadings"]);
    Eigen::VectorXd variance      = Rcpp::as<Eigen::VectorXd>(atom["variance"]);
    Eigen::VectorXd old_precision = Rcpp::as<Eigen::VectorXd>(atom["factor_precision"]);
    const int D                   = working.rows(), B = working.cols();
    if (D < 1 || B < 1 || B > D || scatter.rows() != D || scatter.cols() != D || variance.size() != D || old_precision.size() != B ||
        !std::isfinite(n_times) || n_times <= D + B + 1.0 || !std::isfinite(cov_shape) || cov_shape <= 0.0 ||
        !std::isfinite(cov_rate) || cov_rate <= 0.0 || !std::isfinite(factor_shape) || factor_shape <= 0.0 || !working.allFinite() || !scatter.allFinite() || !variance.allFinite() ||
        !old_precision.allFinite() || (variance.array() <= 0.0).any() || (old_precision.array() <= 0.0).any()) {
        Rcpp::stop("Reversible covariance augmentation needs valid dimensions and more than nodes + factors + 1 effective observations.");
    }
    for (int column = 0; column < B; ++column) for (int row = 0; row < column; ++row) {
        if (working(row, column) != 0.0) Rcpp::stop("Working loadings must be lower triangular.");
    }
    // Only intermediate bridge distributions use a factor shape other than the model value 1/2.
    // Redraw factor precisions and rescale loadings together, preserving their implied covariance.
    for (int column = 0; column < B; ++column) {
        working.col(column) /= std::sqrt(old_precision[column]);
        old_precision[column] = R::rgamma(factor_shape + (D - column) / 2.0, 2.0 / (1.0 + working.col(column).squaredNorm()));
        working.col(column) *= std::sqrt(old_precision[column]);
    }

    // Conditional factor precision is Q plus the variance-weighted loading cross-product.
    Eigen::MatrixXd weighted = working;
    for (int row = 0; row < D; ++row) weighted.row(row) /= variance[row];
    Eigen::MatrixXd precision = working.transpose() * weighted;
    precision.diagonal() += old_precision;
    Eigen::LLT<Eigen::MatrixXd> factor(precision), data_factor(scatter);
    if (factor.info() != Eigen::Success || data_factor.info() != Eigen::Success)
        Rcpp::stop("Augmentation precision and scatter must be positive definite.");
    const Eigen::MatrixXd conditional = factor.solve(Eigen::MatrixXd::Identity(B, B));
    Eigen::LLT<Eigen::MatrixXd> conditional_factor(conditional);
    if (conditional_factor.info() != Eigen::Success)
        Rcpp::stop("Conditional factor covariance must be positive definite.");
    const Eigen::MatrixXd lower = conditional_factor.matrixL();
    Eigen::MatrixXd normals(B, D);
    for (int column = 0; column < D; ++column) for (int row = 0; row < B; ++row) normals(row, column) = R::rnorm(0.0, 1.0);
    const Eigen::MatrixXd mean_map = conditional * weighted.transpose();

    // Draw factor-response cross-products and factor cross-products for a possibly fractional count.
    const Eigen::MatrixXd factor_data = mean_map * scatter + lower * normals * Eigen::MatrixXd(data_factor.matrixU());
    Eigen::MatrixXd bartlett          = Eigen::MatrixXd::Zero(B, B);
    for (int column = 0; column < B; ++column) {
        bartlett(column, column) = std::sqrt(R::rgamma((n_times - D - column) / 2.0, 2.0));
        for (int row = column + 1; row < B; ++row) bartlett(row, column) = R::rnorm(0.0, 1.0);
    }
    const Eigen::MatrixXd noise        = lower * bartlett;
    const Eigen::MatrixXd factor_cross = factor_data * data_factor.solve(Eigen::MatrixXd(factor_data.transpose())) + noise * noise.transpose();

    // Update loadings, then variances, then loadings again so the combined update is reversible.
    for (int sweep = 0; sweep < 2; ++sweep) {
        for (int row = 0; row < D; ++row) {
            const int free                        = std::min(row + 1, B);
            Eigen::MatrixXd conditional_precision = factor_cross.topLeftCorner(free, free) / variance[row];
            conditional_precision.diagonal().array() += 1.0;
            const Eigen::VectorXd information = factor_data.col(row).head(free) / variance[row];
            Eigen::LLT<Eigen::MatrixXd> chol(conditional_precision);
            if (chol.info() != Eigen::Success) Rcpp::stop("Reversible loading precision must be positive definite.");
            Eigen::VectorXd normal(free);
            for (int index = 0; index < free; ++index) normal[index] = R::rnorm(0.0, 1.0);
            const Eigen::VectorXd draw  = chol.solve(information) + chol.matrixU().solve(normal);
            working.row(row).head(free) = draw.transpose();
        }
        if (sweep == 0) {
            const Eigen::MatrixXd product = working * factor_cross;
            for (int row = 0; row < D; ++row) {
                const double residual = scatter(row, row) - 2.0 * working.row(row).dot(factor_data.col(row)) +
                    product.row(row).dot(working.row(row));
                variance[row] = 1.0 / R::rgamma(cov_shape + n_times / 2.0, 1.0 / (cov_rate + std::max(residual, 0.0) / 2.0));
            }
        }
    }
    Rcpp::NumericVector factor_precision(B), variance_result(D);
    for (int column = 0; column < B; ++column) factor_precision[column] = R::rgamma(factor_shape + n_times / 2.0, 2.0 / (1.0 + factor_cross(column, column)));
    for (int column = 0; column < B; ++column) {
        working.col(column) /= std::sqrt(factor_precision[column]);
        factor_precision[column] = R::rgamma(factor_shape + (D - column) / 2.0, 2.0 / (1.0 + working.col(column).squaredNorm()));
        working.col(column) *= std::sqrt(factor_precision[column]);
    }
    std::copy(variance.data(), variance.data() + D, variance_result.begin());

    return move_refresh_covariance(working, factor_precision, variance_result);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Auxiliary covariance proposals for the annealing bridge ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Rcpp::List move_refresh_covariance(const Eigen::MatrixXd&, const Rcpp::NumericVector&, const Rcpp::NumericVector&);
Rcpp::List pdpmvar_eigen_covariance_reversible(const Rcpp::List&, const Eigen::MatrixXd&, double, double, double, double);
double move_covariance_base(const Rcpp::List&, double, double);
double bridge_likelihood(const Rcpp::List&, const Eigen::MatrixXd&, double);

// Set the auxiliary inverse-Gamma variance scale to 15 times the mean variance in center.
// Require positive, finite variances and return one positive scale value.
// Used by both bridge_propose_covariance() and bridge_covariance_density() so forward
// and reverse calculations use exactly the same data-based proposal distribution.
double bridge_variance_rate(const Rcpp::List& center) {
    const Rcpp::NumericMatrix working  = center["working_loadings"];
    const Rcpp::NumericVector variance = center["variance"];
    if (working.nrow() < 1 || working.ncol() < 1 || working.ncol() > working.nrow() || variance.size() != working.nrow()) {
        Rcpp::stop("Invalid auxiliary covariance center dimensions.");
    }
    double sum = 0.0;
    for (double value : variance) {
        if (!std::isfinite(value) || value <= 0.0)
            Rcpp::stop("Auxiliary center variances must be positive and finite.");
        sum += value / variance.size();
    }
    if (!std::isfinite(15.0 * sum)) Rcpp::stop("Auxiliary center variance rate must be finite.");

    return 15.0 * sum;
}

// Draw covariance parameters from the prior specified for one intermediate bridge distribution.
// For D nodes and B factors, return D x B loadings, B precisions, D residual variances
// and a D x D covariance. Working loadings have standard normal priors; shape, rate
// and factor_shape specify the residual-precision and factor-precision Gamma distributions.
// Used when drawing the auxiliary proposal or when an intermediate group has too little
// weighted data for the factor cross-product update. Model priors at the fitted endpoint are unchanged.
Rcpp::List bridge_prior_atom(int D,
                             int B,
                             double shape,
                             double rate,
                             double factor_shape) {
    Eigen::MatrixXd working(D, B);
    for (int column = 0; column < B; ++column) for (int row = 0; row < D; ++row) working(row, column) = R::rnorm(0.0, 1.0);
    for (int column = 0; column < B; ++column) for (int row = 0; row < column; ++row) working(row, column) = 0.0;
    Rcpp::NumericVector precision(B), variance(D);
    for (int column = 0; column < B; ++column) precision[column] = R::rgamma(factor_shape, 2.0);
    for (int row = 0; row < D; ++row) variance[row] = 1.0 / R::rgamma(shape, 1.0 / rate);

    return move_refresh_covariance(working, precision, variance);
}

// Draw the auxiliary covariance proposal used when opening a new group in an annealing bridge.
// For D nodes and B factors, return D x B working loadings, B precisions, D variances
// and a D x D covariance. Loading entries are standard normal, precisions follow
// Gamma(2, 1/2), and residual variances have inverse-Gamma shape 16 and scale
// 15 * mean(center$variance). These proposal distributions do not change the model priors.
// Called by pdpmvar_eigen_covariance_bridge() at the split proposal's starting point.
// [[Rcpp::export(name = ".pdpmvar_eigen_bridge_propose")]]
Rcpp::List bridge_propose_covariance(const Rcpp::List& center) {
    const double rate                 = bridge_variance_rate(center);
    const Rcpp::NumericMatrix working = center["working_loadings"];

    return bridge_prior_atom(working.nrow(), working.ncol(), 16.0, rate, 2.0);
}

// Calculate the auxiliary proposal's log density, including every normalizing constant,
// for the supplied working loadings, factor precisions and residual variances.
// Used in pdpmvar_eigen_covariance_bridge() to account for the difference between
// the auxiliary proposal and the model prior in the final acceptance probability.
// [[Rcpp::export(name = ".pdpmvar_eigen_bridge_density")]]
double bridge_covariance_density(const Rcpp::List& atom, const Rcpp::List& center) {
    const double rate                   = bridge_variance_rate(center);
    const Eigen::MatrixXd working       = Rcpp::as<Eigen::MatrixXd>(atom["working_loadings"]);
    const Rcpp::NumericVector precision = atom["factor_precision"], variance = atom["variance"];
    if (working.rows() < 1 || working.cols() < 1 || working.cols() > working.rows() ||
        precision.size() != working.cols() || variance.size() != working.rows() || !working.allFinite()) {
        Rcpp::stop("Invalid auxiliary covariance atom dimensions or values.");
    }
    for (double v : variance) if (!std::isfinite(v) || v <= 0.0) Rcpp::stop("Auxiliary variances must be positive and finite.");
    for (double q : precision) if (!std::isfinite(q) || q <= 0.0) Rcpp::stop("Auxiliary factor precisions must be positive and finite.");
    double value = move_covariance_base(atom, 16.0, rate);
    for (double q : precision) {
        value += R::dgamma(q, 2.0, 2.0, true) - R::dgamma(q, 0.5, 2.0, true);
    }

    return value;
}

// Update covariance parameters at one step between the auxiliary proposal (beta = 0)
// and the model prior with the full group likelihood (beta = 1). Weight the residual
// cross-products and observation count by beta, and combine the proposal and model
// prior parameters accordingly. scatter is nodes by nodes, D x D; B is factors.
// Use reversible factor cross-product updates when possible, otherwise propose from
// the intermediate prior and accept or reject using the weighted likelihood ratio.
// Called along the bridge for the group that a split creates or a merge removes.
// [[Rcpp::export(name = ".pdpmvar_eigen_bridge_auxiliary")]]
Rcpp::List bridge_auxiliary_step(const Rcpp::List& atom,
                                 const Eigen::MatrixXd& scatter,
                                 double count,
                                 double beta,
                                 const Rcpp::List& center,
                                 int B,
                                 double shape,
                                 double rate) {
    const Rcpp::NumericMatrix working = atom["working_loadings"];
    if (!std::isfinite(beta) || beta < 0.0 || beta > 1.0 || !std::isfinite(count) || count < 0.0 ||
        !std::isfinite(shape) || shape <= 0.0 || !std::isfinite(rate) || rate <= 0.0 || B < 1 || B > scatter.rows() ||
        scatter.rows() < 1 || scatter.cols() != scatter.rows() || !scatter.allFinite() ||
        working.nrow() != scatter.rows() || working.ncol() != B) Rcpp::stop("Invalid auxiliary bridge dimensions or controls.");

    // At beta zero use the auxiliary proposal; at beta one recover the fitted model prior.
    const double intermediate_shape        = beta * shape + (1.0 - beta) * 16.0;
    const double intermediate_rate         = beta * rate + (1.0 - beta) * bridge_variance_rate(center);
    const double factor_shape              = beta * 0.5 + (1.0 - beta) * 2.0;
    const Eigen::MatrixXd weighted_scatter = scatter * beta;
    const double weighted_count            = count * beta;
    Eigen::LLT<Eigen::MatrixXd> scatter_check(weighted_scatter);
    if (weighted_count > scatter.rows() + B + 1.0 && scatter_check.info() == Eigen::Success) {
        return pdpmvar_eigen_covariance_reversible(atom, weighted_scatter, weighted_count,
            intermediate_shape, intermediate_rate, factor_shape);
    }

    // With too little weighted data, use a prior draw and a likelihood-only MH correction.
    const Rcpp::List proposed = bridge_prior_atom(scatter.rows(), B, intermediate_shape, intermediate_rate, factor_shape);
    const double ratio        = bridge_likelihood(proposed, weighted_scatter, weighted_count) - bridge_likelihood(atom, weighted_scatter, weighted_count);
    if (std::log(R::runif(0.0, 1.0)) < std::min(0.0, ratio)) return proposed;

    return atom;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Propose subject assignments for covariance splits ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Calculate a subject's inverse-Wishart predictive log score for a proposed covariance group.
// scale and scatter are nodes by nodes, D x D; scatter is the subject's residual cross-product,
// degrees describes the group's inverse-Wishart distribution and n_times is the subject's
// number of observations. Omit constants common to both candidate assignments.
// Used by pdpmvar_eigen_covariance_restricted() only to guide proposed labels;
// this inverse-Wishart distribution is not the fitted covariance prior.
double pool_predictive(const Eigen::MatrixXd& scale, double degrees, const Eigen::MatrixXd& scatter, double n_times) {
    Eigen::LLT<Eigen::MatrixXd> old_chol(scale), new_chol(scale + scatter);
    if (old_chol.info() != Eigen::Success || new_chol.info() != Eigen::Success)
        Rcpp::stop("Restricted covariance proxy must be positive definite.");
    double score = 0.0;
    for (int row = 0; row < scale.rows(); ++row) score += R::lgammafn((degrees + n_times - row) / 2.0) - R::lgammafn((degrees - row) / 2.0);
    const Eigen::MatrixXd old_upper = old_chol.matrixU(), new_upper = new_chol.matrixU();

    return score + degrees * old_upper.diagonal().array().log().sum() -
        (degrees + n_times) * new_upper.diagonal().array().log().sum();
}

// Keep two selected subjects in opposite groups, then assign the other selected subjects
// using inverse-Wishart predictive scores that do not depend on the current covariance parameters.
// For N subjects and D nodes, supply N cross-product matrices of size D x D and N observation
// counts. Return a label vector of length N and the log probability of the selected assignments.
// With forced labels, evaluate those assignments rather than drawing new ones.
// Used to propose covariance splits and evaluate the reverse split probability of a merge.
// [[Rcpp::export(name = ".pdpmvar_eigen_covariance_restricted")]]
Rcpp::List pdpmvar_eigen_covariance_restricted(const Rcpp::List& scatters,
                                               const Rcpp::NumericVector& times,
                                               const Rcpp::IntegerVector& subjects,
                                               const Rcpp::IntegerVector& anchors,
                                               int h,
                                               int g,
                                               Rcpp::Nullable<Rcpp::NumericVector> forced = R_NilValue) {
    if (scatters.size() == 0 || subjects.size() < 2 || anchors.size() != 2 || h < 1 || g < 1 || h == g || times.size() != scatters.size() ||
        anchors[0] < 1 || anchors[1] < 1 || anchors[0] > scatters.size() || anchors[1] > scatters.size() || anchors[0] == anchors[1]) Rcpp::stop("Invalid restricted covariance dimensions or anchors.");
    const int D = Rcpp::as<Rcpp::NumericMatrix>(scatters[0]).nrow();
    std::vector<bool> seen(scatters.size(), false);
    for (int subject : subjects) {
        if (subject < 1 || subject > scatters.size() || seen[subject - 1])
            Rcpp::stop("Restricted subjects must be distinct valid indices.");
        seen[subject - 1]                = true;
        const Rcpp::NumericMatrix matrix = scatters[subject - 1];
        if (matrix.nrow() != D || matrix.ncol() != D || !std::isfinite(times[subject - 1]) || times[subject - 1] <= 0.0)
            Rcpp::stop("Restricted scatter dimensions and times must agree.");
    }
    if (!seen[anchors[0] - 1] || !seen[anchors[1] - 1])
        Rcpp::stop("Restricted anchors must belong to the selected subjects.");
    if (forced.isNotNull() && Rcpp::NumericVector(forced.get()).size() != scatters.size())
        Rcpp::stop("Forced labels must match the subjects.");

    // Use the pooled residual covariance only to guide the assignment proposal.
    Eigen::MatrixXd pooled = Eigen::MatrixXd::Zero(D, D);
    double count           = 0.0;
    for (int subject : subjects) {
        pooled += Rcpp::as<Eigen::MatrixXd>(scatters[subject - 1]);
        count += times[subject - 1];
    }
    pooled *= (D + 1.0) / count;
    Eigen::LLT<Eigen::MatrixXd> proxy_check(pooled);
    if (proxy_check.info() != Eigen::Success) pooled.diagonal().array() += std::max(pooled.trace() / (D * (D + 1.0)), 1.0) * (D + 1.0) * 1e-6;
    Eigen::MatrixXd first  = pooled + Rcpp::as<Eigen::MatrixXd>(scatters[anchors[0] - 1]);
    Eigen::MatrixXd second = pooled + Rcpp::as<Eigen::MatrixXd>(scatters[anchors[1] - 1]);
    double first_count     = times[anchors[0] - 1], second_count = times[anchors[1] - 1];
    const double degrees   = 2.0 * D + 2.0;

    // Hold the selected subjects in opposite groups, then visit the others in the supplied order.
    Rcpp::NumericVector labels(scatters.size(), h);
    labels[anchors[1] - 1]            = g;
    const bool evaluate               = forced.isNotNull();
    Rcpp::NumericVector forced_labels = evaluate ? Rcpp::NumericVector(forced.get()) : Rcpp::NumericVector(0);
    double probability                = 0.0;
    for (int subject : subjects) {
        if (subject == anchors[0] || subject == anchors[1]) continue;
        const Eigen::MatrixXd scatter = Rcpp::as<Eigen::MatrixXd>(scatters[subject - 1]);
        const double a                = pool_predictive(first, degrees + first_count, scatter, times[subject - 1]);
        const double b                = pool_predictive(second, degrees + second_count, scatter, times[subject - 1]);
        const double maximum          = std::max(a, b);
        const double mass_a           = std::exp(a - maximum), mass_b = std::exp(b - maximum);
        bool choose_first;
        if (evaluate) {
            if (forced_labels[subject - 1] != h && forced_labels[subject - 1] != g)
                Rcpp::stop("Forced restricted labels must use the affected pair.");
            choose_first = forced_labels[subject - 1] == h;
        }
        else {
            const double u = R::runif(0.0, 1.0);
            choose_first   = mass_a >= mass_b ? u <= mass_a / (mass_a + mass_b) : u > mass_b / (mass_a + mass_b);
        }
        labels[subject - 1] = choose_first ? h : g;
        probability += (choose_first ? a : b) - maximum - std::log(mass_a + mass_b);
        if (choose_first) {
            first += scatter;
            first_count += times[subject - 1];
        }
        else {
            second += scatter;
            second_count += times[subject - 1];
        }
    }

    return Rcpp::List::create(Rcpp::Named("labels") = labels, Rcpp::Named("log_probability") = probability);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Likelihoods and covariance updates along the bridge ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Calculate a covariance log likelihood from a residual cross-product matrix and observation count:
// -0.5 * (count * log|Sigma| + trace(Sigma^-1 * scatter)), omitting the Gaussian constant.
// Both covariance and scatter are nodes by nodes, D x D. The bridge also passes differences
// of cross-products and counts, so these inputs may be signed rather than actual data totals.
// Used when comparing covariance parameters and adding likelihood changes along the bridge.
// [[Rcpp::export(name = ".pdpmvar_eigen_covariance_weighted_score")]]
double bridge_likelihood(const Rcpp::List& atom, const Eigen::MatrixXd& scatter, double count) {
    const Eigen::MatrixXd covariance = Rcpp::as<Eigen::MatrixXd>(atom["covariance"]);
    if (covariance.rows() < 1 || covariance.cols() != covariance.rows() || scatter.rows() != covariance.rows() ||
        scatter.cols() != covariance.rows() || !scatter.allFinite() || !covariance.allFinite() || !std::isfinite(count)) Rcpp::stop("Invalid bridge covariance, scatter or effective count.");
    Eigen::LLT<Eigen::MatrixXd> chol(covariance);
    if (chol.info() != Eigen::Success) Rcpp::stop("Bridge covariance must be positive definite.");
    const Eigen::MatrixXd upper = chol.matrixU();
    const Eigen::MatrixXd omega = chol.solve(Eigen::MatrixXd::Identity(covariance.rows(), covariance.rows()));

    return -(count * 2.0 * upper.diagonal().array().log().sum() + omega.cwiseProduct(scatter).sum()) / 2.0;
}

// Declare the covariance update before the bridge step that calls it.
Rcpp::List pdpmvar_eigen_covariance_reversible(const Rcpp::List&, const Eigen::MatrixXd&, double, double, double, double);

// Update one covariance group's parameters at an intermediate weighted likelihood.
// scatter is nodes by nodes, D x D; count is its effective observation count and B is factors.
// Use the reversible cross-product update when there is enough weighted data. Otherwise,
// propose from the model prior and accept or reject using the weighted likelihood ratio.
// Called for the first affected group at each intermediate covariance bridge level.
Rcpp::List covariance_bridge_step(const Rcpp::List& atom,
                                  const Eigen::MatrixXd& scatter,
                                  double count,
                                  int B,
                                  double shape,
                                  double rate) {
    const double old_score = bridge_likelihood(atom, scatter, count);
    Rcpp::List proposed;
    double ratio;
    Eigen::LLT<Eigen::MatrixXd> scatter_check(scatter);
    if (count <= scatter.rows() + B + 1.0 || scatter_check.info() != Eigen::Success) {
        proposed = sampler_new_covariance(scatter.rows(), B, shape, rate);
        ratio    = bridge_likelihood(proposed, scatter, count) - old_score;
    } else {
        return pdpmvar_eigen_covariance_reversible(atom, scatter, count, shape, rate, 0.5);
    }
    if (std::log(R::runif(0.0, 1.0)) < std::min(0.0, ratio)) return proposed;

    return atom;
}

Rcpp::List pdpmvar_eigen_covariance_center(const Eigen::MatrixXd&, int);
Rcpp::List move_propose_covariance(const Rcpp::List&);
double move_covariance_proposal(const Rcpp::List&, const Rcpp::List&);
double move_covariance_base(const Rcpp::List&, double, double);

// Update parameters for the earlier direct-proposal version of an auxiliary covariance step.
// Use either a direct proposal or a weighted-data covariance update, with an acceptance
// correction for the difference between proposal and model prior. scatter is D x D,
// where D is nodes; B is factors and beta is the likelihood weight.
// Retained for comparisons with saved calculations; not called by the current fitting sampler.
Rcpp::List covariance_auxiliary_step(const Rcpp::List& atom,
                                     const Eigen::MatrixXd& scatter,
                                     double count,
                                     double beta,
                                     const Rcpp::List& center,
                                     int B,
                                     double shape,
                                     double rate) {
    const double density = move_covariance_proposal(atom, center), base = move_covariance_base(atom, shape, rate);
    Rcpp::List proposed;
    double ratio;
    if (R::runif(0.0, 1.0) < 0.5) {
        proposed = move_propose_covariance(center);
        ratio = beta * (move_covariance_base(proposed, shape, rate) - base - move_covariance_proposal(proposed, center) + density +
            bridge_likelihood(proposed, scatter, count) - bridge_likelihood(atom, scatter, count));
    } else {
        proposed = covariance_bridge_step(atom, scatter * beta, count * beta, B, shape, rate);
        ratio    = (1.0 - beta) * (move_covariance_proposal(proposed, center) - density - move_covariance_base(proposed, shape, rate) + base);
    }
    if (std::log(R::runif(0.0, 1.0)) < std::min(0.0, ratio)) return proposed;

    return atom;
}

Rcpp::List bridge_propose_covariance(const Rcpp::List&);
double bridge_covariance_density(const Rcpp::List&, const Rcpp::List&);
Rcpp::List bridge_auxiliary_step(const Rcpp::List&, const Eigen::MatrixXd&, double, double, const Rcpp::List&, int, double, double);

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Move between current and proposed covariance groups ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Propose covariance parameters through intermediate distributions for two fixed subject groupings.
// For N subjects and D nodes, labels and times have length N and scatters contains N
// node-by-node residual cross-products, D x D. Only group positions h and g change; B is factors.
// Return the parameter list and log_work, which contributes to the full split/merge acceptance
// ratio. Also return the intermediate levels and, when store_path is true, both groups'
// parameters at the saved levels. This function does not decide whether to accept the full move.
// Called by pdpmvar_eigen_covariance_move(); see Karagiannis and Andrieu (2013), JCGS 22:623-648.
// [[Rcpp::export(name = ".pdpmvar_eigen_covariance_bridge")]]
Rcpp::List pdpmvar_eigen_covariance_bridge(const Rcpp::NumericVector& old_labels,
                                           const Rcpp::NumericVector& new_labels,
                                           const Rcpp::List& original_atoms,
                                           const Rcpp::List& scatters,
                                           const Rcpp::NumericVector& times,
                                           int h,
                                           int g,
                                           int B,
                                           double shape,
                                           double rate,
                                           bool store_path = false) {
    if (scatters.size() == 0 || times.size() != scatters.size() || old_labels.size() != scatters.size() || new_labels.size() != scatters.size() ||
        h < 1 || g < 1 || h == g || h > original_atoms.size() || g > original_atoms.size() || B < 1 || !std::isfinite(shape) || shape <= 0.0 || !std::isfinite(rate) || rate <= 0.0) Rcpp::stop("Invalid covariance bridge dimensions or prior controls.");
    for (int subject = 0; subject < old_labels.size(); ++subject) {
        if (!std::isfinite(times[subject]) || times[subject] <= 0.0 || !std::isfinite(old_labels[subject]) || !std::isfinite(new_labels[subject]) ||
            old_labels[subject] < 1.0 || new_labels[subject] < 1.0 || old_labels[subject] != std::floor(old_labels[subject]) || new_labels[subject] != std::floor(new_labels[subject])) Rcpp::stop("Invalid bridge labels or times.");
        if (old_labels[subject] != h && old_labels[subject] != g && new_labels[subject] != old_labels[subject])
            Rcpp::stop("Bridge may only change the affected pair.");
        if ((old_labels[subject] == h || old_labels[subject] == g) && new_labels[subject] != h && new_labels[subject] != g)
            Rcpp::stop("Bridge labels must remain in the affected pair.");
    }
    // Put more intermediate levels near the proposal and posterior ends of the bridge.
    const double schedule[] = {
        0, 6.1441746022147182e-06, 7.8107637250562746e-06, 9.929405711774312e-06, 1.2622714576991388e-05,
        1.6046560238922624e-05, 2.039908727992137e-05, 2.5932178883090014e-05, 3.2966028485380161e-05, 4.1907660962904021e-05,
        5.3274472797878686e-05, 6.7724149619770231e-05, 8.609268612756602e-05, 0.00010944269832050514, 0.00013912480715655816,
        0.00017685561880001287, 0.0002248167702332953, 0.00028578070084187918, 0.00036327031321807779, 0.00046176157652407537,
        0.00058694049608080218, 0.00074602883383669699, 0.00094819664435372543, 0.001205084233778948, 0.0015314617070032849,
        0.0019460609850855582, 0.0024726231566347743, 0.0031412132848294273, 0.0039898651489941039, 0.0050666293334662542,
        0.0064321084669186346, 0.0081625711531598966, 0.010353737531091752, 0.013125318337102799, 0.016626356107881622,
        0.021041347020468282, 0.026596993576865863, 0.033569223281482541, 0.042289771842033828, 0.053151136398063688,
        0.066608035575090624, 0.083172696493922352, 0.10340045145824957, 0.12786156631908133, 0.15709546888545275,
        0.19154534856146757, 0.23147521650098246, 0.27687819487561, 0.32739298293223945, 0.38225212523075086,
        0.44028635073280697, 0.5, 0.55971364926719303, 0.61774787476924908, 0.67260701706776049,
        0.72312180512438995, 0.76852478349901754, 0.80845465143853246, 0.84290453111454722, 0.8721384336809187,
        0.89659954854175039, 0.91682730350607766, 0.93339196442490935, 0.94684886360193632, 0.95771022815796614,
        0.9664307767185174, 0.97340300642313415, 0.97895865297953177, 0.98337364389211834, 0.98687468166289716,
        0.9896462624689083, 0.99183742884684012, 0.99356789153308134, 0.99493337066653376, 0.99601013485100587,
        0.99685878671517059, 0.99752737684336523, 0.99805393901491446, 0.99846853829299675, 0.99879491576622104,
        0.99905180335564625, 0.99925397116616332, 0.99941305950391923, 0.99953823842347589, 0.99963672968678197,
        0.99971421929915816, 0.99977518322976666, 0.99982314438120001, 0.99986087519284339, 0.99989055730167953,
        0.99991390731387242, 0.99993227585038025, 0.9999467255272021, 0.9999580923390371, 0.99996703397151465,
        0.99997406782111686, 0.99997960091272009, 0.9999839534397611, 0.99998737728542297, 0.99999007059428824,
        0.99999218923627498, 0.99999385582539779, 1
    };
    const int stages = sizeof(schedule) / sizeof(schedule[0]);
    const int D      = Rcpp::as<Rcpp::NumericMatrix>(scatters[0]).nrow();
    if (D < 1 || B > D) Rcpp::stop("Invalid bridge factor dimensions.");
    for (int subject = 0; subject < scatters.size(); ++subject) {
        const Rcpp::NumericMatrix matrix = scatters[subject];
        if (matrix.nrow() != D || matrix.ncol() != D) Rcpp::stop("Bridge scatter dimensions must agree.");
    }
    Eigen::MatrixXd old_scatter[2] = {Eigen::MatrixXd::Zero(D, D), Eigen::MatrixXd::Zero(D, D)};
    Eigen::MatrixXd new_scatter[2] = {Eigen::MatrixXd::Zero(D, D), Eigen::MatrixXd::Zero(D, D)};
    double old_count[2]            = {0.0, 0.0}, new_count[2] = {0.0, 0.0};
    const int groups[2]            = {h, g};
    for (int index = 0; index < 2; ++index) for (int subject = 0; subject < old_labels.size(); ++subject) {
        if (old_labels[subject] == groups[index]) { old_scatter[index] += Rcpp::as<Eigen::MatrixXd>(scatters[subject]); old_count[index] += times[subject]; }
        if (new_labels[subject] == groups[index]) { new_scatter[index] += Rcpp::as<Eigen::MatrixXd>(scatters[subject]); new_count[index] += times[subject]; }
    }
    Rcpp::List atoms = Rcpp::clone(original_atoms);

    // Derive the same auxiliary proposal center for a split and its reverse merge.
    const bool split                     = old_count[1] == 0.0;
    const Eigen::MatrixXd active_scatter = split ? new_scatter[1] : old_scatter[1];
    const double active_count            = split ? new_count[1] : old_count[1];
    if (active_count <= 0.0) Rcpp::stop("The split/merge auxiliary group must be nonempty.");
    const Rcpp::List center = pdpmvar_eigen_covariance_center(active_scatter / active_count, B);
    if (split) atoms[g - 1] = bridge_propose_covariance(center);

    // Add log-density changes as the bridge advances, updating both groups between levels.
    Rcpp::List path(store_path ? stages - 1 : 0);
    double work = 0.0;
    for (int stage = 1; stage < stages; ++stage) {
        Rcpp::checkUserInterrupt();
        if (store_path) path[stage - 1] = Rcpp::List::create(atoms[h - 1], atoms[g - 1]);
        double difference = 0.0;
        for (int index = 0; index < 2; ++index) difference += bridge_likelihood(atoms[groups[index] - 1],
            new_scatter[index] - old_scatter[index], new_count[index] - old_count[index]);
        // Account for using an auxiliary proposal in place of the model prior at the empty end.
        difference += (split ? 1.0 : -1.0) * (move_covariance_base(atoms[g - 1], shape, rate) - bridge_covariance_density(atoms[g - 1], center));
        work += (schedule[stage] - schedule[stage - 1]) * difference;
        if (stage == stages - 1) break;
        const double beta = schedule[stage];
        for (int repeat = 0; repeat < 16; ++repeat) {
            for (int index = 0; index < 2; ++index) {
                const Eigen::MatrixXd scatter = (1.0 - beta) * old_scatter[index] + beta * new_scatter[index];
                const double count            = (1.0 - beta) * old_count[index] + beta * new_count[index];
                if (index == 0) atoms[h - 1] = covariance_bridge_step(atoms[h - 1], scatter, count, B, shape, rate);
                else atoms[g - 1] = bridge_auxiliary_step(atoms[g - 1], active_scatter, active_count,
                    split ? beta : 1.0 - beta, center, B, shape, rate);
            }
        }
    }
    Rcpp::RObject output_path = R_NilValue;
    if (store_path) output_path = path;

    Rcpp::NumericVector temperatures(schedule, schedule + stages);

    return Rcpp::List::create(Rcpp::Named("atoms") = atoms, Rcpp::Named("log_work") = work,
        Rcpp::Named("path") = output_path, Rcpp::Named("temperatures") = temperatures);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Accept or reject a covariance split/merge ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Calculate the log prior probability of subject labels with Beta stick variables integrated out.
// For N subjects, labels has length N. Retain the supplied stick positions and any gaps,
// without making a vector as long as the largest label. Return one log probability.
// Used in the covariance split/merge acceptance ratio without changing the DP prior.
// [[Rcpp::export(name = ".pdpmvar_eigen_covariance_label_prior")]]
double pdpmvar_eigen_covariance_label_prior(const Rcpp::NumericVector& labels, double concentration) {
    if (labels.size() == 0 || !std::isfinite(concentration) || concentration <= 0.0)
        Rcpp::stop("Labels and concentration must be nonempty and positive.");
    std::map<int, int> counts;
    for (double label : labels) {
        if (!std::isfinite(label) || label < 1.0 || label != std::floor(label) || label > std::numeric_limits<int>::max())
            Rcpp::stop("Stick labels must be positive integer indices.");
        ++counts[static_cast<int>(label)];
    }
    int remaining     = labels.size(), previous = 0;
    long double total = 0.0;

    // Include empty-position probabilities analytically rather than storing every gap.
    for (const auto& item : counts) {
        const int gap             = item.first - previous - 1;
        const double ratio        = remaining / concentration;
        const double log_gap_mass = std::isfinite(ratio) ? -std::log1p(ratio) : std::log(concentration) - std::log(concentration + remaining);
        double term               = gap == 0 ? 0.0 : gap * log_gap_mass;
        remaining -= item.second;
        term += R::lbeta(1.0 + item.second, concentration + remaining) + std::log(concentration);
        total += term;
        previous = item.first;
    }

    return static_cast<double>(total);
}

// Calculate log probabilities for selecting one of the supplied empty stick positions.
// Give earlier positions more probability according to their prior expected weights,
// then normalize over the available positions. Return a vector with one log probability
// per empty position, in the same order as empty.
// Used when proposing a covariance split and when evaluating its reverse probability.
Rcpp::NumericVector covariance_empty_probabilities(const std::vector<int>& empty, double concentration) {
    Rcpp::NumericVector probability(empty.size());
    double maximum     = R_NegInf;
    const double decay = concentration < 1.0 ? std::log(concentration) - std::log1p(concentration) : -std::log1p(1.0 / concentration);
    for (int index = 0; index < probability.size(); ++index) {
        probability[index] = (empty[index] - 1.0) * decay;
        maximum            = std::max(maximum, probability[index]);
    }
    long double sum = 0.0;
    for (double value : probability) sum += std::exp(value - maximum);
    const double normalizer = maximum + std::log(static_cast<double>(sum));
    for (double& value : probability) value -= normalizer;

    return probability;
}

// Propose splitting one covariance group or merging two, move the affected parameters
// through an annealing bridge, then accept or reject the full proposal.
// For N subjects and D nodes, residual matrices are D x T_i, where T_i is subject i's
// observation count. Return N labels, the updated covariance parameter list and a move record.
// Include the integrated label prior, subject assignments, empty-position choice and bridge
// contribution in the acceptance ratio. After an accepted merge, draw the unused group's
// parameters from the model prior. Called at the requested interval by the full PDPMVAR sampler.
// [[Rcpp::export(name = ".pdpmvar_eigen_covariance_move")]]
Rcpp::List pdpmvar_eigen_covariance_move(const Rcpp::List& residuals,
                                         const Rcpp::NumericVector& labels,
                                         const Rcpp::List& original_atoms,
                                         double concentration,
                                         int B,
                                         double shape,
                                         double rate,
                                         int slots) {
    const std::vector<int> empty = move_empty_slots(labels, slots);
    if (residuals.size() != labels.size()) Rcpp::stop("Residuals and covariance labels must agree.");
    for (double label : labels) if (label > original_atoms.size()) Rcpp::stop("Move labels must index existing atoms.");
    if (labels.size() < 2) return move_result(labels, original_atoms, "skip", false, NA_REAL);
    const Rcpp::IntegerVector anchors = move_anchors(labels.size());
    const int h                       = labels[anchors[0] - 1];
    int g                             = labels[anchors[1] - 1];
    if (h > slots || g > slots || (h == g && empty.empty())) return move_result(labels, original_atoms, "skip", false, NA_REAL);
    const bool split = h == g;
    const int D      = Rcpp::as<Rcpp::NumericMatrix>(residuals[0]).nrow();
    Rcpp::List atoms = Rcpp::clone(original_atoms);
    while (atoms.size() < slots) atoms.push_back(sampler_new_covariance(D, B, shape, rate));
    Rcpp::List scatters(residuals.size());
    Rcpp::NumericVector times(residuals.size());
    for (int subject = 0; subject < residuals.size(); ++subject) {
        const Eigen::MatrixXd residual = Rcpp::as<Eigen::MatrixXd>(residuals[subject]);
        scatters[subject]              = Eigen::MatrixXd(residual * residual.transpose());
        times[subject]                 = residual.cols();
    }

    // Propose new labels, or evaluate the reverse split if this is a merge.
    Rcpp::NumericVector proposed = Rcpp::clone(labels);
    double probability;
    if (split) {
        // Choose the new stick position using normalized prior expected weights.
        const Rcpp::NumericVector log_probability = covariance_empty_probabilities(empty, concentration);
        std::vector<double> empty_probability(empty.size());
        std::vector<int> order(empty.size());
        for (int index = 0; index < static_cast<int>(empty.size()); ++index) {
            empty_probability[index] = std::exp(log_probability[index]);
            order[index]             = index;
        }
        Rf_revsort(empty_probability.data(), order.data(), empty.size());
        const double uniform = R::runif(0.0, 1.0);
        double cumulative    = 0.0;
        int selected         = order.back();
        for (int index = 0; index < static_cast<int>(empty.size()); ++index) {
            cumulative += empty_probability[index];
            if (uniform <= cumulative) {
                selected = order[index];
                break;
            }
        }
        g                                           = empty[selected];
        const Rcpp::IntegerVector subjects          = move_subjects(labels, h, -1);
        const Rcpp::List allocation                 = pdpmvar_eigen_covariance_restricted(scatters, times, subjects, anchors, h, g);
        const Rcpp::NumericVector allocation_labels = allocation["labels"];
        probability                                 = Rcpp::as<double>(allocation["log_probability"]);
        for (int subject : subjects) proposed[subject - 1] = allocation_labels[subject - 1];
    } else {
        for (int subject = 0; subject < labels.size(); ++subject) if (labels[subject] == g) proposed[subject] = h;
        const Rcpp::List allocation = pdpmvar_eigen_covariance_restricted(scatters, times, move_subjects(labels, h, g), anchors, h, g, labels);
        probability                 = Rcpp::as<double>(allocation["log_probability"]);
    }

    // Include the probability of choosing this empty position and these subject assignments.
    const std::vector<int> proposal_empty     = move_empty_slots(split ? labels : proposed, slots);
    const Rcpp::NumericVector log_probability = covariance_empty_probabilities(proposal_empty, concentration);
    const auto position                       = std::find(proposal_empty.begin(), proposal_empty.end(), g);
    const double log_empty                    = log_probability[std::distance(proposal_empty.begin(), position)];
    const double allocation_ratio             = split ? -log_empty - probability : log_empty + probability;

    // Combine the bridge contribution with the integrated label prior and proposal correction.
    const Rcpp::List bridge = pdpmvar_eigen_covariance_bridge(labels, proposed, atoms, scatters, times, h, g, B, shape, rate, false);
    const double prior_change = pdpmvar_eigen_covariance_label_prior(proposed, concentration) -
        pdpmvar_eigen_covariance_label_prior(labels, concentration);
    const double ratio = Rcpp::as<double>(bridge["log_work"]) + prior_change + allocation_ratio;
    if (std::isnan(ratio)) Rcpp::stop("Non-finite annealed covariance acceptance calculation.");
    const bool accepted     = std::log(R::runif(0.0, 1.0)) < std::min(0.0, ratio);
    Rcpp::List result_atoms = accepted ? Rcpp::as<Rcpp::List>(bridge["atoms"]) : atoms;

    // After an accepted merge, redraw parameters for the group that is now empty.
    if (accepted && !split) result_atoms[g - 1] = sampler_new_covariance(D, B, shape, rate);

    return move_result(accepted ? proposed : labels, result_atoms, split ? "split" : "merge", accepted, ratio);
}
