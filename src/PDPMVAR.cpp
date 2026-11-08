// Functions for drawing PDPMVAR coefficients, proposing groups and running the full sampler.
// Shared random draws and covariance calculations are in sampling_helpers.cpp.
#include <RcppEigen.h>
#include "sampling_helpers.h"
#include <R_ext/Random.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Gaussian calculations for complete coefficient matrices ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Store the Gaussian distribution for one group's complete coefficient matrix.
// For D response nodes and P lagged predictors, arrange the D x P coefficients as
// one vector of length D * P, placing columns one after another. precision and upper
// are coefficient-by-coefficient matrices, (D * P) x (D * P); information and mean
// have length D * P. log_integral is the contribution with coefficients integrated out.
// Used when comparing the two possible groups in a split with full-matrix calculations.
struct CoefficientGroup {
    Eigen::MatrixXd precision;
    Eigen::VectorXd information;
    Eigen::MatrixXd upper;
    Eigen::VectorXd mean;
    double log_integral;
};

// Calculate a Gaussian conditional mean and the contribution with coefficients integrated out.
// precision is coefficient by coefficient, M x M, and information has length M,
// where M = D * P for D response nodes and P lagged predictors.
// Used by pdpmvar_eigen_restricted_split() for the current groups and for adding a subject.
CoefficientGroup integrate_coefficient_group(Eigen::MatrixXd precision,
                                             Eigen::VectorXd information,
                                             double prior_log_determinant_half) {
    Eigen::LLT<Eigen::MatrixXd> factor(precision);
    if (factor.info() != Eigen::Success) {
        Rcpp::stop("Coefficient group precision must be positive definite.");
    }

    // Complete the square and use the Cholesky diagonal for the Gaussian determinant.
    CoefficientGroup group;
    group.mean  = factor.solve(information);
    group.upper = factor.matrixU();
    group.log_integral = information.dot(group.mean) / 2.0 + prior_log_determinant_half -
                          group.upper.diagonal().array().log().sum();
    group.precision   = std::move(precision);
    group.information = std::move(information);

    return group;
}

// Put a group's Gaussian results into an R list for checking from R.
// Return coefficient-by-coefficient precision and Cholesky matrices, M x M, plus
// mean and precision-times-mean vectors of length M, where M is the number of coefficients.
// Used when pdpmvar_eigen_restricted_split() returns its two proposed groups.
Rcpp::List wrap_coefficient_group(const CoefficientGroup& group) {
    Rcpp::NumericVector information(group.information.size());
    Rcpp::NumericVector mean(group.mean.size());
    std::copy(group.information.data(), group.information.data() + group.information.size(), information.begin());
    std::copy(group.mean.data(), group.mean.data() + group.mean.size(), mean.begin());

    return Rcpp::List::create(Rcpp::Named("precision") = group.precision,
        Rcpp::Named("information") = information, Rcpp::Named("upper") = group.upper,
        Rcpp::Named("mean") = mean, Rcpp::Named("log_integral") = group.log_integral);
}

// Calculate each subject's Gaussian contribution for a complete coefficient matrix.
// With D nodes, K lags, P = D * K predictors and T_i observations, subject data are
// D x T_i, predictors are P x T_i and residual covariances are D x D.
// Return one coefficient-by-coefficient precision matrix, (D * P) x (D * P), and one
// precision-times-mean vector of length D * P per subject, together with prior terms.
// Coefficient columns are placed one after another in these vectors.
// Used by the coefficient split/merge method that integrates out the latent factors.
// The memory check rejects full-matrix statistics above 512 MiB before constructing them.
// [[Rcpp::export(name = ".pdpmvar_eigen_statistics")]]
Rcpp::List pdpmvar_eigen_statistics(const Rcpp::List& X,
                                    const Rcpp::List& design,
                                    const Rcpp::List& design_cross,
                                    const Rcpp::List& covariances,
                                    const Rcpp::NumericMatrix& variance) {
    const int n_subjects   = X.size();
    const int n_nodes      = variance.nrow();
    const int n_predictors = variance.ncol();
    const int dimension    = n_nodes * n_predictors;
    if (n_subjects == 0 || n_nodes == 0 || n_predictors == 0 || design.size() != n_subjects ||
        design_cross.size() != n_subjects || covariances.size() != n_subjects) {
        Rcpp::stop("Coefficient statistics need nonempty, matching subject lists and variance dimensions.");
    }
    if (8.0 * n_subjects * dimension * dimension > 512.0 * 1024 * 1024) {
        Rcpp::stop("Dense coefficient cache exceeds 512 MiB; use factor_conditional moves.");
    }

    // Read prior variances column by column, in the same order as the coefficient matrix.
    Eigen::VectorXd inverse_variance(dimension);
    double prior_log_determinant_half = 0.0;
    for (int index = 0; index < dimension; ++index) {
        if (!std::isfinite(variance[index]) || variance[index] <= 0.0) {
            Rcpp::stop("Shared coefficient variances must be positive and finite.");
        }
        inverse_variance[index] = 1.0 / variance[index];
        prior_log_determinant_half -= std::log(variance[index]) / 2.0;
    }

    // Keep one likelihood contribution per subject so proposed groups can sum them.
    Rcpp::List precisions(n_subjects), informations(n_subjects);
    for (int subject = 0; subject < n_subjects; ++subject) {
        Rcpp::checkUserInterrupt();
        Rcpp::NumericMatrix y_r          = X[subject];
        Rcpp::NumericMatrix z_r          = design[subject];
        Rcpp::NumericMatrix cross_r      = design_cross[subject];
        Rcpp::NumericMatrix covariance_r = covariances[subject];
        if (y_r.nrow() != n_nodes || z_r.nrow() != n_predictors || y_r.ncol() != z_r.ncol() ||
            cross_r.nrow() != n_predictors || cross_r.ncol() != n_predictors ||
            covariance_r.nrow() != n_nodes || covariance_r.ncol() != n_nodes) {
            Rcpp::stop("Subject matrices do not match the coefficient-block dimensions.");
        }
        Eigen::Map<const Eigen::MatrixXd> y(y_r.begin(), n_nodes, y_r.ncol());
        Eigen::Map<const Eigen::MatrixXd> z(z_r.begin(), n_predictors, z_r.ncol());
        Eigen::Map<const Eigen::MatrixXd> cross(cross_r.begin(), n_predictors, n_predictors);
        Eigen::Map<const Eigen::MatrixXd> covariance(covariance_r.begin(), n_nodes, n_nodes);
        if (!y.allFinite() || !z.allFinite() || !cross.allFinite() || !covariance.allFinite()) {
            Rcpp::stop("Coefficient statistics require finite subject matrices.");
        }
        Eigen::LLT<Eigen::MatrixXd> factor(covariance);
        if (factor.info() != Eigen::Success) {
            Rcpp::stop("Subject covariance must be positive definite.");
        }

        // Construct the predictor cross-product times covariance precision, one submatrix at a time.
        Eigen::MatrixXd omega = factor.solve(Eigen::MatrixXd::Identity(n_nodes, n_nodes));
        Eigen::MatrixXd precision(dimension, dimension);
        for (int column = 0; column < n_predictors; ++column) {
            for (int row = 0; row < n_predictors; ++row) {
                precision.block(row * n_nodes, column * n_nodes, n_nodes, n_nodes) = cross(row, column) * omega;
            }
        }
        Eigen::MatrixXd information = omega * y * z.transpose();
        Rcpp::NumericVector information_result(dimension);
        std::copy(information.data(), information.data() + dimension, information_result.begin());
        precisions[subject]   = precision;
        informations[subject] = information_result;
    }

    Eigen::MatrixXd prior_precision = inverse_variance.asDiagonal();

    return Rcpp::List::create(Rcpp::Named("precision") = precisions,
        Rcpp::Named("information") = informations, Rcpp::Named("prior_precision") = prior_precision,
        Rcpp::Named("prior_log_determinant_half") = prior_log_determinant_half,
        Rcpp::Named("backend") = "RcppEigen");
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Assign subjects between two proposed full-matrix groups ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Keep two selected subjects in opposite groups, then assign the other selected subjects.
// Use the change in each group's integrated Gaussian likelihood and its log stick weight
// to calculate the assignment probabilities. Each subject receives one label for its
// complete coefficient matrix, with all response rows and lags together.
// Return a label vector of length N, its log proposal probability and two Gaussian groups,
// where N is the total number of subjects. With forced_labels, evaluate those labels
// instead of drawing new ones; otherwise use the supplied uniform random draws.
// Used to propose splits and evaluate reverse split probabilities in the full-matrix method.
// [[Rcpp::export(name = ".pdpmvar_eigen_restricted_split")]]
Rcpp::List pdpmvar_eigen_restricted_split(const Rcpp::List& statistics,
                                          const Rcpp::IntegerVector& subjects,
                                          const Rcpp::IntegerVector& anchors,
                                          const Rcpp::NumericVector& log_weights,
                                          int h,
                                          int g,
                                          Rcpp::Nullable<Rcpp::NumericVector> forced_labels,
                                          const Rcpp::NumericVector& uniforms) {
    Rcpp::List precisions                   = statistics["precision"];
    Rcpp::List informations                 = statistics["information"];
    Rcpp::NumericMatrix prior_r             = statistics["prior_precision"];
    const int n_subjects                    = precisions.size();
    const int dimension                     = prior_r.nrow();
    const double prior_log_determinant_half = Rcpp::as<double>(statistics["prior_log_determinant_half"]);
    if (dimension == 0 || prior_r.ncol() != dimension || informations.size() != n_subjects ||
        anchors.size() != 2 || anchors[0] < 1 || anchors[0] > n_subjects || anchors[1] < 1 ||
        anchors[1] > n_subjects || anchors[0] == anchors[1] || h < 1 || g < 1 || h == g ||
        h > log_weights.size() || g > log_weights.size() || !std::isfinite(prior_log_determinant_half)) {
        Rcpp::stop("Invalid coefficient split dimensions, anchors or cluster indices.");
    }
    Eigen::Map<const Eigen::MatrixXd> prior(prior_r.begin(), dimension, dimension);
    if (!prior.allFinite()) Rcpp::stop("Prior precision must be finite.");
    std::vector<Eigen::Map<const Eigen::MatrixXd>> precision_maps;
    std::vector<Eigen::Map<const Eigen::VectorXd>> information_maps;
    precision_maps.reserve(n_subjects);
    information_maps.reserve(n_subjects);
    for (int subject = 0; subject < n_subjects; ++subject) {
        Rcpp::NumericMatrix matrix = precisions[subject];
        Rcpp::NumericVector vector = informations[subject];
        if (matrix.nrow() != dimension || matrix.ncol() != dimension || vector.size() != dimension) {
            Rcpp::stop("Cached coefficient statistics have inconsistent dimensions.");
        }
        precision_maps.emplace_back(matrix.begin(), dimension, dimension);
        information_maps.emplace_back(vector.begin(), dimension);
        if (!precision_maps.back().allFinite() || !information_maps.back().allFinite()) {
            Rcpp::stop("Cached coefficient statistics must be finite.");
        }
    }
    std::vector<bool> seen(n_subjects, false);
    std::vector<int> others;
    for (int subject : subjects) {
        if (subject < 1 || subject > n_subjects || seen[subject - 1]) {
            Rcpp::stop("Split subjects must be distinct valid subject indices.");
        }
        seen[subject - 1] = true;
        if (subject != anchors[0] && subject != anchors[1]) others.push_back(subject - 1);
    }
    if (!seen[anchors[0] - 1] || !seen[anchors[1] - 1]) {
        Rcpp::stop("Both split anchors must belong to the selected subjects.");
    }
    const bool forced                 = forced_labels.isNotNull();
    Rcpp::NumericVector forced_values = forced ? Rcpp::NumericVector(forced_labels.get()) : Rcpp::NumericVector(0);
    if ((forced && forced_values.size() != n_subjects) || (!forced && uniforms.size() != others.size())) {
        Rcpp::stop("Split labels or uniforms do not match the selected subjects.");
    }

    // Each selected subject starts its own group and stays there throughout the proposal.
    CoefficientGroup first = integrate_coefficient_group(prior + precision_maps[anchors[0] - 1],
        information_maps[anchors[0] - 1], prior_log_determinant_half);
    CoefficientGroup second = integrate_coefficient_group(prior + precision_maps[anchors[1] - 1],
        information_maps[anchors[1] - 1], prior_log_determinant_half);
    Rcpp::NumericVector labels(n_subjects);
    std::fill(labels.begin(), labels.end(), static_cast<double>(h));
    labels[anchors[1] - 1] = g;
    double log_probability = 0.0;
    for (std::size_t index = 0; index < others.size(); ++index) {
        Rcpp::checkUserInterrupt();
        const int subject = others[index];
        CoefficientGroup candidate_first = integrate_coefficient_group(first.precision + precision_maps[subject],
            first.information + information_maps[subject], prior_log_determinant_half);
        CoefficientGroup candidate_second = integrate_coefficient_group(second.precision + precision_maps[subject],
            second.information + information_maps[subject], prior_log_determinant_half);

        // Adding a subject changes the integrated likelihood by its predictive log probability.
        const double score_first  = log_weights[h - 1] + candidate_first.log_integral - first.log_integral;
        const double score_second = log_weights[g - 1] + candidate_second.log_integral - second.log_integral;
        const double maximum      = std::max(score_first, score_second);
        if (!std::isfinite(maximum)) Rcpp::stop("No split candidate has a finite predictive score.");
        const double mass_first  = std::exp(score_first - maximum);
        const double mass_second = std::exp(score_second - maximum);
        const double total       = mass_first + mass_second;
        const double normalizer  = maximum + std::log(total);
        bool choose_first;
        if (forced) {
            if (forced_values[subject] != h && forced_values[subject] != g) {
                Rcpp::stop("Forced split labels must match one of the two cluster indices.");
            }
            choose_first = forced_values[subject] == h;
        } else {
            const double uniform = uniforms[index];
            if (!std::isfinite(uniform) || uniform < 0.0 || uniform > 1.0) {
                Rcpp::stop("Split uniforms must be between zero and one.");
            }
            // Follow R's two-group sampling order: consider the higher-probability group first.
            choose_first = mass_first >= mass_second ? uniform <= mass_first / total : uniform > mass_second / total;
        }
        labels[subject] = choose_first ? h : g;
        log_probability += (choose_first ? score_first : score_second) - normalizer;
        if (choose_first) first = std::move(candidate_first);
        else second = std::move(candidate_second);
    }

    return Rcpp::List::create(Rcpp::Named("labels") = labels,
        Rcpp::Named("log_probability") = log_probability,
        Rcpp::Named("groups") = Rcpp::List::create(wrap_coefficient_group(first), wrap_coefficient_group(second)));
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Draw coefficient matrices conditional on factors ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Draw coefficient matrices for groups that contain subjects, conditional on sampled factors.
// For D nodes, P lagged predictors and N subjects, shared_variance is D x P and
// residual_variances is D x N. Each subject's data and factor contribution are D x T_i,
// predictors are P x T_i, and predictor cross-products are P x P.
// Draw one response row at a time and return a list of response-by-predictor matrices,
// D x P, retaining the parameter values for unused groups.
// Called during every iteration of pdpmvar_eigen_sampler(), before the shrinkage update.
Rcpp::List draw_coefficients_with_design(const Rcpp::List& X,
                                         const Rcpp::List& design,
                                         const Rcpp::List& design_cross,
                                         const Rcpp::List& factor_contributions,
                                         const Rcpp::NumericVector& labels,
                                         const Rcpp::List& atoms,
                                         const Eigen::MatrixXd& shared_variance,
                                         const Eigen::MatrixXd& residual_variances,
                                         const CoefficientDesign* saved_design) {
    const int n_subjects   = X.size();
    const int n_nodes      = shared_variance.rows();
    const int n_predictors = shared_variance.cols();
    const int n_atoms      = atoms.size();
    if (n_subjects == 0 || n_nodes == 0 || n_predictors == 0 || n_atoms == 0 ||
        design.size() != n_subjects || design_cross.size() != n_subjects ||
        factor_contributions.size() != n_subjects || labels.size() != n_subjects ||
        residual_variances.rows() != n_nodes || residual_variances.cols() != n_subjects) {
        Rcpp::stop("Coefficient update inputs must have matching nonempty dimensions.");
    }
    if (!shared_variance.allFinite() || !residual_variances.allFinite() ||
        (shared_variance.array() <= 0.0).any() || (residual_variances.array() <= 0.0).any()) {
        Rcpp::stop("Coefficient and residual variances must be positive and finite.");
    }
    std::vector<std::vector<int>> members(n_atoms);
    // Keep copies of these matrices so converting R integers to doubles cannot invalidate them.
    std::vector<Eigen::MatrixXd> data(n_subjects), contributions(n_subjects);
    CoefficientDesign local_design;
    if (!saved_design) {
        local_design.predictors.resize(n_subjects);
        local_design.cross_products.resize(n_subjects);
    }
    const CoefficientDesign& input = saved_design ? *saved_design : local_design;
    const auto& predictors         = input.predictors;
    const auto& cross_products     = input.cross_products;
    for (int subject = 0; subject < n_subjects; ++subject) {
        const double label = labels[subject];
        if (!std::isfinite(label) || label < 1.0 || label > n_atoms || std::floor(label) != label) {
            Rcpp::stop("Coefficient labels must be valid integer atom indices.");
        }
        members[static_cast<int>(label) - 1].push_back(subject);
        data[subject]           = Rcpp::as<Eigen::MatrixXd>(X[subject]);
        if (!saved_design) {
            local_design.predictors[subject]     = Rcpp::as<Eigen::MatrixXd>(design[subject]);
            local_design.cross_products[subject] = Rcpp::as<Eigen::MatrixXd>(design_cross[subject]);
        }
        contributions[subject]  = Rcpp::as<Eigen::MatrixXd>(factor_contributions[subject]);
        const int n_times       = data[subject].cols();
        if (data[subject].rows() != n_nodes || n_times == 0 || predictors[subject].rows() != n_predictors ||
            predictors[subject].cols() != n_times || contributions[subject].rows() != n_nodes ||
            contributions[subject].cols() != n_times || cross_products[subject].rows() != n_predictors ||
            cross_products[subject].cols() != n_predictors) {
            Rcpp::stop("Subject data, factor contributions and lag predictors must have compatible dimensions.");
        }
        if (!data[subject].allFinite() || !contributions[subject].allFinite() ||
            (!saved_design && (!predictors[subject].allFinite() || !cross_products[subject].allFinite()))) {
            Rcpp::stop("Coefficient update matrices must be finite.");
        }
    }

    // Update occupied groups in order, drawing all coefficients for one response row at a time.
    Rcpp::List result = Rcpp::clone(atoms);
    for (int cluster = 0; cluster < n_atoms; ++cluster) {
        if (members[cluster].empty()) continue;
        Rcpp::checkUserInterrupt();
        Rcpp::List atom             = result[cluster];
        Eigen::MatrixXd coefficient = Rcpp::as<Eigen::MatrixXd>(atom["coefficient"]);
        if (coefficient.rows() != n_nodes || coefficient.cols() != n_predictors) {
            Rcpp::stop("Coefficient atoms must have response rows and stacked-lag predictor columns.");
        }
        for (int response = 0; response < n_nodes; ++response) {
            Eigen::MatrixXd precision   = Eigen::MatrixXd::Zero(n_predictors, n_predictors);
            precision.diagonal()        = shared_variance.row(response).cwiseInverse().transpose();
            Eigen::VectorXd information = Eigen::VectorXd::Zero(n_predictors);
            for (int subject : members[cluster]) {
                const double residual_precision = 1.0 / residual_variances(response, subject);
                const Eigen::VectorXd adjusted = (data[subject].row(response) -
                                                  contributions[subject].row(response)).transpose();
                precision.noalias() += residual_precision * cross_products[subject];
                information.noalias() += residual_precision * predictors[subject] * adjusted;
            }

            // Draw from the Gaussian conditional distribution after adding all member subjects.
            Eigen::LLT<Eigen::MatrixXd> factor(precision);
            if (factor.info() != Eigen::Success) {
                Rcpp::stop("Coefficient conditional precision must be positive definite.");
            }
            Eigen::VectorXd noise(n_predictors);
            for (int predictor = 0; predictor < n_predictors; ++predictor) noise[predictor] = R::rnorm(0.0, 1.0);
            const Eigen::VectorXd draw = factor.solve(information) + factor.matrixU().solve(noise);
            coefficient.row(response)  = draw.transpose();
        }
        atom["coefficient"] = coefficient;
        result[cluster]     = atom;
    }

    return result;
}

// Draw coefficients for calls that supply the predictors as R lists.
// [[Rcpp::export(name = ".pdpmvar_eigen_coefficient_update")]]
Rcpp::List pdpmvar_eigen_coefficient_update(const Rcpp::List& X,
                                            const Rcpp::List& design,
                                            const Rcpp::List& design_cross,
                                            const Rcpp::List& factor_contributions,
                                            const Rcpp::NumericVector& labels,
                                            const Rcpp::List& atoms,
                                            const Eigen::MatrixXd& shared_variance,
                                            const Eigen::MatrixXd& residual_variances) {
    return draw_coefficients_with_design(X, design, design_cross, factor_contributions,
        labels, atoms, shared_variance, residual_variances, nullptr);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Update shared coefficient prior variances ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Draw the shared coefficient prior variances, then lambda squared under the Laplace prior.
// Use coefficient matrices only from groups that contain subjects. Each group contributes
// once, regardless of how many subjects it contains. For D nodes and P predictors,
// return the shared response-by-predictor variance matrix, D x P, and lambda squared.
// Called by pdpmvar_eigen_sampler() after coefficient updates; skipped for the fixed normal prior.
// [[Rcpp::export(name = ".pdpmvar_eigen_shrinkage")]]
Rcpp::List pdpmvar_eigen_shrinkage(const Rcpp::NumericVector& labels, const Rcpp::List& atoms,
                                   const Rcpp::NumericMatrix& shared_variance, double lambda_squared) {
    if (!std::isfinite(lambda_squared) || lambda_squared <= 0.0 || shared_variance.size() == 0) {
        Rcpp::stop("Shared shrinkage needs nonempty variances and positive finite lambda squared.");
    }
    const std::vector<int> counts = count_stick_labels(labels);
    if (counts.size() > atoms.size()) Rcpp::stop("Shrinkage labels must index existing coefficient atoms.");
    Rcpp::NumericMatrix sum_squared(shared_variance.nrow(), shared_variance.ncol());
    int occupied = 0;
    for (int cluster = 0; cluster < counts.size(); ++cluster) {
        if (counts[cluster] == 0) continue;
        ++occupied;
        Rcpp::List atom                 = atoms[cluster];
        Rcpp::NumericMatrix coefficient = atom["coefficient"];
        if (coefficient.nrow() != shared_variance.nrow() || coefficient.ncol() != shared_variance.ncol()) {
            Rcpp::stop("Coefficient atoms must match the shared variance matrix.");
        }
        for (int index = 0; index < coefficient.size(); ++index) {
            if (!std::isfinite(coefficient[index])) Rcpp::stop("Coefficient atoms must be finite.");
            sum_squared[index] += coefficient[index] * coefficient[index];
        }
    }

    // GIG shape is 1 minus half the number of occupied coefficient groups.
    Rcpp::NumericMatrix variance = Rcpp::clone(shared_variance);
    long double sum_variance     = 0.0;
    for (int index = 0; index < variance.size(); ++index) {
        if (!std::isfinite(variance[index]) || variance[index] <= 0.0) {
            Rcpp::stop("Shared coefficient variances must be positive and finite.");
        }
        variance[index] = draw_shared_gig(1.0 - occupied / 2.0, sum_squared[index], lambda_squared);
        if (!std::isfinite(variance[index]) || variance[index] <= 0.0) {
            Rcpp::stop("A GIG variance draw overflowed or underflowed.");
        }
        sum_variance += variance[index];
    }

    // The global rate has its original Gamma(1, 2) prior. R::rgamma takes scale = 1 / rate.
    const double rate           = 2.0 + static_cast<double>(sum_variance) / 2.0;
    const double updated_lambda = R::rgamma(variance.size() + 1.0, 1.0 / rate);

    return Rcpp::List::create(Rcpp::Named("shared_variance") = variance,
                              Rcpp::Named("lambda_squared") = updated_lambda);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Assign subjects to coefficient groups ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Draw one coefficient-group label per subject using the full residual covariance likelihood.
// Only groups whose weights exceed the subject's slice value are eligible. Their likelihoods
// are not multiplied by the weights again. One label covers every row and lag of the matrix.
// For N subjects and C available coefficient groups, return N labels and two
// subjects-by-groups matrices, N x C, with log likelihoods and assignment probabilities.
// Called during each sampler iteration when at most 200 groups are available;
// the R assignment function handles larger numbers of groups.
Rcpp::List assign_coefficients_with_design(const Rcpp::List& X,
                                           const Rcpp::List& design,
                                           const Rcpp::List& coefficient_atoms,
                                           const Rcpp::NumericVector& covariance_labels,
                                           const Rcpp::List& covariance_atoms,
                                           const Rcpp::NumericVector& weights,
                                           const Rcpp::NumericVector& slices,
                                           const Rcpp::NumericVector& uniforms,
                                           const CoefficientDesign* saved_design) {
    const int n_subjects = X.size();
    const int n_atoms    = coefficient_atoms.size();
    if (n_subjects == 0 || design.size() != n_subjects || covariance_labels.size() != n_subjects ||
        slices.size() != n_subjects || uniforms.size() != n_subjects || weights.size() != n_atoms || n_atoms == 0) {
        Rcpp::stop("Coefficient assignment inputs must have matching nonempty dimensions.");
    }

    // Keep scores for all components, leaving ineligible ones at minus infinity.
    Rcpp::NumericVector labels(n_subjects);
    Rcpp::NumericMatrix scores(n_subjects, n_atoms), probabilities(n_subjects, n_atoms);
    std::fill(scores.begin(), scores.end(), R_NegInf);
    std::vector<Eigen::MatrixXd> lowers(covariance_atoms.size()), coefficients(n_atoms);
    std::vector<double> determinants(covariance_atoms.size());
    std::vector<bool> ready(covariance_atoms.size(), false), coefficient_ready(n_atoms, false);
    for (int index = 0; index < weights.size(); ++index) {
        if (!std::isfinite(weights[index]) || weights[index] < 0.0)
            Rcpp::stop("Slice weights must be nonnegative and finite.");
    }
    for (int subject = 0; subject < n_subjects; ++subject) {
        Rcpp::checkUserInterrupt();
        const Eigen::MatrixXd data       = Rcpp::as<Eigen::MatrixXd>(X[subject]);
        Eigen::MatrixXd local_predictors;
        if (!saved_design) local_predictors = Rcpp::as<Eigen::MatrixXd>(design[subject]);
        const Eigen::MatrixXd& predictors = saved_design ? saved_design->predictors[subject] : local_predictors;
        const double label               = covariance_labels[subject];
        if (!std::isfinite(label) || std::floor(label) != label || label < 1 || label > covariance_atoms.size() ||
            data.rows() == 0 || data.cols() != predictors.cols() || predictors.rows() == 0 ||
            !data.allFinite() || (!saved_design && !predictors.allFinite()) || !std::isfinite(slices[subject]) || slices[subject] < 0.0) {
            Rcpp::stop("Invalid subject data, covariance labels or slices.");
        }
        const int covariance_index = static_cast<int>(label) - 1;
        cache_likelihood_covariance(covariance_atoms, covariance_index, data.rows(), lowers, determinants, ready);
        if (lowers[covariance_index].rows() != data.rows())
            Rcpp::stop("All subjects must share the same node dimension.");
        std::vector<int> candidates;
        std::vector<double> log_scores;
        for (int cluster = 0; cluster < n_atoms; ++cluster) {
            if (weights[cluster] <= slices[subject]) continue;
            if (!coefficient_ready[cluster]) {
                Rcpp::List atom            = coefficient_atoms[cluster];
                coefficients[cluster]      = Rcpp::as<Eigen::MatrixXd>(atom["coefficient"]);
                coefficient_ready[cluster] = true;
            }
            if (coefficients[cluster].rows() != data.rows() || coefficients[cluster].cols() != predictors.rows() ||
                !coefficients[cluster].allFinite()) Rcpp::stop("Coefficient atom dimensions must match the lag predictors.");
            const Eigen::MatrixXd residual = data - coefficients[cluster] * predictors;
            const double score             = residual_log_likelihood(residual, lowers[covariance_index], determinants[covariance_index]);
            scores(subject, cluster)       = score;
            candidates.push_back(cluster);
            log_scores.push_back(score);
        }
        labels[subject] = sample_slice_candidate(candidates, log_scores, uniforms[subject], probabilities, subject);
    }

    return Rcpp::List::create(Rcpp::Named("labels") = labels, Rcpp::Named("log_likelihood") = scores,
                              Rcpp::Named("probabilities") = probabilities);
}

// Draw coefficient labels for calls that supply the predictors as an R list.
// [[Rcpp::export(name = ".pdpmvar_eigen_assign_coefficients")]]
Rcpp::List pdpmvar_eigen_assign_coefficients(const Rcpp::List& X,
                                             const Rcpp::List& design,
                                             const Rcpp::List& coefficient_atoms,
                                             const Rcpp::NumericVector& covariance_labels,
                                             const Rcpp::List& covariance_atoms,
                                             const Rcpp::NumericVector& weights,
                                             const Rcpp::NumericVector& slices,
                                             const Rcpp::NumericVector& uniforms) {
    return assign_coefficients_with_design(X, design, coefficient_atoms, covariance_labels,
        covariance_atoms, weights, slices, uniforms, nullptr);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Store coefficient statistics conditional on factors ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Calculate Gaussian statistics after subtracting sampled factor contributions.
// For D response nodes, P lagged predictors and N subjects, keep N predictor cross-products,
// P x P, and N variance-weighted response-predictor cross-products, D x P.
// Subjects with exactly the same D residual variances share one column in group_variance,
// a nodes-by-variance-groups matrix, D x G. This shares calculations only; it does not
// assign those subjects to the same coefficient or covariance cluster.
// Used by the factor-conditional coefficient split/merge move to avoid (D * P) x (D * P) matrices.
static Rcpp::List row_statistics_with_design(const Rcpp::List& adjusted,
                                      const Rcpp::List& design,
                                      const Rcpp::List& design_cross,
                                      const Eigen::MatrixXd& residual_variances,
                                      const Eigen::MatrixXd& variance,
                                      const CoefficientDesign* saved_design) {
    const int N = adjusted.size(), D = variance.rows(), P = variance.cols();
    if (N < 1 || D < 1 || P < 1 || design.size() != N || design_cross.size() != N ||
        residual_variances.rows() != D || residual_variances.cols() != N || !variance.allFinite() ||
        !residual_variances.allFinite() || (variance.array() <= 0).any() || (residual_variances.array() <= 0).any()) {
        Rcpp::stop("Row statistics require matching dimensions and positive finite variances.");
    }

    // Equal residual variance columns share calculations, without sharing coefficient labels.
    Rcpp::List information(N);
    std::vector<Eigen::VectorXd> patterns;
    Rcpp::IntegerVector group(N);
    for (int i = 0; i < N; ++i) {
        const Eigen::MatrixXd y     = Rcpp::as<Eigen::MatrixXd>(adjusted[i]);
        Eigen::MatrixXd local_predictors;
        if (!saved_design) {
            local_predictors            = Rcpp::as<Eigen::MatrixXd>(design[i]);
            const Eigen::MatrixXd cross = Rcpp::as<Eigen::MatrixXd>(design_cross[i]);
            if (cross.rows() != P || cross.cols() != P || !cross.allFinite() || !local_predictors.allFinite()) {
                Rcpp::stop("Invalid row-statistic subject matrices.");
            }
        }
        const Eigen::MatrixXd& z = saved_design ? saved_design->predictors[i] : local_predictors;
        if (y.rows() != D || z.rows() != P || y.cols() != z.cols() || !y.allFinite()) {
            Rcpp::stop("Invalid row-statistic subject matrices.");
        }
        Eigen::MatrixXd b = y * z.transpose();
        for (int row = 0; row < D; ++row) b.row(row) /= residual_variances(row, i);
        information[i] = b;

        int g = 0;
        while (g < static_cast<int>(patterns.size()) &&
            !(patterns[g].array() == residual_variances.col(i).array()).all()) ++g;
        if (g == static_cast<int>(patterns.size())) patterns.push_back(residual_variances.col(i));
        group[i] = g + 1;
    }

    // Store each distinct set of node variances once, alongside each subject's column number.
    Eigen::MatrixXd grouped(D, patterns.size());
    for (int g = 0; g < static_cast<int>(patterns.size()); ++g) grouped.col(g) = patterns[g];

    return Rcpp::List::create(Rcpp::Named("design_cross") = design_cross,
        Rcpp::Named("information") = information, Rcpp::Named("variance") = variance,
        Rcpp::Named("group_variance") = grouped, Rcpp::Named("variance_group") = group,
        Rcpp::Named("n_subjects") = N, Rcpp::Named("representation") = "factor_rows",
        Rcpp::Named("backend") = "RcppEigen");
}

// Prepare row statistics for calls that supply the predictors as R lists.
// [[Rcpp::export(name = ".pdpmvar_eigen_row_statistics")]]
Rcpp::List pdpmvar_eigen_row_statistics(const Rcpp::List& adjusted,
                                        const Rcpp::List& design,
                                        const Rcpp::List& design_cross,
                                        const Eigen::MatrixXd& residual_variances,
                                        const Eigen::MatrixXd& variance) {
    return row_statistics_with_design(adjusted, design, design_cross, residual_variances, variance, nullptr);
}

// Keep the subject statistics needed for coefficient calculations conditional on factors.
// N is subjects, D is nodes, P is predictors and G is distinct sets of node residual variances.
// Store N predictor cross-product matrices, P x P, and N variance-weighted
// response-predictor cross-products, D x P. Prior variances are D x P and
// distinct residual variance columns are D x G.
// Used by row_group(), row_restricted() and the factor-conditional split/merge functions.
// Reuse the row-specific sampler's cross-products when supplied. Otherwise own copies.
struct RowStatistics {
    int N, D, P, G;
    Eigen::MatrixXd variance, group_variance;
    std::vector<Eigen::MatrixXd> local_cross, information;
    const std::vector<Eigen::MatrixXd>& cross;
    std::vector<int> group;
    // Read and check the matrices returned by pdpmvar_eigen_row_statistics().
    explicit RowStatistics(const Rcpp::List& statistics, const CoefficientDesign* saved_design = nullptr) :
        cross(saved_design ? saved_design->cross_products : local_cross) {
        variance                      = Rcpp::as<Eigen::MatrixXd>(statistics["variance"]);
        group_variance                = Rcpp::as<Eigen::MatrixXd>(statistics["group_variance"]);
        const Rcpp::List crosses      = statistics["design_cross"], infos = statistics["information"];
        const Rcpp::IntegerVector ids = Rcpp::as<Rcpp::IntegerVector>(statistics["variance_group"]);
        N                             = crosses.size();
        D                             = variance.rows();
        P                             = variance.cols();
        G                             = group_variance.cols();
        if (N < 1 || D < 1 || P < 1 || G < 1 || group_variance.rows() != D || infos.size() != N || ids.size() != N ||
            !variance.allFinite() || !group_variance.allFinite() || (variance.array() <= 0).any() ||
            (group_variance.array() <= 0).any()) Rcpp::stop("Invalid compact coefficient statistics.");
        for (int i = 0; i < N; ++i) {
            if (!saved_design) {
                local_cross.push_back(Rcpp::as<Eigen::MatrixXd>(crosses[i]));
                const Eigen::MatrixXd& subject_cross = local_cross.back();
                if (subject_cross.rows() != P || subject_cross.cols() != P || !subject_cross.allFinite()) {
                    Rcpp::stop("Invalid compact subject dimensions or values.");
                }
            }
            information.push_back(Rcpp::as<Eigen::MatrixXd>(infos[i]));
            if (information.back().rows() != D || information.back().cols() != P || ids[i] < 1 ||
                ids[i] > G || !information.back().allFinite()) Rcpp::stop("Invalid compact subject dimensions or values.");
            group.push_back(ids[i] - 1);
        }
    }
};

// Store the summed statistics for one coefficient group conditional on sampled factors.
// Keep G predictor-by-predictor sums, P x P, and one response-by-predictor sum, D x P,
// where D is nodes, P is predictors and G is the number of distinct residual variance columns.
// Used when adding subjects to either proposed group; no full coefficient precision is stored.
struct RowGroup {
    std::vector<Eigen::MatrixXd> cross;
    Eigen::MatrixXd information;
    int count       = 0;
    double integral = 0;
    // Start with no subjects and zero summed contributions for every variance group.
    explicit RowGroup(const RowStatistics& s) : information(Eigen::MatrixXd::Zero(s.D, s.P)) {
        for (int g = 0; g < s.G; ++g) cross.push_back(Eigen::MatrixXd::Zero(s.P, s.P));
    }
};

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Calculate Gaussian distributions for groups conditional on factors ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Calculate the predictor-by-predictor precision matrix, P x P, for one response row.
// Start with that row's prior precision, then add the group's predictor cross-products
// weighted by the appropriate node residual variances. P is the number of predictors.
// Used by row_integral() and row_wrap(); row is counted from zero within C++.
Eigen::MatrixXd row_precision(const RowStatistics& s, const RowGroup& group, int row) {
    Eigen::MatrixXd precision = Eigen::MatrixXd::Zero(s.P, s.P);
    precision.diagonal()      = s.variance.row(row).cwiseInverse().transpose();
    for (int g = 0; g < s.G; ++g) precision.noalias() += group.cross[g] / s.group_variance(row, g);

    return precision;
}

// Integrate out each response row, then add its log contribution for the full coefficient matrix.
// Each calculation uses a predictor-by-predictor precision matrix, P x P, where P is predictors.
// Include the coefficient prior once per group, not once per subject. An empty group
// integrates to one, so its log contribution is zero. Factor-prior terms cancel because
// the same sampled factors are held fixed when comparing coefficient groups.
// Used whenever row_group() or row_add() evaluates a group.
double row_integral(const RowStatistics& s, const RowGroup& group) {
    if (group.count == 0) return 0.0;
    long double value = 0;
    for (int row = 0; row < s.D; ++row) {
        Rcpp::checkUserInterrupt();
        Eigen::LLT<Eigen::MatrixXd> factor(row_precision(s, group, row));
        if (factor.info() != Eigen::Success) Rcpp::stop("Conditional coefficient precision must be positive definite.");
        const Eigen::VectorXd b     = group.information.row(row).transpose();
        const Eigen::VectorXd mean  = factor.solve(b);
        const Eigen::MatrixXd upper = factor.matrixU();
        value += b.dot(mean) / 2.0 - s.variance.row(row).array().log().sum() / 2.0 - upper.diagonal().array().log().sum();
    }

    return static_cast<double>(value);
}

// Add the selected subjects' statistics and integrate out their shared coefficient matrix.
// Return G predictor-by-predictor sums, P x P, summed variance-weighted response-predictor
// cross-products, D x P, and the integrated log contribution. D is nodes, P is predictors
// and G is the number of distinct residual variance columns.
// Used to form starting groups, compare merged groups and draw matrices after a move.
// Subject numbers supplied from R start at one; subtract one before indexing the stored statistics.
RowGroup row_group(const RowStatistics& s, const Rcpp::IntegerVector& subjects) {
    RowGroup group(s);
    std::vector<bool> seen(s.N, false);
    for (int subject : subjects) {
        if (subject < 1 || subject > s.N || seen[subject - 1])
            Rcpp::stop("Row-group subjects must be distinct valid indices.");
        const int i = subject - 1;
        seen[i]     = true;
        group.cross[s.group[i]] += s.cross[i];
        group.information += s.information[i];
        ++group.count;
    }
    group.integral = row_integral(s, group);

    return group;
}

// Add one subject to a proposed coefficient group and recalculate its integrated log contribution.
// Update the predictor sum for that subject's residual variance column and add its
// variance-weighted response-predictor cross-products. The matrices remain P x P and D x P,
// for P predictors and D nodes.
// Used by row_restricted() to compare the two possible assignments without rebuilding each group.
// Unlike row_group(), subject here is already counted from zero within C++.
RowGroup row_add(const RowStatistics& s, const RowGroup& old, int subject) {
    RowGroup result = old;
    result.cross[s.group[subject]] += s.cross[subject];
    result.information += s.information[subject];
    ++result.count;
    result.integral = row_integral(s, result);

    return result;
}

// Calculate conditional coefficient means and, if requested, draw a full coefficient matrix.
// For D response nodes and P predictors, return means and precision-times-mean values as
// vectors of length D * P, placing columns one after another. If noise is supplied, it
// must be a response-by-predictor matrix, D x P, of standard normal draws; return the
// drawn coefficient matrix in the same shape. No (D * P) x (D * P) matrix is constructed.
// Used for R checks of group calculations and to draw both matrices after a factor-conditional move.
Rcpp::List row_wrap(const RowStatistics& s, const RowGroup& group,
                    Rcpp::Nullable<Rcpp::NumericMatrix> noise = R_NilValue) {
    const bool draw = noise.isNotNull();
    Eigen::MatrixXd normals;
    if (draw) {
        normals = Rcpp::as<Eigen::MatrixXd>(noise.get());
        if (normals.rows() != s.D || normals.cols() != s.P || !normals.allFinite())
            Rcpp::stop("Invalid coefficient normal noise.");
    }

    // Calculate each row separately; supplied normal draws turn the means into coefficient draws.
    Eigen::MatrixXd mean(s.D, s.P), coefficient(s.D, s.P);
    for (int row = 0; row < s.D; ++row) {
        Eigen::LLT<Eigen::MatrixXd> factor(row_precision(s, group, row));
        if (factor.info() != Eigen::Success) Rcpp::stop("Conditional coefficient precision must be positive definite.");
        mean.row(row) = factor.solve(group.information.row(row).transpose()).transpose();
        if (draw) coefficient.row(row) = mean.row(row) + factor.matrixU().solve(normals.row(row).transpose()).transpose();
    }

    // Return vectors in R's matrix order: every entry of column 1, then column 2, and so on.
    Rcpp::NumericVector values(mean.size()), info(mean.size());
    std::copy(mean.data(), mean.data() + mean.size(), values.begin());
    std::copy(group.information.data(), group.information.data() + mean.size(), info.begin());
    Rcpp::List result = Rcpp::List::create(Rcpp::Named("mean") = values, Rcpp::Named("information") = info,
        Rcpp::Named("log_integral") = group.integral);
    if (draw) result["coefficient"] = coefficient;

    return result;
}

// Return Gaussian results for a selected group, with factors held fixed.
// For D response nodes and P predictors, means and precision-times-mean values are vectors
// of length D * P. An optional coefficient draw is D x P, using supplied D x P normal draws.
// Called by the R helper .pdpmvar_coefficient_row_group() for separate checks of these calculations.
// [[Rcpp::export(name = ".pdpmvar_eigen_row_group")]]
Rcpp::List pdpmvar_eigen_row_group(const Rcpp::List& statistics, const Rcpp::IntegerVector& subjects,
                                   Rcpp::Nullable<Rcpp::NumericMatrix> noise = R_NilValue) {
    const RowStatistics s(statistics);

    return row_wrap(s, row_group(s, subjects), noise);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Assign subjects between two proposed groups conditional on factors ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Keep subject labels, the two proposed coefficient groups and their log assignment probability.
// labels has length N, where N is the number of subjects. The field named probability
// stores a log probability. Used by row_restricted() and the factor-conditional split/merge move.
struct RowAllocation {
    Rcpp::NumericVector labels;
    RowGroup first, second;
    double probability = 0;
    // Start the label vector and both empty groups before placing the two selected subjects.
    explicit RowAllocation(const RowStatistics& s) : labels(s.N), first(s), second(s) {}
};

// Assign the selected subjects between two groups while holding two subjects in opposite groups.
// For each possible assignment, add Gaussian log contributions across all response rows
// before choosing a single label for the subject's entire coefficient matrix.
// weights contains log stick weights. With forced labels, evaluate the probability of
// those assignments; otherwise use the supplied uniform draws. Return N labels and
// the two groups' summed statistics, where N is the total number of subjects.
// Used to propose a factor-conditional split and evaluate the reverse proposal for a merge.
RowAllocation row_restricted(const RowStatistics& s,
                             const Rcpp::IntegerVector& subjects,
                             const Rcpp::IntegerVector& anchors,
                             const Rcpp::NumericVector& weights,
                             int h,
                             int g,
                             Rcpp::Nullable<Rcpp::NumericVector> forced,
                             const Rcpp::NumericVector& uniforms) {
    if (anchors.size() != 2 || anchors[0] < 1 || anchors[0] > s.N || anchors[1] < 1 || anchors[1] > s.N ||
        anchors[0] == anchors[1] || h < 1 || g < 1 || h == g || std::max(h, g) > weights.size()) Rcpp::stop("Invalid row-split anchors or weights.");
    std::vector<bool> seen(s.N, false);
    std::vector<int> others;
    for (int subject : subjects) {
        if (subject < 1 || subject > s.N || seen[subject - 1])
            Rcpp::stop("Split subjects must be distinct valid indices.");
        seen[subject - 1] = true;
        if (subject != anchors[0] && subject != anchors[1]) others.push_back(subject - 1);
    }
    if (!seen[anchors[0] - 1] || !seen[anchors[1] - 1]) Rcpp::stop("Both anchors must be in the split union.");
    const bool evaluate              = forced.isNotNull();
    const Rcpp::NumericVector values = evaluate ? Rcpp::NumericVector(forced.get()) : Rcpp::NumericVector(0);
    if ((evaluate && values.size() != s.N) || (!evaluate && uniforms.size() != others.size()))
        Rcpp::stop("Invalid row-split labels or uniforms.");

    // Place the selected subjects in opposite groups before assigning the others.
    RowAllocation result(s);
    result.first  = row_group(s, Rcpp::IntegerVector::create(anchors[0]));
    result.second = row_group(s, Rcpp::IntegerVector::create(anchors[1]));
    std::fill(result.labels.begin(), result.labels.end(), static_cast<double>(h));
    result.labels[anchors[1] - 1] = g;
    for (std::size_t j = 0; j < others.size(); ++j) {
        Rcpp::checkUserInterrupt();
        const int i    = others[j];
        RowGroup first = row_add(s, result.first, i), second = row_add(s, result.second, i);

        // Sum across response rows before choosing a single full-matrix label.
        const double a       = weights[h - 1] + first.integral - result.first.integral;
        const double b       = weights[g - 1] + second.integral - result.second.integral;
        const double maximum = std::max(a, b);
        if (!std::isfinite(maximum)) Rcpp::stop("No finite conditional coefficient allocation score.");
        const double pa = std::exp(a - maximum), pb = std::exp(b - maximum), total = pa + pb;
        bool select;
        if (evaluate) {
            if (values[i] != h && values[i] != g) Rcpp::stop("Invalid forced conditional coefficient label.");
            select = values[i] == h;
        } else {
            if (!std::isfinite(uniforms[j]) || uniforms[j] < 0 || uniforms[j] > 1)
                Rcpp::stop("Invalid row-split uniform.");
            select = pa >= pb ? uniforms[j] <= pa / total : uniforms[j] > pb / total;
        }
        result.labels[i] = select ? h : g;
        result.probability += (select ? a : b) - maximum - std::log(total);
        if (select) result.first = std::move(first);
        else result.second = std::move(second);
    }

    return result;
}

// Return a factor-conditional split's labels, log assignment probability and two Gaussian groups.
// The label vector has length N, where N is the number of subjects; group means and
// precision-times-mean values have length D * P, for D response nodes and P predictors.
// Called from R to check the proposed assignments and forward/reverse probabilities.
// [[Rcpp::export(name = ".pdpmvar_eigen_row_restricted")]]
Rcpp::List pdpmvar_eigen_row_restricted(const Rcpp::List& statistics,
                                        const Rcpp::IntegerVector& subjects,
                                        const Rcpp::IntegerVector& anchors,
                                        const Rcpp::NumericVector& weights,
                                        int h,
                                        int g,
                                        Rcpp::Nullable<Rcpp::NumericVector> forced,
                                        const Rcpp::NumericVector& uniforms) {
    const RowStatistics s(statistics);
    const RowAllocation result = row_restricted(s, subjects, anchors, weights, h, g, forced, uniforms);

    return Rcpp::List::create(Rcpp::Named("labels") = result.labels, Rcpp::Named("log_probability") = result.probability,
        Rcpp::Named("groups") = Rcpp::List::create(row_wrap(s, result.first), row_wrap(s, result.second)));
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Split/merge moves conditional on sampled factors ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Propose a coefficient split/merge with sampled factors held fixed, then accept or reject it.
// Use Gaussian row integrals to compare the groups, keeping one label for each full matrix.
// After either decision, draw both affected coefficient matrices from their conditional distributions.
// For D nodes and P predictors, adjusted subject data are D x T_i, predictors are P x T_i,
// residual_variances is D x N and variance is D x P, where N is subjects and T_i is observations.
// Return N labels, the updated coefficient parameter list and a record of the move.
// Called by pdpmvar_eigen_sampler() when the factor_conditional coefficient move is scheduled.
Rcpp::List move_coefficient_rows_with_design(const Rcpp::List& adjusted,
                                             const Rcpp::List& design,
                                             const Rcpp::List& design_cross,
                                             const Eigen::MatrixXd& residual_variances,
                                             const Rcpp::NumericVector& labels,
                                             const Rcpp::List& atoms,
                                             const Rcpp::NumericMatrix& variance,
                                             double concentration,
                                             int slots,
                                             const CoefficientDesign* saved_design) {
    const std::vector<int> empty = move_empty_slots(labels, slots);
    if (labels.size() != adjusted.size() || !std::isfinite(concentration) || concentration <= 0)
        Rcpp::stop("Invalid conditional move labels or concentration.");
    for (double label : labels) if (label > atoms.size()) Rcpp::stop("Conditional move labels must index atoms.");
    if (labels.size() < 2) return move_result(labels, atoms, "skip", false, NA_REAL);
    const Rcpp::IntegerVector anchors = move_anchors(labels.size());
    const int h                       = labels[anchors[0] - 1];
    int g                             = labels[anchors[1] - 1];
    if (h > slots || g > slots) return move_result(labels, atoms, "skip", false, NA_REAL);
    const bool split = h == g;
    if (split && empty.empty()) return move_result(labels, atoms, "skip", false, NA_REAL);

    // Draw auxiliary stick weights, keeping the model's prior and counts at later positions.
    const Rcpp::NumericVector weights = pdpmvar_eigen_prefix_weights(labels, concentration, slots);
    const Rcpp::List statistics = row_statistics_with_design(adjusted, design, design_cross, residual_variances,
        Rcpp::as<Eigen::MatrixXd>(variance), saved_design);
    const RowStatistics s(statistics, saved_design);
    if (split) g = empty[static_cast<int>(R_unif_index(empty.size()))];
    const Rcpp::IntegerVector subjects = move_subjects(labels, h, g);
    Rcpp::NumericVector uniforms(split ? subjects.size() - 2 : 0);
    for (double& u : uniforms) u = R::runif(0.0, 1.0);
    const RowAllocation allocation = row_restricted(s, subjects, anchors, weights, h, g,
        split ? Rcpp::Nullable<Rcpp::NumericVector>(R_NilValue) : Rcpp::Nullable<Rcpp::NumericVector>(labels), uniforms);
    Rcpp::NumericVector proposed = Rcpp::clone(labels);
    for (int i : subjects) proposed[i - 1] = split ? allocation.labels[i - 1] : static_cast<double>(h);

    // Compare the two groups with their merged version, reusing the split calculations.
    const RowGroup merged = row_group(s, subjects);
    double target         = (split ? 1 : -1) * (allocation.first.integral + allocation.second.integral - merged.integral);
    for (int i : subjects) target += weights[static_cast<int>(proposed[i - 1]) - 1] - weights[static_cast<int>(labels[i - 1]) - 1];

    // Include the assignment probability and the chance of selecting an empty stick position.
    const std::vector<int> reverse_empty = split ? empty : move_empty_slots(proposed, slots);
    const double density                 = allocation.probability - std::log(static_cast<double>(reverse_empty.size()));
    const double ratio                   = target + (split ? -density : density);
    if (std::isnan(ratio)) Rcpp::stop("Invalid conditional coefficient split/merge ratio.");
    const bool accepted                     = std::log(R::runif(0.0, 1.0)) < std::min(0.0, ratio);
    const Rcpp::NumericVector result_labels = accepted ? proposed : labels;

    // Both matrices were integrated out of the decision, so draw them after either outcome.
    Rcpp::List result_atoms = Rcpp::clone(atoms);
    while (result_atoms.size() < std::max(h, g)) result_atoms.push_back(sampler_new_coefficient(variance));
    for (int cluster : {h, g}) {
        const RowGroup group = row_group(s, move_subjects(result_labels, cluster, -1));
        Rcpp::NumericMatrix noise(s.D, s.P);
        for (double& z : noise) z = R::rnorm(0.0, 1.0);
        const Rcpp::List conditional = row_wrap(s, group, noise);
        result_atoms[cluster - 1]    = Rcpp::List::create(Rcpp::Named("coefficient") = conditional["coefficient"]);
    }

    return move_result(result_labels, result_atoms, split ? "split" : "merge", accepted, ratio);
}

// Try a split or merge for calls that supply the predictors as R lists.
// [[Rcpp::export(name = ".pdpmvar_eigen_row_move")]]
Rcpp::List pdpmvar_eigen_row_move(const Rcpp::List& adjusted,
                                  const Rcpp::List& design,
                                  const Rcpp::List& design_cross,
                                  const Eigen::MatrixXd& residual_variances,
                                  const Rcpp::NumericVector& labels,
                                  const Rcpp::List& atoms,
                                  const Rcpp::NumericMatrix& variance,
                                  double concentration,
                                  int slots) {
    return move_coefficient_rows_with_design(adjusted, design, design_cross, residual_variances,
        labels, atoms, variance, concentration, slots, nullptr);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Split/merge moves with factors integrated out ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Sum the selected subjects' Gaussian contributions, add the prior once, and integrate coefficients out.
// For D response nodes and P predictors, return coefficient-by-coefficient precision and
// Cholesky matrices, (D * P) x (D * P), and Gaussian vectors of length D * P.
// Used to compare current and proposed groups and draw their coefficient matrices
// in the split/merge method that integrates out the factors.
Rcpp::List move_coefficient_group(const Rcpp::List& statistics, const Rcpp::IntegerVector& subjects) {
    Eigen::MatrixXd precision   = Rcpp::as<Eigen::MatrixXd>(statistics["prior_precision"]);
    Eigen::VectorXd information = Eigen::VectorXd::Zero(precision.rows());
    Rcpp::List precisions       = statistics["precision"], informations = statistics["information"];
    for (int subject : subjects) {
        precision += Rcpp::as<Eigen::MatrixXd>(precisions[subject - 1]);
        information += Rcpp::as<Eigen::VectorXd>(informations[subject - 1]);
    }
    Rcpp::NumericMatrix precision_r = Rcpp::wrap(precision);
    Rcpp::NumericVector information_r(information.size());
    std::copy(information.data(), information.data() + information.size(), information_r.begin());

    return pdpmvar_eigen_integral(precision_r, information_r,
        Rcpp::as<double>(statistics["prior_log_determinant_half"]));
}

// Calculate the acceptance ratio for a full-matrix coefficient split/merge.
// Compare integrated Gaussian contributions and log stick weights, then include the
// forward/reverse assignment probabilities and the probability of choosing an empty stick position.
// Input label vectors have length N, where N is subjects. Return the log acceptance ratio,
// log target change, log forward proposal probability and log reverse proposal probability.
// Used by pdpmvar_eigen_coefficient_move() and separate checks of its acceptance calculation.
// [[Rcpp::export(name = ".pdpmvar_eigen_coefficient_move_ratio")]]
Rcpp::List pdpmvar_eigen_coefficient_move_ratio(const Rcpp::NumericVector& old_labels,
                                                const Rcpp::NumericVector& new_labels,
                                                const Rcpp::List& statistics,
                                                const Rcpp::NumericVector& log_weights,
                                                const Rcpp::IntegerVector& anchors,
                                                int slots) {
    move_validate_ratio(old_labels, new_labels, log_weights, anchors, slots);
    Rcpp::List precisions = statistics["precision"];
    if (precisions.size() != old_labels.size()) Rcpp::stop("Coefficient statistics and move labels must agree.");
    const int h                        = old_labels[anchors[0] - 1];
    const bool split                   = h == old_labels[anchors[1] - 1];
    const int g                        = split ? new_labels[anchors[1] - 1] : old_labels[anchors[1] - 1];
    const Rcpp::IntegerVector subjects = move_subjects(old_labels, h, g);

    // Include the coefficient prior once for each current or proposed group.
    const Rcpp::List old_h = move_coefficient_group(statistics, move_subjects(old_labels, h));
    const Rcpp::List old_g = move_coefficient_group(statistics, move_subjects(old_labels, g));
    const Rcpp::List new_h = move_coefficient_group(statistics, move_subjects(new_labels, h));
    const Rcpp::List new_g = move_coefficient_group(statistics, move_subjects(new_labels, g));
    double target = Rcpp::as<double>(new_h["log_integral"]) + Rcpp::as<double>(new_g["log_integral"]) -
        Rcpp::as<double>(old_h["log_integral"]) - Rcpp::as<double>(old_g["log_integral"]);
    long double weight_change = 0.0;
    for (int subject : subjects) weight_change += log_weights[static_cast<int>(new_labels[subject - 1]) - 1] -
        log_weights[static_cast<int>(old_labels[subject - 1]) - 1];
    target += static_cast<double>(weight_change);

    // A split chooses an empty position and assignments; the reverse merge makes neither choice.
    const std::vector<int> empty = move_empty_slots(split ? old_labels : new_labels, slots);
    Rcpp::Nullable<Rcpp::NumericVector> forced(split ? new_labels : old_labels);
    const Rcpp::List allocation = pdpmvar_eigen_restricted_split(statistics, subjects, anchors,
        log_weights, h, g, forced, Rcpp::NumericVector(0));
    const double density = -std::log(static_cast<double>(empty.size())) + Rcpp::as<double>(allocation["log_probability"]);
    const double forward = split ? density : 0.0;
    const double reverse = split ? 0.0 : density;

    return Rcpp::List::create(Rcpp::Named("log_ratio") = target + reverse - forward,
        Rcpp::Named("target_change") = target, Rcpp::Named("log_forward") = forward, Rcpp::Named("log_reverse") = reverse);
}

// Propose splitting one coefficient group or merging two, with latent factors integrated out.
// Choose two subjects, propose group labels, calculate the acceptance probability and draw
// both affected coefficient matrices after accepting or rejecting the move.
// For D nodes and P predictors, data are D x T_i, predictors are P x T_i and each
// subject covariance is D x D, where T_i is that subject's number of observations.
// Return N labels, updated D x P coefficient matrices and a move record, for N subjects.
// Called by pdpmvar_eigen_sampler() when the collapsed coefficient move is scheduled.
// [[Rcpp::export(name = ".pdpmvar_eigen_coefficient_move")]]
Rcpp::List pdpmvar_eigen_coefficient_move(const Rcpp::List& X,
                                          const Rcpp::List& design,
                                          const Rcpp::List& design_cross,
                                          const Rcpp::List& covariances,
                                          const Rcpp::NumericVector& labels,
                                          const Rcpp::List& atoms,
                                          const Rcpp::NumericMatrix& variance,
                                          double concentration,
                                          int slots) {
    const std::vector<int> empty = move_empty_slots(labels, slots);
    for (double label : labels) if (label > atoms.size()) Rcpp::stop("Move labels must index existing atoms.");
    if (labels.size() < 2) return move_result(labels, atoms);
    const Rcpp::IntegerVector anchors = move_anchors(labels.size());
    const int h                       = labels[anchors[0] - 1];
    int g                             = labels[anchors[1] - 1];
    if (h > slots || g > slots) return move_result(labels, atoms);
    const bool split = h == g;
    if (split && empty.empty()) return move_result(labels, atoms);

    // The extra move uses only the selected stick positions; ordinary slice sampling keeps the tail.
    const Rcpp::NumericVector log_weights = pdpmvar_eigen_prefix_weights(labels, concentration, slots);
    const Rcpp::List statistics           = pdpmvar_eigen_statistics(X, design, design_cross, covariances, variance);
    Rcpp::NumericVector proposed          = Rcpp::clone(labels);
    if (split) {
        g                                  = empty[static_cast<int>(R_unif_index(empty.size()))];
        const Rcpp::IntegerVector subjects = move_subjects(labels, h);
        Rcpp::NumericVector uniforms(subjects.size() - 2);
        for (double& uniform : uniforms) uniform = R::runif(0.0, 1.0);
        const Rcpp::List allocation = pdpmvar_eigen_restricted_split(statistics, subjects, anchors,
            log_weights, h, g, R_NilValue, uniforms);
        const Rcpp::NumericVector allocation_labels = allocation["labels"];
        for (int subject : subjects) proposed[subject - 1] = allocation_labels[subject - 1];
    } else {
        for (int subject = 0; subject < labels.size(); ++subject) if (labels[subject] == g) proposed[subject] = h;
    }

    // Apply the forward/reverse proposal correction before accepting or rejecting the move.
    const Rcpp::List ratio = pdpmvar_eigen_coefficient_move_ratio(labels, proposed, statistics, log_weights, anchors, slots);
    const double log_ratio = Rcpp::as<double>(ratio["log_ratio"]);
    if (std::isnan(log_ratio)) Rcpp::stop("Non-finite coefficient split/merge acceptance calculation.");
    const bool accepted                     = std::log(R::runif(0.0, 1.0)) < std::min(0.0, log_ratio);
    const Rcpp::NumericVector result_labels = accepted ? proposed : labels;

    // Both matrices were integrated out of the decision, so draw them after either outcome.
    Rcpp::List result_atoms = Rcpp::clone(atoms);
    while (result_atoms.size() < std::max(h, g)) result_atoms.push_back(sampler_new_coefficient(variance));
    for (int cluster : {h, g}) {
        const Rcpp::List group      = move_coefficient_group(statistics, move_subjects(result_labels, cluster));
        const Eigen::VectorXd mean  = Rcpp::as<Eigen::VectorXd>(group["mean"]);
        const Eigen::MatrixXd upper = Rcpp::as<Eigen::MatrixXd>(group["upper"]);
        Eigen::VectorXd noise(mean.size());
        for (int index = 0; index < noise.size(); ++index) noise[index] = R::rnorm(0.0, 1.0);
        const Eigen::VectorXd draw = mean + upper.triangularView<Eigen::Upper>().solve(noise);
        Rcpp::NumericMatrix coefficient(variance.nrow(), variance.ncol());
        std::copy(draw.data(), draw.data() + draw.size(), coefficient.begin());
        result_atoms[cluster - 1] = Rcpp::List::create(Rcpp::Named("coefficient") = coefficient);
    }

    return move_result(result_labels, result_atoms, split ? "split" : "merge", accepted, log_ratio);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Keep R and C++ random draws in sequence ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Read R's random-number state before sampling and save the updated state when sampling finishes.
// Used once in pdpmvar_eigen_sampler() so set.seed() controls all C++ draws.
// Automatic Rcpp random-number handling is disabled for that function to avoid loading a second state.
struct SamplerRngState {
    // Read the current seed on entry; save the updated seed when this sampler call ends.
    SamplerRngState() { GetRNGstate(); }
    ~SamplerRngState() { PutRNGstate(); }
};

// Save the current C++ random-number state before calling an R function, then read it back afterward.
// Used in pdpmvar_eigen_sampler() when R draws cluster labels or prints progress.
// This keeps a single sequence of random draws even if an R function also draws random values.
struct SamplerCallbackRng {
    // Make C++ draws visible to R before the call, then continue from R's updated seed.
    SamplerCallbackRng() { PutRNGstate(); }
    ~SamplerCallbackRng() { GetRNGstate(); }
};

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Run MCMC iterations and collect posterior summaries ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Run the full PDPMVAR sampler and collect posterior summaries for each subject.
// Called by pdpmvar() after R centers the data if requested, prepares lagged predictors
// and chooses starting groups and parameters. D is nodes, P = D * K is predictors,
// N is subjects and M is the number of retained iterations; K is the number of lags.
// Return coefficient summaries as response-by-predictor-by-subject arrays, D x P x N,
// and covariance summaries as node-by-node-by-subject arrays, D x D x N.
// Label histories are M x N and co-clustering probabilities are N x N.
// Optional draw arrays add retained iteration as the last dimension: D x P x N x M
// and D x D x N x M. Cluster counts and split/merge records also include burn-in.
// [[Rcpp::export(name = ".pdpmvar_eigen_sampler", rng = false)]]
Rcpp::List pdpmvar_eigen_sampler(const Rcpp::List& X,
                                 const Rcpp::List& design,
                                 const Rcpp::List& design_cross,
                                 const Rcpp::List& initial_state,
                                 const Rcpp::List& control,
                                 const Rcpp::List& callbacks) {
    // Read dimensions and iteration settings, then check the supplied starting values.
    const int n_subjects                = X.size();
    Rcpp::NumericMatrix shared_variance = Rcpp::clone(Rcpp::as<Rcpp::NumericMatrix>(initial_state["shared_variance"]));
    const int n_nodes                   = shared_variance.nrow();
    const int n_predictors              = shared_variance.ncol();
    const int n_burnin                  = Rcpp::as<int>(control["n_burnin"]);
    const int n_mcmc                    = Rcpp::as<int>(control["n_mcmc"]);
    const int n_factors                 = Rcpp::as<int>(control["n_factors"]);
    if (n_subjects == 0 || n_nodes == 0 || n_predictors == 0 || design.size() != n_subjects ||
        design_cross.size() != n_subjects || n_burnin < 0 || n_mcmc < 1 || n_factors < 1 || n_factors > n_nodes ||
        static_cast<double>(n_burnin) + n_mcmc > std::numeric_limits<int>::max()) {
        Rcpp::stop("Invalid sampler data dimensions or iteration controls.");
    }
    const int total_iterations             = n_burnin + n_mcmc;
    const double coefficient_concentration = Rcpp::as<double>(control["coefficient_concentration"]);
    const double covariance_concentration  = Rcpp::as<double>(control["covariance_concentration"]);
    const double cov_shape                 = Rcpp::as<double>(control["cov_shape"]);
    const double cov_rate                  = Rcpp::as<double>(control["cov_rate"]);
    const bool laplace                     = Rcpp::as<std::string>(control["coefficient_prior"]) == "laplace";
    const bool store_draws                 = Rcpp::as<bool>(control["store_draws"]);
    const bool verbose                     = Rcpp::as<bool>(control["verbose"]);
    const int coefficient_interval         = Rcpp::as<int>(control["coefficient_split_merge_every"]);
    const int covariance_interval          = Rcpp::as<int>(control["split_merge_every"]);
    const int coefficient_slots            = Rcpp::as<int>(control["coefficient_split_merge_slots"]);
    const int covariance_slots             = Rcpp::as<int>(control["split_merge_slots"]);
    const bool factor_conditional = control.containsElementNamed("coefficient_split_merge") &&
        Rcpp::as<std::string>(control["coefficient_split_merge"]) == "factor_conditional";
    if (!std::isfinite(coefficient_concentration) || coefficient_concentration <= 0.0 ||
        !std::isfinite(covariance_concentration) || covariance_concentration <= 0.0 ||
        !std::isfinite(cov_shape) || cov_shape <= 0.0 || !std::isfinite(cov_rate) || cov_rate <= 0.0 ||
        coefficient_interval < 0 || covariance_interval < 0 || coefficient_slots < 2 || covariance_slots < 2) {
        Rcpp::stop("Invalid sampler concentration, covariance prior or move controls.");
    }

    // Work with copies so the starting values supplied from R remain unchanged.
    Rcpp::List coefficient_atoms           = Rcpp::clone(Rcpp::as<Rcpp::List>(initial_state["coefficient_atoms"]));
    Rcpp::List covariance_atoms            = Rcpp::clone(Rcpp::as<Rcpp::List>(initial_state["covariance_atoms"]));
    Rcpp::NumericVector coefficient_labels = Rcpp::clone(Rcpp::as<Rcpp::NumericVector>(initial_state["coefficient_labels"]));
    Rcpp::NumericVector covariance_labels  = Rcpp::clone(Rcpp::as<Rcpp::NumericVector>(initial_state["covariance_labels"]));
    double lambda_squared                  = Rcpp::as<double>(initial_state["lambda_squared"]);
    if (coefficient_labels.size() != n_subjects || covariance_labels.size() != n_subjects ||
        !std::isfinite(lambda_squared) || lambda_squared <= 0.0) Rcpp::stop("Invalid initial sampler state.");
    sampler_counts(coefficient_labels, coefficient_atoms.size());
    sampler_counts(covariance_labels, covariance_atoms.size());
    std::vector<Eigen::MatrixXd> data(n_subjects), predictors(n_subjects);
    for (int subject = 0; subject < n_subjects; ++subject) {
        data[subject]       = Rcpp::as<Eigen::MatrixXd>(X[subject]);
        predictors[subject] = Rcpp::as<Eigen::MatrixXd>(design[subject]);
        if (data[subject].rows() != n_nodes || predictors[subject].rows() != n_predictors ||
            data[subject].cols() == 0 || data[subject].cols() != predictors[subject].cols() ||
            !data[subject].allFinite() || !predictors[subject].allFinite()) Rcpp::stop("Invalid sampler subject matrices.");
    }
    Rcpp::Function progress            = callbacks["message"];
    Rcpp::Function assign_coefficients = callbacks["assign_coefficients"];
    Rcpp::Function assign_covariances  = callbacks["assign_covariances"];

    // Keep running sums for posterior summaries and matrices of saved labels and diagnostics.
    const R_xlen_t coefficient_size  = static_cast<R_xlen_t>(n_nodes) * n_predictors;
    const R_xlen_t covariance_size   = static_cast<R_xlen_t>(n_nodes) * n_nodes;
    const R_xlen_t coefficient_total = coefficient_size * n_subjects;
    const R_xlen_t covariance_total  = covariance_size * n_subjects;
    Rcpp::NumericVector coefficient_sum(coefficient_total), positive_sum(coefficient_total), covariance_sum(covariance_total);
    coefficient_sum.attr("dim") = Rcpp::IntegerVector::create(n_nodes, n_predictors, n_subjects);
    positive_sum.attr("dim")    = coefficient_sum.attr("dim");
    covariance_sum.attr("dim")  = Rcpp::IntegerVector::create(n_nodes, n_nodes, n_subjects);
    Rcpp::NumericMatrix coefficient_membership(n_mcmc, n_subjects), covariance_membership(n_mcmc, n_subjects);
    Rcpp::NumericMatrix coefficient_similarity(n_subjects, n_subjects), covariance_similarity(n_subjects, n_subjects);
    Rcpp::NumericMatrix cluster_counts(total_iterations, 2), parameter_traces(n_mcmc, 3);
    cluster_counts.attr("dimnames") = Rcpp::List::create(R_NilValue, Rcpp::CharacterVector::create("coefficient", "covariance"));
    parameter_traces.attr("dimnames") = Rcpp::List::create(R_NilValue,
        Rcpp::CharacterVector::create("log_likelihood", "mean_node_variance", "coefficient_rms"));
    Rcpp::NumericMatrix shared_variance_sum(n_nodes, n_predictors);
    Rcpp::NumericVector lambda_trace(n_mcmc);
    Rcpp::IntegerVector iterations(total_iterations);
    Rcpp::CharacterVector coefficient_types(total_iterations), covariance_types(total_iterations);
    Rcpp::LogicalVector coefficient_accepted(total_iterations), covariance_accepted(total_iterations);
    Rcpp::NumericVector coefficient_ratios(total_iterations, NA_REAL), covariance_ratios(total_iterations, NA_REAL);
    for (int iteration = 0; iteration < total_iterations; ++iteration) {
        iterations[iteration]        = iteration + 1;
        coefficient_types[iteration] = "none";
        covariance_types[iteration]  = "none";
    }

    // Full draw arrays are optional because they can require much more memory than summary sums.
    Rcpp::NumericVector coefficient_draws(0), covariance_draws(0);
    if (store_draws) {
        coefficient_draws             = Rcpp::NumericVector(coefficient_total * n_mcmc);
        covariance_draws              = Rcpp::NumericVector(covariance_total * n_mcmc);
        coefficient_draws.attr("dim") = Rcpp::IntegerVector::create(n_nodes, n_predictors, n_subjects, n_mcmc);
        covariance_draws.attr("dim")  = Rcpp::IntegerVector::create(n_nodes, n_nodes, n_subjects, n_mcmc);
    }

    // Use R's current seed for the whole chain.
    SamplerRngState rng_state;
    for (int iteration = 0; iteration < total_iterations; ++iteration) {
        Rcpp::checkUserInterrupt();

        // Try an extra coefficient split/merge at the requested iteration interval.
        if (coefficient_interval > 0 && (iteration + 1) % coefficient_interval == 0) {
            Rcpp::List move;
            if (factor_conditional) {
                // Draw factors and hold them fixed while comparing coefficient groups.
                // Draw new factors again before the ordinary coefficient update below.
                Rcpp::List adjusted(n_subjects);
                Eigen::MatrixXd variances(n_nodes, n_subjects);
                for (int subject = 0; subject < n_subjects; ++subject) {
                    const Rcpp::List a             = coefficient_atoms[static_cast<int>(coefficient_labels[subject]) - 1];
                    const Rcpp::List atom          = covariance_atoms[static_cast<int>(covariance_labels[subject]) - 1];
                    const Eigen::MatrixXd working  = Rcpp::as<Eigen::MatrixXd>(atom["working_loadings"]);
                    const Eigen::MatrixXd residual = data[subject] - Rcpp::as<Eigen::MatrixXd>(a["coefficient"]) * predictors[subject];
                    Eigen::MatrixXd noise(working.cols(), data[subject].cols());
                    for (int index = 0; index < noise.size(); ++index) noise.data()[index] = R::rnorm(0.0, 1.0);
                    const Eigen::MatrixXd factors = pdpmvar_eigen_factors(residual, atom, noise);
                    adjusted[subject]             = Eigen::MatrixXd(data[subject] - working * factors);
                    variances.col(subject)        = Rcpp::as<Eigen::VectorXd>(atom["variance"]);
                }
                move = pdpmvar_eigen_row_move(adjusted, design, design_cross, variances,
                    coefficient_labels, coefficient_atoms, shared_variance, coefficient_concentration, coefficient_slots);
            } else {
                Rcpp::List subject_covariances(n_subjects);
                for (int subject = 0; subject < n_subjects; ++subject) {
                    Rcpp::List atom              = covariance_atoms[static_cast<int>(covariance_labels[subject]) - 1];
                    subject_covariances[subject] = atom["covariance"];
                }
                move = pdpmvar_eigen_coefficient_move(X, design, design_cross, subject_covariances,
                    coefficient_labels, coefficient_atoms, shared_variance, coefficient_concentration, coefficient_slots);
            }
            coefficient_labels              = Rcpp::as<Rcpp::NumericVector>(move["labels"]);
            coefficient_atoms               = Rcpp::as<Rcpp::List>(move["atoms"]);
            Rcpp::List record               = move["record"];
            coefficient_types[iteration]    = Rcpp::as<std::string>(record["type"]);
            coefficient_accepted[iteration] = Rcpp::as<bool>(record["accepted"]);
            coefficient_ratios[iteration]   = Rcpp::as<double>(record["log_ratio"]);
        }

        // Draw coefficient weights and slices, create any newly needed groups, then assign subjects.
        Rcpp::List slice            = pdpmvar_eigen_slice_weights(coefficient_labels, coefficient_concentration);
        Rcpp::NumericVector weights = slice["weights"], slices = slice["slices"];
        Rcpp::List next_coefficients(weights.size());
        for (int cluster = 0; cluster < weights.size(); ++cluster) {
            next_coefficients[cluster] = cluster < coefficient_atoms.size() ? coefficient_atoms[cluster] :
                static_cast<SEXP>(sampler_new_coefficient(shared_variance));
        }
        coefficient_atoms = next_coefficients;
        Rcpp::NumericVector uniforms(n_subjects);
        Rcpp::List assignment;
        // With more than 200 components, let R draw labels using its weighted sampling function.
        if (weights.size() > 200) {
            SamplerCallbackRng callback_rng;
            assignment = assign_coefficients(X, design, coefficient_atoms, covariance_labels,
                covariance_atoms, slice);
        } else {
            for (int subject = 0; subject < n_subjects; ++subject) uniforms[subject] = R::runif(0.0, 1.0);
            assignment = pdpmvar_eigen_assign_coefficients(X, design, coefficient_atoms, covariance_labels,
                covariance_atoms, weights, slices, uniforms);
        }
        coefficient_labels = Rcpp::as<Rcpp::NumericVector>(assignment["labels"]);

        // Draw fresh factors under the new labels before updating coefficient rows.
        Rcpp::List contributions(n_subjects);
        Eigen::MatrixXd residual_variances(n_nodes, n_subjects);
        for (int subject = 0; subject < n_subjects; ++subject) {
            Rcpp::List coefficient_atom       = coefficient_atoms[static_cast<int>(coefficient_labels[subject]) - 1];
            Rcpp::List covariance_atom        = covariance_atoms[static_cast<int>(covariance_labels[subject]) - 1];
            const Eigen::MatrixXd coefficient = Rcpp::as<Eigen::MatrixXd>(coefficient_atom["coefficient"]);
            const Eigen::MatrixXd working     = Rcpp::as<Eigen::MatrixXd>(covariance_atom["working_loadings"]);
            const Eigen::MatrixXd residual    = data[subject] - coefficient * predictors[subject];
            Eigen::MatrixXd noise(n_factors, residual.cols());
            for (int time = 0; time < residual.cols(); ++time) {
                for (int factor = 0; factor < n_factors; ++factor) noise(factor, time) = R::rnorm(0.0, 1.0);
            }
            const Eigen::MatrixXd factors   = pdpmvar_eigen_factors(residual, covariance_atom, noise);
            contributions[subject]          = Eigen::MatrixXd(working * factors);
            residual_variances.col(subject) = Rcpp::as<Eigen::VectorXd>(covariance_atom["variance"]);
        }
        coefficient_atoms = pdpmvar_eigen_coefficient_update(X, design, design_cross, contributions,
            coefficient_labels, coefficient_atoms, Rcpp::as<Eigen::MatrixXd>(shared_variance), residual_variances);

        // Update shared shrinkage using occupied groups, then redraw unused matrices from the updated prior.
        const std::vector<int> coefficient_counts = sampler_counts(coefficient_labels, coefficient_atoms.size());
        if (laplace) {
            Rcpp::List shrinkage = pdpmvar_eigen_shrinkage(coefficient_labels, coefficient_atoms, shared_variance, lambda_squared);
            shared_variance      = Rcpp::as<Rcpp::NumericMatrix>(shrinkage["shared_variance"]);
            lambda_squared       = Rcpp::as<double>(shrinkage["lambda_squared"]);
        }
        for (int cluster = 0; cluster < coefficient_atoms.size(); ++cluster) {
            if (coefficient_counts[cluster] == 0) coefficient_atoms[cluster] = sampler_new_coefficient(shared_variance);
        }

        // Calculate residuals from the new coefficient matrices for all covariance updates.
        Rcpp::List residuals(n_subjects);
        for (int subject = 0; subject < n_subjects; ++subject) {
            Rcpp::List atom    = coefficient_atoms[static_cast<int>(coefficient_labels[subject]) - 1];
            residuals[subject] = Eigen::MatrixXd(data[subject] - Rcpp::as<Eigen::MatrixXd>(atom["coefficient"]) * predictors[subject]);
        }

        // Try the additional covariance split/merge without changing coefficient labels.
        if (covariance_interval > 0 && (iteration + 1) % covariance_interval == 0) {
            Rcpp::List move = pdpmvar_eigen_covariance_move(residuals, covariance_labels, covariance_atoms,
                covariance_concentration, n_factors, cov_shape, cov_rate, covariance_slots);
            covariance_labels              = Rcpp::as<Rcpp::NumericVector>(move["labels"]);
            covariance_atoms               = Rcpp::as<Rcpp::List>(move["atoms"]);
            Rcpp::List record              = move["record"];
            covariance_types[iteration]    = Rcpp::as<std::string>(record["type"]);
            covariance_accepted[iteration] = Rcpp::as<bool>(record["accepted"]);
            covariance_ratios[iteration]   = Rcpp::as<double>(record["log_ratio"]);
        }

        // Draw covariance weights and slices, create any newly needed groups, then assign subjects.
        slice   = pdpmvar_eigen_slice_weights(covariance_labels, covariance_concentration);
        weights = Rcpp::as<Rcpp::NumericVector>(slice["weights"]);
        slices  = Rcpp::as<Rcpp::NumericVector>(slice["slices"]);
        Rcpp::List next_covariances(weights.size());
        for (int cluster = 0; cluster < weights.size(); ++cluster) {
            next_covariances[cluster] = cluster < covariance_atoms.size() ? covariance_atoms[cluster] :
                static_cast<SEXP>(sampler_new_covariance(n_nodes, n_factors, cov_shape, cov_rate));
        }
        covariance_atoms = next_covariances;
        if (weights.size() > 200) {
            SamplerCallbackRng callback_rng;
            assignment = assign_covariances(residuals, covariance_atoms, slice);
        } else {
            for (int subject = 0; subject < n_subjects; ++subject) uniforms[subject] = R::runif(0.0, 1.0);
            assignment = pdpmvar_eigen_assign_covariances(residuals, covariance_atoms, weights, slices, uniforms);
        }
        covariance_labels = Rcpp::as<Rcpp::NumericVector>(assignment["labels"]);

        // Pool residual columns within each covariance group, then draw that group's parameters.
        const std::vector<int> covariance_counts = sampler_counts(covariance_labels, covariance_atoms.size());
        for (int cluster = 0; cluster < covariance_atoms.size(); ++cluster) {
            if (covariance_counts[cluster] == 0) {
                covariance_atoms[cluster] = sampler_new_covariance(n_nodes, n_factors, cov_shape, cov_rate);
                continue;
            }
            int n_times = 0;
            for (int subject = 0; subject < n_subjects; ++subject) {
                if (covariance_labels[subject] == cluster + 1) n_times += data[subject].cols();
            }
            Eigen::MatrixXd pooled(n_nodes, n_times);
            int offset = 0;
            for (int subject = 0; subject < n_subjects; ++subject) {
                if (covariance_labels[subject] != cluster + 1) continue;
                const Eigen::MatrixXd residual             = Rcpp::as<Eigen::MatrixXd>(residuals[subject]);
                pooled.middleCols(offset, residual.cols()) = residual;
                offset += residual.cols();
            }
            covariance_atoms[cluster] = pdpmvar_eigen_covariance_update(pooled,
                Rcpp::as<Rcpp::List>(covariance_atoms[cluster]), cov_shape, cov_rate);
        }

        // Count groups throughout burn-in as well as the retained iterations.
        cluster_counts(iteration, 0) = std::count_if(coefficient_counts.begin(), coefficient_counts.end(), [](int n) { return n > 0; });
        cluster_counts(iteration, 1) = std::count_if(covariance_counts.begin(), covariance_counts.end(), [](int n) { return n > 0; });

        // After burn-in, save labels and add each subject's parameters to the posterior sums.
        if (iteration >= n_burnin) {
            const int retained     = iteration - n_burnin;
            lambda_trace[retained] = std::sqrt(lambda_squared);
            for (int index = 0; index < shared_variance.size(); ++index) shared_variance_sum[index] += shared_variance[index];
            parameter_traces(retained, 0) = pdpmvar_eigen_total_likelihood(residuals, covariance_labels, covariance_atoms);
            double mean_variance          = 0.0, mean_square = 0.0;
            for (int subject = 0; subject < n_subjects; ++subject) {
                coefficient_membership(retained, subject) = coefficient_labels[subject];
                covariance_membership(retained, subject)  = covariance_labels[subject];

                // Count how often each pair of subjects shares a label, regardless of its numeric value.
                for (int other = 0; other < n_subjects; ++other) {
                    coefficient_similarity(subject, other) += coefficient_labels[subject] == coefficient_labels[other];
                    covariance_similarity(subject, other)  += covariance_labels[subject] == covariance_labels[other];
                }
                Rcpp::List coefficient_atom     = coefficient_atoms[static_cast<int>(coefficient_labels[subject]) - 1];
                Rcpp::List covariance_atom      = covariance_atoms[static_cast<int>(covariance_labels[subject]) - 1];
                Rcpp::NumericMatrix coefficient = coefficient_atom["coefficient"], covariance = covariance_atom["covariance"];
                long double squares             = 0.0, diagonal_sum = 0.0;
                for (int index = 0; index < coefficient_size; ++index) {
                    const R_xlen_t position = subject * coefficient_size + index;
                    coefficient_sum[position] += coefficient[index];
                    positive_sum[position] += coefficient[index] > 0.0;
                    squares += coefficient[index] * coefficient[index];
                    if (store_draws) coefficient_draws[retained * coefficient_total + position] = coefficient[index];
                }
                for (int index = 0; index < covariance_size; ++index) {
                    const R_xlen_t position = subject * covariance_size + index;
                    covariance_sum[position] += covariance[index];
                    if (store_draws) covariance_draws[retained * covariance_total + position] = covariance[index];
                }
                for (int node = 0; node < n_nodes; ++node) diagonal_sum += covariance(node, node);
                mean_square += static_cast<double>(squares / coefficient_size) / n_subjects;
                mean_variance += static_cast<double>(diagonal_sum / n_nodes) / n_subjects;
            }
            parameter_traces(retained, 1) = mean_variance;
            parameter_traces(retained, 2) = std::sqrt(mean_square);
        }

        // Let R print progress after saving any retained results.
        if (verbose && ((iteration + 1) % 100 == 0 || iteration + 1 == total_iterations)) {
            std::ostringstream text;
            text << "Iteration " << iteration + 1 << ": " << cluster_counts(iteration, 0)
                 << " coefficient clusters, " << cluster_counts(iteration, 1) << " covariance clusters";
            SamplerCallbackRng callback_rng;
            progress(text.str());
        }
    }

    // Divide accumulated sums by the retained iteration count to obtain posterior summaries.
    Rcpp::NumericVector sign_tail(coefficient_total);
    sign_tail.attr("dim") = coefficient_sum.attr("dim");
    for (R_xlen_t index = 0; index < coefficient_total; ++index) {
        coefficient_sum[index] /= n_mcmc;
        positive_sum[index] /= n_mcmc;
        sign_tail[index] = 2.0 * std::min(positive_sum[index], 1.0 - positive_sum[index]);
    }
    for (R_xlen_t index = 0; index < covariance_total; ++index) covariance_sum[index] /= n_mcmc;
    for (int index = 0; index < shared_variance_sum.size(); ++index) shared_variance_sum[index] /= n_mcmc;
    for (int index = 0; index < coefficient_similarity.size(); ++index) {
        coefficient_similarity[index] /= n_mcmc;
        covariance_similarity[index] /= n_mcmc;
    }

    // Return both the posterior summaries and the last labels and parameter values.
    Rcpp::List final_state = Rcpp::List::create(Rcpp::Named("coefficient_atoms") = coefficient_atoms,
        Rcpp::Named("covariance_atoms") = covariance_atoms, Rcpp::Named("coefficient_labels") = coefficient_labels,
        Rcpp::Named("covariance_labels") = covariance_labels, Rcpp::Named("shared_variance") = shared_variance,
        Rcpp::Named("lambda_squared") = lambda_squared);
    Rcpp::RObject draws = R_NilValue;
    if (store_draws) draws = Rcpp::List::create(Rcpp::Named("coefficients") = coefficient_draws,
                                               Rcpp::Named("covariance") = covariance_draws);
    Rcpp::RObject lambda_result = laplace ? static_cast<SEXP>(lambda_trace) : R_NilValue;

    return Rcpp::List::create(Rcpp::Named("coefficient_mean") = coefficient_sum,
        Rcpp::Named("covariance_mean") = covariance_sum, Rcpp::Named("coefficient_sign_probability") = positive_sum,
        Rcpp::Named("coefficient_sign_tail") = sign_tail, Rcpp::Named("coefficient_labels") = coefficient_membership,
        Rcpp::Named("covariance_labels") = covariance_membership, Rcpp::Named("coefficient_coclustering") = coefficient_similarity,
        Rcpp::Named("covariance_coclustering") = covariance_similarity, Rcpp::Named("cluster_counts") = cluster_counts,
        Rcpp::Named("parameter_traces") = parameter_traces,
        Rcpp::Named("split_merge") = sampler_move_trace(iterations, covariance_types, covariance_accepted, covariance_ratios),
        Rcpp::Named("coefficient_split_merge") = sampler_move_trace(iterations, coefficient_types, coefficient_accepted, coefficient_ratios),
        Rcpp::Named("lambda") = lambda_result, Rcpp::Named("shared_variance_mean") = shared_variance_sum,
        Rcpp::Named("final_state") = final_state, Rcpp::Named("draws") = draws);
}
