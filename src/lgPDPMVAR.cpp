// Coefficient draws, group moves and sampling for the lag-specific PDPMVAR model.
// The common mixture weights and covariance updates are in sampling_helpers.cpp.
#include "sampling_helpers.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <R_ext/Random.h>
#if defined(PDPMVAR_ACCELERATE) && defined(__clang__) && __has_include(<vecLib/cblas_new.h>)
#define PDPMVAR_HAVE_ACCELERATE
#define ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Draw coefficient rows across lags ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Draw one response row jointly across all occupied groups at all lags.
// Called after the group assignments and factor draws in each iteration.
// For D nodes, K lags and T_i observations, each adjusted response is
// D x T_i and its predictors are (D * K) x T_i, stacked by lag. A row draw
// has D * sum(H_k) entries, with D predictors for each of the H_k occupied
// groups at lag k. Return the updated D x D matrices in lag and stick order.
// [[Rcpp::export(name = ".lgpdpmvar_eigen_coefficient_update")]]
Rcpp::List lgpdpmvar_eigen_coefficient_update(const Rcpp::List& adjusted,
                                              const Rcpp::List& design,
                                              const Rcpp::List& design_cross,
                                              const Rcpp::List& labels,
                                              const Rcpp::List& atoms,
                                              const Rcpp::List& shared_variance,
                                              const Eigen::MatrixXd& residual_variances) {
    const int n_subjects = adjusted.size();
    const int n_nodes    = residual_variances.rows();
    const int n_lags     = labels.size();
    if (n_nodes == 0 || n_subjects == 0 || n_lags == 0 || design.size() != n_subjects ||
        design_cross.size() != n_subjects || atoms.size() != n_lags || shared_variance.size() != n_lags ||
        residual_variances.cols() != n_subjects || !residual_variances.allFinite() ||
        (residual_variances.array() <= 0).any()) Rcpp::stop("Invalid lag coefficient update dimensions or variances.");

    // Assign positions in the Gaussian draw only to groups with subjects.
    List result = Rcpp::clone(atoms);
    std::vector<std::vector<int>> offsets(n_lags), subject_offsets(n_subjects, std::vector<int>(n_lags));
    std::vector<MatrixXd> variances(n_lags);
    Eigen::Index dimension = 0;
    for (int lag = 0; lag < n_lags; ++lag) {
        const List lag_atoms           = atoms[lag];
        const NumericVector lag_labels = labels[lag];
        if (lag_labels.size() != n_subjects) Rcpp::stop("Each lag needs one label per subject.");
        const auto counts = sampler_counts(lag_labels, lag_atoms.size());
        variances[lag]    = Rcpp::as<MatrixXd>(shared_variance[lag]);
        if (variances[lag].rows() != n_nodes || variances[lag].cols() != n_nodes ||
            !variances[lag].allFinite() || (variances[lag].array() <= 0).any()) {
            Rcpp::stop("Each lag needs a positive, finite D by D prior variance matrix.");
        }
        offsets[lag].resize(lag_atoms.size(), -1);
        for (int group = 0; group < lag_atoms.size(); ++group) {
            if (counts[group] == 0) continue;
            offsets[lag][group] = dimension;
            dimension += n_nodes;
        }
        for (int subject = 0; subject < n_subjects; ++subject) {
            subject_offsets[subject][lag] = offsets[lag][static_cast<int>(lag_labels[subject]) - 1];
        }
    }
    // Check the size of one response row's precision before allocating it.
    if (8.0 * dimension * dimension > 512.0 * 1024 * 1024) {
        Rcpp::stop("The lag coefficient precision exceeds 512 MiB for one response row.");
    }

    std::vector<MatrixXd> cross(n_subjects), information(n_subjects);
    for (int subject = 0; subject < n_subjects; ++subject) {
        const MatrixXd predictors = Rcpp::as<MatrixXd>(design[subject]);
        const MatrixXd response   = Rcpp::as<MatrixXd>(adjusted[subject]);
        cross[subject]            = Rcpp::as<MatrixXd>(design_cross[subject]);
        if (predictors.rows() != n_nodes * n_lags || response.rows() != n_nodes ||
            predictors.cols() != response.cols() || cross[subject].rows() != predictors.rows() ||
            cross[subject].cols() != predictors.rows() || !predictors.allFinite() ||
            !response.allFinite() || !cross[subject].allFinite()) Rcpp::stop("Invalid subject regression matrices.");
        information[subject] = predictors * response.transpose();
    }

    // Store one draw per response in columns, then copy the entries to their lag matrices.
    MatrixXd draws(dimension, n_nodes);
    for (int response = 0; response < n_nodes; ++response) {
        Rcpp::checkUserInterrupt();
        MatrixXd precision = MatrixXd::Zero(dimension, dimension);
        VectorXd rhs       = VectorXd::Zero(dimension);
        for (int lag = 0; lag < n_lags; ++lag) {
            for (int offset : offsets[lag]) {
                if (offset < 0) continue;
                precision.diagonal().segment(offset, n_nodes) =
                    variances[lag].row(response).cwiseInverse().transpose();
            }
        }

        // Add each subject's cross-products within and between lags.
        // Cholesky uses the lower triangle, so only that half is filled.
        for (int subject = 0; subject < n_subjects; ++subject) {
            const double weight = 1.0 / residual_variances(response, subject);
            for (int lag = 0; lag < n_lags; ++lag) {
                const int row = subject_offsets[subject][lag];
                rhs.segment(row, n_nodes) += weight * information[subject].col(response).segment(lag * n_nodes, n_nodes);
                for (int other = 0; other <= lag; ++other) {
                    const int column = subject_offsets[subject][other];
                    precision.block(row, column, n_nodes, n_nodes) +=
                        weight * cross[subject].block(lag * n_nodes, other * n_nodes, n_nodes, n_nodes);
                }
            }
        }

        Eigen::LLT<MatrixXd> factor(precision);
        if (factor.info() != Eigen::Success) Rcpp::stop("Lag coefficient precision must be positive definite.");
        VectorXd noise(dimension);
        for (Eigen::Index index = 0; index < dimension; ++index) noise[index] = R::rnorm(0.0, 1.0);
        draws.col(response) = factor.solve(rhs) + factor.matrixU().solve(noise);
    }
    for (int lag = 0; lag < n_lags; ++lag) {
        List lag_atoms = result[lag];
        for (int group = 0; group < lag_atoms.size(); ++group) {
            const int offset = offsets[lag][group];
            if (offset < 0) continue;
            List atom           = lag_atoms[group];
            atom["coefficient"] = MatrixXd(draws.middleRows(offset, n_nodes).transpose());
        }
    }
    return result;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Assemble subject coefficients and draw assignment uniforms ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Draw a vector of N uniforms, one per subject, for the slice assignments.
// The sampler uses it to draw both coefficient and covariance labels.
static NumericVector lag_assignment_uniforms(int n_subjects) {
    NumericVector uniforms(n_subjects);
    for (double& uniform : uniforms) uniform = R::runif(0.0, 1.0);
    return uniforms;
}

// Build one subject's coefficient matrix, nodes by lagged predictors,
// D x (D * K), by placing its selected D x D matrices side by side in lag
// order. D is nodes and K is lags. Used for residuals and subject summaries.
static MatrixXd lag_subject_coefficient(const List& atoms, const List& labels, int subject, int n_nodes) {
    const int n_lags = atoms.size();
    MatrixXd coefficient(n_nodes, n_nodes * n_lags);
    for (int lag = 0; lag < n_lags; ++lag) {
        const List lag_atoms                           = atoms[lag];
        const NumericVector lag_labels                 = labels[lag];
        const List atom                                = lag_atoms[static_cast<int>(lag_labels[subject]) - 1];
        coefficient.middleCols(lag * n_nodes, n_nodes) = Rcpp::as<MatrixXd>(atom["coefficient"]);
    }
    return coefficient;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Integrate and draw lag matrices using the factor covariance ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Prepare one lag's regression quantities for the moves that integrate
// out factors. For D response nodes, P predictors and B factors, store a
// P x P predictor cross-product and D x P weighted response cross-product
// per subject. For each covariance group, express the inverse covariance
// as a diagonal matrix minus W W', with W of size D x B.
struct LagGaussianData {
    MatrixXd variance;
    std::vector<MatrixXd> gram, information, W;
    std::vector<VectorXd> inverse_variance;
    NumericVector covariance_labels;

    LagGaussianData(const List& responses, const List& design, const List& design_cross,
                    const List& atoms, const NumericVector& labels, const NumericMatrix& prior)
        : variance(Rcpp::as<MatrixXd>(prior)), W(atoms.size()), inverse_variance(atoms.size()),
          covariance_labels(labels) {
        if (responses.size() != labels.size() || responses.size() != design.size() ||
            responses.size() != design_cross.size() || !variance.allFinite() ||
            (variance.array() <= 0).any()) Rcpp::stop("Invalid collapsed lag data or prior variances.");
        const auto counts = sampler_counts(labels, atoms.size());
        for (int group = 0; group < atoms.size(); ++group) {
            if (counts[group] == 0) continue;
            const List atom        = atoms[group];
            const MatrixXd loading = Rcpp::as<MatrixXd>(atom["working_loadings"]);
            const VectorXd v       = Rcpp::as<VectorXd>(atom["variance"]);
            const VectorXd q       = Rcpp::as<VectorXd>(atom["factor_precision"]);
            if (v.size() != variance.rows() || loading.rows() != v.size() || loading.cols() != q.size() ||
                !loading.allFinite() || !v.allFinite() || !q.allFinite() ||
                (v.array() <= 0).any() || (q.array() <= 0).any()) {
                Rcpp::stop("Invalid collapsed lag factor covariance.");
            }
            inverse_variance[group] = v.cwiseInverse();
            const MatrixXd weighted = inverse_variance[group].asDiagonal() * loading;
            MatrixXd precision      = loading.transpose() * weighted;
            precision.diagonal()   += q;
            Eigen::LLT<MatrixXd> factor(precision);
            if (factor.info() != Eigen::Success) Rcpp::stop("Factor precision must be positive definite.");
            W[group] = factor.matrixL().solve(weighted.transpose()).transpose();
        }
        for (int subject = 0; subject < responses.size(); ++subject) {
            const int group      = static_cast<int>(labels[subject]) - 1;
            const MatrixXd Y     = Rcpp::as<MatrixXd>(responses[subject]);
            const MatrixXd Z     = Rcpp::as<MatrixXd>(design[subject]);
            const MatrixXd cross = Rcpp::as<MatrixXd>(design_cross[subject]);
            if (Y.rows() != variance.rows() || Z.rows() != variance.cols() || Y.cols() != Z.cols() ||
                cross.rows() != Z.rows() || cross.cols() != Z.rows() ||
                !Y.allFinite() || !Z.allFinite() || !cross.allFinite()) {
                Rcpp::stop("Invalid collapsed lag regression matrices.");
            }
            const MatrixXd YZ = Y * Z.transpose();
            gram.push_back(cross);
            information.push_back(inverse_variance[group].asDiagonal() * YZ -
                                  W[group] * (W[group].transpose() * YZ));
        }
    }
};

// Store the integrated score for one group and the Gaussian factors when
// a coefficient draw will follow. The coefficient matrix is D x P, with
// D response nodes and P predictors. V has D * P rows and B times the
// total predictor rank across covariance groups in its columns.
struct LagGaussianGroup {
    double log_integral = 0;
    MatrixXd V, schur_upper;
    VectorXd whitened_mean;
    std::vector<MatrixXd> upper_blocks;
    double bytes = 0.0;
};

// Form the smaller precision correction I - V'V for a collapsed group
// score. If V has D * P rows and R columns, return an R x R matrix.
// Only its lower triangle is used by the Cholesky calculation.
static MatrixXd lag_schur(const MatrixXd& V) {
    const Eigen::Index width = V.cols();
    MatrixXd schur           = MatrixXd::Identity(width, width);
#ifdef PDPMVAR_HAVE_ACCELERATE
    // The current Accelerate interface is available from macOS 13.3.
    if (__builtin_available(macOS 13.3, *)) {
        if (width >= 128 && V.rows() <= std::numeric_limits<int>::max()) {
            cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans, static_cast<int>(width),
                static_cast<int>(V.rows()), -1.0, V.data(), static_cast<int>(V.rows()),
                1.0, schur.data(), static_cast<int>(width));
            return schur;
        }
    }
#endif
    schur.selfadjointView<Eigen::Lower>().rankUpdate(V.transpose(), -1.0);
    return schur;
}

// Integrate one group's D x P lag matrix, with D response nodes and P
// predictors. The precision is Q0 - U U', where Q0 has D diagonal blocks
// of size P x P. Rescale U with the Cholesky factor of Q0 to obtain V,
// then use I - V'V to calculate the determinant and quadratic terms.
// Split/merge and transfer moves use the score for the acceptance ratio
// and keep the Gaussian factors when they can be reused for a draw.
static LagGaussianGroup lag_gaussian_group(const LagGaussianData& data,
                                           const std::vector<int>& members,
                                           bool keep_factor,
                                           double budget = R_PosInf) {
    const int D = data.variance.rows();
    const int P = data.variance.cols();
    LagGaussianGroup result;
    if (members.empty()) return result;

    std::vector<MatrixXd> grams(data.W.size()), roots(data.W.size());
    MatrixXd information = MatrixXd::Zero(D, P);
    for (int subject : members) {
        const int group = static_cast<int>(data.covariance_labels[subject]) - 1;
        if (grams[group].size() == 0) grams[group] = data.gram[subject];
        else grams[group] += data.gram[subject];
        information += data.information[subject];
    }
    Eigen::Index width = 0;
    for (int group = 0; group < grams.size(); ++group) {
        if (grams[group].size() == 0 || data.W[group].cols() == 0) continue;
        Eigen::SelfAdjointEigenSolver<MatrixXd> eigen(grams[group]);
        if (eigen.info() != Eigen::Success) Rcpp::stop("Could not factor the lag predictor cross-product.");
        // Ignore roundoff-sized eigenvalues from unavailable or dependent predictors.
        const double threshold = eigen.eigenvalues().maxCoeff() * std::numeric_limits<double>::epsilon() * P;
        const int rank         = (eigen.eigenvalues().array() > threshold).count();
        roots[group]           = eigen.eigenvectors().rightCols(rank) *
            eigen.eigenvalues().tail(rank).cwiseSqrt().asDiagonal();
        width += rank * data.W[group].cols();
    }
    const Eigen::Index dimension = static_cast<Eigen::Index>(D) * P;
    if (8.0 * width * (dimension + width) > 512.0 * 1024 * 1024) {
        Rcpp::stop("Low-rank lag coefficient workspace exceeds 512 MiB; use factor_conditional.");
    }

    const double factor_bytes = 8.0 * (width * (static_cast<double>(dimension) + width) +
        static_cast<double>(dimension) * (P + 1));
    keep_factor = keep_factor && factor_bytes <= budget;
    MatrixXd V(dimension, width);
    VectorXd h(dimension);
    if (keep_factor) result.upper_blocks.reserve(D);
    double log_determinant = 0.0;
    for (int node = 0; node < D; ++node) {
        MatrixXd precision = data.variance.row(node).cwiseInverse().asDiagonal();
        MatrixXd U(P, width);
        Eigen::Index start = 0;
        for (int group = 0; group < grams.size(); ++group) {
            if (grams[group].size() == 0) continue;
            precision.noalias() += data.inverse_variance[group][node] * grams[group];
            for (int factor = 0; factor < data.W[group].cols(); ++factor) {
                const int rank            = roots[group].cols();
                U.middleCols(start, rank) = data.W[group](node, factor) * roots[group];
                start += rank;
            }
        }
        Eigen::LLT<MatrixXd> factor(precision);
        if (factor.info() != Eigen::Success) Rcpp::stop("Lag coefficient block must be positive definite.");
        if (width > 0) V.middleRows(node * P, P) = factor.matrixL().solve(U);
        h.segment(node * P, P) = factor.matrixL().solve(information.row(node).transpose());
        log_determinant += factor.matrixLLT().diagonal().array().log().sum();
        if (keep_factor) result.upper_blocks.emplace_back(factor.matrixU());
    }

    double gain = h.squaredNorm();
    if (width > 0) {
        Eigen::LLT<MatrixXd> factor(lag_schur(V));
        if (factor.info() != Eigen::Success) Rcpp::stop("Low-rank lag coefficient precision must be positive definite.");
        const VectorXd rhs        = V.transpose() * h;
        const VectorXd adjustment = factor.matrixL().solve(rhs);
        gain            += adjustment.squaredNorm();
        log_determinant += factor.matrixLLT().diagonal().array().log().sum();
        if (keep_factor) {
            h.noalias() += V * factor.matrixU().solve(adjustment);
            result.schur_upper = factor.matrixU();
        }
    }
    result.log_integral = (gain - data.variance.array().log().sum()) / 2.0 - log_determinant;
    if (keep_factor) {
        result.V             = std::move(V);
        result.whitened_mean = std::move(h);
        result.bytes         = factor_bytes;
    }
    return result;
}

// Draw the D x P lag matrix from a group's Gaussian distribution after
// a move is accepted or rejected. D is response nodes and P is predictors.
// In the rescaled coordinates, the covariance is I + V (I - V'V)^{-1} V'.
static MatrixXd draw_lag_gaussian(const LagGaussianGroup& group) {
    const int D = group.upper_blocks.size();
    const int P = group.upper_blocks.front().cols();
    VectorXd noise(group.whitened_mean.size());
    for (Eigen::Index i = 0; i < noise.size(); ++i) noise[i] = R::rnorm(0, 1);
    if (group.V.cols() > 0) {
        VectorXd extra(group.V.cols());
        for (Eigen::Index i = 0; i < extra.size(); ++i) extra[i] = R::rnorm(0, 1);
        noise.noalias() += group.V * group.schur_upper.triangularView<Eigen::Upper>().solve(extra);
    }
    MatrixXd draw(D, P);
    for (int node = 0; node < D; ++node) {
        const auto upper    = group.upper_blocks[node].triangularView<Eigen::Upper>();
        const VectorXd mean = upper.solve(group.whitened_mean.segment(node * P, P));
        draw.row(node)      = mean.transpose() + upper.solve(noise.segment(node * P, P)).transpose();
    }
    return draw;
}

// Save Gaussian factorizations that may be needed to redraw matrices after
// a split, merge or transfer. Match each entry by its subjects and limit
// the memory used by the saved factorizations within one proposal.
struct LagFactorCache {
    std::vector<std::vector<int>> members;
    std::vector<LagGaussianGroup> groups;
    double remaining = 128.0 * 1024 * 1024;

    // Find a saved factorization by its subject set, ignoring subject order.
    int position(std::vector<int> subjects) const {
        std::sort(subjects.begin(), subjects.end());
        for (int i = 0; i < members.size(); ++i) {
            if (members[i] == subjects) return i;
        }
        return -1;
    }
};

// Return one group's integrated Gaussian log score for a move.
// Save its factorization only if the group is marked for a possible redraw
// and the factorization fits within the remaining memory budget.
static double lag_group_score(const LagGaussianData& data,
                              const std::vector<int>& members,
                              LagFactorCache* cache) {
    const int position = cache ? cache->position(members) : -1;
    if (position < 0) return lag_gaussian_group(data, members, false).log_integral;
    auto& saved = cache->groups[position];
    if (saved.bytes == 0.0) {
        // Use the same subject order as when calculating the factorization from scratch.
        saved = lag_gaussian_group(data, cache->members[position], true, cache->remaining);
        cache->remaining -= saved.bytes;
    }
    return saved.log_integral;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Split or merge groups with factors integrated out ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Keep the subject assignments and integrated scores of the two groups
// formed in a restricted split. The label vector has N entries for N
// subjects. Member positions start at zero. The collapsed move also uses
// this record when evaluating the split that would reverse a merge.
struct LagSplit {
    NumericVector labels;
    std::vector<int> first, second;
    double first_score, second_score;
    double log_probability = 0.0;

    explicit LagSplit(int n) : labels(n) {}
};

// Place the two selected subjects in opposite groups, then use integrated
// predictive scores to assign the remaining subjects in the affected groups.
// For a merge, follow the old assignments to calculate their reverse
// probability. Used by the collapsed move and its acceptance-ratio check.
static LagSplit lag_restricted_split(const LagGaussianData& data,
                                     const Rcpp::IntegerVector& subjects,
                                     const Rcpp::IntegerVector& anchors,
                                     const NumericVector& weights,
                                     int h,
                                     int g,
                                     const NumericVector* forced,
                                     const NumericVector& uniforms,
                                     LagFactorCache* cache = nullptr) {
    LagSplit split(data.gram.size());
    split.first.push_back(anchors[0] - 1);
    split.second.push_back(anchors[1] - 1);
    std::fill(split.labels.begin(), split.labels.end(), h);
    split.labels[anchors[1] - 1] = g;
    split.first_score            = lag_group_score(data, split.first, cache);
    split.second_score           = lag_group_score(data, split.second, cache);
    int index                    = 0;
    for (int subject : subjects) {
        if (subject == anchors[0] || subject == anchors[1]) continue;
        Rcpp::checkUserInterrupt();
        split.first.push_back(subject - 1);
        split.second.push_back(subject - 1);
        const double first   = lag_group_score(data, split.first, cache);
        const double second  = lag_group_score(data, split.second, cache);
        const double a       = weights[h - 1] + first - split.first_score;
        const double b       = weights[g - 1] + second - split.second_score;
        const double maximum = std::max(a, b);
        if (!std::isfinite(maximum)) Rcpp::stop("No finite collapsed lag split score.");
        const double mass_a = std::exp(a - maximum), mass_b = std::exp(b - maximum);
        const double total  = mass_a + mass_b;
        bool choose_first;
        if (forced) choose_first = (*forced)[subject - 1] == h;
        else {
            const double u = uniforms[index++];
            choose_first   = mass_a >= mass_b ? u <= mass_a / total : u > mass_b / total;
        }
        split.labels[subject - 1] = choose_first ? h : g;
        split.log_probability += (choose_first ? a : b) - maximum - std::log(total);
        if (choose_first) {
            split.first_score = first;
            split.second.pop_back();
        } else {
            split.second_score = second;
            split.first.pop_back();
        }
    }
    return split;
}

// Combine the integrated Gaussian scores, stick weights and proposal
// probabilities into a log acceptance ratio for a specified split or merge.
// Used by the collapsed move and the check against the dense calculation.
static List lag_collapsed_ratio(const LagGaussianData& data,
                                const LagSplit& split,
                                const NumericVector& old_labels,
                                const NumericVector& new_labels,
                                const NumericVector& weights,
                                const Rcpp::IntegerVector& subjects,
                                bool splitting,
                                int slots,
                                LagFactorCache* cache = nullptr) {
    std::vector<int> members;
    for (int subject : subjects) members.push_back(subject - 1);
    const double merged = lag_group_score(data, members, cache);
    double target       = (splitting ? 1 : -1) * (split.first_score + split.second_score - merged);
    for (int subject : members) {
        target += weights[static_cast<int>(new_labels[subject]) - 1] -
            weights[static_cast<int>(old_labels[subject]) - 1];
    }
    const double density = split.log_probability -
        std::log(static_cast<double>(move_empty_slots(splitting ? old_labels : new_labels, slots).size()));
    const double forward = splitting ? density : 0.0;
    const double reverse = splitting ? 0.0 : density;
    return List::create(Rcpp::Named("log_ratio") = target + reverse - forward,
        Rcpp::Named("target_change") = target, Rcpp::Named("log_forward") = forward,
        Rcpp::Named("log_reverse") = reverse);
}

// Check a specified split or merge at fixed stick weights using the
// low-rank Gaussian calculation. The tests call this function from R
// to compare forward and reverse ratios with the dense calculation.
// [[Rcpp::export(name = ".lgpdpmvar_eigen_collapsed_ratio")]]
Rcpp::List lgpdpmvar_eigen_collapsed_ratio(const Rcpp::List& responses,
                                           const Rcpp::List& design,
                                           const Rcpp::List& design_cross,
                                           const Rcpp::List& covariance_atoms,
                                           const Rcpp::NumericVector& covariance_labels,
                                           const Rcpp::NumericMatrix& variance,
                                           const Rcpp::NumericVector& old_labels,
                                           const Rcpp::NumericVector& new_labels,
                                           const Rcpp::NumericVector& weights,
                                           const Rcpp::IntegerVector& anchors,
                                           int slots) {
    move_validate_ratio(old_labels, new_labels, weights, anchors, slots);
    const LagGaussianData data(responses, design, design_cross, covariance_atoms, covariance_labels, variance);
    if (old_labels.size() != responses.size()) Rcpp::stop("Lag labels must match the subject count.");
    const int h          = old_labels[anchors[0] - 1];
    const bool splitting = h == old_labels[anchors[1] - 1];
    const int g          = splitting ? new_labels[anchors[1] - 1] : old_labels[anchors[1] - 1];
    const auto subjects  = move_subjects(old_labels, h, g);
    const LagSplit split = lag_restricted_split(data, subjects, anchors, weights, h, g,
        splitting ? &new_labels : &old_labels, NumericVector(0));
    return lag_collapsed_ratio(data, split, old_labels, new_labels, weights, subjects, splitting, slots);
}

// Propose one split or merge at a lag using partial residuals and the
// current covariance groups. The lag sweep calls this when factors are
// integrated out. Redraw both affected D x D matrices after the decision,
// where D is the number of nodes.
static List lag_collapsed_move(const List& responses,
                               const List& design,
                               const List& design_cross,
                               const List& covariance_atoms,
                               const NumericVector& covariance_labels,
                               const NumericVector& labels,
                               const List& atoms,
                               const NumericMatrix& variance,
                               double concentration,
                               int slots) {
    const auto empty = move_empty_slots(labels, slots);
    if (labels.size() < 2) return move_result(labels, atoms);
    const auto anchors = move_anchors(labels.size());
    const int h        = labels[anchors[0] - 1];
    int g              = labels[anchors[1] - 1];
    if (h > slots || g > slots) return move_result(labels, atoms);
    const bool splitting = h == g;
    if (splitting && empty.empty()) return move_result(labels, atoms);

    const NumericVector weights = pdpmvar_eigen_prefix_weights(labels, concentration, slots);
    const LagGaussianData data(responses, design, design_cross, covariance_atoms, covariance_labels, variance);
    if (splitting) g = empty[static_cast<int>(R_unif_index(empty.size()))];
    const auto subjects = move_subjects(labels, h, g);
    NumericVector uniforms(splitting ? subjects.size() - 2 : 0);
    for (double& u : uniforms) u = R::runif(0, 1);
    LagFactorCache cache;
    std::vector<int> first, second, merged;
    for (int subject : subjects) {
        merged.push_back(subject - 1);
        if (labels[subject - 1] == h) first.push_back(subject - 1);
        else second.push_back(subject - 1);
    }
    if (!splitting) {
        cache.members.push_back(std::move(first));
        cache.members.push_back(std::move(second));
    }
    cache.members.push_back(std::move(merged));
    cache.groups.resize(cache.members.size());
    const LagSplit split = lag_restricted_split(data, subjects, anchors, weights, h, g,
        splitting ? nullptr : &labels, uniforms, &cache);
    NumericVector proposed = Rcpp::clone(labels);
    for (int subject : subjects) proposed[subject - 1] = splitting ? split.labels[subject - 1] : h;
    const List ratio       = lag_collapsed_ratio(data, split, labels, proposed, weights, subjects, splitting, slots, &cache);
    const double log_ratio = Rcpp::as<double>(ratio["log_ratio"]);
    if (std::isnan(log_ratio)) Rcpp::stop("Invalid collapsed lag split/merge ratio.");
    const bool accepted               = std::log(R::runif(0, 1)) < std::min(0.0, log_ratio);
    const NumericVector result_labels = accepted ? proposed : labels;
    List result_atoms                 = Rcpp::clone(atoms);
    while (result_atoms.size() < std::max(h, g)) result_atoms.push_back(sampler_new_coefficient(variance));
    for (int group : {h, g}) {
        std::vector<int> members;
        for (int subject : subjects) {
            if (result_labels[subject - 1] == group) members.push_back(subject - 1);
        }
        if (members.empty()) {
            result_atoms[group - 1] = sampler_new_coefficient(variance);
            continue;
        }
        const int position  = cache.position(members);
        const bool saved    = position >= 0 && cache.groups[position].bytes > 0.0;
        const MatrixXd draw = saved ? draw_lag_gaussian(cache.groups[position]) :
            draw_lag_gaussian(lag_gaussian_group(data, members, true));
        result_atoms[group - 1] = List::create(Rcpp::Named("coefficient") = draw);
    }
    return move_result(result_labels, result_atoms, splitting ? "split" : "merge", accepted, log_ratio);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Propose and check single-subject transfers ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Redraw one affected coefficient matrix, D x P, after a transfer.
// D is response nodes and P is predictors. Reuse the saved Gaussian factors
// when available, and draw from the prior if the group is empty.
static List refresh_lag_atom(const LagGaussianData& data,
                             const std::vector<int>& members,
                             const NumericMatrix& variance,
                             const LagFactorCache& cache) {
    if (members.empty()) return sampler_new_coefficient(variance);
    const int position  = cache.position(members);
    const bool saved    = position >= 0 && cache.groups[position].bytes > 0.0;
    const MatrixXd draw = saved ? draw_lag_gaussian(cache.groups[position]) :
        draw_lag_gaussian(lag_gaussian_group(data, members, true));
    return List::create(Rcpp::Named("coefficient") = draw);
}

// Measure average prediction error after scaling by the subject's
// covariance. The ridge fit predicts a D x T_i response from P x T_i
// lagged predictors. D is nodes, P is predictors and T_i is observations.
// The subject and destination guides use this error to propose transfers.
static double relocation_error(const MatrixXd& Y,
                               const MatrixXd& Z,
                               const MatrixXd& lower,
                               const MatrixXd& gram,
                               const MatrixXd& information,
                               double ridge) {
    MatrixXd residual = Y;
    if (gram.size() > 0) {
        MatrixXd precision = gram;
        precision.diagonal().array() += ridge;
        Eigen::LLT<MatrixXd> factor(precision);
        if (factor.info() != Eigen::Success) Rcpp::stop("Could not fit the lag relocation proposal.");
        const MatrixXd coefficient = factor.solve(information);
        residual.noalias() -= coefficient.transpose() * Z;
    }
    const MatrixXd standardized = lower.triangularView<Eigen::Lower>().solve(residual);
    return standardized.squaredNorm() / Y.size();
}

// Score possible destinations by predicting the subject from a ridge fit
// to each group's other members. Return one score per stick position in
// the proposal, with empty positions excluded from the guided scores.
// Removing the subject first gives the same scores for the reverse move.
static VectorXd relocation_scores(const List& responses,
                                  const List& design,
                                  const List& design_cross,
                                  const List& covariance_atoms,
                                  const NumericVector& covariance_labels,
                                  const NumericVector& labels,
                                  int subject,
                                  int slots) {
    const MatrixXd Y     = Rcpp::as<MatrixXd>(responses[subject]);
    const MatrixXd Z     = Rcpp::as<MatrixXd>(design[subject]);
    const MatrixXd gram  = Rcpp::as<MatrixXd>(design_cross[subject]);
    const int P          = Z.rows();
    const double ridge   = std::max(1e-8, 0.1 * gram.trace() / P);
    const List atom      = covariance_atoms[static_cast<int>(covariance_labels[subject]) - 1];
    const MatrixXd lower = likelihood_covariance_lower(Rcpp::as<MatrixXd>(atom["covariance"]), Y.rows());

    std::vector<MatrixXd> precision(slots), information(slots);
    for (int i = 0; i < labels.size(); ++i) {
        const int group = static_cast<int>(labels[i]) - 1;
        if (i == subject || group >= slots) continue;
        if (precision[group].size() == 0) {
            precision[group]   = MatrixXd::Zero(P, P);
            information[group] = MatrixXd::Zero(P, Y.rows());
        }
        precision[group] += Rcpp::as<MatrixXd>(design_cross[i]);
        const MatrixXd zi = Rcpp::as<MatrixXd>(design[i]);
        const MatrixXd yi = Rcpp::as<MatrixXd>(responses[i]);
        information[group].noalias() += zi * yi.transpose();
    }

    // Standardize prediction errors by the subject's residual covariance.
    // A temperature of 0.05 is on the per-entry Gaussian log-score scale.
    VectorXd scores = VectorXd::Constant(slots, R_NegInf);
    for (int group = 0; group < slots; ++group) {
        if (precision[group].size() == 0) continue;
        scores[group] = -relocation_error(Y, Z, lower, precision[group], information[group], ridge) / 0.1;
    }
    return scores;
}

// Turn destination scores into a vector of proposal probabilities.
// The guided part uses occupied alternatives. Mix in 20% uniform over
// all positions other than the current one. Use uniform probabilities
// alone if there is no occupied alternative. Used in both directions.
static VectorXd relocation_probabilities(const VectorXd& scores, int excluded) {
    VectorXd probability  = scores;
    probability[excluded] = R_NegInf;
    const double maximum  = probability.maxCoeff();
    double uniform_mass   = 1.0;
    if (maximum != R_NegInf) {
        probability  = (probability.array() - maximum).exp();
        probability  = 0.8 * probability / probability.sum();
        uniform_mass = 0.2;
    } else {
        probability.setZero();
    }
    for (int group = 0; group < probability.size(); ++group) {
        if (group != excluded) probability[group] += uniform_mass / (probability.size() - 1);
    }
    return probability;
}

// Return one selection probability per subject, favoring those whose
// prediction improves most in another occupied group. Pool each group's
// P x P predictor cross-product and P x D predictor-response cross-product,
// with P predictors and D nodes. Leave each subject out of its own fit.
// Recompute these probabilities after the proposed transfer for the reverse.
static VectorXd relocation_subject_probabilities(const List& responses,
                                                 const List& design,
                                                 const List& design_cross,
                                                 const List& covariance_atoms,
                                                 const NumericVector& covariance_labels,
                                                 const NumericVector& labels,
                                                 int slots) {
    const int N = labels.size();
    std::vector<MatrixXd> cross(N), information(N), gram_by_group(slots), info_by_group(slots);
    std::vector<MatrixXd> lower(covariance_atoms.size());
    std::vector<int> count(slots, 0);
    for (int i = 0; i < N; ++i) {
        const int group = static_cast<int>(labels[i]) - 1;
        if (group >= slots) continue;
        cross[i]         = Rcpp::as<MatrixXd>(design_cross[i]);
        const MatrixXd Y = Rcpp::as<MatrixXd>(responses[i]);
        const MatrixXd Z = Rcpp::as<MatrixXd>(design[i]);
        information[i]   = Z * Y.transpose();
        if (++count[group] == 1) {
            gram_by_group[group] = cross[i];
            info_by_group[group] = information[i];
        } else {
            gram_by_group[group] += cross[i];
            info_by_group[group] += information[i];
        }
    }

    VectorXd scores = VectorXd::Constant(N, R_NegInf);
    int eligible    = 0;
    for (int i = 0; i < N; ++i) {
        const int group = static_cast<int>(labels[i]) - 1;
        if (group >= slots) continue;
        ++eligible;
        const MatrixXd Y    = Rcpp::as<MatrixXd>(responses[i]);
        const MatrixXd Z    = Rcpp::as<MatrixXd>(design[i]);
        const int cov_group = static_cast<int>(covariance_labels[i]) - 1;
        if (lower[cov_group].size() == 0) {
            const List atom  = covariance_atoms[cov_group];
            lower[cov_group] = likelihood_covariance_lower(Rcpp::as<MatrixXd>(atom["covariance"]), Y.rows());
        }
        const double ridge   = std::max(1e-8, 0.1 * cross[i].trace() / Z.rows());
        const MatrixXd gram  = count[group] > 1 ? MatrixXd(gram_by_group[group] - cross[i]) : MatrixXd();
        const MatrixXd info  = count[group] > 1 ? MatrixXd(info_by_group[group] - information[i]) : MatrixXd();
        const double current = relocation_error(Y, Z, lower[cov_group], gram, info, ridge);
        double best          = current;
        for (int other = 0; other < slots; ++other) {
            if (other == group || count[other] == 0) continue;
            const double error = relocation_error(Y, Z, lower[cov_group],
                gram_by_group[other], info_by_group[other], ridge);
            best = std::min(best, error);
        }
        // Use the same per-entry Gaussian score scale as the destination guide.
        scores[i] = (current - best) / 0.1;
    }
    if (eligible == 0) return VectorXd::Zero(N);
    VectorXd probability = (scores.array() - scores.maxCoeff()).exp();
    probability          = 0.8 * probability / probability.sum();
    for (int i = 0; i < N; ++i) {
        if (labels[i] <= slots) probability[i] += 0.2 / eligible;
    }
    return probability;
}

// Draw an index from normalized probabilities, counting from one.
// The guided transfer uses it to select a subject or a destination.
static int draw_relocation_index(const VectorXd& probability) {
    const double uniform = R::runif(0.0, 1.0);
    double cumulative    = 0.0;
    int selected         = 0;
    for (int index = 0; index < probability.size(); ++index) {
        if (probability[index] == 0.0) continue;
        selected = index + 1;
        cumulative += probability[index];
        if (uniform < cumulative) break;
    }
    return selected;
}

// Propose a single-subject transfer at one lag with the factors integrated
// out. The lag sweep calls this on responses of size nodes by time points,
// D x T_i. Compare the source and destination groups before and after the
// transfer. The optional fixed pair and weights let tests check a specified
// move without drawing a proposal.
// [[Rcpp::export(name = ".lgpdpmvar_eigen_relocation_move")]]
Rcpp::List lgpdpmvar_eigen_relocation_move(const Rcpp::List& responses,
                                           const Rcpp::List& design,
                                           const Rcpp::List& design_cross,
                                           const Rcpp::List& covariance_atoms,
                                           const Rcpp::NumericVector& covariance_labels,
                                           const Rcpp::NumericVector& labels,
                                           const Rcpp::List& atoms,
                                           const Rcpp::NumericMatrix& variance,
                                           double concentration,
                                           int slots,
                                           int subject     = 0,
                                           int destination = 0,
                                           Rcpp::Nullable<Rcpp::NumericVector> fixed_weights = R_NilValue,
                                           bool evaluate_only = false,
                                           std::string method = "uniform") {
    if (labels.size() == 0 || labels.size() != responses.size() || slots < 2 ||
        subject < 0 || subject > labels.size() || destination < 0 || destination > slots ||
        ((subject == 0) != (destination == 0)) ||
        (evaluate_only && (subject == 0 || fixed_weights.isNull()))) {
        Rcpp::stop("Invalid lag relocation inputs.");
    }
    if (method != "uniform" && method != "destination" && method != "guided") Rcpp::stop("Invalid lag relocation method.");
    sampler_counts(labels, atoms.size());
    sampler_counts(covariance_labels, covariance_atoms.size());
    if (covariance_labels.size() != labels.size() || design.size() != labels.size() ||
        design_cross.size() != labels.size()) Rcpp::stop("Invalid lag relocation data dimensions.");
    const bool random_pair     = subject == 0;
    const bool guide_subject   = method == "guided";
    double log_subject_forward = -std::log(static_cast<double>(labels.size()));
    if (guide_subject) {
        const VectorXd probability = relocation_subject_probabilities(responses, design, design_cross,
            covariance_atoms, covariance_labels, labels, slots);
        if (random_pair) subject = draw_relocation_index(probability);
        if (subject == 0 || probability[subject - 1] == 0.0) return move_result(labels, atoms);
        log_subject_forward = std::log(probability[subject - 1]);
    } else if (random_pair) {
        subject = static_cast<int>(R_unif_index(labels.size())) + 1;
    }
    const int source = labels[subject - 1];
    if (source > slots) return move_result(labels, atoms);
    if (destination == source) Rcpp::stop("Relocation destination must differ from the source.");

    double log_destination_forward = -std::log(static_cast<double>(slots - 1));
    double log_destination_reverse = log_destination_forward;
    if (method != "uniform") {
        const VectorXd scores = relocation_scores(responses, design, design_cross,
            covariance_atoms, covariance_labels, labels, subject - 1, slots);
        const VectorXd forward = relocation_probabilities(scores, source - 1);
        if (random_pair) destination = draw_relocation_index(forward);
        const VectorXd reverse  = relocation_probabilities(scores, destination - 1);
        log_destination_forward = std::log(forward[destination - 1]);
        log_destination_reverse = std::log(reverse[source - 1]);
    } else if (random_pair) {
        destination = static_cast<int>(R_unif_index(slots - 1)) + 1;
        if (destination >= source) ++destination;
    }

    double log_subject_reverse = log_subject_forward;
    if (guide_subject) {
        NumericVector proposed = Rcpp::clone(labels);
        proposed[subject - 1]  = destination;
        const VectorXd reverse = relocation_subject_probabilities(responses, design, design_cross,
            covariance_atoms, covariance_labels, proposed, slots);
        log_subject_reverse = std::log(reverse[subject - 1]);
    }
    const double log_forward    = log_subject_forward + log_destination_forward;
    const double log_reverse    = log_subject_reverse + log_destination_reverse;
    const NumericVector weights = fixed_weights.isNotNull() ? NumericVector(fixed_weights.get()) :
        pdpmvar_eigen_prefix_weights(labels, concentration, slots);
    if (weights.size() < slots || !std::isfinite(weights[source - 1]) ||
        !std::isfinite(weights[destination - 1])) Rcpp::stop("Invalid lag relocation stick weights.");

    const LagGaussianData data(responses, design, design_cross, covariance_atoms, covariance_labels, variance);
    std::vector<int> old_source, old_destination, new_source, new_destination;
    for (int i = 0; i < labels.size(); ++i) {
        if (labels[i] == source) old_source.push_back(i);
        if (labels[i] == destination) old_destination.push_back(i);
        if (labels[i] == source && i != subject - 1) new_source.push_back(i);
        if (labels[i] == destination || i == subject - 1) new_destination.push_back(i);
    }

    // Save factorizations that fit within the proposal's memory budget. An empty
    // group has log integral zero and needs no Gaussian factorization.
    LagFactorCache cache;
    if (!evaluate_only) {
        cache.members = {old_source, old_destination, new_source, new_destination};
        cache.groups.resize(cache.members.size());
    }
    const double old_score = lag_group_score(data, old_source, &cache) +
        lag_group_score(data, old_destination, &cache);
    const double new_score = lag_group_score(data, new_source, &cache) +
        lag_group_score(data, new_destination, &cache);
    const double gaussian  = new_score - old_score;
    const double weight    = weights[destination - 1] - weights[source - 1];
    const double log_ratio = gaussian + weight + (log_reverse - log_forward);
    if (!std::isfinite(log_ratio)) Rcpp::stop("Invalid lag relocation log ratio.");
    const bool accepted = !evaluate_only && std::log(R::runif(0, 1)) < std::min(0.0, log_ratio);
    const List record   = List::create(Rcpp::Named("type") = "relocate",
        Rcpp::Named("accepted") = accepted, Rcpp::Named("log_ratio") = log_ratio,
        Rcpp::Named("subject") = subject, Rcpp::Named("source") = source,
        Rcpp::Named("destination") = destination, Rcpp::Named("gaussian") = gaussian,
        Rcpp::Named("stick_weight") = weight, Rcpp::Named("log_forward") = log_forward,
        Rcpp::Named("log_reverse") = log_reverse,
        Rcpp::Named("log_subject_forward") = log_subject_forward,
        Rcpp::Named("log_subject_reverse") = log_subject_reverse);
    if (evaluate_only) return record;

    NumericVector result_labels = Rcpp::clone(labels);
    if (accepted) result_labels[subject - 1] = destination;
    List result_atoms = Rcpp::clone(atoms);
    while (result_atoms.size() < destination) result_atoms.push_back(sampler_new_coefficient(variance));
    result_atoms[source - 1]      = refresh_lag_atom(data, accepted ? new_source : old_source, variance, cache);
    result_atoms[destination - 1] = refresh_lag_atom(data,
        accepted ? new_destination : old_destination, variance, cache);
    return List::create(Rcpp::Named("labels") = result_labels, Rcpp::Named("atoms") = result_atoms,
        Rcpp::Named("record") = record, Rcpp::Named("log_weights") = weights);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Apply coefficient group moves across lags ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Sweep through the K lag partitions, trying a split or merge or a
// single-subject transfer at each lag. For D nodes and T_i observations,
// the input responses are D x T_i and stacked predictors are (D * K) x T_i.
// Remove the other lag contributions before each move. Use full covariances
// to integrate out factors, or adjusted responses and diagonal residual
// variances to hold factors fixed. Called before the joint coefficient draw.
// [[Rcpp::export(name = ".lgpdpmvar_eigen_coefficient_moves")]]
Rcpp::List lgpdpmvar_eigen_coefficient_moves(const Rcpp::List& responses,
                                             const Rcpp::List& design,
                                             const Rcpp::List& design_cross,
                                             const Rcpp::List& initial_labels,
                                             const Rcpp::List& initial_atoms,
                                             const Rcpp::List& variance,
                                             const Eigen::MatrixXd& residual_variances,
                                             const Rcpp::NumericVector& concentration,
                                             int slots,
                                             Rcpp::Nullable<Rcpp::List> covariances = R_NilValue,
                                             Rcpp::Nullable<Rcpp::NumericVector> covariance_labels = R_NilValue,
                                             bool relocate                 = false,
                                             std::string relocation_method = "uniform") {
    const bool collapsed = covariances.isNotNull();
    const bool low_rank  = covariance_labels.isNotNull();
    if ((low_rank && !collapsed) || (relocate && !low_rank)) Rcpp::stop("Supply factor covariance atoms for the collapsed lag move.");
    const NumericVector groups     = low_rank ? NumericVector(covariance_labels.get()) : NumericVector();
    const List subject_covariances = collapsed ? List(covariances.get()) : List();
    const int n_subjects           = responses.size();
    const int n_lags               = initial_labels.size();
    const int n_nodes              = n_subjects > 0 ? Rcpp::as<NumericMatrix>(responses[0]).nrow() : 0;
    if (n_subjects == 0 || n_lags == 0 || n_nodes == 0 || design.size() != n_subjects ||
        design_cross.size() != n_subjects || initial_atoms.size() != n_lags ||
        variance.size() != n_lags || concentration.size() != n_lags ||
        (collapsed ? (low_rank ? groups.size() != n_subjects : subject_covariances.size() != n_subjects) :
         (residual_variances.rows() != n_nodes || residual_variances.cols() != n_subjects))) {
        Rcpp::stop("Invalid lag split/merge dimensions.");
    }
    for (int lag = 0; lag < n_lags; ++lag) {
        const NumericVector lag_labels = initial_labels[lag];
        const List lag_atoms           = initial_atoms[lag];
        if (lag_labels.size() != n_subjects) Rcpp::stop("Each lag needs one label per subject.");
        sampler_counts(lag_labels, lag_atoms.size());
    }

    List labels = Rcpp::clone(initial_labels);
    List atoms  = Rcpp::clone(initial_atoms);
    List records(n_lags);
    std::vector<MatrixXd> predictors(n_subjects), cross(n_subjects), residuals(n_subjects);
    for (int subject = 0; subject < n_subjects; ++subject) {
        predictors[subject]     = Rcpp::as<MatrixXd>(design[subject]);
        cross[subject]          = Rcpp::as<MatrixXd>(design_cross[subject]);
        const MatrixXd response = Rcpp::as<MatrixXd>(responses[subject]);
        const int n_predictors  = n_nodes * n_lags;
        if (response.rows() != n_nodes || response.cols() != predictors[subject].cols() ||
            predictors[subject].rows() != n_predictors || cross[subject].rows() != n_predictors ||
            cross[subject].cols() != n_predictors) Rcpp::stop("Invalid lag split/merge regression matrices.");
        residuals[subject] = response -
            lag_subject_coefficient(atoms, labels, subject, n_nodes) * predictors[subject];
    }

    for (int lag = 0; lag < n_lags; ++lag) {
        const int start                = lag * n_nodes;
        const List old_atoms           = atoms[lag];
        const NumericVector old_labels = labels[lag];
        List partial(n_subjects), lag_design(n_subjects), lag_cross(n_subjects);
        for (int subject = 0; subject < n_subjects; ++subject) {
            const List atom = old_atoms[static_cast<int>(old_labels[subject]) - 1];
            const auto z    = predictors[subject].middleRows(start, n_nodes);
            residuals[subject].noalias() += Rcpp::as<MatrixXd>(atom["coefficient"]) * z;
            partial[subject]    = residuals[subject];
            lag_design[subject] = MatrixXd(z);
            lag_cross[subject]  = MatrixXd(cross[subject].block(start, start, n_nodes, n_nodes));
        }

        const List move = relocate ?
            lgpdpmvar_eigen_relocation_move(partial, lag_design, lag_cross, subject_covariances, groups,
                old_labels, old_atoms, variance[lag], concentration[lag], slots,
                0, 0, R_NilValue, false, relocation_method) :
            low_rank ?
            lag_collapsed_move(partial, lag_design, lag_cross, subject_covariances, groups,
                old_labels, old_atoms, variance[lag], concentration[lag], slots) :
            collapsed ?
            pdpmvar_eigen_coefficient_move(partial, lag_design, lag_cross, subject_covariances,
                old_labels, old_atoms, variance[lag], concentration[lag], slots) :
            pdpmvar_eigen_row_move(partial, lag_design, lag_cross, residual_variances,
                old_labels, old_atoms, variance[lag], concentration[lag], slots);
        const List new_atoms           = move["atoms"];
        const NumericVector new_labels = move["labels"];

        // Even a rejected move can refresh the affected matrices. Use those
        // draws when forming the next lag's partial residual.
        for (int subject = 0; subject < n_subjects; ++subject) {
            const List atom = new_atoms[static_cast<int>(new_labels[subject]) - 1];
            residuals[subject].noalias() -= Rcpp::as<MatrixXd>(atom["coefficient"]) *
                predictors[subject].middleRows(start, n_nodes);
        }
        atoms[lag]   = new_atoms;
        labels[lag]  = new_labels;
        records[lag] = move["record"];
    }

    return List::create(Rcpp::Named("labels") = labels, Rcpp::Named("atoms") = atoms,
        Rcpp::Named("records") = records);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Run the lag-specific sampler and collect results ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Run the sampler called by lgpdpmvar() and save its estimates and traces.
// For D nodes, K lags, N subjects and M retained draws, return coefficient
// summaries of size D x (D * K) x N with lag blocks side by side. Covariance
// summaries are D x D x N. Each lag and covariance get their own M x N
// label matrix, with retained iterations in rows and subjects in columns.
// [[Rcpp::export(name = ".lgpdpmvar_eigen_sampler")]]
Rcpp::List lgpdpmvar_eigen_sampler(const Rcpp::List& X,
                                   const Rcpp::List& design,
                                   const Rcpp::List& design_cross,
                                   const Rcpp::List& initial_state,
                                   const Rcpp::List& control) {
    List atoms                        = Rcpp::clone(Rcpp::as<List>(initial_state["coefficient_atoms"]));
    List labels                       = Rcpp::clone(Rcpp::as<List>(initial_state["coefficient_labels"]));
    List variance                     = Rcpp::clone(Rcpp::as<List>(initial_state["shared_variance"]));
    NumericVector lambda_squared      = Rcpp::clone(Rcpp::as<NumericVector>(initial_state["lambda_squared"]));
    List covariance_atoms             = Rcpp::clone(Rcpp::as<List>(initial_state["covariance_atoms"]));
    NumericVector covariance_labels   = Rcpp::clone(Rcpp::as<NumericVector>(initial_state["covariance_labels"]));
    const NumericVector concentration = control["coefficient_concentration"];
    const int n_lags                  = labels.size();
    const int n_subjects              = X.size();
    const MatrixXd first              = Rcpp::as<MatrixXd>(X[0]);
    const int n_nodes                 = first.rows();
    const int n_factors               = Rcpp::as<int>(control["n_factors"]);
    const int n_burnin                = Rcpp::as<int>(control["n_burnin"]);
    const int n_mcmc                  = Rcpp::as<int>(control["n_mcmc"]);
    const int total                   = n_burnin + n_mcmc;
    const int interval                = Rcpp::as<int>(control["split_merge_every"]);
    const int slots                   = Rcpp::as<int>(control["split_merge_slots"]);
    const int coefficient_interval    = control.containsElementNamed("coefficient_split_merge_every") ?
        Rcpp::as<int>(control["coefficient_split_merge_every"]) : 0;
    const int coefficient_slots = control.containsElementNamed("coefficient_split_merge_slots") ?
        Rcpp::as<int>(control["coefficient_split_merge_slots"]) : 10;
    const std::string relocation_method = control.containsElementNamed("coefficient_relocation_method") ?
        Rcpp::as<std::string>(control["coefficient_relocation_method"]) : "uniform";
    const int relocation_interval = control.containsElementNamed("coefficient_relocation_every") ?
        Rcpp::as<int>(control["coefficient_relocation_every"]) : 0;
    const bool collapsed = control.containsElementNamed("coefficient_split_merge") &&
        Rcpp::as<std::string>(control["coefficient_split_merge"]) == "collapsed";
    const double cov_concentration = Rcpp::as<double>(control["covariance_concentration"]);
    const double cov_shape         = Rcpp::as<double>(control["cov_shape"]);
    const double cov_rate          = Rcpp::as<double>(control["cov_rate"]);
    const bool laplace             = Rcpp::as<bool>(control["laplace"]);
    const bool store_draws         = Rcpp::as<bool>(control["store_draws"]);
    const bool verbose             = Rcpp::as<bool>(control["verbose"]);

    // Store each subject's full predictors, (D * K) x T_i, and the D x T_i
    // predictor block for each lag. K is lags and T_i is subject i's time count.
    std::vector<MatrixXd> data(n_subjects), predictors(n_subjects), residuals(n_subjects);
    std::vector<List> lag_design;
    lag_design.reserve(n_lags);
    for (int lag = 0; lag < n_lags; ++lag) lag_design.emplace_back(n_subjects);
    for (int subject = 0; subject < n_subjects; ++subject) {
        data[subject]       = Rcpp::as<MatrixXd>(X[subject]);
        predictors[subject] = Rcpp::as<MatrixXd>(design[subject]);
        for (int lag = 0; lag < n_lags; ++lag) {
            lag_design[lag][subject] = MatrixXd(predictors[subject].middleRows(lag * n_nodes, n_nodes));
        }
    }

    // Accumulate subject summaries without saving all draws unless requested.
    const R_xlen_t covariance_size   = static_cast<R_xlen_t>(n_nodes) * n_nodes;
    const R_xlen_t coefficient_size  = covariance_size * n_lags;
    const R_xlen_t coefficient_total = coefficient_size * n_subjects;
    const R_xlen_t covariance_total  = covariance_size * n_subjects;
    NumericVector coefficient_sum(coefficient_total), positive_sum(coefficient_total), covariance_sum(covariance_total);
    coefficient_sum.attr("dim") = Rcpp::IntegerVector::create(n_nodes, n_nodes * n_lags, n_subjects);
    positive_sum.attr("dim")    = coefficient_sum.attr("dim");
    covariance_sum.attr("dim")  = Rcpp::IntegerVector::create(n_nodes, n_nodes, n_subjects);

    // Save one label matrix and one co-clustering matrix per partition.
    List coefficient_membership(n_lags), coefficient_similarity(n_lags);
    NumericMatrix covariance_membership(n_mcmc, n_subjects), covariance_similarity(n_subjects, n_subjects);
    NumericMatrix cluster_counts(total, n_lags + 1), parameter_traces(n_mcmc, 3);
    NumericMatrix lambda_trace(n_mcmc, n_lags), shared_variance_sum(n_nodes, n_nodes * n_lags);
    Rcpp::CharacterVector lag_names(n_lags), count_names(n_lags + 1);
    for (int lag = 0; lag < n_lags; ++lag) {
        coefficient_membership[lag] = NumericMatrix(n_mcmc, n_subjects);
        coefficient_similarity[lag] = NumericMatrix(n_subjects, n_subjects);
        lag_names[lag]              = "lag_" + std::to_string(lag + 1);
        count_names[lag]            = lag_names[lag];
    }
    count_names[n_lags]              = "covariance";
    coefficient_membership.names()   = lag_names;
    coefficient_similarity.names()   = lag_names;
    Rcpp::colnames(cluster_counts)   = count_names;
    Rcpp::colnames(lambda_trace)     = lag_names;
    Rcpp::colnames(parameter_traces) = Rcpp::CharacterVector::create("log_likelihood", "mean_node_variance", "coefficient_rms");

    // Record proposal outcomes at every iteration, including burn-in.
    Rcpp::IntegerVector iterations = Rcpp::seq(1, total);
    Rcpp::CharacterVector move_types(total, "none");
    Rcpp::LogicalVector move_accepted(total);
    NumericVector move_ratios(total, NA_REAL);
    std::vector<Rcpp::CharacterVector> lag_move_types;
    std::vector<Rcpp::LogicalVector> lag_move_accepted;
    std::vector<NumericVector> lag_move_ratios;
    for (int lag = 0; lag < n_lags; ++lag) {
        lag_move_types.emplace_back(total, "none");
        lag_move_accepted.emplace_back(total);
        lag_move_ratios.emplace_back(total, NA_REAL);
    }
    std::vector<Rcpp::CharacterVector> relocation_types;
    std::vector<Rcpp::LogicalVector> relocation_accepted;
    std::vector<NumericVector> relocation_ratios;
    for (int lag = 0; lag < n_lags; ++lag) {
        relocation_types.emplace_back(total, "none");
        relocation_accepted.emplace_back(total);
        relocation_ratios.emplace_back(total, NA_REAL);
    }
    Rcpp::IntegerMatrix relocation_subjects(total, n_lags), relocation_sources(total, n_lags),
        relocation_destinations(total, n_lags);
    std::fill(relocation_subjects.begin(), relocation_subjects.end(), NA_INTEGER);
    std::fill(relocation_sources.begin(), relocation_sources.end(), NA_INTEGER);
    std::fill(relocation_destinations.begin(), relocation_destinations.end(), NA_INTEGER);

    // Optional draw arrays add retained iterations as the fourth dimension.
    NumericVector coefficient_draws, covariance_draws;
    if (store_draws) {
        coefficient_draws             = NumericVector(coefficient_total * n_mcmc);
        covariance_draws              = NumericVector(covariance_total * n_mcmc);
        coefficient_draws.attr("dim") = Rcpp::IntegerVector::create(n_nodes, n_nodes * n_lags, n_subjects, n_mcmc);
        covariance_draws.attr("dim")  = Rcpp::IntegerVector::create(n_nodes, n_nodes, n_subjects, n_mcmc);
    }

    for (int iteration = 0; iteration < total; ++iteration) {
        Rcpp::checkUserInterrupt();
        for (int subject = 0; subject < n_subjects; ++subject) {
            residuals[subject] = data[subject] -
                lag_subject_coefficient(atoms, labels, subject, n_nodes) * predictors[subject];
        }

        // Update each lag's labels with the other lag contributions held fixed.
        // Carry the updated residual forward so the next lag sees the new labels.
        for (int lag = 0; lag < n_lags; ++lag) {
            const List old_atoms           = atoms[lag];
            const NumericVector old_labels = labels[lag];
            const NumericMatrix lag_var    = variance[lag];
            const List slice               = pdpmvar_eigen_slice_weights(old_labels, concentration[lag]);
            const NumericVector weights    = slice["weights"];
            List lag_atoms(weights.size()), partial(n_subjects);
            for (int group = 0; group < lag_atoms.size(); ++group) {
                if (group < old_atoms.size()) lag_atoms[group] = old_atoms[group];
                else lag_atoms[group] = sampler_new_coefficient(lag_var);
            }
            for (int subject = 0; subject < n_subjects; ++subject) {
                const List atom = old_atoms[static_cast<int>(old_labels[subject]) - 1];
                residuals[subject] += Rcpp::as<MatrixXd>(atom["coefficient"]) *
                    predictors[subject].middleRows(lag * n_nodes, n_nodes);
                partial[subject] = residuals[subject];
            }

            const List assignment = pdpmvar_eigen_assign_coefficients(partial, lag_design[lag], lag_atoms,
                covariance_labels, covariance_atoms, weights, slice["slices"], lag_assignment_uniforms(n_subjects));
            const NumericVector next_labels = assignment["labels"];
            for (int subject = 0; subject < n_subjects; ++subject) {
                const List atom = lag_atoms[static_cast<int>(next_labels[subject]) - 1];
                residuals[subject] -= Rcpp::as<MatrixXd>(atom["coefficient"]) *
                    predictors[subject].middleRows(lag * n_nodes, n_nodes);
            }
            atoms[lag]  = lag_atoms;
            labels[lag] = next_labels;
        }

        // Keep the new labels, matrices and proposal records for the scheduled moves.
        const auto save_moves = [&](const List& moves, bool relocation = false) {
            labels             = Rcpp::as<List>(moves["labels"]);
            atoms              = Rcpp::as<List>(moves["atoms"]);
            const List records = moves["records"];
            for (int lag = 0; lag < n_lags; ++lag) {
                const List record = records[lag];
                if (relocation) {
                    relocation_types[lag][iteration]    = Rcpp::as<std::string>(record["type"]);
                    relocation_accepted[lag][iteration] = Rcpp::as<bool>(record["accepted"]);
                    relocation_ratios[lag][iteration]   = Rcpp::as<double>(record["log_ratio"]);
                    if (record.containsElementNamed("subject")) {
                        relocation_subjects(iteration, lag)     = Rcpp::as<int>(record["subject"]);
                        relocation_sources(iteration, lag)      = Rcpp::as<int>(record["source"]);
                        relocation_destinations(iteration, lag) = Rcpp::as<int>(record["destination"]);
                    }
                } else {
                    lag_move_types[lag][iteration]    = Rcpp::as<std::string>(record["type"]);
                    lag_move_accepted[lag][iteration] = Rcpp::as<bool>(record["accepted"]);
                    lag_move_ratios[lag][iteration]   = Rcpp::as<double>(record["log_ratio"]);
                }
            }
        };

        const bool try_move       = coefficient_interval > 0 && (iteration + 1) % coefficient_interval == 0;
        const bool try_relocation = relocation_interval > 0 && (iteration + 1) % relocation_interval == 0;
        if (try_move && collapsed) {
            save_moves(lgpdpmvar_eigen_coefficient_moves(X, design, design_cross,
                labels, atoms, variance, MatrixXd(), concentration, coefficient_slots,
                covariance_atoms, covariance_labels));
        }
        if (try_relocation) {
            save_moves(lgpdpmvar_eigen_coefficient_moves(X, design, design_cross,
                labels, atoms, variance, MatrixXd(), concentration, coefficient_slots,
                covariance_atoms, covariance_labels, true, relocation_method), true);
        }
        if ((try_move && collapsed) || try_relocation) {
            // Recalculate the residuals before drawing factors under the new labels and matrices.
            for (int subject = 0; subject < n_subjects; ++subject) {
                residuals[subject] = data[subject] -
                    lag_subject_coefficient(atoms, labels, subject, n_nodes) * predictors[subject];
            }
        }

        // Draw factors using the full residuals. Subtract their contribution
        // from the observations before the coefficient draws.
        List adjusted(n_subjects);
        MatrixXd residual_variances(n_nodes, n_subjects);
        for (int subject = 0; subject < n_subjects; ++subject) {
            const List atom = covariance_atoms[static_cast<int>(covariance_labels[subject]) - 1];
            MatrixXd noise(n_factors, data[subject].cols());
            for (Eigen::Index index = 0; index < noise.size(); ++index) noise.data()[index] = R::rnorm(0.0, 1.0);
            const MatrixXd factors          = pdpmvar_eigen_factors(residuals[subject], atom, noise);
            adjusted[subject]               = MatrixXd(data[subject] - Rcpp::as<MatrixXd>(atom["working_loadings"]) * factors);
            residual_variances.col(subject) = Rcpp::as<VectorXd>(atom["variance"]);
        }

        // The factor-conditional move holds these draws fixed.
        if (try_move && !collapsed) {
            save_moves(lgpdpmvar_eigen_coefficient_moves(adjusted, design, design_cross,
                labels, atoms, variance, residual_variances, concentration, coefficient_slots));
        }

        // Draw each response row jointly across the occupied groups and lags.
        atoms = lgpdpmvar_eigen_coefficient_update(adjusted, design, design_cross, labels, atoms, variance, residual_variances);

        // Update shrinkage using occupied groups, then redraw empty matrices from the prior.
        for (int lag = 0; lag < n_lags; ++lag) {
            const NumericVector lag_labels = labels[lag];
            List lag_atoms                 = atoms[lag];
            const auto counts              = sampler_counts(lag_labels, lag_atoms.size());
            cluster_counts(iteration, lag) = std::count_if(counts.begin(), counts.end(), [](int n) { return n > 0; });
            if (laplace) {
                const List shrinkage = pdpmvar_eigen_shrinkage(lag_labels, lag_atoms, variance[lag], lambda_squared[lag]);
                variance[lag]        = shrinkage["shared_variance"];
                lambda_squared[lag]  = Rcpp::as<double>(shrinkage["lambda_squared"]);
            }
            const NumericMatrix lag_var = variance[lag];
            for (int group = 0; group < lag_atoms.size(); ++group) {
                if (counts[group] == 0) lag_atoms[group] = sampler_new_coefficient(lag_var);
            }
        }

        // Subtract all lag contributions to form the residuals for covariance updates.
        List full_residuals(n_subjects);
        std::vector<MatrixXd> coefficients(n_subjects);
        for (int subject = 0; subject < n_subjects; ++subject) {
            coefficients[subject]   = lag_subject_coefficient(atoms, labels, subject, n_nodes);
            full_residuals[subject] = MatrixXd(data[subject] - coefficients[subject] * predictors[subject]);
        }
        if (interval > 0 && (iteration + 1) % interval == 0) {
            const List move = pdpmvar_eigen_covariance_move(full_residuals, covariance_labels, covariance_atoms,
                cov_concentration, n_factors, cov_shape, cov_rate, slots);
            covariance_labels        = Rcpp::as<NumericVector>(move["labels"]);
            covariance_atoms         = Rcpp::as<List>(move["atoms"]);
            const List record        = move["record"];
            move_types[iteration]    = Rcpp::as<std::string>(record["type"]);
            move_accepted[iteration] = Rcpp::as<bool>(record["accepted"]);
            move_ratios[iteration]   = Rcpp::as<double>(record["log_ratio"]);
        }

        // Draw covariance weights and slices, then update subjects' covariance labels.
        const List slice            = pdpmvar_eigen_slice_weights(covariance_labels, cov_concentration);
        const NumericVector weights = slice["weights"];
        List next_covariances(weights.size());
        for (int group = 0; group < next_covariances.size(); ++group) {
            if (group < covariance_atoms.size()) next_covariances[group] = covariance_atoms[group];
            else next_covariances[group] = sampler_new_covariance(n_nodes, n_factors, cov_shape, cov_rate);
        }
        covariance_atoms      = next_covariances;
        const List assignment = pdpmvar_eigen_assign_covariances(full_residuals, covariance_atoms,
            weights, slice["slices"], lag_assignment_uniforms(n_subjects));
        covariance_labels = Rcpp::as<NumericVector>(assignment["labels"]);

        // Pool residual time points within each group to update its covariance.
        const auto counts = sampler_counts(covariance_labels, covariance_atoms.size());
        std::vector<std::vector<int>> members(covariance_atoms.size());
        for (int subject = 0; subject < n_subjects; ++subject) {
            members[static_cast<int>(covariance_labels[subject]) - 1].push_back(subject);
        }
        for (int group = 0; group < covariance_atoms.size(); ++group) {
            if (counts[group] == 0) {
                covariance_atoms[group] = sampler_new_covariance(n_nodes, n_factors, cov_shape, cov_rate);
                continue;
            }
            Eigen::Index n_times = 0;
            for (int subject : members[group]) n_times += data[subject].cols();
            MatrixXd pooled(n_nodes, n_times);
            Eigen::Index offset = 0;
            for (int subject : members[group]) {
                const MatrixXd residual                    = Rcpp::as<MatrixXd>(full_residuals[subject]);
                pooled.middleCols(offset, residual.cols()) = residual;
                offset += residual.cols();
            }
            covariance_atoms[group] = pdpmvar_eigen_covariance_update(pooled, covariance_atoms[group], cov_shape, cov_rate);
        }
        cluster_counts(iteration, n_lags) = std::count_if(counts.begin(), counts.end(), [](int n) { return n > 0; });

        if (verbose && ((iteration + 1) % 100 == 0 || iteration + 1 == total)) {
            Rcpp::Rcout << "Iteration " << iteration + 1 << " of " << total << '\n';
        }
        if (iteration < n_burnin) continue;

        // Save the retained labels and count how often each pair shares a group.
        const int retained = iteration - n_burnin;
        for (int lag = 0; lag < n_lags; ++lag) {
            const NumericVector lag_labels = labels[lag];
            const NumericMatrix lag_var    = variance[lag];
            NumericMatrix membership       = coefficient_membership[lag];
            NumericMatrix similarity       = coefficient_similarity[lag];
            lambda_trace(retained, lag)    = std::sqrt(lambda_squared[lag]);
            for (R_xlen_t index = 0; index < covariance_size; ++index) {
                shared_variance_sum[lag * covariance_size + index] += lag_var[index];
            }
            for (int subject = 0; subject < n_subjects; ++subject) {
                membership(retained, subject) = lag_labels[subject];
                for (int other = 0; other < n_subjects; ++other) {
                    similarity(subject, other) += lag_labels[subject] == lag_labels[other];
                }
            }
        }

        double mean_square = 0.0, mean_variance = 0.0;
        for (int subject = 0; subject < n_subjects; ++subject) {
            covariance_membership(retained, subject) = covariance_labels[subject];
            for (int other = 0; other < n_subjects; ++other) {
                covariance_similarity(subject, other) += covariance_labels[subject] == covariance_labels[other];
            }
            const List atom                = covariance_atoms[static_cast<int>(covariance_labels[subject]) - 1];
            const NumericMatrix covariance = atom["covariance"];
            const MatrixXd& coefficient    = coefficients[subject];
            mean_square += coefficient.squaredNorm() / coefficient_size / n_subjects;
            for (int node = 0; node < n_nodes; ++node) mean_variance += covariance(node, node) / n_nodes / n_subjects;
            for (R_xlen_t index = 0; index < coefficient_size; ++index) {
                const R_xlen_t position = subject * coefficient_size + index;
                const double value      = coefficient.data()[index];
                coefficient_sum[position] += value;
                positive_sum[position]    += value > 0.0;
                if (store_draws) coefficient_draws[retained * coefficient_total + position] = value;
            }
            for (R_xlen_t index = 0; index < covariance_size; ++index) {
                const R_xlen_t position = subject * covariance_size + index;
                covariance_sum[position] += covariance[index];
                if (store_draws) covariance_draws[retained * covariance_total + position] = covariance[index];
            }
        }
        parameter_traces(retained, 0) = pdpmvar_eigen_total_likelihood(full_residuals, covariance_labels, covariance_atoms);
        parameter_traces(retained, 1) = mean_variance;
        parameter_traces(retained, 2) = std::sqrt(mean_square);
    }

    // Turn sums and counts into posterior means and probabilities.
    NumericVector sign_tail(coefficient_total);
    sign_tail.attr("dim") = coefficient_sum.attr("dim");
    for (R_xlen_t index = 0; index < coefficient_total; ++index) {
        coefficient_sum[index] /= n_mcmc;
        positive_sum[index]    /= n_mcmc;
        sign_tail[index] = 2.0 * std::min(positive_sum[index], 1.0 - positive_sum[index]);
    }
    for (double& value : covariance_sum) value /= n_mcmc;
    for (double& value : covariance_similarity) value /= n_mcmc;
    for (double& value : shared_variance_sum) value /= n_mcmc;
    for (int lag = 0; lag < n_lags; ++lag) {
        NumericMatrix similarity = coefficient_similarity[lag];
        for (double& value : similarity) value /= n_mcmc;
    }

    // Return the last sampled state along with the posterior summaries and move records.
    List final_state = List::create(Rcpp::Named("coefficient_atoms") = atoms,
        Rcpp::Named("coefficient_labels") = labels, Rcpp::Named("covariance_atoms") = covariance_atoms,
        Rcpp::Named("covariance_labels") = covariance_labels, Rcpp::Named("shared_variance") = variance,
        Rcpp::Named("lambda_squared") = lambda_squared);
    Rcpp::RObject draws = R_NilValue;
    if (store_draws) draws = List::create(Rcpp::Named("coefficients") = coefficient_draws,
                                         Rcpp::Named("covariance") = covariance_draws);
    Rcpp::RObject lambda_result = laplace ? static_cast<SEXP>(lambda_trace) : R_NilValue;
    List coefficient_move_traces(n_lags);
    coefficient_move_traces.names() = lag_names;
    for (int lag = 0; lag < n_lags; ++lag) {
        coefficient_move_traces[lag] = sampler_move_trace(iterations, lag_move_types[lag],
            lag_move_accepted[lag], lag_move_ratios[lag]);
    }
    List relocation_traces(n_lags);
    relocation_traces.names() = lag_names;
    for (int lag = 0; lag < n_lags; ++lag) {
        const Rcpp::IntegerVector subjects     = relocation_subjects(Rcpp::_, lag);
        const Rcpp::IntegerVector sources      = relocation_sources(Rcpp::_, lag);
        const Rcpp::IntegerVector destinations = relocation_destinations(Rcpp::_, lag);
        relocation_traces[lag]                 = Rcpp::DataFrame::create(
            Rcpp::Named("iteration") = iterations, Rcpp::Named("type") = relocation_types[lag],
            Rcpp::Named("accepted") = relocation_accepted[lag], Rcpp::Named("log_ratio") = relocation_ratios[lag],
            Rcpp::Named("subject") = subjects, Rcpp::Named("source") = sources,
            Rcpp::Named("destination") = destinations);
    }
    return List::create(Rcpp::Named("coefficient_mean") = coefficient_sum,
        Rcpp::Named("covariance_mean") = covariance_sum, Rcpp::Named("coefficient_sign_probability") = positive_sum,
        Rcpp::Named("coefficient_sign_tail") = sign_tail, Rcpp::Named("coefficient_labels") = coefficient_membership,
        Rcpp::Named("covariance_labels") = covariance_membership, Rcpp::Named("coefficient_coclustering") = coefficient_similarity,
        Rcpp::Named("covariance_coclustering") = covariance_similarity, Rcpp::Named("cluster_counts") = cluster_counts,
        Rcpp::Named("parameter_traces") = parameter_traces, Rcpp::Named("lambda") = lambda_result,
        Rcpp::Named("shared_variance_mean") = shared_variance_sum, Rcpp::Named("final_state") = final_state,
        Rcpp::Named("split_merge") = sampler_move_trace(iterations, move_types, move_accepted, move_ratios),
        Rcpp::Named("coefficient_split_merge") = coefficient_move_traces,
        Rcpp::Named("coefficient_relocation") = relocation_traces, Rcpp::Named("draws") = draws);
}
