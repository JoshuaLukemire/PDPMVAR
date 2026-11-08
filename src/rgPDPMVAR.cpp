// Row-specific coefficient mixtures and the rgPDPMVAR sampling loop.
#include "sampling_helpers.h"
#include <algorithm>
#include <cmath>
#include <limits>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Assemble coefficients and draw assignment uniforms ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Used for coefficient and covariance assignments. Return N independent
// uniform draws, one for each of the N subjects.
static NumericVector row_assignment_uniforms(int n_subjects) {
    NumericVector uniforms(n_subjects);
    for (double& uniform : uniforms) uniform = R::runif(0.0, 1.0);
    return uniforms;
}

// Assemble the subject's nodes-by-lagged-predictors matrix, D x (D * K),
// by selecting one row for each of the D response nodes. Within each row,
// predictors are ordered by lag. Used for residuals and posterior summaries.
static MatrixXd row_subject_coefficient(const List& atoms,
                                        const List& labels,
                                        int subject,
                                        int n_predictors) {
    MatrixXd coefficient(atoms.size(), n_predictors);
    for (int node = 0; node < atoms.size(); ++node) {
        const List row_atoms           = atoms[node];
        const NumericVector row_labels = labels[node];
        const List atom                = row_atoms[static_cast<int>(row_labels[subject]) - 1];
        coefficient.row(node)          = Rcpp::as<MatrixXd>(atom["coefficient"]);
    }
    return coefficient;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Try single-subject transfers across response nodes ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Store one record per iteration for a response node's transfers. The
// sampler returns these records when coefficient relocation is enabled.
struct RowRelocationTrace {
    Rcpp::CharacterVector types;
    Rcpp::LogicalVector accepted;
    NumericVector ratios;
    Rcpp::IntegerVector subjects, sources, destinations;

    explicit RowRelocationTrace(int iterations) :
        types(iterations, "none"), accepted(iterations), ratios(iterations, NA_REAL),
        subjects(iterations, NA_INTEGER), sources(iterations, NA_INTEGER), destinations(iterations, NA_INTEGER) {}

    // Save the outcome of this node's scheduled transfer attempt.
    void save(int iteration, const List& record) {
        types[iteration]    = Rcpp::as<std::string>(record["type"]);
        accepted[iteration] = Rcpp::as<bool>(record["accepted"]);
        ratios[iteration]   = Rcpp::as<double>(record["log_ratio"]);
        if (record.containsElementNamed("subject")) {
            subjects[iteration]     = Rcpp::as<int>(record["subject"]);
            sources[iteration]      = Rcpp::as<int>(record["source"]);
            destinations[iteration] = Rcpp::as<int>(record["destination"]);
        }
    }

    // Present the saved records in iteration order for the fitted object.
    Rcpp::DataFrame as_data_frame(const Rcpp::IntegerVector& iterations) const {
        return Rcpp::DataFrame::create(
            Rcpp::Named("iteration") = iterations, Rcpp::Named("type") = types,
            Rcpp::Named("accepted") = accepted, Rcpp::Named("log_ratio") = ratios,
            Rcpp::Named("subject") = subjects, Rcpp::Named("source") = sources,
            Rcpp::Named("destination") = destinations);
    }
};

// Try one guided transfer per response node before drawing fresh factors.
// Condition on the other response rows under each subject's full covariance.
// Each subject's conditional response is 1 x T_i, with T_i observations.
// Update the affected labels and rows in place and return the updated
// full residuals, D x T_i per subject, for the subsequent factor draws.
static std::vector<MatrixXd> relocate_response_rows(const std::vector<MatrixXd>& data,
                                                     const std::vector<MatrixXd>& predictors,
                                                     const List& design,
                                                     const List& design_cross,
                                                     const List& covariance_atoms,
                                                     const NumericVector& covariance_labels,
                                                     const List& variance,
                                                     const NumericVector& concentration,
                                                     int slots,
                                                     List& atoms,
                                                     List& labels,
                                                     std::vector<RowRelocationTrace>& traces,
                                                     int iteration) {
    const int n_nodes      = atoms.size();
    const int n_subjects   = data.size();
    const int n_predictors = predictors.front().rows();
    std::vector<MatrixXd> residuals(n_subjects);
    for (int subject = 0; subject < n_subjects; ++subject) {
        residuals[subject] = data[subject] -
            row_subject_coefficient(atoms, labels, subject, n_predictors) * predictors[subject];
    }

    // Reuse each group's precision matrix across the node transfers.
    const auto counts = sampler_counts(covariance_labels, covariance_atoms.size());
    std::vector<MatrixXd> precision(covariance_atoms.size());
    for (int group = 0; group < covariance_atoms.size(); ++group) {
        if (counts[group] == 0) continue;
        const List atom      = covariance_atoms[group];
        const MatrixXd lower = likelihood_covariance_lower(Rcpp::as<MatrixXd>(atom["covariance"]), n_nodes);
        const MatrixXd inverse_lower =
            lower.triangularView<Eigen::Lower>().solve(MatrixXd::Identity(n_nodes, n_nodes));
        precision[group] = inverse_lower.transpose() * inverse_lower;
    }

    for (int node = 0; node < n_nodes; ++node) {
        Rcpp::checkUserInterrupt();
        List scalar_covariances(covariance_atoms.size()), response(n_subjects);
        std::vector<Eigen::RowVectorXd> adjustment(covariance_atoms.size());
        for (int group = 0; group < covariance_atoms.size(); ++group) {
            if (counts[group] == 0) continue;
            const double conditional_variance = 1.0 / precision[group](node, node);
            adjustment[group]                 = conditional_variance * precision[group].row(node);
            adjustment[group][node]           = 0.0;
            NumericMatrix covariance(1, 1);
            covariance(0, 0) = conditional_variance;
            scalar_covariances[group] = List::create(
                Rcpp::Named("covariance") = covariance, Rcpp::Named("variance") = NumericVector::create(conditional_variance),
                Rcpp::Named("working_loadings") = NumericMatrix(1, 0), Rcpp::Named("factor_precision") = NumericVector(0));
        }
        for (int subject = 0; subject < n_subjects; ++subject) {
            const int group = static_cast<int>(covariance_labels[subject]) - 1;
            response[subject] = MatrixXd(data[subject].row(node) + adjustment[group] * residuals[subject]);
        }

        // The lag transfer accepts any Gaussian response. Here it sees
        // one response row and all of its lagged predictors.
        const List move = lgpdpmvar_eigen_relocation_move(response, design, design_cross,
            scalar_covariances, covariance_labels, labels[node], atoms[node], variance[node],
            concentration[node], slots, 0, 0, R_NilValue, false, "guided");
        const List record = move["record"];
        traces[node].save(iteration, record);
        if (!record.containsElementNamed("subject")) continue;

        const List next_atoms           = move["atoms"];
        const NumericVector next_labels = move["labels"];
        atoms[node]                     = next_atoms;
        labels[node]                    = next_labels;

        // Both affected atoms are redrawn after either decision. Use the
        // refreshed residuals when conditioning the next response's move.
        for (int subject = 0; subject < n_subjects; ++subject) {
            const List atom = next_atoms[static_cast<int>(next_labels[subject]) - 1];
            const MatrixXd coefficient = Rcpp::as<MatrixXd>(atom["coefficient"]);
            residuals[subject].row(node) = data[subject].row(node) - coefficient * predictors[subject];
        }
    }
    return residuals;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Run the row-specific sampler and collect results ----
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Run the sampler called by rgpdpmvar() and save its estimates and traces.
// For D nodes, K lags, N subjects and M retained draws, return coefficient
// summaries of size D x (D * K) x N with lag blocks side by side. Covariance
// summaries are D x D x N. Each response node and covariance get their own M x N
// label matrix, with retained iterations in rows and subjects in columns.
// [[Rcpp::export(name = ".rgpdpmvar_eigen_sampler")]]
Rcpp::List rgpdpmvar_eigen_sampler(const Rcpp::List& X,
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
    const int n_subjects              = X.size();
    const MatrixXd first              = Rcpp::as<MatrixXd>(X[0]);
    const int n_nodes                 = first.rows();
    const int n_predictors            = Rcpp::as<MatrixXd>(design[0]).rows();
    const int n_factors               = Rcpp::as<int>(control["n_factors"]);
    const int n_burnin                = Rcpp::as<int>(control["n_burnin"]);
    const int n_mcmc                  = Rcpp::as<int>(control["n_mcmc"]);
    const int total                   = n_burnin + n_mcmc;
    const int interval                = Rcpp::as<int>(control["split_merge_every"]);
    const int slots                   = Rcpp::as<int>(control["split_merge_slots"]);
    const int coefficient_interval    = Rcpp::as<int>(control["coefficient_split_merge_every"]);
    const int coefficient_slots       = Rcpp::as<int>(control["coefficient_split_merge_slots"]);
    const int relocation_interval     = control.containsElementNamed("coefficient_relocation_every") ?
        Rcpp::as<int>(control["coefficient_relocation_every"]) : 0;
    const double cov_concentration    = Rcpp::as<double>(control["covariance_concentration"]);
    const double cov_shape            = Rcpp::as<double>(control["cov_shape"]);
    const double cov_rate             = Rcpp::as<double>(control["cov_rate"]);
    const bool store_draws            = Rcpp::as<bool>(control["store_draws"]);
    const bool verbose                = Rcpp::as<bool>(control["verbose"]);

    // Check each subject's (D * K) x T_i predictors and (D * K) x (D * K)
    // cross-products once. Reuse them for all rows and iterations.
    CoefficientDesign saved_design;
    saved_design.predictors.resize(n_subjects);
    saved_design.cross_products.resize(n_subjects);
    const auto& predictors = saved_design.predictors;
    std::vector<MatrixXd> data(n_subjects), adjusted(n_subjects);
    List zero_contributions(n_subjects);
    if (design.size() != n_subjects || design_cross.size() != n_subjects) {
        Rcpp::stop("Each subject needs predictors and predictor cross-products.");
    }
    for (int subject = 0; subject < n_subjects; ++subject) {
        data[subject]                        = Rcpp::as<MatrixXd>(X[subject]);
        saved_design.predictors[subject]     = Rcpp::as<MatrixXd>(design[subject]);
        saved_design.cross_products[subject] = Rcpp::as<MatrixXd>(design_cross[subject]);
        const MatrixXd& cross                = saved_design.cross_products[subject];
        if (predictors[subject].rows() != n_predictors || predictors[subject].cols() != data[subject].cols() ||
            cross.rows() != n_predictors || cross.cols() != n_predictors ||
            !predictors[subject].allFinite() || !cross.allFinite()) {
            Rcpp::stop("Subject predictors and cross-products must have matching dimensions and finite entries.");
        }
        zero_contributions[subject] = NumericMatrix(1, data[subject].cols());
    }

    // Accumulate subject summaries without saving all draws unless requested.
    const R_xlen_t covariance_size   = static_cast<R_xlen_t>(n_nodes) * n_nodes;
    const R_xlen_t coefficient_size  = static_cast<R_xlen_t>(n_nodes) * n_predictors;
    const R_xlen_t coefficient_total = coefficient_size * n_subjects;
    const R_xlen_t covariance_total  = covariance_size * n_subjects;
    NumericVector coefficient_sum(coefficient_total), positive_sum(coefficient_total), covariance_sum(covariance_total);
    coefficient_sum.attr("dim") = Rcpp::IntegerVector::create(n_nodes, n_predictors, n_subjects);
    positive_sum.attr("dim")    = coefficient_sum.attr("dim");
    covariance_sum.attr("dim")  = Rcpp::IntegerVector::create(n_nodes, n_nodes, n_subjects);

    // Save one label matrix and one co-clustering matrix per partition.
    List coefficient_membership(n_nodes), coefficient_similarity(n_nodes);
    NumericMatrix covariance_membership(n_mcmc, n_subjects), covariance_similarity(n_subjects, n_subjects);
    NumericMatrix cluster_counts(total, n_nodes + 1), parameter_traces(n_mcmc, 3);
    NumericMatrix lambda_trace(n_mcmc, n_nodes), shared_variance_sum(n_nodes, n_predictors);
    Rcpp::CharacterVector node_names(n_nodes), count_names(n_nodes + 1);
    for (int node = 0; node < n_nodes; ++node) {
        coefficient_membership[node] = NumericMatrix(n_mcmc, n_subjects);
        coefficient_similarity[node] = NumericMatrix(n_subjects, n_subjects);
        node_names[node]             = "node_" + std::to_string(node + 1);
        count_names[node]            = node_names[node];
    }
    count_names[n_nodes]             = "covariance";
    coefficient_membership.names()   = node_names;
    coefficient_similarity.names()   = node_names;
    Rcpp::colnames(cluster_counts)   = count_names;
    Rcpp::colnames(lambda_trace)     = node_names;
    Rcpp::colnames(parameter_traces) = Rcpp::CharacterVector::create("log_likelihood", "mean_node_variance", "coefficient_rms");

    // Record proposal outcomes at every iteration, including burn-in.
    Rcpp::IntegerVector iterations = Rcpp::seq(1, total);
    Rcpp::CharacterVector move_types(total, "none");
    Rcpp::LogicalVector move_accepted(total);
    NumericVector move_ratios(total, NA_REAL);
    std::vector<Rcpp::CharacterVector> row_move_types;
    std::vector<Rcpp::LogicalVector> row_move_accepted;
    std::vector<NumericVector> row_move_ratios;
    for (int node = 0; node < n_nodes; ++node) {
        row_move_types.emplace_back(total, "none");
        row_move_accepted.emplace_back(total);
        row_move_ratios.emplace_back(total, NA_REAL);
    }
    std::vector<RowRelocationTrace> relocation_traces;
    if (relocation_interval > 0) {
        relocation_traces.reserve(n_nodes);
        for (int node = 0; node < n_nodes; ++node) relocation_traces.emplace_back(total);
    }

    // Optional draw arrays add retained iterations as the fourth dimension.
    NumericVector coefficient_draws, covariance_draws;
    if (store_draws) {
        coefficient_draws             = NumericVector(coefficient_total * n_mcmc);
        covariance_draws              = NumericVector(covariance_total * n_mcmc);
        coefficient_draws.attr("dim") = Rcpp::IntegerVector::create(n_nodes, n_predictors, n_subjects, n_mcmc);
        covariance_draws.attr("dim")  = Rcpp::IntegerVector::create(n_nodes, n_nodes, n_subjects, n_mcmc);
    }

    // The benchmark can save selected entries in a matrix with entries in
    // rows and retained iterations in columns. Indices follow R's array order.
    NumericVector selected_indices;
    std::vector<R_xlen_t> selected_positions;
    NumericMatrix selected_draws;
    if (control.containsElementNamed("selected_coefficient_indices")) {
        selected_indices = Rcpp::as<NumericVector>(control["selected_coefficient_indices"]);
        if (store_draws || selected_indices.size() == 0 || selected_indices.size() > std::numeric_limits<int>::max()) {
            Rcpp::stop("Select at least one coefficient entry with full draw storage turned off.");
        }
        for (double index : selected_indices) {
            if (!std::isfinite(index) || index < 1 || index > coefficient_total || std::floor(index) != index) {
                Rcpp::stop("Selected coefficient indices must be integers within the subject coefficient array.");
            }
            selected_positions.push_back(static_cast<R_xlen_t>(index) - 1);
        }
        selected_draws = NumericMatrix(selected_positions.size(), n_mcmc);
    }

    for (int iteration = 0; iteration < total; ++iteration) {
        Rcpp::checkUserInterrupt();

        // Transfers integrate out the factors, so run them before drawing
        // fresh factors for the ordinary assignments and coefficient updates.
        const bool try_relocation = relocation_interval > 0 && (iteration + 1) % relocation_interval == 0;
        std::vector<MatrixXd> relocated_residuals;
        if (try_relocation) {
            relocated_residuals = relocate_response_rows(data, predictors, design, design_cross,
                covariance_atoms, covariance_labels, variance, concentration, coefficient_slots,
                atoms, labels, relocation_traces, iteration);
        }

        // Draw factors under the current full coefficient matrices and covariances.
        MatrixXd residual_variances(n_nodes, n_subjects);
        for (int subject = 0; subject < n_subjects; ++subject) {
            const List atom = covariance_atoms[static_cast<int>(covariance_labels[subject]) - 1];
            const MatrixXd residual = try_relocation ? std::move(relocated_residuals[subject]) :
                MatrixXd(data[subject] - row_subject_coefficient(atoms, labels, subject, n_predictors) * predictors[subject]);
            MatrixXd noise(n_factors, data[subject].cols());
            for (Eigen::Index index = 0; index < noise.size(); ++index) noise.data()[index] = R::rnorm(0.0, 1.0);
            const MatrixXd factors          = pdpmvar_eigen_factors(residual, atom, noise);
            adjusted[subject]               = data[subject] - Rcpp::as<MatrixXd>(atom["working_loadings"]) * factors;
            residual_variances.col(subject) = Rcpp::as<VectorXd>(atom["variance"]);
        }

        // Given the factors, each response has its own slice assignments and
        // Gaussian regressions. Each regression uses all lags at once.
        const bool try_move = coefficient_interval > 0 && (iteration + 1) % coefficient_interval == 0;
        for (int node = 0; node < n_nodes; ++node) {
            const NumericVector old_labels = labels[node];
            const List old_atoms           = atoms[node];
            NumericMatrix row_variance     = variance[node];
            const List slice               = pdpmvar_eigen_slice_weights(old_labels, concentration[node]);
            const NumericVector weights    = slice["weights"];
            List row_atoms(weights.size()), response(n_subjects), row_covariances(covariance_atoms.size());
            for (int group = 0; group < row_atoms.size(); ++group) {
                if (group < old_atoms.size()) row_atoms[group] = old_atoms[group];
                else row_atoms[group] = sampler_new_coefficient(row_variance);
            }
            for (int subject = 0; subject < n_subjects; ++subject) {
                response[subject] = MatrixXd(adjusted[subject].row(node));
            }
            for (int group = 0; group < covariance_atoms.size(); ++group) {
                const List atom               = covariance_atoms[group];
                const NumericVector variances = atom["variance"];
                NumericMatrix covariance(1, 1);
                covariance(0, 0)       = variances[node];
                row_covariances[group] = List::create(Rcpp::Named("covariance") = covariance);
            }

            const List assignment = assign_coefficients_with_design(response, design, row_atoms,
                covariance_labels, row_covariances, weights, slice["slices"], row_assignment_uniforms(n_subjects),
                &saved_design);
            NumericVector row_labels    = assignment["labels"];
            const MatrixXd residual_row = residual_variances.row(node);

            if (try_move) {
                const List move = move_coefficient_rows_with_design(response, design, design_cross, residual_row,
                    row_labels, row_atoms, row_variance, concentration[node], coefficient_slots, &saved_design);
                row_labels                         = Rcpp::as<NumericVector>(move["labels"]);
                row_atoms                          = Rcpp::as<List>(move["atoms"]);
                const List record                  = move["record"];
                row_move_types[node][iteration]    = Rcpp::as<std::string>(record["type"]);
                row_move_accepted[node][iteration] = Rcpp::as<bool>(record["accepted"]);
                row_move_ratios[node][iteration]   = Rcpp::as<double>(record["log_ratio"]);
            }
            row_atoms = draw_coefficients_with_design(response, design, design_cross, zero_contributions,
                row_labels, row_atoms, Rcpp::as<MatrixXd>(row_variance), residual_row, &saved_design);

            // Only occupied rows inform shrinkage. Refresh the others under
            // the new variances before they can receive a subject again.
            const auto counts               = sampler_counts(row_labels, row_atoms.size());
            cluster_counts(iteration, node) = std::count_if(counts.begin(), counts.end(), [](int n) { return n > 0; });
            const List shrinkage            = pdpmvar_eigen_shrinkage(row_labels, row_atoms, row_variance, lambda_squared[node]);
            row_variance                    = Rcpp::as<NumericMatrix>(shrinkage["shared_variance"]);
            lambda_squared[node]            = Rcpp::as<double>(shrinkage["lambda_squared"]);
            for (int group = 0; group < row_atoms.size(); ++group) {
                if (counts[group] == 0) row_atoms[group] = sampler_new_coefficient(row_variance);
            }
            labels[node]   = row_labels;
            atoms[node]    = row_atoms;
            variance[node] = row_variance;
        }

        // Subtract all lag contributions to form the residuals for covariance updates.
        List full_residuals(n_subjects);
        std::vector<MatrixXd> coefficients(n_subjects);
        for (int subject = 0; subject < n_subjects; ++subject) {
            coefficients[subject]   = row_subject_coefficient(atoms, labels, subject, n_predictors);
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
            weights, slice["slices"], row_assignment_uniforms(n_subjects));
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
        cluster_counts(iteration, n_nodes) = std::count_if(counts.begin(), counts.end(), [](int n) { return n > 0; });

        if (verbose && ((iteration + 1) % 100 == 0 || iteration + 1 == total)) {
            Rcpp::Rcout << "Iteration " << iteration + 1 << " of " << total << '\n';
        }
        if (iteration < n_burnin) continue;

        // Save the retained labels and count how often each pair shares a group.
        const int retained = iteration - n_burnin;
        for (int node = 0; node < n_nodes; ++node) {
            const NumericVector row_labels   = labels[node];
            const NumericMatrix row_variance = variance[node];
            NumericMatrix membership          = coefficient_membership[node];
            NumericMatrix similarity          = coefficient_similarity[node];
            lambda_trace(retained, node)     = std::sqrt(lambda_squared[node]);
            for (R_xlen_t index = 0; index < n_predictors; ++index) {
                shared_variance_sum(node, index) += row_variance[index];
            }
            for (int subject = 0; subject < n_subjects; ++subject) {
                membership(retained, subject) = row_labels[subject];
                for (int other = 0; other < n_subjects; ++other) {
                    similarity(subject, other) += row_labels[subject] == row_labels[other];
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
        for (int entry = 0; entry < selected_draws.nrow(); ++entry) {
            const R_xlen_t position       = selected_positions[entry];
            const MatrixXd& coefficient   = coefficients[position / coefficient_size];
            selected_draws(entry, retained) = coefficient.data()[position % coefficient_size];
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
    for (int node = 0; node < n_nodes; ++node) {
        NumericMatrix similarity = coefficient_similarity[node];
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
    if (!selected_positions.empty()) {
        draws = List::create(Rcpp::Named("coefficients") = selected_draws,
                             Rcpp::Named("coefficient_indices") = selected_indices);
    }
    List coefficient_move_traces(n_nodes);
    coefficient_move_traces.names() = node_names;
    for (int node = 0; node < n_nodes; ++node) {
        coefficient_move_traces[node] = sampler_move_trace(iterations, row_move_types[node],
            row_move_accepted[node], row_move_ratios[node]);
    }
    List result = List::create(Rcpp::Named("coefficient_mean") = coefficient_sum,
        Rcpp::Named("covariance_mean") = covariance_sum, Rcpp::Named("coefficient_sign_probability") = positive_sum,
        Rcpp::Named("coefficient_sign_tail") = sign_tail, Rcpp::Named("coefficient_labels") = coefficient_membership,
        Rcpp::Named("covariance_labels") = covariance_membership, Rcpp::Named("coefficient_coclustering") = coefficient_similarity,
        Rcpp::Named("covariance_coclustering") = covariance_similarity, Rcpp::Named("cluster_counts") = cluster_counts,
        Rcpp::Named("parameter_traces") = parameter_traces, Rcpp::Named("lambda") = lambda_trace,
        Rcpp::Named("shared_variance_mean") = shared_variance_sum, Rcpp::Named("final_state") = final_state,
        Rcpp::Named("split_merge") = sampler_move_trace(iterations, move_types, move_accepted, move_ratios),
        Rcpp::Named("coefficient_split_merge") = coefficient_move_traces, Rcpp::Named("draws") = draws);
    if (relocation_interval > 0) {
        List transfers(n_nodes);
        transfers.names() = node_names;
        for (int node = 0; node < n_nodes; ++node) {
            transfers[node] = relocation_traces[node].as_data_frame(iterations);
        }
        result["coefficient_relocation"] = transfers;
    }
    return result;
}
