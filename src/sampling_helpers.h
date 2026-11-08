#ifndef PDPMVAR_SAMPLING_HELPERS_H
#define PDPMVAR_SAMPLING_HELPERS_H
#include <RcppEigen.h>
#include <vector>

// Shared draws, DP bookkeeping and factor-covariance routines used by model samplers.
double draw_shared_gig(double, double, double);
std::vector<int> count_stick_labels(const Rcpp::NumericVector&, int minimum_size = 1);
Rcpp::List pdpmvar_eigen_slice_weights(const Rcpp::NumericVector&, double);
Rcpp::NumericVector pdpmvar_eigen_prefix_weights(const Rcpp::NumericVector&, double, int);
Rcpp::List pdpmvar_eigen_integral(const Rcpp::NumericMatrix&, const Rcpp::NumericVector&, double);
Eigen::MatrixXd pdpmvar_eigen_factors(const Eigen::MatrixXd&, const Rcpp::List&, const Eigen::MatrixXd&);
Rcpp::List pdpmvar_eigen_covariance_update(const Eigen::MatrixXd&, const Rcpp::List&, double, double);
Rcpp::List pdpmvar_eigen_assign_covariances(const Rcpp::List&, const Rcpp::List&, const Rcpp::NumericVector&,
    const Rcpp::NumericVector&, const Rcpp::NumericVector&);
double pdpmvar_eigen_total_likelihood(const Rcpp::List&, const Rcpp::NumericVector&, const Rcpp::List&);
Rcpp::List pdpmvar_eigen_covariance_move(const Rcpp::List&, const Rcpp::NumericVector&,
    const Rcpp::List&, double, int, double, double, int);

// Subject predictors checked once by the row-specific sampler. For P lagged
// predictors and T_i times, each design is P x T_i and each cross-product is P x P.
struct CoefficientDesign {
    std::vector<Eigen::MatrixXd> predictors, cross_products;
};

// Coefficient mixture calculations shared across models.
Rcpp::List pdpmvar_eigen_coefficient_update(const Rcpp::List&, const Rcpp::List&,
    const Rcpp::List&, const Rcpp::List&, const Rcpp::NumericVector&, const Rcpp::List&,
    const Eigen::MatrixXd&, const Eigen::MatrixXd&);
Rcpp::List draw_coefficients_with_design(const Rcpp::List&, const Rcpp::List&,
    const Rcpp::List&, const Rcpp::List&, const Rcpp::NumericVector&, const Rcpp::List&,
    const Eigen::MatrixXd&, const Eigen::MatrixXd&, const CoefficientDesign*);
Rcpp::List pdpmvar_eigen_shrinkage(const Rcpp::NumericVector&, const Rcpp::List&,
    const Rcpp::NumericMatrix&, double);
Rcpp::List pdpmvar_eigen_assign_coefficients(const Rcpp::List&, const Rcpp::List&,
    const Rcpp::List&, const Rcpp::NumericVector&, const Rcpp::List&,
    const Rcpp::NumericVector&, const Rcpp::NumericVector&, const Rcpp::NumericVector&);
Rcpp::List assign_coefficients_with_design(const Rcpp::List&, const Rcpp::List&,
    const Rcpp::List&, const Rcpp::NumericVector&, const Rcpp::List&,
    const Rcpp::NumericVector&, const Rcpp::NumericVector&, const Rcpp::NumericVector&, const CoefficientDesign*);

Rcpp::List pdpmvar_eigen_row_move(const Rcpp::List&, const Rcpp::List&,
    const Rcpp::List&, const Eigen::MatrixXd&, const Rcpp::NumericVector&,
    const Rcpp::List&, const Rcpp::NumericMatrix&, double, int);
Rcpp::List move_coefficient_rows_with_design(const Rcpp::List&, const Rcpp::List&,
    const Rcpp::List&, const Eigen::MatrixXd&, const Rcpp::NumericVector&,
    const Rcpp::List&, const Rcpp::NumericMatrix&, double, int, const CoefficientDesign*);

Rcpp::List pdpmvar_eigen_coefficient_move(const Rcpp::List&, const Rcpp::List&,
    const Rcpp::List&, const Rcpp::List&, const Rcpp::NumericVector&, const Rcpp::List&,
    const Rcpp::NumericMatrix&, double, int);
Rcpp::List lgpdpmvar_eigen_relocation_move(const Rcpp::List&, const Rcpp::List&, const Rcpp::List&,
    const Rcpp::List&, const Rcpp::NumericVector&, const Rcpp::NumericVector&, const Rcpp::List&,
    const Rcpp::NumericMatrix&, double, int, int, int, Rcpp::Nullable<Rcpp::NumericVector>, bool, std::string);

// Shared covariance-factorization and slice-category utilities.
Eigen::MatrixXd likelihood_covariance_lower(const Eigen::MatrixXd&, int);
double residual_log_likelihood(const Eigen::MatrixXd&, const Eigen::MatrixXd&, double);
void cache_likelihood_covariance(const Rcpp::List&, int, int, std::vector<Eigen::MatrixXd>&,
    std::vector<double>&, std::vector<bool>&);
int sample_slice_candidate(const std::vector<int>&, const std::vector<double>&, double, Rcpp::NumericMatrix&, int);

// Shared atom initialization, fixed-stick split/merge indices and trace format.
Rcpp::List sampler_new_coefficient(const Rcpp::NumericMatrix&);
Rcpp::List sampler_new_covariance(int, int, double, double);
std::vector<int> sampler_counts(const Rcpp::NumericVector&, int);
Rcpp::DataFrame sampler_move_trace(const Rcpp::IntegerVector&, const Rcpp::CharacterVector&,
    const Rcpp::LogicalVector&, const Rcpp::NumericVector&);
Rcpp::IntegerVector move_anchors(int);
std::vector<int> move_empty_slots(const Rcpp::NumericVector&, int);
void move_validate_ratio(const Rcpp::NumericVector&, const Rcpp::NumericVector&,
    const Rcpp::NumericVector&, const Rcpp::IntegerVector&, int);
Rcpp::IntegerVector move_subjects(const Rcpp::NumericVector&, int, int g = -1);
Rcpp::List move_result(const Rcpp::NumericVector&, const Rcpp::List&, const char* type = "skip",
    bool accepted = false, double ratio = NA_REAL);
#endif
