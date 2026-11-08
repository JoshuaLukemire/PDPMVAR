# Load the independent R calculations into an isolated environment for parity tests.
# Tests call these references explicitly; the package namespace contains only
# native wrappers and input-preparation helpers.
.pdpmvar_test_reference <- new.env(parent = asNamespace("PDPMVAR"))
sys.source(testthat::test_path("reference", "sampling_helpers.R"), .pdpmvar_test_reference)
sys.source(testthat::test_path("reference", "pdpmvar.R"), .pdpmvar_test_reference)
