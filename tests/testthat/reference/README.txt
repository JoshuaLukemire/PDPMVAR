These R calculations are maintained validation references, not production sampler options.

helper-reference-functions.R loads them into an isolated .pdpmvar_test_reference
environment. Tests use that environment explicitly; production never sources it.
The old backend controls remain here so existing kernel and full-driver parity
checks can exercise the independent R calculations and native kernels with common
random numbers. Production helpers expose no backend choice.

Optional benchmark parity scripts can explicitly load these files into their own
reference environment. Keep the published model/priors consistent when maintaining
both native code and these references. Never source archived experimental samplers.
