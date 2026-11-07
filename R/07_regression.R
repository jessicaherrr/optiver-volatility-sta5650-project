# Owner: C
# Purpose: Test H3 and H4: nested lm models with anova() partial F-tests, diagnostics,
#   VIF, Durbin-Watson, clustered SEs, ridge/lasso/PCR, and a Diebold-Mariano test.
#   Compare out-of-sample RMSPE under random K-fold and a time-ordered split.
# Input: data/processed/features.parquet, data/processed/common_factors.parquet,
#   data/processed/time_order.parquet
# Output: figures/regression_*.png, report/regression_results.md

source("R/utils.R")
