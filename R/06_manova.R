# Owner: B
# Purpose: Test H2 on a table with one row per stock: Box's M, MANOVA with Pillai's
#   trace, Bonferroni intervals, and profile analysis of the stock groups.
# Input: data/processed/features.parquet, data/processed/stock_clusters.parquet
# Output: figures/profile_*.png, report/manova_results.md

source("R/utils.R")
