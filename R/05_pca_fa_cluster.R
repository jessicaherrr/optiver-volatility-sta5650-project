# Owner: B
# Purpose: Test H1 on the 3,830 x 112 log volatility matrix: PCA on the correlation
#   matrix, scree plot, factor analysis with varimax, then hierarchical clustering
#   and k-means of stocks on their loadings.
# Input: data/processed/features.parquet
# Output: data/processed/common_factors.parquet (for R/07_regression.R),
#   data/processed/stock_clusters.parquet (for R/06_manova.R),
#   figures/pca_*.png, figures/fa_*.png, figures/cluster_*.png

source("R/utils.R")
