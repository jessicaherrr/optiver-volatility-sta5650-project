# Owner: C
# Purpose: Measure volatility persistence with ACF plots, ordering windows by
#   time_rank from time_order.parquet.
# Input: data/processed/features.parquet, data/processed/time_order.parquet
# Output: figures/acf_*.png

source("R/utils.R")
