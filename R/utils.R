# Owner: Shared by A, B, and C
# Purpose: Helpers used by all of the R analysis scripts.
# Input: data/processed/features.parquet (read by load_features)
# Output: None. Source this file to define rmspe() and load_features().

# Root mean squared percentage error, the competition metric.
# Matches rmspe() in src/metrics.py.
rmspe <- function(y_true, y_pred) {
  if (length(y_true) != length(y_pred)) {
    stop("y_true and y_pred have different lengths")
  }
  if (any(y_true == 0)) {
    stop("y_true contains a zero, so RMSPE is undefined")
  }
  sqrt(mean(((y_true - y_pred) / y_true)^2))
}

# Read the feature table that Python writes. The default path assumes the working
# directory is the project root, which is automatic when you open the .Rproj.
load_features <- function(path = "data/processed/features.parquet") {
  if (!file.exists(path)) {
    stop("Cannot find ", path, ". See 'Get the data' in README.md.")
  }
  arrow::read_parquet(path)
}
