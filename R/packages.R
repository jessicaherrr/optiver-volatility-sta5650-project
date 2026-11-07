# Install any R packages this project needs that are missing.
# Usage, from the project root: source("R/packages.R")
# When the project's renv is active, packages go into its project library.

if (identical(unname(getOption("repos")["CRAN"]), "@CRAN@") ||
    is.null(getOption("repos"))) {
  options(repos = c(CRAN = "https://cloud.r-project.org"))
}

required_packages <- c(
  "arrow", "tidyverse", "psych", "GPArotation", "car", "MASS", "MVN", "heplots",
  "lmtest", "sandwich", "glmnet", "pls", "forecast", "corrplot", "cluster",
  "factoextra", "rmarkdown", "mcmcse", "tseries"
)

is_missing <- !vapply(
  required_packages,
  function(pkg) nzchar(system.file(package = pkg)),
  logical(1)
)
missing_packages <- required_packages[is_missing]

if (length(missing_packages) > 0) {
  # Call the utils version directly, so that the binary-only setting in .Rprofile
  # applies. renv's replacement for install.packages() can still choose source.
  utils::install.packages(missing_packages)
} else {
  message("All required R packages are already installed.")
}
