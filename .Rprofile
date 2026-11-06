# On macOS and Windows, install prebuilt packages only. CRAN's source version of a
# package can be newer than its binary, and renv would then try to compile the
# locked version from source. Compiling arrow this way can fail.
if (.Platform$pkgType != "source") {
  options(pkgType = "binary", install.packages.check.source = "no")
}

source("renv/activate.R")
