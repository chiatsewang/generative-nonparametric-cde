# R packages needed by CTMTF and DRF (condgen_benchmark/methods/{ctmtf,drf}/*.R).
# Usage: Rscript R/install.R   (or: make install-r)
# Requires R >= 4.5. mlt 1.9-0 needs R >= 4.6, so on older R we install mlt 1.8-4
# from the CRAN archive. mlt depends on nloptr, which can fail to compile from
# source on some systems.
packages <- c(
  "reticulate", "mlt", "tram", "trtf", "basefun", "partykit", "drf"
)
cran <- "https://cloud.r-project.org"
mlt_archive <- paste0(cran, "/src/contrib/Archive/mlt/mlt_1.8-4.tar.gz")
# CRAN dependencies of mlt 1.8-4 (the archive install does not fetch them).
mlt_deps <- c(
  "basefun", "variables", "BB", "alabama", "quadprog", "coneproj", "sandwich",
  "numDeriv", "survival", "Matrix", "nloptr", "mvtnorm", "icenReg"
)

install_r_packages <- function(packages, lib = NULL) {
  have <- function() rownames(installed.packages(lib.loc = lib))
  missing <- setdiff(packages, have())
  if (length(missing) == 0L) {
    message("All R packages are already installed: ", paste(packages, collapse = ", "))
    return(invisible())
  }
  message("Installing missing R packages: ", paste(missing, collapse = ", "))
  if ("mlt" %in% missing && getRversion() < "4.6") {
    install.packages(setdiff(mlt_deps, have()), lib = lib, repos = cran)
    install.packages(mlt_archive, lib = lib, repos = NULL, type = "source")
    missing <- setdiff(missing, "mlt")
  }
  if (length(missing) > 0L) install.packages(missing, lib = lib, repos = cran)
  still_missing <- setdiff(packages, have())
  if (length(still_missing) > 0L) {
    stop("Failed to install: ", paste(still_missing, collapse = ", "))
  }
}

# Run only under Rscript, so tests can source() this file and call the function.
if (sys.nframe() == 0L) install_r_packages(packages)
