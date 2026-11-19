# R packages needed by CTMTF and DRF (condgen_benchmark/methods/{ctmtf,drf}/*.R).
# Usage: Rscript R/install.R   (or: make install-r)
# Requires R >= 4.5. mlt 1.9-0 needs R >= 4.6, so on R 4.5 install mlt 1.8-4.
# mlt depends on nloptr, which can fail to compile from source on some systems.
packages <- c(
  "reticulate", "mlt", "tram", "trtf", "basefun", "partykit", "drf"
)

missing <- setdiff(packages, rownames(installed.packages()))
if (length(missing) == 0L) {
  message("All R packages are already installed: ", paste(packages, collapse = ", "))
} else {
  message("Installing missing R packages: ", paste(missing, collapse = ", "))
  install.packages(missing, repos = "https://cloud.r-project.org")
  still_missing <- setdiff(packages, rownames(installed.packages()))
  if (length(still_missing) > 0L) {
    stop("Failed to install: ", paste(still_missing, collapse = ", "))
  }
}
