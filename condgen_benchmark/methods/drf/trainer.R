suppressPackageStartupMessages(library(drf))
suppressPackageStartupMessages(library(reticulate))

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list()
  i <- 1L
  while (i <= length(args)) {
    key <- sub("^--", "", args[[i]])
    key <- gsub("-", "_", key)
    if (i == length(args)) stop("Missing value for argument: ", args[[i]])
    out[[key]] <- args[[i + 1L]]
    i <- i + 2L
  }
  out
}

read_npz_array <- function(np, path, key) {
  z <- np$load(normalizePath(path, winslash = "/", mustWork = TRUE), allow_pickle = FALSE)
  arr <- py_to_r(z$`__getitem__`(key))
  if (is.null(dim(arr))) arr <- matrix(arr, ncol = 1L)
  if (length(dim(arr)) == 1L) arr <- matrix(arr, ncol = 1L)
  storage.mode(arr) <- "double"
  arr
}

save_train_npz <- function(np, path, X_train, Y_train, X_valid, Y_valid) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  invisible(np$savez(
    normalizePath(path, winslash = "/", mustWork = FALSE),
    X_train = np$array(X_train, dtype = "float64"),
    Y_train = np$array(Y_train, dtype = "float64"),
    X_valid = np$array(X_valid, dtype = "float64"),
    Y_valid = np$array(Y_valid, dtype = "float64")
  ))
}

args <- parse_args()

if (!is.null(args$python_bin)) {
  reticulate::use_python(normalizePath(args$python_bin, winslash = "/", mustWork = TRUE), required = TRUE)
}

required <- c("train_npz", "valid_npz", "model_out", "train_data_out", "seed",
              "num_trees", "min_node_size", "splitting_rule")
missing <- setdiff(required, names(args))
if (length(missing) > 0L) stop("Missing arguments: ", paste(missing, collapse = ", "))

set.seed(as.integer(args$seed))
np <- import("numpy", convert = FALSE)

X_train <- read_npz_array(np, args$train_npz, "X")
Y_train <- read_npz_array(np, args$train_npz, "Y")
X_valid <- read_npz_array(np, args$valid_npz, "X")
Y_valid <- read_npz_array(np, args$valid_npz, "Y")

cat("[DRF/R] Training data shape:\n")
cat("  X_train:", paste(dim(X_train), collapse = " x "), "\n")
cat("  Y_train:", paste(dim(Y_train), collapse = " x "), "\n")
cat("[DRF/R] Validation data shape:\n")
cat("  X_valid:", paste(dim(X_valid), collapse = " x "), "\n")
cat("  Y_valid:", paste(dim(Y_valid), collapse = " x "), "\n")

fit <- drf(
  X = X_train,
  Y = Y_train,
  num.trees = as.integer(args$num_trees),
  min.node.size = as.integer(args$min_node_size),
  splitting.rule = args$splitting_rule
)

dir.create(dirname(args$model_out), recursive = TRUE, showWarnings = FALSE)
saveRDS(fit, file = normalizePath(args$model_out, winslash = "/", mustWork = FALSE))
save_train_npz(np, args$train_data_out, X_train, Y_train, X_valid, Y_valid)

cat("\n[DRF/R] Model summary:\n")
print(fit)

# Do not compute variable importance by default.
# In DRF this can be much slower than the model fit itself, especially on Windows.
cat("\n[DRF/R] Variable importance: skipped.\n")
