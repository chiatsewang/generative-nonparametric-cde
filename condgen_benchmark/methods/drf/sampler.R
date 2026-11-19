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

predict_weights <- function(fit, X_test) {
  if (exists("predict_drf", where = asNamespace("drf"), inherits = FALSE)) {
    pred <- drf::predict_drf(fit, X_test)
  } else {
    pred <- predict(fit, newdata = X_test)
  }

  if (is.list(pred) && !is.null(pred$weights)) {
    weights <- pred$weights
  } else if (is.list(pred)) {
    weights <- pred[[1L]]
  } else {
    stop("Could not extract DRF weights from prediction output.")
  }

  weights <- as.matrix(weights)
  storage.mode(weights) <- "double"
  weights
}

sample_from_weights <- function(weights, Y_train, num_samples, seed) {
  set.seed(seed)

  if (length(dim(Y_train)) == 1L) Y_train <- matrix(Y_train, ncol = 1L)

  n_train <- nrow(Y_train)
  y_dim <- ncol(Y_train)

  if (ncol(weights) != n_train && nrow(weights) == n_train) {
    weights <- t(weights)
  }
  if (ncol(weights) != n_train) {
    stop("Weight shape ", paste(dim(weights), collapse = " x "),
         " is incompatible with Y_train shape ", paste(dim(Y_train), collapse = " x "), ".")
  }

  row_sums <- rowSums(weights)
  if (any(row_sums <= 0)) stop("At least one DRF weight row has non-positive total mass.")
  weights <- weights / row_sums

  n_test <- nrow(weights)
  Y_hat <- array(NA_real_, dim = c(n_test, num_samples, y_dim))

  for (i in seq_len(n_test)) {
    idx <- sample.int(n_train, size = num_samples, replace = TRUE, prob = weights[i, ])
    if (y_dim == 1L) {
      Y_hat[i, , 1L] <- Y_train[idx, 1L]
    } else {
      Y_hat[i, , ] <- Y_train[idx, , drop = FALSE]
    }
  }

  list(Y_hat = Y_hat, weights = weights)
}

args <- parse_args()

if (!is.null(args$python_bin)) {
  reticulate::use_python(normalizePath(args$python_bin, winslash = "/", mustWork = TRUE), required = TRUE)
}

required <- c("model", "train_npz", "test_npz", "out", "seed", "num_samples")
missing <- setdiff(required, names(args))
if (length(missing) > 0L) stop("Missing arguments: ", paste(missing, collapse = ", "))

np <- import("numpy", convert = FALSE)

fit <- readRDS(normalizePath(args$model, winslash = "/", mustWork = TRUE))
X_test <- read_npz_array(np, args$test_npz, "X")
Y_true <- read_npz_array(np, args$test_npz, "Y")
Y_train <- read_npz_array(np, args$train_npz, "Y")

cat("[DRF/R] Test data shape:\n")
cat("  X_test:", paste(dim(X_test), collapse = " x "), "\n")
cat("  Y_true:", paste(dim(Y_true), collapse = " x "), "\n")
cat("[DRF/R] Training response shape:\n")
cat("  Y_train:", paste(dim(Y_train), collapse = " x "), "\n")

weights <- predict_weights(fit, X_test)
cat("\n[DRF/R] Raw weights shape:", paste(dim(weights), collapse = " x "), "\n")

sampled <- sample_from_weights(
  weights = weights,
  Y_train = Y_train,
  num_samples = as.integer(args$num_samples),
  seed = as.integer(args$seed)
)

cat("[DRF/R] Normalized weights shape:", paste(dim(sampled$weights), collapse = " x "), "\n")
cat("[DRF/R] First five row sums:", paste(head(rowSums(sampled$weights), 5L), collapse = ", "), "\n")

dir.create(dirname(args$out), recursive = TRUE, showWarnings = FALSE)
invisible(np$savez(
  normalizePath(args$out, winslash = "/", mustWork = FALSE),
  Y_hat = np$array(sampled$Y_hat, dtype = "float64"),
  X = np$array(X_test, dtype = "float64"),
  Y_true = np$array(Y_true, dtype = "float64")
))

cat("\n[DRF/R] Saved samples to:", args$out, "\n")
cat("[DRF/R] Output shapes:\n")
cat("  Y_hat:", paste(dim(sampled$Y_hat), collapse = " x "), "(N_test x num_samples x y_dim)\n")
cat("  X:", paste(dim(X_test), collapse = " x "), "\n")
cat("  Y_true:", paste(dim(Y_true), collapse = " x "), "\n")
