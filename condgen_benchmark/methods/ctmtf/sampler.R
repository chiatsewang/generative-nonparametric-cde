# condgen_benchmark/methods/ctmtf/sampler.R

suppressPackageStartupMessages({
  library(reticulate)
  library(mlt)
  library(tram)
  library(trtf)
  library(basefun)
  library(partykit)
})

parse_args <- function() {
  x <- commandArgs(trailingOnly = TRUE)
  if (length(x) %% 2 != 0) {
    stop("Arguments must be supplied as --key value pairs")
  }

  out <- list()
  i <- 1
  while (i <= length(x)) {
    key <- sub("^--", "", x[i])
    out[[key]] <- x[i + 1]
    i <- i + 2
  }
  out
}

load_npz_xy <- function(npz_path) {
  py <- Sys.which("python")
  if (nzchar(py)) {
    reticulate::use_python(py, required = TRUE)
  }

  np <- import("numpy")
  dat <- np$load(npz_path, allow_pickle = TRUE)

  X <- py_to_r(dat[["X"]])
  Y <- py_to_r(dat[["Y"]])

  X <- as.matrix(X)
  Y <- as.matrix(Y)

  if (is.null(dim(X))) {
    X <- matrix(X, ncol = 1)
  }
  if (is.null(dim(Y))) {
    Y <- matrix(Y, ncol = 1)
  }

  list(X = X, Y = Y)
}

extract_quantile_matrix <- function(pred_obj, n_test, num_samples) {
  # predict.traforest() usually returns one object per newdata row.
  if (is.list(pred_obj)) {
    if (length(pred_obj) != n_test) {
      stop(
        paste0(
          "Unexpected list length returned by predict.traforest: got ",
          length(pred_obj), ", expected ", n_test
        )
      )
    }

    out <- matrix(NA_real_, nrow = n_test, ncol = num_samples)
    for (i in seq_len(n_test)) {
      q_i <- as.numeric(pred_obj[[i]])
      if (length(q_i) != num_samples) {
        stop(
          paste0(
            "Unexpected number of predicted quantiles for test point ", i,
            ": got ", length(q_i), ", expected ", num_samples
          )
        )
      }
      out[i, ] <- q_i
    }
    return(out)
  }

  obj <- as.matrix(pred_obj)

  if (identical(dim(obj), c(n_test, num_samples))) {
    return(obj)
  }
  if (identical(dim(obj), c(num_samples, n_test))) {
    return(t(obj))
  }

  vals <- as.numeric(obj)
  if (length(vals) == n_test * num_samples) {
    return(matrix(vals, nrow = n_test, ncol = num_samples, byrow = TRUE))
  }

  stop(
    paste0(
      "Unexpected output shape returned by predict.traforest: ",
      paste(dim(obj), collapse = " x ")
    )
  )
}

args <- parse_args()
data_name <- args$data_name
seed_now <- as.integer(args$seed)
num_samples <- as.integer(args$num_samples)

if (!is.finite(num_samples) || num_samples <= 0) {
  stop("num_samples must be a positive integer")
}

project_root <- getwd()
data_dir <- file.path(
  project_root, "datasets", "synthetic", data_name, paste0("seed_", seed_now)
)
workspace_dir <- file.path(
  project_root, "workspaces", "ctmtf", data_name, paste0("seed_", seed_now)
)

test_npz <- file.path(data_dir, "test.npz")
model_path <- file.path(workspace_dir, "model.rds")
out_path <- file.path(workspace_dir, "sampled.npz")
sample_log_path <- file.path(workspace_dir, "sample_log.csv")

cat("\n=============================\n")
cat("CTMTF sample start\n")
cat("data_name:", data_name, "\n")
cat("seed:", seed_now, "\n")
cat("num_samples:", num_samples, "\n")
cat("test_npz:", test_npz, "\n")
cat("model_path:", model_path, "\n")
cat("out_path:", out_path, "\n")
cat("=============================\n\n")

test <- load_npz_xy(test_npz)
X_test <- test$X
Y_true <- test$Y

if (ncol(Y_true) != 1) {
  stop("CTMTF is univariate only, but Y has dimension != 1")
}

model_obj <- readRDS(model_path)
fit <- model_obj$fit

n_test <- nrow(X_test)
x_dim <- ncol(X_test)
y_dim <- ncol(Y_true)

if (!is.null(model_obj$x_names) && length(model_obj$x_names) != x_dim) {
  stop("The test X dimension does not match the fitted CTMTF model")
}

x_names <- if (!is.null(model_obj$x_names)) {
  model_obj$x_names
} else {
  paste0("x", seq_len(x_dim))
}

X_test_df <- as.data.frame(X_test)
names(X_test_df) <- x_names

cat("Test X dim:", paste(dim(X_test), collapse = " x "), "\n")
cat("Test Y dim:", paste(dim(Y_true), collapse = " x "), "\n")

sampling_time <- system.time({
  set.seed(100000 + seed_now)

  # Use one common set of random probability levels for all test points.
  # Only the marginal conditional samples at each test point are needed, so
  # independence across different test points is unnecessary. This allows
  # all test points to be processed in a single predict() call.
  probs <- runif(num_samples, min = 1e-6, max = 1 - 1e-6)

  pred_all <- predict(
    fit,
    newdata = X_test_df,
    prob = probs,
    type = "quantile"
  )

  Y_mat <- extract_quantile_matrix(pred_all, n_test, num_samples)
})[["elapsed"]]

Y_hat <- array(Y_mat, dim = c(n_test, num_samples, 1))

np <- import("numpy")
np$savez(
  out_path,
  Y_hat = Y_hat,
  X = X_test,
  Y_true = Y_true
)

sample_log <- data.frame(
  method = "ctmtf",
  backend = "TransformationForest",
  data_name = data_name,
  seed = seed_now,
  test_n = n_test,
  x_dim = x_dim,
  y_dim = y_dim,
  num_samples = num_samples,
  sampling_time = as.numeric(sampling_time),
  out_path = out_path
)

write.csv(sample_log, sample_log_path, row.names = FALSE)

cat("\nCTMTF sample done\n")
cat("sampling_time:", sampling_time, "sec\n")
cat("saved samples:", out_path, "\n")
cat("Y_hat shape:", paste(dim(Y_hat), collapse = " x "), "\n")
