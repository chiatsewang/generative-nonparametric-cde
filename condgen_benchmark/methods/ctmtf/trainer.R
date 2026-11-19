# condgen_benchmark/methods/ctmtf/trainer.R

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

fit_ctmtf <- function(X_train, Y_train, seed_now) {
  if (ncol(Y_train) != 1) {
    stop("CTMTF requires y_dim = 1")
  }

  x_names <- paste0("x", seq_len(ncol(X_train)))
  y_name <- "y"

  train_data <- data.frame(y = as.numeric(Y_train[, 1]), X_train)
  names(train_data) <- c(y_name, x_names)

  y_range <- range(train_data[[y_name]], na.rm = TRUE)
  if (any(!is.finite(y_range)) || diff(y_range) <= 0) {
    stop("The response must contain at least two distinct finite values")
  }

  y_var <- numeric_var(y_name, support = y_range)
  b_y <- Bernstein_basis(y_var, order = 5, ui = "increasing")
  base_model <- ctm(b_y)

  formula_y <- as.formula(
    paste(y_name, "~", paste(x_names, collapse = " + "))
  )

  set.seed(seed_now)

  cat("Fit unconditional base transformation model\n")
  fit_mlt <- mlt(base_model, data = train_data)

  cat("Fit transformation forest\n")
  fit_forest <- traforest(
    fit_mlt,
    formula = formula_y,
    data = train_data,
    ntree = 100,
    update = FALSE
  )

  list(
    fit = fit_forest,
    x_names = x_names,
    y_name = y_name,
    support = y_range,
    basis_order = 5L,
    ntree = 100L
  )
}

args <- parse_args()

if (!is.null(args$python_bin)) {
  reticulate::use_python(args$python_bin, required = TRUE)
}
data_name <- args$data_name
seed_now <- as.integer(args$seed)
project_root <- getwd()

data_dir <- file.path(
  project_root, "datasets", "synthetic", data_name, paste0("seed_", seed_now)
)
workspace_dir <- file.path(
  project_root, "workspaces", "ctmtf", data_name, paste0("seed_", seed_now)
)

train_npz <- file.path(data_dir, "train.npz")
model_path <- file.path(workspace_dir, "model.rds")
train_log_path <- file.path(workspace_dir, "train_log.csv")

dir.create(workspace_dir, recursive = TRUE, showWarnings = FALSE)

cat("\n=============================\n")
cat("CTMTF train start\n")
cat("data_name:", data_name, "\n")
cat("seed:", seed_now, "\n")
cat("train_npz:", train_npz, "\n")
cat("workspace_dir:", workspace_dir, "\n")
cat("=============================\n\n")

train <- load_npz_xy(train_npz)
X_train <- train$X
Y_train <- train$Y

cat("Train X dim:", paste(dim(X_train), collapse = " x "), "\n")
cat("Train Y dim:", paste(dim(Y_train), collapse = " x "), "\n")

if (ncol(Y_train) != 1) {
  stop("CTMTF is univariate only, but Y has dimension != 1")
}

training_time <- system.time({
  model_obj <- fit_ctmtf(X_train, Y_train, seed_now)
})[["elapsed"]]

saveRDS(model_obj, model_path, compress = "xz")

train_log <- data.frame(
  method = "ctmtf",
  backend = "TransformationForest",
  data_name = data_name,
  seed = seed_now,
  train_n = nrow(X_train),
  x_dim = ncol(X_train),
  y_dim = ncol(Y_train),
  basis_order = model_obj$basis_order,
  ntree = model_obj$ntree,
  training_time = as.numeric(training_time),
  model_path = model_path
)

write.csv(train_log, train_log_path, row.names = FALSE)

cat("\nCTMTF train done\n")
cat("training_time:", training_time, "sec\n")
cat("saved model:", model_path, "\n")
cat("model size:", file.info(model_path)$size / 1024^2, "MB\n")
