# Owner: C
# Purpose: Follow-up to R/04b_stationarity.R. Remove the intraday pattern and re-run the
#   dependence checks, relate factor strength to market volatility, check whether PC2
#   and PC3 are stable, and redo batch means with batches of whole trading days.
# Input: data/raw/train.csv, data/processed/time_order.parquet,
#   data/processed/day_slots.parquet, report/stationarity_results.json
# Output: figures/intraday_pattern.png, figures/pc1_share_vs_market_vol.png,
#   report/intraday_results.json
# Run from the project root: Rscript R/04c_intraday_and_stability.R

library(arrow)
library(tseries)
library(mcmcse)

SEED <- 5650
ALPHA <- 0.05
K <- 10
N_SHUFFLES <- 50
ACF_MAX_LAG <- 100
ROLL_WIDTH <- 200
ROLL_STEP <- 50
BOOT_REPS <- 5000
BOOT_BLOCKS <- c(4, 8, 12, 16)
BOOT_MAIN_BLOCK <- 8
DAYS_PER_BATCH <- c(6, 7, 9)
MIN_WINDOWS_TO_PLOT <- 30

COLOR_PRIMARY <- "#2a78d6"
COLOR_BASELINE <- "#eb6834"
COLOR_GRID <- "#e8e7e2"
COLOR_TEXT <- "#52514e"

results <- list(seed = SEED)

# ---- Helpers ----------------------------------------------------------------

# Run a test and record whether tseries warned that the p-value is at its table bound.
run_with_warning_flag <- function(test_fun, x) {
  at_bound <- FALSE
  fit <- withCallingHandlers(
    test_fun(x),
    warning = function(w) {
      if (grepl("p-value", conditionMessage(w))) {
        at_bound <<- TRUE
        invokeRestart("muffleWarning")
      }
    }
  )
  list(statistic = unname(fit$statistic), p_value = fit$p.value,
       lag = unname(fit$parameter), p_at_table_bound = at_bound)
}

# Name the 2x2 ADF/KPSS outcome.
classify_tests <- function(adf_rejects, kpss_rejects) {
  if (adf_rejects && !kpss_rejects) return("stationary")
  if (adf_rejects && kpss_rejects) return("possible structural breaks")
  if (!adf_rejects && !kpss_rejects) return("inconclusive")
  "nonstationary"
}

# Run ADF and both KPSS lag options on one series.
stationarity_tests <- function(x) {
  adf <- run_with_warning_flag(adf.test, x)
  kpss <- run_with_warning_flag(function(v) kpss.test(v, null = "Level"), x)
  kpss_long <- run_with_warning_flag(function(v) kpss.test(v, null = "Level", lshort = FALSE), x)
  list(adf = adf, kpss = kpss, kpss_long_lag = kpss_long,
       adf_rejects = adf$p_value < ALPHA, kpss_rejects = kpss$p_value < ALPHA,
       kpss_long_lag_rejects = kpss_long$p_value < ALPHA,
       class = classify_tests(adf$p_value < ALPHA, kpss$p_value < ALPHA))
}

# Open a PNG device with shared plot styling.
open_figure <- function(path, width, height, mfrow = c(1, 1)) {
  png(path, width = width, height = height, res = 200)
  par(mfrow = mfrow, mar = c(4.2, 4.6, 3, 1), mgp = c(2.6, 0.7, 0), las = 1, bty = "n",
      col.axis = COLOR_TEXT, col.lab = COLOR_TEXT, col.main = "#0b0b0b",
      cex.main = 1.05, font.main = 1, tcl = -0.25)
}

# Return the leading eigenvalues and eigenvectors of a data matrix's correlation matrix.
top_pcs <- function(X, k = 3) {
  e <- eigen(cor(X), symmetric = TRUE)
  list(values = e$values, vectors = e$vectors[, seq_len(k), drop = FALSE],
       share = e$values[seq_len(k)] / ncol(X))
}

# Return the absolute cosine similarity between two unit vectors.
abs_cosine <- function(a, b) abs(sum(a * b))

# Return the two principal angles, in degrees, between the spans of two 2-column bases.
principal_angles <- function(A, B) {
  singular_values <- svd(crossprod(A, B))$d
  acos(pmin(singular_values, 1)) * 180 / pi
}

# Assign rows 1..n to K equal contiguous batches.
contiguous_batches <- function(n, K) floor((seq_len(n) - 1) * K / n) + 1

# Summarize one series for the before-and-after table.
describe_series <- function(x, bound) {
  acf_values <- as.numeric(acf(x, lag.max = ACF_MAX_LAG, plot = FALSE)$acf)[-1]
  is_peak <- c(FALSE, diff(sign(diff(acf_values))) < 0, FALSE)
  pgram <- spec.pgram(x, taper = 0, detrend = TRUE, plot = FALSE)
  in_range <- 1 / pgram$freq >= 2 & 1 / pgram$freq <= 100
  ar1 <- arima(x, order = c(1, 0, 0), method = "ML")
  list(
    variance = var(x),
    acf = acf_values,
    acf_at = setNames(as.list(acf_values[c(1, 2, 3, 6, 7, 13, 26, 39, 50, 100)]),
                      paste0("lag", c(1, 2, 3, 6, 7, 13, 26, 39, 50, 100))),
    acf_min = min(acf_values),
    lags_inside_bound = sum(abs(acf_values) < bound),
    first_lag_inside_bound = if (any(abs(acf_values) < bound)) which(abs(acf_values) < bound)[1] else NA,
    peak_lags = which(is_peak),
    peak_values = acf_values[is_peak],
    dominant_period = 1 / pgram$freq[in_range][which.max(pgram$spec[in_range])],
    dominant_period_power_ratio = max(pgram$spec[in_range]) / median(pgram$spec[in_range]),
    ar1_phi = unname(coef(ar1)["ar1"]),
    ar1_se = unname(sqrt(ar1$var.coef["ar1", "ar1"])),
    ess_implied_by_ar1 = length(x) * (1 - coef(ar1)[["ar1"]]) / (1 + coef(ar1)[["ar1"]]),
    tests = stationarity_tests(x),
    ess_sqroot = unname(mcmcse::ess(x, size = "sqroot")),
    ess_cuberoot = unname(mcmcse::ess(x, size = "cuberoot"))
  )
}

# Summarize the stationarity tests and ESS across stocks.
describe_stocks <- function(Y) {
  series <- lapply(seq_len(ncol(Y)), function(j) Y[!is.na(Y[, j]), j])
  tests <- lapply(series, stationarity_tests)
  classes <- vapply(tests, `[[`, character(1), "class")
  levels <- c("stationary", "possible structural breaks", "inconclusive", "nonstationary")
  list(
    adf_reject_share = mean(vapply(tests, `[[`, logical(1), "adf_rejects")),
    kpss_reject_share = mean(vapply(tests, `[[`, logical(1), "kpss_rejects")),
    kpss_long_lag_reject_share = mean(vapply(tests, `[[`, logical(1), "kpss_long_lag_rejects")),
    class_counts = as.list(table(factor(classes, levels = levels))),
    ess_median_sqroot = median(vapply(series, function(s) unname(mcmcse::ess(s, size = "sqroot")), numeric(1))),
    ess_median_cuberoot = median(vapply(series, function(s) unname(mcmcse::ess(s, size = "cuberoot")), numeric(1)))
  )
}

# ---- Load data ---------------------------------------------------------------

train <- arrow::read_csv_arrow("data/raw/train.csv")
time_order <- as.data.frame(arrow::read_parquet("data/processed/time_order.parquet"))
time_order <- time_order[order(time_order$time_rank), ]
day_slots <- as.data.frame(arrow::read_parquet("data/processed/day_slots.parquet"))
day_slots <- day_slots[match(time_order$time_id, day_slots$time_id), ]

stocks <- sort(unique(train$stock_id))
n <- nrow(time_order)
p <- length(stocks)
stopifnot(n == 3830, p == 112, !anyNA(day_slots$slot), !anyNA(day_slots$day_id))

Y <- matrix(NA_real_, n, p, dimnames = list(time_order$time_rank, stocks))
Y[cbind(match(train$time_id, time_order$time_id), match(train$stock_id, stocks))] <- log(train$target)
Y_filled <- apply(Y, 2, function(y) replace(y, is.na(y), mean(y, na.rm = TRUE)))
m_t <- rowMeans(Y, na.rm = TRUE)
slot <- day_slots$slot
day_id <- day_slots$day_id
acf_bound <- qnorm(0.975) / sqrt(n)

# ---- Item 2: Remove the intraday pattern --------------------------------------

# Subtract each stock's mean in each slot. Missing cells stay missing, and they are set
# to 0 (their slot mean) only for PCA steps.
Y_deseason <- Y
for (s in unique(slot)) {
  rows <- slot == s
  Y_deseason[rows, ] <- sweep(Y[rows, , drop = FALSE], 2, colMeans(Y[rows, , drop = FALSE], na.rm = TRUE))
}
Y_deseason[is.nan(Y_deseason)] <- NA
Y_deseason_filled <- replace(Y_deseason, is.na(Y_deseason), 0)
m_deseason <- rowMeans(Y_deseason, na.rm = TRUE)

full_raw <- top_pcs(Y_filled)
full_deseason <- top_pcs(Y_deseason_filled)

slot_values <- sort(unique(slot))
slot_table <- data.frame(
  slot = slot_values,
  windows = as.integer(table(slot)[as.character(slot_values)]),
  mean_m_t = as.numeric(tapply(m_t, slot, mean)[as.character(slot_values)]),
  sd_m_t = as.numeric(tapply(m_t, slot, sd)[as.character(slot_values)])
)

day_means <- ave(m_t, day_id)
day_means_deseason <- ave(m_deseason, day_id)
results$deseasonalize <- list(
  slot_table = slot_table,
  raw = describe_series(m_t, acf_bound),
  deseasonalized = describe_series(m_deseason, acf_bound),
  stocks_raw = describe_stocks(Y),
  stocks_deseasonalized = describe_stocks(Y_deseason),
  pc_share_raw = full_raw$share,
  pc_share_deseasonalized = full_deseason$share,
  abs_cosine_pc1_raw_vs_deseasonalized = abs_cosine(full_raw$vectors[, 1], full_deseason$vectors[, 1]),
  m_t_variance_share_removed = 1 - var(m_deseason) / var(m_t),
  m_t_variance_share_between_days = sum((day_means - mean(m_t))^2) / sum((m_t - mean(m_t))^2),
  m_deseason_variance_share_between_days =
    sum((day_means_deseason - mean(m_deseason))^2) / sum((m_deseason - mean(m_deseason))^2)
)

shown <- slot_table$windows >= MIN_WINDOWS_TO_PLOT
half_width <- qnorm(0.975) * slot_table$sd_m_t / sqrt(slot_table$windows)
open_figure("figures/intraday_pattern.png", 2000, 1700, mfrow = c(2, 1))
plot(slot_table$slot[shown], slot_table$mean_m_t[shown], type = "n",
     ylim = range((slot_table$mean_m_t + half_width)[shown], (slot_table$mean_m_t - half_width)[shown]),
     xaxt = "n", xlab = "Slot (0 = boundary window; slots count back along the recovered order)",
     ylab = "Average m_t", main = "Average market log volatility by slot (with 95% intervals)")
axis(1, at = slot_table$slot[shown])
grid(nx = NA, ny = NULL, col = COLOR_GRID, lty = 1)
segments(slot_table$slot[shown], (slot_table$mean_m_t - half_width)[shown],
         slot_table$slot[shown], (slot_table$mean_m_t + half_width)[shown], col = COLOR_PRIMARY, lwd = 1.5)
lines(slot_table$slot[shown], slot_table$mean_m_t[shown], col = COLOR_PRIMARY, lwd = 2)
points(slot_table$slot[shown], slot_table$mean_m_t[shown], pch = 21, bg = COLOR_PRIMARY, col = "white", cex = 1.3)

acf_raw <- results$deseasonalize$raw$acf
acf_after <- results$deseasonalize$deseasonalized$acf
plot(seq_len(ACF_MAX_LAG), acf_raw, type = "n", ylim = range(c(acf_raw, acf_after, 0)),
     xlab = "Lag (windows in recovered order)", ylab = "Autocorrelation",
     main = "ACF of m_t before and after removing slot means")
grid(nx = NA, ny = NULL, col = COLOR_GRID, lty = 1)
abline(h = 0, col = COLOR_TEXT)
lines(seq_len(ACF_MAX_LAG), acf_raw, col = COLOR_BASELINE, lwd = 1.5)
lines(seq_len(ACF_MAX_LAG), acf_after, col = COLOR_PRIMARY, lwd = 2)
legend("topright", legend = c("Before", "After"), col = c(COLOR_BASELINE, COLOR_PRIMARY),
       lwd = c(1.5, 2), bty = "n", cex = 0.85, horiz = TRUE, text.col = COLOR_TEXT)
dev.off()

# ---- Item 4, 7, 8: Stability across batches -----------------------------------

time_batch <- contiguous_batches(n, K)

# Measure one set of 10 batches against the full-sample PCs of the same data.
measure_batches <- function(batch, X = Y_filled, full = full_raw) {
  fits <- lapply(seq_len(K), function(b) top_pcs(X[batch == b, , drop = FALSE]))
  rows <- t(vapply(fits, function(fit) {
    angles <- principal_angles(fit$vectors[, 2:3], full$vectors[, 2:3])
    c(pc1_share = fit$share[1],
      pc1_cosine = abs_cosine(fit$vectors[, 1], full$vectors[, 1]),
      pc2_cosine = abs_cosine(fit$vectors[, 2], full$vectors[, 2]),
      pc3_cosine = abs_cosine(fit$vectors[, 3], full$vectors[, 3]),
      pc2_vs_full_pc3 = abs_cosine(fit$vectors[, 2], full$vectors[, 3]),
      pc3_vs_full_pc2 = abs_cosine(fit$vectors[, 3], full$vectors[, 2]),
      angle_small = angles[1], angle_large = angles[2],
      eigen_ratio_2_3 = fit$values[2] / fit$values[3])
  }, numeric(9)))
  pairs <- combn(K, 2)
  pair_angles <- apply(pairs, 2, function(ij) {
    principal_angles(fits[[ij[1]]]$vectors[, 2:3], fits[[ij[2]]]$vectors[, 2:3])[2]
  })
  list(rows = rows, pair_large_angle = pair_angles)
}

set.seed(SEED)
time_measures <- measure_batches(time_batch)
# These draws repeat the shuffles in R/04b_stationarity.R, which sets the same seed and
# draws no other random numbers first. The check below confirms that.
random_measures <- lapply(seq_len(N_SHUFFLES), function(s) measure_batches(sample(time_batch)))

earlier <- jsonlite::fromJSON("report/stationarity_results.json")$batches
random_cube <- simplify2array(lapply(random_measures, `[[`, "rows"))
random_batch_means <- apply(random_cube, c(1, 2), mean)
stopifnot(
  isTRUE(all.equal(unname(time_measures$rows[, "pc1_share"]), earlier$by_batch$time_pc1_share, tolerance = 1e-8)),
  isTRUE(all.equal(unname(random_batch_means[, "pc1_share"]), earlier$by_batch$random_pc1_share, tolerance = 1e-8)),
  isTRUE(all.equal(unname(random_batch_means[, "pc1_cosine"]), earlier$by_batch$random_abs_cosine, tolerance = 1e-8))
)

# Compare time batches with the distribution of shuffle-level results.
compare_measure <- function(name, time_rows = time_measures$rows, cube = random_cube) {
  values <- time_rows[, name]
  shuffle_means <- apply(cube[, name, ], 2, mean)
  shuffle_sds <- apply(cube[, name, ], 2, sd)
  list(
    time_mean = mean(values), time_sd = sd(values), time_min = min(values), time_max = max(values),
    random_mean = mean(shuffle_means),
    random_mean_q025 = unname(quantile(shuffle_means, 0.025)),
    random_mean_q975 = unname(quantile(shuffle_means, 0.975)),
    random_mean_sd = mean(shuffle_sds),
    random_min = min(cube[, name, ]), random_max = max(cube[, name, ])
  )
}

pair_random_means <- vapply(random_measures, function(m) mean(m$pair_large_angle), numeric(1))
pair_random_maxes <- vapply(random_measures, function(m) max(m$pair_large_angle), numeric(1))

score_slot_share <- function(scores) {
  1 - sum(tapply(scores, slot, function(v) sum((v - mean(v))^2))) / sum((scores - mean(scores))^2)
}
scores_raw <- scale(Y_filled) %*% eigen(cor(Y_filled), symmetric = TRUE)$vectors[, 1:3]

results$pc_stability <- list(
  full_sample_eigenvalues = full_raw$values[1:4],
  full_sample_eigen_ratio_2_3 = full_raw$values[2] / full_raw$values[3],
  by_batch = data.frame(batch = seq_len(K), time_measures$rows,
                        random_pc2_cosine = random_batch_means[, "pc2_cosine"],
                        random_pc3_cosine = random_batch_means[, "pc3_cosine"],
                        random_angle_large = random_batch_means[, "angle_large"]),
  pc2_cosine = compare_measure("pc2_cosine"),
  pc3_cosine = compare_measure("pc3_cosine"),
  angle_small = compare_measure("angle_small"),
  angle_large = compare_measure("angle_large"),
  eigen_ratio_2_3 = compare_measure("eigen_ratio_2_3"),
  batches_where_order_swaps = sum(time_measures$rows[, "pc2_vs_full_pc3"] > time_measures$rows[, "pc2_cosine"]),
  random_batches_where_order_swaps_share = mean(random_cube[, "pc2_vs_full_pc3", ] > random_cube[, "pc2_cosine", ]),
  pairwise_large_angle_time_mean = mean(time_measures$pair_large_angle),
  pairwise_large_angle_time_max = max(time_measures$pair_large_angle),
  pairwise_large_angle_random_mean = mean(pair_random_means),
  pairwise_large_angle_random_max = max(pair_random_maxes),
  score_variance_share_explained_by_slot = c(
    pc1 = score_slot_share(scores_raw[, 1]), pc2 = score_slot_share(scores_raw[, 2]),
    pc3 = score_slot_share(scores_raw[, 3])
  )
)

# Extra check: repeat the batch comparison after removing the intraday pattern.
set.seed(SEED)
time_deseason <- measure_batches(time_batch, Y_deseason_filled, full_deseason)
random_deseason <- lapply(seq_len(N_SHUFFLES), function(s) {
  measure_batches(sample(time_batch), Y_deseason_filled, full_deseason)
})
cube_deseason <- simplify2array(lapply(random_deseason, `[[`, "rows"))
results$pc_stability_deseasonalized <- list(
  full_sample_eigenvalues = full_deseason$values[1:4],
  pc1_share = compare_measure("pc1_share", time_deseason$rows, cube_deseason),
  pc1_cosine = compare_measure("pc1_cosine", time_deseason$rows, cube_deseason),
  pc2_cosine = compare_measure("pc2_cosine", time_deseason$rows, cube_deseason),
  pc3_cosine = compare_measure("pc3_cosine", time_deseason$rows, cube_deseason),
  angle_small = compare_measure("angle_small", time_deseason$rows, cube_deseason),
  angle_large = compare_measure("angle_large", time_deseason$rows, cube_deseason),
  pairwise_large_angle_time_mean = mean(time_deseason$pair_large_angle),
  pairwise_large_angle_random_mean = mean(vapply(random_deseason, function(m) mean(m$pair_large_angle), numeric(1)))
)

time_shares <- time_measures$rows[, "pc1_share"]
shuffle_share_sds <- apply(random_cube[, "pc1_share", ], 2, sd)
results$pc1_share_se <- list(
  full_sample_share = full_raw$share[1],
  time_batch_mean_share = mean(time_shares),
  time_batch_sd = sd(time_shares),
  time_batch_se = sd(time_shares) / sqrt(K),
  time_batch_interval = mean(time_shares) + c(-1, 1) * qt(0.975, K - 1) * sd(time_shares) / sqrt(K),
  random_batch_sd = mean(shuffle_share_sds),
  random_batch_sd_pooled = sd(as.numeric(random_cube[, "pc1_share", ])),
  random_batch_se = mean(shuffle_share_sds) / sqrt(K),
  se_ratio_time_to_random = sd(time_shares) / mean(shuffle_share_sds)
)

# ---- Item 6: Batch means with batches of whole days ---------------------------

# Return the vech of the outer product of the first 3 PC scores.
vech_scores <- function(X) {
  scores <- scale(X) %*% eigen(cor(X), symmetric = TRUE)$vectors[, 1:3]
  pairs <- rbind(c(1, 1), c(1, 2), c(1, 3), c(2, 2), c(2, 3), c(3, 3))
  out <- apply(pairs, 1, function(ij) scores[, ij[1]] * scores[, ij[2]])
  colnames(out) <- apply(pairs, 1, function(ij) paste0("s", ij[1], "s", ij[2]))
  out
}

vech_raw <- vech_scores(Y_filled)
vech_deseason <- vech_scores(Y_deseason_filled)
typical_length <- jsonlite::fromJSON("report/day_boundary_results.json")$days$typical_length

# Run batch means at one batch size and one lugsail setting.
batch_means_row <- function(days, r) {
  size <- days * typical_length
  run <- function(x) {
    fit <- mcmcse::mcse.multi(x, method = "bm", size = size, r = r, blather = TRUE)
    list(multi_ess = mcmcse::multiESS(x, covmat = fit$cov),
         adjustment_used = fit$Adjustment_Used, method = fit$method,
         s1s1_mcse = sqrt(fit$cov[1, 1] / n), det_ok = min(eigen(fit$cov, only.values = TRUE)$values) > 0)
  }
  raw <- run(vech_raw)
  after <- run(vech_deseason)
  list(
    days = days, r = r, batch_size = size, n_batches = floor(n / size),
    inner_batch_size = if (r > 1) floor(size / r) else NA,
    inner_batch_is_whole_days = if (r > 1) floor(size / r) %% typical_length == 0 else NA,
    batches_exceed_dimension = floor(n / size) > ncol(vech_raw),
    multi_ess_raw = raw$multi_ess, multi_ess_deseasonalized = after$multi_ess,
    adjustment_raw = raw$adjustment_used, adjustment_deseasonalized = after$adjustment_used,
    s1s1_mcse_raw = raw$s1s1_mcse, s1s1_mcse_deseasonalized = after$s1s1_mcse,
    m_t_ess_raw = unname(mcmcse::ess(m_t, size = size, r = r)),
    m_t_ess_deseasonalized = unname(mcmcse::ess(m_deseason, size = size, r = r))
  )
}

settings <- expand.grid(days = DAYS_PER_BATCH, r = c(1, 3))
results$batch_means_whole_days <- list(
  typical_day_length = typical_length,
  rows = lapply(seq_len(nrow(settings)), function(i) batch_means_row(settings$days[i], settings$r[i]))
)

# Extra check: the PC1 share is a ratio, because high-volatility windows raise its top
# and bottom together. This linearized series accounts for that. Loadings stay fixed.
z_raw <- scale(Y_filled)
pc1_scores <- as.numeric(z_raw %*% full_raw$vectors[, 1])
total_squares <- rowSums(z_raw^2)
share_linearized <- (pc1_scores^2 - full_raw$share[1] * total_squares) / p
results$pc1_share_se$linearized_batch_means <- lapply(seq_len(nrow(settings)), function(i) {
  size <- settings$days[i] * typical_length
  list(days = settings$days[i], r = settings$r[i], batch_size = size,
       se = unname(mcmcse::mcse(share_linearized, size = size, r = settings$r[i])$se))
})
results$pc1_share_se$numerator_only_se_at_batch_83 <-
  unname(mcmcse::mcse(pc1_scores^2, size = 83, r = 3)$se) / p

# ---- Item 3: Factor strength against market volatility ------------------------

starts <- seq(1, n - ROLL_WIDTH + 1, by = ROLL_STEP)

rolling_measures <- function(X, m) {
  data.frame(
    start = starts,
    pc1_share = vapply(starts, function(s) top_pcs(X[s:(s + ROLL_WIDTH - 1), ], 1)$share, numeric(1)),
    mean_m = vapply(starts, function(s) mean(m[s:(s + ROLL_WIDTH - 1)]), numeric(1))
  )
}

# Resample whole blocks of the rolling results and recompute the Spearman correlation.
block_bootstrap <- function(x, y, block, reps) {
  count <- length(x)
  blocks_needed <- ceiling(count / block)
  draws <- vapply(seq_len(reps), function(i) {
    first <- sample.int(count - block + 1, blocks_needed, replace = TRUE)
    picked <- as.vector(outer(first, seq_len(block) - 1, `+`))[seq_len(count)]
    cor(x[picked], y[picked], method = "spearman")
  }, numeric(1))
  list(block = block, block_span_windows = ROLL_WIDTH + (block - 1) * ROLL_STEP,
       se = sd(draws), q025 = unname(quantile(draws, 0.025)), q975 = unname(quantile(draws, 0.975)),
       share_above_zero = mean(draws > 0))
}

summarize_rolling <- function(roll) {
  list(
    spearman = cor(roll$pc1_share, roll$mean_m, method = "spearman"),
    pc1_share_quantiles = as.list(quantile(roll$pc1_share, c(0, 0.25, 0.5, 0.75, 1))),
    bootstrap = lapply(BOOT_BLOCKS, function(b) block_bootstrap(roll$pc1_share, roll$mean_m, b, BOOT_REPS))
  )
}

non_overlapping <- function(X, m) {
  firsts <- seq(1, n - ROLL_WIDTH + 1, by = ROLL_WIDTH)
  share <- vapply(firsts, function(s) top_pcs(X[s:(s + ROLL_WIDTH - 1), ], 1)$share, numeric(1))
  level <- vapply(firsts, function(s) mean(m[s:(s + ROLL_WIDTH - 1)]), numeric(1))
  test <- cor.test(share, level, method = "spearman", exact = TRUE)
  list(blocks = length(firsts), spearman = unname(test$estimate), p_value = test$p.value)
}

set.seed(SEED + 1)
rolling_raw <- rolling_measures(Y_filled, m_t)
rolling_deseason <- rolling_measures(Y_deseason_filled, m_t)

results$factor_vs_vol <- list(
  width = ROLL_WIDTH, step = ROLL_STEP, n_rolling = length(starts), overlap_windows = ROLL_WIDTH - ROLL_STEP,
  bootstrap_reps = BOOT_REPS, main_block = BOOT_MAIN_BLOCK,
  mean_m_min = min(rolling_raw$mean_m), mean_m_max = max(rolling_raw$mean_m),
  raw = c(summarize_rolling(rolling_raw), list(non_overlapping = non_overlapping(Y_filled, m_t))),
  deseasonalized = c(summarize_rolling(rolling_deseason),
                     list(non_overlapping = non_overlapping(Y_deseason_filled, m_t)))
)

main_boot <- results$factor_vs_vol$raw$bootstrap[[which(BOOT_BLOCKS == BOOT_MAIN_BLOCK)]]
open_figure("figures/pc1_share_vs_market_vol.png", 1800, 1500)
plot(rolling_raw$mean_m, rolling_raw$pc1_share, type = "n",
     xlab = "Rolling mean of m_t (200 windows)", ylab = "Rolling PC1 variance share (200 windows)",
     main = "Factor strength against market volatility")
grid(col = COLOR_GRID, lty = 1)
points(rolling_raw$mean_m, rolling_raw$pc1_share, pch = 21, bg = COLOR_PRIMARY, col = "white", cex = 1.4)
legend("bottomright", bty = "n", cex = 0.85, text.col = COLOR_TEXT,
       legend = c(sprintf("Spearman correlation: %.3f", results$factor_vs_vol$raw$spearman),
                  sprintf("95%% block bootstrap interval: %.3f to %.3f", main_boot$q025, main_boot$q975),
                  sprintf("%d overlapping rolling windows, step %d", length(starts), ROLL_STEP)))
dev.off()

# ---- Save ----------------------------------------------------------------------

jsonlite::write_json(results, "report/intraday_results.json", auto_unbox = TRUE, digits = NA,
                     pretty = TRUE, dataframe = "rows", matrix = "rowmajor")
cat("Done. Results saved to report/intraday_results.json\n")
