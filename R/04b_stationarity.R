# Owner: C
# Purpose: Check whether the 3,830 time windows can be treated as replicates before
#   PCA or factor analysis: stationarity tests, serial dependence, effective sample
#   size, covariance stability across batches, the effect of level shifts on PC1, and
#   batch means standard errors. Extra checks measure a repeating cycle in the ACF.
# Input: data/raw/train.csv, data/processed/time_order.parquet
# Output: figures/stationarity_market_logrv.png, figures/acf_market_logrv.png,
#   figures/batch_pca_stability.png, report/stationarity_results.json,
#   report/batch_pca_stability.md, report/per_stock_stationarity.md
# Run from the project root: Rscript R/04b_stationarity.R

library(arrow)
library(tseries)
library(mcmcse)

SEED <- 5650
ROLL_WINDOW <- 100
ACF_MAX_LAG <- 100
K <- 10
N_SHUFFLES <- 50
ALPHA <- 0.05

COLOR_PRIMARY <- "#2a78d6"
COLOR_BASELINE <- "#eb6834"
COLOR_TRACE <- "#b5b4ad"
COLOR_GRID <- "#e8e7e2"
COLOR_TEXT <- "#52514e"
CYCLE_HARMONICS <- 3

set.seed(SEED)
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

# Run ADF and KPSS on one series and classify the result.
stationarity_tests <- function(x) {
  adf <- run_with_warning_flag(adf.test, x)
  kpss <- run_with_warning_flag(function(v) kpss.test(v, null = "Level"), x)
  adf_rejects <- adf$p_value < ALPHA
  kpss_rejects <- kpss$p_value < ALPHA
  list(adf = adf, kpss = kpss, adf_rejects = adf_rejects, kpss_rejects = kpss_rejects,
       class = classify_tests(adf_rejects, kpss_rejects))
}

# Return the PC1 variance share and PC1 loadings of a data matrix's correlation matrix.
pc1_of <- function(X) {
  e <- eigen(cor(X), symmetric = TRUE)
  list(share = e$values[1] / ncol(X), loadings = e$vectors[, 1], values = e$values)
}

# Return the absolute cosine similarity between two loading vectors.
abs_cosine <- function(a, b) {
  abs(sum(a * b)) / sqrt(sum(a^2) * sum(b^2))
}

# Assign rows 1..n to K equal contiguous batches.
contiguous_batches <- function(n, K) {
  floor((seq_len(n) - 1) * K / n) + 1
}

# Subtract each column's mean within each batch.
demean_within <- function(X, batch) {
  out <- X
  for (b in unique(batch)) {
    rows <- batch == b
    out[rows, ] <- scale(X[rows, , drop = FALSE], center = TRUE, scale = FALSE)
  }
  out
}

# Return centered rolling means and SDs for a window of width w.
rolling_stats <- function(x, w) {
  weights <- rep(1 / w, w)
  roll_mean <- as.numeric(stats::filter(x, weights, sides = 2))
  roll_mean_sq <- as.numeric(stats::filter(x^2, weights, sides = 2))
  list(mean = roll_mean, sd = sqrt((roll_mean_sq - roll_mean^2) * w / (w - 1)))
}

# Write a data frame as a Markdown table, rounding numeric columns.
markdown_table <- function(df, digits = 4) {
  cells <- lapply(df, function(column) {
    if (is.numeric(column) && !is.integer(column)) formatC(column, digits = digits, format = "f")
    else as.character(column)
  })
  rows <- do.call(paste, c(cells, sep = " | "))
  c(paste0("| ", paste(names(df), collapse = " | "), " |"),
    paste0("|", paste(rep(" --- ", ncol(df)), collapse = "|"), "|"),
    paste0("| ", rows, " |"))
}

# Open a PNG device with shared plot styling.
open_figure <- function(path, width, height, mfrow = c(1, 1), left = 4.6) {
  png(path, width = width, height = height, res = 200)
  par(mfrow = mfrow, mar = c(4.2, left, 3, 1), mgp = c(2.6, 0.7, 0), las = 1, bty = "n",
      col.axis = COLOR_TEXT, col.lab = COLOR_TEXT, col.main = "#0b0b0b",
      cex.main = 1.05, font.main = 1, tcl = -0.25)
}

# ---- Load data ---------------------------------------------------------------

train <- arrow::read_csv_arrow("data/raw/train.csv")
time_order <- as.data.frame(arrow::read_parquet("data/processed/time_order.parquet"))
time_order <- time_order[order(time_order$time_rank), ]

stocks <- sort(unique(train$stock_id))
n <- nrow(time_order)
p <- length(stocks)
stopifnot(n == 3830, p == 112, setequal(train$time_id, time_order$time_id),
          all(train$target > 0), !anyDuplicated(train[c("stock_id", "time_id")]))

Y <- matrix(NA_real_, n, p, dimnames = list(time_order$time_rank, stocks))
Y[cbind(match(train$time_id, time_order$time_id), match(train$stock_id, stocks))] <- log(train$target)

missing_by_stock <- colSums(is.na(Y))
missing_by_stock <- missing_by_stock[missing_by_stock > 0]
results$data <- list(
  n_windows = n, n_stocks = p, cells_total = n * p, cells_missing = sum(is.na(Y)),
  missing_by_stock = as.list(missing_by_stock)
)

# Missing cells are filled with column means for PCA steps only.
Y_filled <- apply(Y, 2, function(y) replace(y, is.na(y), mean(y, na.rm = TRUE)))
m_t <- rowMeans(Y, na.rm = TRUE)

# ---- Step 1: Visual check ------------------------------------------------------

roll <- rolling_stats(m_t, ROLL_WINDOW)
results$visual <- list(
  roll_window = ROLL_WINDOW, m_mean = mean(m_t), m_sd = sd(m_t),
  roll_mean_min = min(roll$mean, na.rm = TRUE), roll_mean_max = max(roll$mean, na.rm = TRUE),
  roll_sd_min = min(roll$sd, na.rm = TRUE), roll_sd_max = max(roll$sd, na.rm = TRUE),
  roll_mean_max_rank = time_order$time_rank[which.max(roll$mean)]
)

ranks <- time_order$time_rank
open_figure("figures/stationarity_market_logrv.png", 2000, 1500, mfrow = c(2, 1))
plot(ranks, m_t, type = "n", xlab = "Recovered time rank", ylab = "Mean log(target)",
     main = "Market log volatility over the recovered time order", xaxt = "n",
     xlim = c(0, n - 1), ylim = c(min(m_t), max(m_t) + 0.55))
axis(1, at = seq(0, 3500, 500))
grid(nx = NA, ny = NULL, col = COLOR_GRID, lty = 1)
lines(ranks, m_t, col = COLOR_TRACE, lwd = 0.6)
lines(ranks, roll$mean, col = COLOR_PRIMARY, lwd = 2)
legend("top", legend = c("m_t (cross-sectional mean)", "Rolling mean (100 windows)"),
       col = c(COLOR_TRACE, COLOR_PRIMARY), lwd = c(1, 2), bty = "n", cex = 0.8,
       text.col = COLOR_TEXT, horiz = TRUE)
plot(ranks, roll$sd, type = "n", xlab = "Recovered time rank", ylab = "Rolling SD of m_t",
     main = "Rolling SD of market log volatility (100 windows)", xaxt = "n", xlim = c(0, n - 1))
axis(1, at = seq(0, 3500, 500))
grid(nx = NA, ny = NULL, col = COLOR_GRID, lty = 1)
lines(ranks, roll$sd, col = COLOR_PRIMARY, lwd = 2)
dev.off()

# ---- Step 2: Formal tests ------------------------------------------------------

market_tests <- stationarity_tests(m_t)

# Tests need a complete series. Stocks with missing windows use their observed
# windows in order, so these few gaps are treated as adjacent.
stock_series <- lapply(seq_len(p), function(j) Y[!is.na(Y[, j]), j])
stock_tests <- lapply(stock_series, stationarity_tests)

adf_rejects <- vapply(stock_tests, `[[`, logical(1), "adf_rejects")
kpss_rejects <- vapply(stock_tests, `[[`, logical(1), "kpss_rejects")
stock_class <- vapply(stock_tests, `[[`, character(1), "class")
class_levels <- c("stationary", "possible structural breaks", "inconclusive", "nonstationary")

# Extra check: KPSS can over-reject on persistent series when its lag is short, so
# repeat it with the longer lag option.
kpss_long <- function(x) {
  run_with_warning_flag(function(v) kpss.test(v, null = "Level", lshort = FALSE), x)
}
stock_kpss_long <- lapply(stock_series, kpss_long)

results$tests <- list(
  alpha = ALPHA,
  market = market_tests,
  market_kpss_long_lag = kpss_long(m_t),
  stock_kpss_long_lag_reject_share = mean(vapply(stock_kpss_long, function(s) s$p_value < ALPHA, logical(1))),
  stocks = list(
    n = p,
    adf_reject_share = mean(adf_rejects),
    kpss_reject_share = mean(kpss_rejects),
    class_counts = as.list(table(factor(stock_class, levels = class_levels))),
    adf_p_at_bound = sum(vapply(stock_tests, function(s) s$adf$p_at_table_bound, logical(1))),
    kpss_p_at_bound = sum(vapply(stock_tests, function(s) s$kpss$p_at_table_bound, logical(1)))
  )
)

# ---- Step 3: Dependence --------------------------------------------------------

acf_fit <- acf(m_t, lag.max = ACF_MAX_LAG, plot = FALSE)
acf_values <- as.numeric(acf_fit$acf)[-1]
acf_bound <- qnorm(0.975) / sqrt(n)

open_figure("figures/acf_market_logrv.png", 2000, 1100)
plot(seq_len(ACF_MAX_LAG), acf_values, type = "n", ylim = range(c(0, acf_values, acf_bound, -acf_bound)),
     xlab = "Lag (windows in recovered order)", ylab = "Autocorrelation",
     main = "ACF of market log volatility (m_t), lags 1 to 100")
grid(nx = NA, ny = NULL, col = COLOR_GRID, lty = 1)
abline(h = 0, col = COLOR_TEXT, lwd = 1)
abline(h = c(-acf_bound, acf_bound), col = COLOR_BASELINE, lwd = 1.2, lty = 2)
segments(seq_len(ACF_MAX_LAG), 0, seq_len(ACF_MAX_LAG), acf_values, col = COLOR_PRIMARY, lwd = 2)
legend("topright", legend = c("Sample ACF", "95% white-noise bounds"),
       col = c(COLOR_PRIMARY, COLOR_BASELINE), lwd = c(2, 1.2), lty = c(1, 2), bty = "n", cex = 0.8,
       text.col = COLOR_TEXT)
dev.off()

ar1_fit <- arima(m_t, order = c(1, 0, 0), method = "ML")
ar1_phi <- unname(coef(ar1_fit)["ar1"])
inside_bound <- which(abs(acf_values) < acf_bound)
is_peak <- c(FALSE, diff(sign(diff(acf_values))) < 0, FALSE)

# Find the strongest repeating period between 2 and 100 windows.
pgram <- spec.pgram(m_t, taper = 0, detrend = TRUE, plot = FALSE)
in_range <- 1 / pgram$freq >= 2 & 1 / pgram$freq <= 100
dominant_period <- 1 / pgram$freq[in_range][which.max(pgram$spec[in_range])]

ess_of <- function(x, size, ...) unname(mcmcse::ess(x, size = size, ...))
ess_market <- c(sqroot = ess_of(m_t, "sqroot"), cuberoot = ess_of(m_t, "cuberoot"))
ess_market_r1 <- c(sqroot = ess_of(m_t, "sqroot", r = 1), cuberoot = ess_of(m_t, "cuberoot", r = 1))
ess_stocks <- cbind(
  sqroot = vapply(stock_series, ess_of, numeric(1), size = "sqroot"),
  cuberoot = vapply(stock_series, ess_of, numeric(1), size = "cuberoot")
)
stopifnot(all(is.finite(ess_market)), all(is.finite(ess_stocks)))

results$dependence <- list(
  acf_lag1 = acf_values[1], acf_lag5 = acf_values[5], acf_lag10 = acf_values[10],
  acf_lag20 = acf_values[20], acf_lag50 = acf_values[50], acf_lag100 = acf_values[100],
  white_noise_bound = acf_bound,
  first_lag_inside_bound = if (length(inside_bound)) inside_bound[1] else NA,
  lags_inside_bound_count = length(inside_bound),
  acf_min_lags_1_to_100 = min(acf_values),
  acf_local_peak_lags = which(is_peak),
  acf_local_peak_values = acf_values[is_peak],
  dominant_period_windows = dominant_period,
  ar1_phi = ar1_phi, ar1_se = unname(sqrt(ar1_fit$var.coef["ar1", "ar1"])),
  ar1_half_life_windows = log(0.5) / log(ar1_phi),
  ess_implied_by_ar1 = n * (1 - ar1_phi) / (1 + ar1_phi),
  n = n,
  batch_size_sqroot = floor(sqrt(n)), batch_size_cuberoot = floor(n^(1 / 3)),
  ess_market = as.list(ess_market),
  ess_market_share_of_n = as.list(ess_market / n),
  ess_market_classic_bm_r1 = as.list(ess_market_r1),
  ess_stock_median = as.list(apply(ess_stocks, 2, median)),
  ess_stock_q25 = as.list(apply(ess_stocks, 2, quantile, probs = 0.25, names = FALSE)),
  ess_stock_q75 = as.list(apply(ess_stocks, 2, quantile, probs = 0.75, names = FALSE)),
  ess_stock_min = as.list(apply(ess_stocks, 2, min)),
  ess_stock_max = as.list(apply(ess_stocks, 2, max))
)

per_stock <- data.frame(
  stock_id = as.integer(stocks),
  n = lengths(stock_series),
  adf_stat = vapply(stock_tests, function(s) s$adf$statistic, numeric(1)),
  adf_p = vapply(stock_tests, function(s) s$adf$p_value, numeric(1)),
  kpss_stat = vapply(stock_tests, function(s) s$kpss$statistic, numeric(1)),
  kpss_p = vapply(stock_tests, function(s) s$kpss$p_value, numeric(1)),
  class = stock_class,
  ess_sqroot = ess_stocks[, "sqroot"],
  ess_cuberoot = ess_stocks[, "cuberoot"]
)
writeLines(c(
  "# Per-stock stationarity results",
  "",
  "Generated by `R/04b_stationarity.R`. `tseries` limits p-values to 0.01-0.10 for KPSS and 0.01-0.99 for ADF, so values at those limits are bounds. Stocks with missing windows use their observed windows in order.",
  "",
  markdown_table(per_stock, digits = 3)
), "report/per_stock_stationarity.md")

# ---- Step 4: Covariance stability across batches -------------------------------

full_pca <- pc1_of(Y_filled)
time_batch <- contiguous_batches(n, K)

batch_stats <- function(batch) {
  t(vapply(seq_len(K), function(b) {
    fit <- pc1_of(Y_filled[batch == b, , drop = FALSE])
    c(share = fit$share, cosine = abs_cosine(fit$loadings, full_pca$loadings))
  }, numeric(2)))
}

time_stats <- batch_stats(time_batch)
random_stats <- lapply(seq_len(N_SHUFFLES), function(s) batch_stats(sample(time_batch)))
random_share <- do.call(rbind, lapply(random_stats, function(m) m[, "share"]))
random_cosine <- do.call(rbind, lapply(random_stats, function(m) m[, "cosine"]))

summarize_batches <- function(time_values, random_values) {
  random_sds <- apply(random_values, 1, sd)
  list(
    time_mean = mean(time_values), time_sd = sd(time_values),
    time_min = min(time_values), time_max = max(time_values),
    random_mean = mean(random_values),
    random_mean_sd = mean(random_sds),
    random_sd_q025 = unname(quantile(random_sds, 0.025)),
    random_sd_q975 = unname(quantile(random_sds, 0.975)),
    random_min = min(random_values), random_max = max(random_values),
    random_batch_q025 = unname(quantile(random_values, 0.025)),
    random_batch_q975 = unname(quantile(random_values, 0.975)),
    sd_ratio_time_to_random = sd(time_values) / mean(random_sds),
    shuffles_with_sd_at_least_time = sum(random_sds >= sd(time_values))
  )
}

results$batches <- list(
  K = K, batch_size = as.integer(table(time_batch)[1]), n_shuffles = N_SHUFFLES,
  full_sample_pc1_share = full_pca$share,
  corr_batch_mean_m_with_pc1_share = cor(tapply(m_t, time_batch, mean), time_stats[, "share"]),
  pc1_share = summarize_batches(time_stats[, "share"], random_share),
  pc1_cosine = summarize_batches(time_stats[, "cosine"], random_cosine),
  by_batch = data.frame(
    batch = seq_len(K),
    time_pc1_share = time_stats[, "share"],
    time_abs_cosine = time_stats[, "cosine"],
    random_pc1_share = colMeans(random_share),
    random_abs_cosine = colMeans(random_cosine)
  )
)

writeLines(c(
  "# PC1 stability across batches",
  "",
  sprintf("Generated by `R/04b_stationarity.R`. Each batch has %d windows. Random columns are averaged over %d shuffles. The batch number has no meaning for random batches.", results$batches$batch_size, N_SHUFFLES),
  "",
  markdown_table(results$batches$by_batch, digits = 6),
  "",
  "| Summary | Time batches | Random batches |",
  "| --- | --- | --- |",
  sprintf("| PC1 share: mean | %.4f | %.4f |", results$batches$pc1_share$time_mean, results$batches$pc1_share$random_mean),
  sprintf("| PC1 share: SD across batches | %.4f | %.4f |", results$batches$pc1_share$time_sd, results$batches$pc1_share$random_mean_sd),
  sprintf("| Absolute cosine: mean | %.6f | %.6f |", results$batches$pc1_cosine$time_mean, results$batches$pc1_cosine$random_mean),
  sprintf("| Absolute cosine: SD across batches | %.6f | %.6f |", results$batches$pc1_cosine$time_sd, results$batches$pc1_cosine$random_mean_sd)
), "report/batch_pca_stability.md")

draw_batch_panel <- function(time_values, random_values, ylab, title, show_legend) {
  band <- quantile(random_values, c(0.025, 0.975))
  limits <- range(c(time_values, band))
  headroom <- if (show_legend) 0.22 * diff(limits) else 0
  plot(seq_len(K), time_values, type = "n", ylim = c(limits[1], limits[2] + headroom),
       xaxt = "n", xlab = "Batch (time order for time batches)", ylab = "", main = title)
  title(ylab = ylab, line = 4)
  axis(1, at = seq_len(K))
  grid(nx = NA, ny = NULL, col = COLOR_GRID, lty = 1)
  rect(0.5, band[1], K + 0.5, band[2], col = adjustcolor(COLOR_BASELINE, alpha.f = 0.18), border = NA)
  abline(h = mean(random_values), col = COLOR_BASELINE, lwd = 2)
  lines(seq_len(K), time_values, col = COLOR_PRIMARY, lwd = 2)
  points(seq_len(K), time_values, pch = 21, bg = COLOR_PRIMARY, col = "white", cex = 1.3)
  if (show_legend) {
    legend("top", legend = c("Time batches", "Random: mean", "Random: 95% of batches"),
           col = c(COLOR_PRIMARY, COLOR_BASELINE, adjustcolor(COLOR_BASELINE, alpha.f = 0.3)),
           lwd = c(2, 2, 8), bty = "n", cex = 0.75, text.col = COLOR_TEXT, horiz = TRUE)
  }
}

open_figure("figures/batch_pca_stability.png", 2000, 1700, mfrow = c(2, 1), left = 5.6)
draw_batch_panel(time_stats[, "share"], random_share, "PC1 variance share",
                 "PC1 share by batch: time order vs. 50 random shuffles", TRUE)
draw_batch_panel(time_stats[, "cosine"], random_cosine, "|cosine| with full-sample PC1",
                 "PC1 loading similarity by batch (same legend as above)", FALSE)
dev.off()

# ---- Step 5: Do level shifts inflate PC1? --------------------------------------

Y_demeaned <- demean_within(Y_filled, time_batch)
demeaned_pca <- pc1_of(Y_demeaned)
random_demeaned_share <- vapply(
  seq_len(N_SHUFFLES),
  function(s) pc1_of(demean_within(Y_filled, sample(time_batch)))$share,
  numeric(1)
)
m_batch_means <- ave(m_t, time_batch)

results$level_shift <- list(
  pc1_share_raw = full_pca$share,
  pc1_share_demeaned = demeaned_pca$share,
  drop_percentage_points = 100 * (full_pca$share - demeaned_pca$share),
  relative_drop = (full_pca$share - demeaned_pca$share) / full_pca$share,
  pc1_share_demeaned_random_batches_mean = mean(random_demeaned_share),
  abs_cosine_raw_vs_demeaned_pc1 = abs_cosine(full_pca$loadings, demeaned_pca$loadings),
  m_t_variance_share_between_batches = sum((m_batch_means - mean(m_t))^2) / sum((m_t - mean(m_t))^2),
  pc2_share_raw = full_pca$values[2] / p,
  pc3_share_raw = full_pca$values[3] / p,
  pc2_share_demeaned = demeaned_pca$values[2] / p,
  pc3_share_demeaned = demeaned_pca$values[3] / p
)

# ---- Extra check: How much do the cycle and level shifts explain? ---------------

# Regress out sine and cosine terms at the dominant period. This assumes the period
# is steady across the sample, which the recovered order cannot fully confirm.
cycle_terms <- do.call(cbind, lapply(seq_len(CYCLE_HARMONICS), function(h) {
  angle <- 2 * pi * h * (seq_len(n) - 1) / dominant_period
  cbind(sin(angle), cos(angle))
}))
design_cycle <- cbind(1, cycle_terms)
design_both <- cbind(model.matrix(~ factor(time_batch) - 1), cycle_terms)

r_squared <- function(design, y) {
  1 - sum(qr.resid(qr(design), y)^2) / sum((y - mean(y))^2)
}

residual_summary <- function(design) {
  m_residual <- qr.resid(qr(design), m_t)
  residual_pca <- pc1_of(qr.resid(qr(design), Y_filled))
  residual_acf <- as.numeric(acf(m_residual, lag.max = ACF_MAX_LAG, plot = FALSE)$acf)[-1]
  tests <- stationarity_tests(m_residual)
  list(
    m_t_r_squared = r_squared(design, m_t),
    pc1_share = residual_pca$share,
    abs_cosine_with_full_pc1 = abs_cosine(residual_pca$loadings, full_pca$loadings),
    acf_lag1 = residual_acf[1],
    acf_at_rounded_period = residual_acf[round(dominant_period)],
    acf_lag50 = residual_acf[50],
    acf_lag100 = residual_acf[100],
    lags_inside_white_noise_bound = sum(abs(residual_acf) < acf_bound),
    ess_sqroot = ess_of(m_residual, "sqroot"),
    ess_cuberoot = ess_of(m_residual, "cuberoot"),
    tests = tests,
    kpss_long_lag = kpss_long(m_residual)
  )
}

results$cycle_extra <- list(
  dominant_period_windows = dominant_period,
  harmonics = CYCLE_HARMONICS,
  cycle_removed = residual_summary(design_cycle),
  batch_means_and_cycle_removed = residual_summary(design_both)
)

# ---- Step 6: Batch means standard errors ---------------------------------------

# With fewer batches than dimensions the covariance estimate is singular, so the
# 112 stocks are reduced to the first 3 PC scores, which gives 6 vech dimensions.
pca_full <- eigen(cor(Y_filled), symmetric = TRUE)
scores <- scale(Y_filled) %*% pca_full$vectors[, 1:3]
colnames(scores) <- c("PC1", "PC2", "PC3")

vech_pairs <- rbind(c(1, 1), c(1, 2), c(1, 3), c(2, 2), c(2, 3), c(3, 3))
vech_names <- apply(vech_pairs, 1, function(pair) paste0("s", pair[1], "s", pair[2]))
Y_vech <- apply(vech_pairs, 1, function(pair) scores[, pair[1]] * scores[, pair[2]])
colnames(Y_vech) <- vech_names

bm <- mcmcse::mcse.multi(Y_vech, method = "bm", blather = TRUE)
mv_ess <- mcmcse::multiESS(Y_vech, covmat = bm$cov)
mv_ess_default <- mcmcse::multiESS(Y_vech)
n_batches <- floor(n / bm$size)
stopifnot(n_batches > ncol(Y_vech))

results$batch_means <- list(
  dimension = ncol(Y_vech),
  vech_names = vech_names,
  batch_size = bm$size,
  n_batches = n_batches,
  batches_exceed_dimension = n_batches > ncol(Y_vech),
  method_used = bm$method,
  adjustment_used = bm$Adjustment_Used,
  message = bm$message,
  estimated_mean = setNames(as.list(as.numeric(bm$est)), vech_names),
  mcse = setNames(as.list(sqrt(diag(bm$cov) / n)), vech_names),
  sample_variance_of_vech = setNames(as.list(apply(Y_vech, 2, var)), vech_names),
  bm_covariance = bm$cov,
  bm_covariance_eigenvalues = eigen(bm$cov, symmetric = TRUE, only.values = TRUE)$values,
  multi_ess = mv_ess,
  multi_ess_default_call = mv_ess_default,
  multi_ess_share_of_n = mv_ess / n,
  min_ess_for_0_05_precision = mcmcse::minESS(p = ncol(Y_vech))
)

# ---- Save ----------------------------------------------------------------------

jsonlite::write_json(results, "report/stationarity_results.json", auto_unbox = TRUE,
                     digits = NA, pretty = TRUE, dataframe = "rows", matrix = "rowmajor")
cat("Done. Results saved to report/stationarity_results.json\n")
