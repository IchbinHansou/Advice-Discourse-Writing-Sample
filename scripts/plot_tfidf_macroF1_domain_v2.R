this_file <- tryCatch(normalizePath(rstudioapi::getActiveDocumentContext()$path),
                      error = function(e) "")
if (this_file == "") {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  this_file <- sub("^--file=", "", file_arg)
  this_file <- normalizePath(this_file)
}

root <- normalizePath(file.path(dirname(this_file), ".."))
data_dir <- file.path(root, "data_new")

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(tidyr)
  library(stringr)
})

# =========================
# Helpers: pick existing file
# =========================
pick_existing <- function(dir, candidates) {
  full <- file.path(dir, candidates)
  hit <- full[file.exists(full)][1]
  if (is.na(hit) || length(hit) == 0) {
    stop(
      "No candidate file found in: ", dir, "\n",
      "Tried:\n  - ", paste(candidates, collapse = "\n  - ")
    )
  }
  hit
}

# =========================
# Label canonicalization
# =========================
LABELS_CANON <- c("ADVICE", "DISCLOSURE")   # for computation
LABELS_PLOT  <- c("Advice", "Disclosure")   # for display

canon_label <- function(x) {
  x <- toupper(trimws(as.character(x)))
  x <- ifelse(x == "STORY", "DISCLOSURE", x)  # key: internal STORY -> paper DISCLOSURE
  x
}

macro_f1 <- function(gold, pred) {
  f1s <- c()
  for (lab in LABELS_CANON) {
    tp <- sum(gold == lab & pred == lab)
    fp <- sum(gold != lab & pred == lab)
    fn <- sum(gold == lab & pred != lab)
    prec <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
    rec  <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
    f1   <- ifelse(prec + rec == 0, 0, 2 * prec * rec / (prec + rec))
    f1s <- c(f1s, f1)
  }
  mean(f1s)
}

paired_bootstrap <- function(df, B = 2000, seed = 1004) {
  set.seed(seed)
  n <- nrow(df)
  stats <- replicate(B, {
    idx <- sample.int(n, n, replace = TRUE)
    d <- df[idx, ]
    m_raw <- macro_f1(d$gold_label, d$pred_raw)
    m_cln <- macro_f1(d$gold_label, d$pred_clean)
    c(raw = m_raw, clean = m_cln, delta = m_cln - m_raw)
  })
  as.data.frame(t(stats))
}

summarize_boot <- function(boot_df) {
  boot_df %>%
    summarize(
      raw_mean = mean(raw),
      raw_lo = quantile(raw, 0.025),
      raw_hi = quantile(raw, 0.975),
      clean_mean = mean(clean),
      clean_lo = quantile(clean, 0.025),
      clean_hi = quantile(clean, 0.975),
      delta_mean = mean(delta),
      delta_lo = quantile(delta, 0.025),
      delta_hi = quantile(delta, 0.975),
      p_two_sided = min(1, 2 * min(mean(delta <= 0), mean(delta >= 0)))
    )
}

make_long <- function(sum_df, domain_name) {
  tibble(
    domain = domain_name,
    condition = c("raw", "clean"),
    mean = c(sum_df$raw_mean, sum_df$clean_mean),
    lo = c(sum_df$raw_lo, sum_df$clean_lo),
    hi = c(sum_df$raw_hi, sum_df$clean_hi),
    delta_mean = sum_df$delta_mean,
    delta_lo = sum_df$delta_lo,
    delta_hi = sum_df$delta_hi,
    p = sum_df$p_two_sided
  )
}

# =========================
# Main: pick your current files
# =========================
raw_candidates <- c(
  "preds_tfidf_raw_gold.csv",
  "preds_tfidf_raw_norm_gold.csv",
  "preds_tfidf_raw_norm_FROZEN_gold.csv",
  "preds_tfidf_rawnorm_gold.csv"
)

clean_candidates <- c(
  "preds_tfidf_clean_gold.csv",
  "preds_tfidf_cleaned_v1_gold.csv",
  "preds_tfidf_clean_v1_gold.csv",
  "preds_tfidf_clean_v1_FROZEN_gold.csv"
)

raw_path <- pick_existing(data_dir, raw_candidates)
cln_path <- pick_existing(data_dir, clean_candidates)

cat("Using RAW file:  ", raw_path, "\n")
cat("Using CLEAN file:", cln_path, "\n")

read_preds <- function(path) {
  df <- read_csv(path, show_col_types = FALSE)
  
  # accept either pred or pred_label
  if ("pred_label" %in% names(df)) {
    df <- df %>% rename(pred = pred_label)
  }
  if (!("pred" %in% names(df))) stop("Missing column 'pred' (or 'pred_label') in: ", path)
  if (!all(c("doc_id", "domain", "gold_label") %in% names(df))) {
    stop("Missing required columns (doc_id, domain, gold_label) in: ", path)
  }
  
  df %>%
    mutate(
      doc_id = as.character(doc_id),
      domain = tolower(as.character(domain)),
      gold_label = canon_label(gold_label),
      pred = canon_label(pred)
    )
}

raw <- read_preds(raw_path)
cln <- read_preds(cln_path)

# join paired by doc_id
df <- raw %>%
  select(doc_id, domain, gold_label, pred_raw = pred) %>%
  inner_join(
    cln %>% select(doc_id, pred_clean = pred),
    by = "doc_id"
  )

# keep binary only
df <- df %>% filter(gold_label %in% LABELS_CANON)

# overall + per-domain
domains <- c("overall", sort(unique(df$domain)))

plot_df <- bind_rows(lapply(domains, function(dom) {
  dsub <- if (dom == "overall") df else df %>% filter(domain == dom)
  boot <- paired_bootstrap(dsub, B = 2000, seed = 1004)
  sumd <- summarize_boot(boot)
  make_long(sumd, dom)
}))

# pretty facet labels
plot_df <- plot_df %>%
  mutate(
    domain = factor(domain, levels = domains),
    domain = fct_recode(domain,
                        "Overall" = "overall",
                        "AITA" = "aita",
                        "Confession" = "confession",
                        "RA" = "ra"),
    condition = factor(condition, levels = c("raw", "clean"),
                       labels = c("Raw", "Clean"))
  )

# per-domain annotation label (Delta + p) with per-facet y
ann <- plot_df %>%
  group_by(domain) %>%
  summarize(
    delta_mean = first(delta_mean),
    delta_lo = first(delta_lo),
    delta_hi = first(delta_hi),
    p = first(p),
    y = max(hi) + 0.02,
    .groups = "drop"
  ) %>%
  mutate(
    p_fmt = ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)),
    lab = sprintf("Delta=%.3f [%.3f, %.3f], p=%s", delta_mean, delta_lo, delta_hi, p_fmt)
  )

p <- ggplot(plot_df, aes(x = condition, y = mean, color = condition)) +
  geom_errorbar(aes(ymin = lo, ymax = hi),
                position = position_dodge(width = 0.45),
                linewidth = 0.4, width = 0.15) +
  geom_point(position = position_dodge(width = 0.45), size = 2.6) +
  facet_wrap(~domain, nrow = 1) +
  geom_text(data = ann, aes(x = 1.5, y = y, label = lab),
            inherit.aes = FALSE, size = 3.2) +
  coord_cartesian(ylim = c(0.35, 0.80)) +
  labs(
    x = NULL,
    y = "Macro-F1 (95% paired bootstrap CI)",
    title = "TF–IDF baseline on gold (binary)",
    subtitle = "Raw vs artifact-removed text (Clean)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold")
  )

out_pdf <- file.path(data_dir, "figure_tfidf_macroF1_domain_facet.pdf")
out_png <- file.path(data_dir, "figure_tfidf_macroF1_domain_facet.png")

ggsave(out_pdf, p, width = 11, height = 3.6)
ggsave(out_png, p, width = 11, height = 3.6, dpi = 300)

cat("Wrote: ", out_pdf, "\n")
cat("Wrote: ", out_png, "\n")
