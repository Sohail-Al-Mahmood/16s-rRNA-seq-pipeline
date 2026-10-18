# =========================================================
# LOAD LIBRARIES
# =========================================================

library(phyloseq)
library(qiime2R)
library(DESeq2)
library(vegan)
library(ggplot2)
library(ggpubr)
library(dplyr)
library(reshape2)

setwd("E:/thesis 2")

# =========================================================
# IMPORT DATA
# (tree removed: not needed without Faith's PD)
# =========================================================

physeq <- qza_to_phyloseq(
  features = "filtered-table-clustered-97.qza",
  taxonomy = "taxonomy.qza",
  metadata = "metadata.tsv"
)

physeq

# =========================================================
# FIX METADATA
# =========================================================

meta <- data.frame(sample_data(physeq))
meta <- meta[rownames(meta) %in% sample_names(physeq), ]

sample_data(physeq) <- sample_data(meta)

gc()

# =========================================================
# REMOVE ZERO-READ SAMPLES
# =========================================================

physeq <- prune_samples(sample_sums(physeq) > 0, physeq)

# Check original sequencing depth
summary(sample_sums(physeq))

quantile(
  sample_sums(physeq),
  probs = c(0, 0.01, 0.05, 0.10, 0.25, 0.50,
            0.75, 0.90, 0.95, 0.99, 1)
)

hist(
  sample_sums(physeq),
  breaks = 50,
  main = "Read depth before filtering",
  xlab = "Reads per sample"
)

# =========================================================
# REMOVE LOW-DEPTH SAMPLES
# ---------------------------------------------------------
# IMPORTANT:
# Do not rarefy using min(sample_sums(physeq)) if even one
# sample has extremely low reads, e.g. 1 read.
# Choose a biologically reasonable minimum depth.
# =========================================================

min_depth <- 1000   # <-- Change this based on your read-depth distribution

physeq_filtered <- prune_samples(
  sample_sums(physeq) >= min_depth,
  physeq
)

physeq_filtered

nsamples(physeq_filtered)
ntaxa(physeq_filtered)

summary(sample_sums(physeq_filtered))

# =========================================================
# RAREFY FOR DIVERSITY ANALYSIS
# =========================================================

rarefied <- rarefy_even_depth(
  physeq_filtered,
  sample.size = min(sample_sums(physeq_filtered)),
  replace = FALSE,
  rngseed = 123,
  verbose = TRUE
)

# Confirm rarefaction worked
summary(sample_sums(rarefied))

ntaxa(rarefied)
nsamples(rarefied)

# =========================================================
# EXTRACT OTU MATRIX
# Samples in rows, taxa in columns
# =========================================================

otu_mat <- as(otu_table(rarefied), "matrix")

if (taxa_are_rows(rarefied)) {
  otu_mat <- t(otu_mat)
}

dim(otu_mat)

# Confirm orientation
stopifnot(nrow(otu_mat) == nsamples(rarefied))
stopifnot(ncol(otu_mat) == ntaxa(rarefied))

# =========================================================
# ALPHA DIVERSITY METRICS
# =========================================================

alpha <- estimate_richness(
  rarefied,
  measures = c("Observed", "Chao1", "ACE", "Shannon", "Simpson")
)

summary(alpha)

# Good's coverage = 1 - (singletons / total reads)
singletons <- rowSums(otu_mat == 1)
total_reads <- rowSums(otu_mat)

goods_coverage <- 1 - (singletons / total_reads)

# Confirm row order before binding
stopifnot(identical(rownames(alpha), names(goods_coverage)))

# Combine all metrics with metadata
alpha_df <- cbind(
  alpha,
  Goods = goods_coverage,
  data.frame(sample_data(rarefied))
)

# Check columns
colnames(alpha_df)

# Check alpha diversity values
summary(alpha_df[, c("Observed", "Chao1", "ACE",
                     "Shannon", "Simpson", "Goods")])

# Ensure grouping variables are factors
alpha_df$Group <- factor(alpha_df$Group)
alpha_df$BioProject <- factor(alpha_df$BioProject)

table(alpha_df$Group)
table(alpha_df$BioProject)

# =========================================================
# LINEAR MODELS: BATCH-ADJUSTED
# =========================================================

# Helper function to fit models safely
fit_alpha_lm <- function(data, response) {
  df <- data[complete.cases(data[, c(response, "Group", "BioProject")]), ]
  
  df$Group <- droplevels(df$Group)
  df$BioProject <- droplevels(df$BioProject)
  
  cat("\n----------------------------------------\n")
  cat("Model:", response, "\n")
  cat("Samples used:", nrow(df), "\n")
  cat("Group levels:\n")
  print(table(df$Group))
  cat("BioProject levels:\n")
  print(table(df$BioProject))
  cat("----------------------------------------\n\n")
  
  model <- lm(
    as.formula(paste(response, "~ Group + BioProject")),
    data = df
  )
  
  summary(model)
}

lm_Observed <- fit_alpha_lm(alpha_df, "Observed")
lm_Chao1    <- fit_alpha_lm(alpha_df, "Chao1")
lm_ACE      <- fit_alpha_lm(alpha_df, "ACE")
lm_Shannon  <- fit_alpha_lm(alpha_df, "Shannon")
lm_Simpson  <- fit_alpha_lm(alpha_df, "Simpson")
lm_Goods    <- fit_alpha_lm(alpha_df, "Goods")

# =========================================================
# THESIS THEME
# =========================================================

theme_thesis <- function() {
  theme_classic(
    base_size = 11,
    base_family = "Times New Roman"
  ) +
    theme(
      axis.title = element_text(
        face = "plain",
        size = 11
      ),
      axis.text = element_text(
        face = "plain",
        size = 10
      ),
      axis.text.x = element_text(
        family = "Times New Roman",
        face = "plain",
        size = 10
      ),
      axis.text.y = element_text(
        family = "Times New Roman",
        face = "plain",
        size = 10
      ),
      legend.position = "none",
      panel.border = element_rect(
        colour = "black",
        fill = NA,
        linewidth = 0.7
      )
    )
}

# =========================================================
# ALPHA DIVERSITY PLOT FUNCTION
# =========================================================

plot_alpha_thesis <- function(df,
                              y,
                              file,
                              ylim_vals = NULL,
                              ylab_text = y) {
  
  p <- ggplot(
    df,
    aes(
      x = Group,
      y = .data[[y]],
      colour = Group
    )
  ) +
    geom_boxplot(
      outlier.shape = NA,
      width = 0.5,
      linewidth = 0.8,
      fill = "white"
    ) +
    stat_boxplot(
      geom = "errorbar",
      width = 0.30,
      linewidth = 0.8
    ) +
    labs(
      x = NULL,
      y = ylab_text
    ) +
    theme_thesis() +
    scale_y_continuous(
      expand = expansion(mult = c(0.02, 0.15))
    )
  
  if (!is.null(ylim_vals)) {
    p <- p + coord_cartesian(ylim = ylim_vals)
  }
  
  ggsave(
    filename = file,
    plot = p,
    width = 2.5,
    height = 2.5,
    units = "in",
    dpi = 600,
    compression = "lzw"
  )
  
  return(p)
}

# =========================================================
# ALPHA DIVERSITY PLOTS
# =========================================================

plot_alpha_thesis(
  alpha_df,
  "Observed",
  "Observed_plot.tiff",
  c(0, max(alpha_df$Observed) * 1.15),
  "Observed species"
)

plot_alpha_thesis(
  alpha_df,
  "Chao1",
  "Chao1_plot.tiff",
  c(0, max(alpha_df$Chao1, na.rm = TRUE) * 1.15),
  "Chao1 richness"
)

plot_alpha_thesis(
  alpha_df,
  "ACE",
  "ACE_plot.tiff",
  c(0, max(alpha_df$ACE, na.rm = TRUE) * 1.15),
  "ACE richness"
)

plot_alpha_thesis(
  alpha_df,
  "Shannon",
  "Shannon_plot.tiff",
  c(0, max(alpha_df$Shannon, na.rm = TRUE) * 1.15),
  "Shannon diversity"
)

plot_alpha_thesis(
  alpha_df,
  "Simpson",
  "Simpson_plot.tiff",
  c(0, 1),
  "Simpson index"
)

plot_alpha_thesis(
  alpha_df,
  "Goods",
  "Goods_plot.tiff",
  c(min(alpha_df$Goods, na.rm = TRUE) * 0.98, 1),
  "Good's coverage"
)