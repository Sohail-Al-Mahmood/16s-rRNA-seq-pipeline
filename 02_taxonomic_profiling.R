# LOAD LIBRARIES
library(phyloseq)
library(qiime2R)
library(DESeq2)
library(vegan)
library(ggplot2)
library(ggpubr)
library(dplyr)
library(picante)
library(reshape2)

setwd("E:/Temp/Thesis/datasets7/diversity")

# IMPORT DATA

physeq <- qza_to_phyloseq(features = "table-filtered-1.qza",tree = "rooted-tree.qza", taxonomy = "taxonomy.qza", metadata = "metadata.tsv")

# FIX METADATA
meta <- data.frame(sample_data(physeq))
meta <- meta[rownames(meta) %in% sample_names(physeq), ]
sample_data(physeq) <- sample_data(meta)
gc()

# REMOVE ZERO SAMPLES
physeq <- prune_samples(sample_sums(physeq) > 0, physeq)

# RAREFY (for diversity only)
rarefy <- rarefy_even_depth(
  physeq,
  sample.size = min(sample_sums(physeq)),
  replace = FALSE,
  rngseed = 123,
  verbose = TRUE
)

# From rarefied phyloseq object
otu <- as(otu_table(rarefy), "matrix")

#Alpha diversity (batch-adjusted)
alpha <- estimate_richness(rarefy,
                           measures = c("Shannon","Chao1"))

alpha_df <- cbind(alpha, as(sample_data(rarefy), "data.frame"))




# Faith PD (phylogenetic diversity)
tree <- phy_tree(rarefy)
# OTU matrix extract (samples in rows, taxa in columns)
otu_mat <- as(otu_table(rarefy), "matrix")
if(taxa_are_rows(rarefy)){
  otu_mat <- t(otu_mat)
}
faith_pd <- pd(otu_mat, tree, include.root = TRUE)
faith <- data.frame(FaithPD = faith_pd$PD, row.names = rownames(alpha))

# Combine all metrics with metadata
alpha_df2 <- cbind(alpha,faith, data.frame(sample_data(rarefy)))

# Linear model (batch-adjusted)
lm_Shannon=lm(Shannon ~ Group + BioProject, data = alpha_df)
summary(lm_Shannon)
lm_chao1 <- lm(Chao1 ~ Group + BioProject, data = alpha_df)
summary(lm_chao1)
lm_FaithPD=lm(FaithPD ~ Group + BioProject, data = alpha_df2)
summary(lm_FaithPD)


### Alpha diversity plot
## Chao1


p1 <- ggplot(alpha_df, aes(x = Group, y = Chao1, colour = Group)) +
  geom_boxplot(notch = TRUE, outlier.shape = NA, outlier.colour = "#D2691E", lwd = 0.6) + ylim(150,800)+
  stat_boxplot(geom = "errorbar", width = 0.1, lwd = 0.6) +
  scale_fill_manual(values = c("white", "white")) +
  ylab("Chao1") +
  xlab("") +
  theme_classic() +
  theme(
    axis.text.x = element_text(angle = 360, hjust = 1),
    legend.position = "none"
  ) +
  guides(fill = "none")
tiff("Chao1_plot.tiff", width = 2, height = 2, units = "in", res = 300)
print(p1)

dev.off()

#Shannon


p2=ggplot(alpha_df, aes(x = Group, y = Shannon, colour = Group)) +ylab("Shanon")+
  geom_boxplot(notch = TRUE, outlier.shape = NA, lwd = 0.6) + ylim(2.5,6.5)+
  stat_boxplot(geom = "errorbar", width = 0.1, lwd = 0.6) +
  scale_fill_manual(values = c("white", "white")) +
  xlab("") + ylab("Shanon") +
  theme_classic() +
  theme(
    axis.text.x = element_text(angle = 360, hjust = 1),
    legend.position = "none"
  ) +
  guides(fill = "none")

tiff("Shannon_plot.tiff", width = 2, height = 2, units = "in", res = 300)
print(p2)
dev.off()



#FaithPD



p3=ggplot(alpha_df2, aes(x = Group, y = FaithPD, colour = Group)) + ylab("FaithPD")+
  geom_boxplot(notch = TRUE, outlier.shape = NA, lwd = 0.6) + ylim(10,50)+
  stat_boxplot(geom = "errorbar", width = 0.1, lwd = 0.6) +
  scale_fill_manual(values = c("white", "white")) +
  xlab("") + ylab("FaithPD") +
  theme_classic() +
  theme(
    axis.text.x = element_text(angle = 360, hjust = 1),
    legend.position = "none"
  ) +
  guides(fill = "none")
tiff("FaithPD_plot.tiff", width = 2, height = 2, units = "in", res = 300)
print(p3)
dev.off()




## Beta diversity (batch-adjusted)
set.seed(123)



## 1. Relative abundance

relab_physeq <- transform_sample_counts(rarefy, function(x) x / sum(x))


## 2. FILTER for UniFrac stability

#rarefy_filt <- prune_taxa(taxa_sums(rarefy) > 5, rarefy)


ntaxa(relab_physeq)
length(tree$tip.label)


## 3. DISTANCE MATRICES


# Bray-Curtis (relative abundance OK)
dist_bray <- phyloseq::distance(relab_physeq, method = "bray")


## 4. SAMPLE METADATA## -

samples_df <- data.frame(sample_data(relab_physeq))



## 5. PERMANOVA (batch-adjusted)

adonis2(dist_bray ~ Group + BioProject, data = samples_df)


## 6. PCoA - Bray Curtis

ord_bray <- ordinate(relab_physeq, method = "PCoA", distance = dist_bray)
eig <- ord_bray$values$Eigenvalues

var_explained <- eig / sum(eig) * 100

PC1 <- round(var_explained[1], 2)
PC2 <- round(var_explained[2], 2)

windowsFonts(Times=windowsFont("Times New Roman"))
p4 <- plot_ordination(relab_physeq, ord_bray, color = "Group", shape = "Group") +
  geom_point(size = 1,alpha = 0.7) +
  scale_colour_manual(values = c('blue','red')) +
  stat_ellipse(aes(group = Group)) +
  theme_test() +
  xlab(paste0("PCoA1 (", PC1, "%)")) +
  ylab(paste0("PCoA2 (", PC2, "%)")) + theme(
    axis.title.x = element_text(size = 8),
    axis.title.y = element_text(size = 8)
  )+
  font("x.text", face=1,size=8)+
  font("y.text", face=1,size=8)+font("legend.text", size=8, face=2)+font("legend.title", size=8,face=2)+
  theme(legend.text = element_text(family="Times"))

ggsave("PCoA_bray.tiff", p4, width=2.6, height=2, dpi=300)
print(p4)
dev.off()




library(phyloseq)
library(qiime2R)
library(ggplot2)
library(dplyr)
library(MMUPHin)



# =====================================================
# MMUPHin BATCH CORRECTION
# =====================================================

otu <- as(otu_table(physeq), "matrix")

if(!taxa_are_rows(physeq)){
  otu <- t(otu)
}
meta <- data.frame(sample_data(physeq))

adj <- adjust_batch(
  feature_abd = otu,
  batch = "BioProject",
  covariates = "Group",
  data = meta,
  control = list(verbose = TRUE)
)
otu_adj <- adj$feature_abd_adj

physeq_adj <- phyloseq(
  otu_table(otu_adj, taxa_are_rows = TRUE),
  tax_table(physeq),
  sample_data(physeq)
)

# =====================================================
# FILTERING
# =====================================================

physeq_filtered <- filter_taxa(
  physeq_adj,
  function(x) sum(x > 0) > (0.1 * length(x)),
  TRUE
)

# =====================================================
# TAXONOMIC LEVELS
# =====================================================

ranks <- c("Phylum","Family","Genus")

for(rank in ranks){
  
  cat("Processing:", rank, "\n")
  
  tax_glommed <- tax_glom(
    physeq_filtered,
    taxrank = rank
  )
  
  tax_glommed <- filter_taxa(
    tax_glommed,
    function(x) sum(x) > 5,
    TRUE
  )
  
  merged <- merge_samples(
    tax_glommed,
    "Group"
  )
  
  # ==========================================
  # EXPORT OTU TABLE
  # ==========================================
  
  otu_export <- as(
    otu_table(merged),
    "matrix"
  )
  
  if(!taxa_are_rows(merged)){
    otu_export <- t(otu_export)
  }
  
  taxa_export <- as(
    tax_table(merged),
    "matrix"
  )
  
  taxa_names_vec <- as.character(
    taxa_export[, rank]
  )
  
  taxa_names_vec[is.na(taxa_names_vec)] <- "Unclassified"
  
  taxa_names_vec <- make.unique(
    taxa_names_vec
  )
  
  rownames(otu_export) <- taxa_names_vec
  
  write.csv(
    otu_export,
    paste0(
      "otu_",
      tolower(rank),
      "_batch_adjusted.csv"
    )
  )
  
  write.csv(
    taxa_export,
    paste0(
      "taxa_",
      tolower(rank),
      "_batch_adjusted.csv"
    )
  )
  
  # ==========================================
  # RELATIVE ABUNDANCE
  # ==========================================
  
  rel_abund <- transform_sample_counts(
    tax_glommed,
    function(x) x/sum(x)*100
  )
  
  rel_df <- psmelt(rel_abund) %>%
    group_by(Group, !!sym(rank)) %>%
    summarise(
      Abundance = mean(
        Abundance,
        na.rm = TRUE
      ),
      .groups = "drop"
    )
  
  top_taxa <- rel_df %>%
    group_by(!!sym(rank)) %>%
    summarise(
      Total = sum(Abundance)
    ) %>%
    arrange(desc(Total)) %>%
    slice(1:15) %>%
    pull(!!sym(rank))
  
  rel_df$Taxa <- ifelse(
    rel_df[[rank]] %in% top_taxa,
    rel_df[[rank]],
    "Other"
  )
  
  tiff(
    paste0(
      tolower(rank),
      "_top15_batch_adjusted.tiff"
    ),
    height = 4,
    width = 6,
    units = "in",
    res = 300
  )

  
 p5= ggplot(rel_df, aes(x=Group, y=Abundance, fill=Taxa)) +
    geom_bar(stat="identity", position="fill", width=0.7) +
    scale_fill_manual(values = rainbow(16)) +
    theme_classic() +
   guides(fill = guide_legend(ncol = 5)) + 
    coord_flip() +
    theme( axis.text.y = element_text(size = 6),   # 👈 bacteria names font size
           axis.text.x = element_text(size = 6),
           legend.title = element_blank(),
           legend.key.size = unit(0.3, "cm"),
           legend.position = "bottom",
           legend.text = element_text(family="Times New Roman", size=6),
           legend.text.align = 0,
           axis.line = element_line(size = 0.23),
           axis.ticks = element_line(size = .34),
           axis.ticks.length = unit(.08, "cm"))
 
 tiff(paste0(tolower(rank), "_top15_batch.tiff"),
      height = 2.4, width = 3.8, units="in", res=300, bg="white")
  print(p5)
  
  dev.off()
  
  cat("Done:", rank, "\n")
}


library(patchwork)

# Top row
top_row <- p1 + p2 + p3

# Bottom row
bottom_row <- p4 + p5 +
  plot_layout(widths = c(1, 1.4))

# Final figure
final_plot <-
  top_row /
  bottom_row +
  plot_annotation(
    tag_levels = "a"
  ) &
  theme(
    plot.tag = element_text(
      face = "bold",
      size = 14
    )
  )

# Preview
final_plot

# Save
ggsave(
  "Combined_Figure.tiff",
  final_plot,
  width = 11,
  height = 9,
  units = "in",
  dpi = 600,
  compression = "lzw",
  bg = "white"
)


