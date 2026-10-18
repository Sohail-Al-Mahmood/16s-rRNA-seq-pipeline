# 16S rRNA-Seq Pipeline for Microbiome-Based CRC Classification

This repository contains a reproducible analysis workflow for 16S rRNA sequencing data focused on colorectal cancer (CRC) vs. healthy control classification. The pipeline combines microbiome diversity analysis, taxonomic profiling, machine learning, and model explainability using SHAP.

The project is organized as a research workflow rather than a packaged software library, and the scripts are designed to run in a data directory containing processed microbiome tables and metadata.

## Project overview

The workflow includes:

- Alpha and beta diversity analysis using R and phyloseq
- Taxonomic profiling and community-level summaries
- Imbalanced-class machine learning on genus-level CLR-transformed features
- Stacking ensemble classification and evaluation
- SHAP-based interpretation to identify the most important microbial genera

## Repository structure

- `01_alpha_beta_diversity.R` — alpha diversity, beta diversity, rarefaction, and plotting workflow
- `02_taxonomic_profiling.R` — taxonomic profiling and diversity-related summaries
- `03_ml_imbalanced.py` — baseline models trained on the original imbalanced dataset
- `04_ml_stacking_pipeline.py` — stacked ensemble pipeline with SMOTE applied only within CV folds
- `05_shap_analysis.py` — SHAP analysis of the final stack model and generation of importance plots

## Expected input data

### Python ML scripts
The machine-learning scripts expect a data directory containing:

- `metadata.csv` with at least:
  - `SampleID`
  - `Group` (for example `tumor`, `crc`, `healthy`, `normal`, `control`)
- `final_genus_list.txt` — one genus name per line
- One of the following:
  - `genus_clr.csv` with rows = samples and columns = genera, or
  - `clr_data.csv` + `final.csv` where ASV-level CLR values are summed per genus

### R diversity scripts
The R scripts assume QIIME 2-style inputs such as:

- `filtered-table-clustered-97.qza`
- `taxonomy.qza`
- `metadata.tsv`
- optional rooted tree or other phylogenetic files depending on the analysis

## Setup

### Python environment
Install the required Python packages:

```bash
pip install pandas numpy scikit-learn imbalanced-learn lightgbm shap matplotlib
```

### R environment
Install the required R packages:

```r
install.packages(c(
  "phyloseq",
  "qiime2R",
  "DESeq2",
  "vegan",
  "ggplot2",
  "ggpubr",
  "dplyr",
  "reshape2",
  "picante"
))
```

## Usage

### Diversity analysis

```bash
Rscript 01_alpha_beta_diversity.R
Rscript 02_taxonomic_profiling.R
```

### Machine learning

```bash
python 03_ml_imbalanced.py --data-dir data --out-dir results_imbalanced
python 04_ml_stacking_pipeline.py --data-dir data --out-dir results_stacked
```

### SHAP analysis

Run the script from a directory that contains the required input files:

```bash
python 05_shap_analysis.py
```

## Outputs

Depending on the script, the project generates:

- diversity plots and summary tables
- taxonomic summary outputs
- ML performance metrics for CRC vs. healthy classification
- ROC/AUC, F1, precision, recall, specificity, and average precision summaries
- SHAP summary plots and CSV files ranking genera by feature importance

## Notes

- This repository expects the raw microbiome inputs to be prepared before running the analysis scripts.
- Sample identifiers must be consistent across metadata and abundance tables.
- The machine-learning scripts use a genus-level feature matrix and treat disease labels in a binary manner (`tumor` vs. `normal`).
- For reproducibility, keep the same feature list, metadata, and seed settings across runs.

## Typical project workflow

1. Prepare Qiime2 or processed taxonomic abundance tables.
2. Run diversity analysis scripts.
3. Build the genus-level feature matrix and metadata alignment.
4. Train baseline models and the stacked ensemble.
5. Interpret the final model with SHAP to identify key genera associated with CRC.

## License

This project does not include a formal license file. If you plan to reuse this code, confirm the repository owner's licensing requirements before redistribution or publication.
