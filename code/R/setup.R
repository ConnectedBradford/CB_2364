options(warn = -1)

pkgs <- c(
  "dplyr", "sjPlot", "tidyr", "car", "forcats", "broom", 
  "interactions", "ggeffects", "ggplot2", "gtsummary", "lubridate",
  "gt", "webshot2", "Cairo", "performance", "labelled", "flextable",
  "officer", "magrittr", "gtExtras", "caret", "purrr", "tibble",
  "pROC"
)

invisible(lapply(pkgs, function(p) {
  suppressPackageStartupMessages(
    library(p, character.only = TRUE, warn.conflicts = FALSE)
  )
}))

options(repr.plot.width = 10, repr.plot.height = 8, repr.plot.res = 300)
options(jupyter.plot_mimetypes = "image/svg+xml")
options(device = function(...) Cairo::CairoPNG(...))

save_docs <- FALSE

if (use_mock_set) {
  results_dir <- file.path(getwd(), "Results_toy")
} else {
  results_dir <- file.path(getwd(), "Results")
}

if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
}

labels_map <- list(
  gender        = "Sex",
  ethnicity     = "Ethnicity",      
  IMD           = "IMD (quintile)", 
  Has_HV        = "Has 2y HV",
  age_months    = "Age at FSP (months)", 
  ASQ_GLD       = "ASQ-3 GLD",
  ASQ_FGLD_dom  = "ASQ-3 GLD (Domain)",
  ASQ_Composite = "ASQ-3 Composite Score"
)