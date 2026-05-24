prepare_df <- function(df,
                       factor_cols = c("gender", "ethnicity_group", "IMD19_quintile", "Has_HV", "Has_ASQ", "ASQ_FGLD_dom", "ASQ_GLD"), # just default setting
                       numeric_cols = NULL,
                       reference_levels = list(
                         ethnicity_group = "White British",
                         gender = "Female",
                         IMD19_quintile = "5",
                         Has_HV = "0",
                         Has_ASQ = "0",
                         ASQ_FGLD_dom = "0",
                         ASQ_GLD = "0"
                       )) {
  
  if ("gender" %in% names(df)) {
    df$gender[df$gender == "Unknown/Other"] <- NA
  }
  
  if (!is.null(numeric_cols)) {
    existing_num_cols <- numeric_cols[numeric_cols %in% names(df)]
    
    df[existing_num_cols] <- lapply(df[existing_num_cols], function(x) {
      suppressWarnings(as.numeric(as.character(x)))
    })
    
    factor_cols <- setdiff(factor_cols, existing_num_cols)
  }
  
  existing_factor_cols <- factor_cols[factor_cols %in% names(df)]
  df[existing_factor_cols] <- lapply(df[existing_factor_cols], factor)
  
  for (col in names(reference_levels)) {
    if (col %in% existing_factor_cols) {
      ref <- reference_levels[[col]]
      lev <- levels(df[[col]])
      if (!ref %in% lev) {
        warning(sprintf(
          "Reference '%s' is not a level of %s. Existing levels: %s",
          ref, col, paste(lev, collapse = ", ")
        ))
        next
      }
      df[[col]] <- relevel(df[[col]], ref = ref)
    }
  }

  cat("Reference levels used in this logit model:\n")
  invisible(
    sapply(existing_factor_cols, function(col) {
      cat(sprintf("  %s reference: %s\n", col, levels(df[[col]])[1]))
    })
  )
  cat("\n")
  
  rename_map <- list(
    ethnicity = "ethnicity_group",
    IMD = "IMD19_quintile",
    age_months = "age_fsp_months" 
   )

  for (new_name in names(rename_map)) {
    old_name <- rename_map[[new_name]]
    if (old_name %in% names(df)) {
      df <- df %>% rename(!!new_name := all_of(old_name))
    }
  }
  
  existing_labels <- labels_map[names(labels_map) %in% names(df)]

  df <- df %>% 
    set_variable_labels(.labels = existing_labels)
  
  return(df)
}
                       
droplevels_keep_labels <- function(df) {
  labels <- lapply(df, function(x) attr(x, "label"))

  df2 <- droplevels(df)

  for (nm in names(labels)) {
    if (!is.null(labels[[nm]])) {
      attr(df2[[nm]], "label") <- labels[[nm]]
    }
  }

  df2
}