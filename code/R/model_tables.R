get_perf <- function(x) {
  perf <- performance::model_performance(x) %>% as.data.frame()
  if ("n_Obs" %in% names(perf)) perf$nobs <- perf$n_Obs
  if (!"nobs" %in% names(perf)) perf$nobs <- stats::nobs(x)
  r2_val <- performance::r2(x)
  
  if (inherits(x, "glm")) {
    perf$R2_Tjur <- as.numeric(r2_val[[1]])
    perf$R2_adjusted <- NA
  } else {
    perf$R2_Tjur <- NA
    perf$R2_adjusted <- if("R2_adjusted" %in% names(perf)) perf$R2_adjusted else as.numeric(r2_val[[1]])
    perf$R2 <- if("R2" %in% names(perf)) perf$R2 else as.numeric(r2_val[[1]])
  }
  
  return(perf)
}


make_reg_tbl <- function(m, ref_symbol = "Reference") {
  is_glm <- inherits(m, "glm")

  tbl_regression(
    m,
    exponentiate = is_glm,
    conf.int = TRUE,
    conf.level = 0.95
    # add_estimate_to_reference_rows = TRUE
  ) %>%
    add_glance_table(
      glance_fun = get_perf,
      include = c("nobs", "R2_Tjur", "R2_adjusted"),
      label = list(
        nobs ~ "No. Obs.",
        R2_Tjur ~ "R² Tjur",
        R2_adjusted ~ "R² (Adj.)"
      )
    ) %>%
    modify_table_body(~ .x %>%
      dplyr::filter(
        !(row_type == "glance_statistic" &
          dplyr::if_all(
            tidyselect::where(is.numeric),
            ~ is.na(.x)
          ))
      )
    ) %>%
    modify_table_styling(
      # columns = label,
      columns = everything(),
      rows = row_type == "glance_statistic",
      text_format = "bold"
    ) %>%
    bold_labels() %>%
    bold_p() %>%
    modify_fmt_fun(
      p.value = function(x) style_pvalue(x, digits = 3)
    ) %>%
    modify_header(
      label    ~ "**Predictors**",
      estimate ~ ifelse(is_glm, "**OR (95% CI)**", "**$\\beta$ (95% CI)**"),
      p.value  ~ "***p***"
    ) %>%
    modify_table_styling(
      columns = c(estimate, ci),
      rows = row_type %in% c("label", "level") & !dplyr::coalesce(reference_row, FALSE) &
         !is.na(ci) & ci != "",
      cols_merge_pattern = "{estimate} ({conf.low} - {conf.high})"
    ) %>%
    modify_table_styling(columns = ci, hide = TRUE) %>%
    modify_table_styling(
      columns = estimate,
      rows = reference_row,
      text_format = "italic",
      missing_symbol = ref_symbol,
      align = "center"
    )
}


pick_mid_model <- function(models) {
  n <- length(models)
  if (n == 0) return(NA_integer_)

  mid <- (n + 1) / 2

  cand <- which(vapply(
    models,
    function(m) length(stats::coef(m)) - 1 > 1,
    logical(1)
  ))

  if (length(cand) == 0) return(NA_integer_)

  cand[which.min(abs(cand - mid))]
}


make_merged_table <- function(models,
                              dv_labels,
                              sort_predictors = TRUE) {
  # bold spanners
  dv_labels <- paste0("**", dv_labels, "**")

  # choose which table shows "Reference"
  fill_idx <- pick_mid_model(models)

  # build individual tables
  tbl_list <- vector("list", length(models))
  for (i in seq_along(models)) {
    ref_symbol <- if (!is.na(fill_idx) && i == fill_idx) "Reference" else ""
    tbl_list[[i]] <- make_reg_tbl(models[[i]], ref_symbol = ref_symbol)
  }

  # merge
  out <- gtsummary::tbl_merge(tbl_list, tab_spanner = dv_labels)

  # optional sorting
  if (isTRUE(sort_predictors)) {
  asq_vars <- c("ASQ-3")

  out <- out %>%
    gtsummary::modify_table_body(~ .x %>%
      dplyr::mutate(.orig_order = dplyr::row_number()) %>%
      dplyr::arrange(
        row_type == "glance_statistic",  
        dplyr::if_else(
          row_type == "glance_statistic",
          .orig_order, 
          dplyr::case_when(
            grepl("^ASQ", variable) ~ 0,                                
            TRUE ~ as.numeric(factor(variable)) + 1                   
          )
        )
      ) %>%
      dplyr::select(-.orig_order)
    )
}
  out
}
                              
make_gt <- function(tbl, padding_left_px = 50, label_col = "label") {
  tbl %>%
    # gtsummary::modify_table_body(~ .x %>%
    #   dplyr::mutate(
    #     label = dplyr::if_else(
    #       row_type == "label" & grepl("^[a-z]", label),
    #       stringr::str_to_sentence(label), 
    #       label
    #     )
    #   )
    # ) %>%
    as_gt() %>%
    gt::tab_style(
      style = gt::cell_text(indent = gt::px(padding_left_px)),
      locations = list(
        gt::cells_body(columns = -gt::all_of(label_col)),
        gt::cells_column_labels(columns = -gt::all_of(label_col))
      )
    )
}


