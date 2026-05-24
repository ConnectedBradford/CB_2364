make_outputs <- function(results_dir, name) {
  list(
    html = file.path(results_dir, paste0(name, "_fig.html")),
    rds  = file.path(results_dir, paste0(name, "_model.rds")),
    rds_gt = file.path(results_dir, paste0(name, "_fig_gt.rds")),
    docx = file.path(results_dir, paste0(name, "_fig.docx"))
  )
}

safe_docx <- function(gt_tbl, filename) {
  tryCatch(
    {
      gt::gtsave(gt_tbl, filename = filename)
      message("DOCX saved: ", filename)
      TRUE
    },
    error = function(e) {
      message("DOCX NOT saved (ignored): ", filename)
      message("Reason: ", conditionMessage(e))
      FALSE
    }
  )
}
