# get_confusion_matrix <- function(model, data = NULL, outcome, threshold = 0.5) {
  
#   if (is.null(data)) data <- model.frame(model)
  
#   actual <- data[[outcome]]
#   if (is.factor(actual)) actual <- as.numeric(as.character(actual))
  
#   pred_prob <- predict(model, newdata = data, type = "response")
#   pred_class <- ifelse(pred_prob >= threshold, 1, 0)
  
#   TP <- sum(pred_class == 1 & actual == 1, na.rm = TRUE)
#   FP <- sum(pred_class == 1 & actual == 0, na.rm = TRUE)
#   TN <- sum(pred_class == 0 & actual == 0, na.rm = TRUE)
#   FN <- sum(pred_class == 0 & actual == 1, na.rm = TRUE)
  
#   tibble::tibble(
#     `Predicted / Actual` = c(
#       "Predicted GLD = 1",
#       "Predicted GLD = 0"
#     ),
#     `Actual GLD = 1` = c(
#       paste0(TP, " (True Positive)"),
#       paste0(FN, " (False Negative)")
#     ),
#     `Actual GLD = 0` = c(
#       paste0(FP, " (False Positive)"),
#       paste0(TN, " (True Negative)")
#     )
#   )
# }



get_classification <- function(model, data = NULL, outcome, threshold = 0.5) {
  
  if (is.null(data)) data <- model.frame(model)
  
  actual <- data[[outcome]]
  if (is.factor(actual)) actual <- as.character(actual)
  actual <- as.numeric(actual)
  
  pred_prob <- predict(model, newdata = data, type = "response")
  pred_class <- ifelse(pred_prob >= threshold, 1, 0)
  
  cm <- caret::confusionMatrix(
    data = factor(pred_class, levels = c(0, 1)),
    reference = factor(actual, levels = c(0, 1)),
    positive = "1"
  )
    
  roc_obj <- pROC::roc(
    response = actual,
    predictor = pred_prob,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  
  auc_val <- as.numeric(pROC::auc(roc_obj))
  
  cm_tbl <- as.data.frame.matrix(cm$table) %>%
    tibble::rownames_to_column("Predicted") %>%
    dplyr::rename(
      `Actual GLD = 0` = `0`,
      `Actual GLD = 1` = `1`
    )
  
  metrics_tbl <- tibble::tibble(
    Threshold = threshold,
    AUC = auc_val,
    Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    Specificity = as.numeric(cm$byClass["Specificity"]),
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    `Positive predictive value` = as.numeric(cm$byClass["Pos Pred Value"]),
    `Negative predictive value` = as.numeric(cm$byClass["Neg Pred Value"])
  )
  
  list(
    confusion_matrix = cm_tbl,
    metrics = metrics_tbl,
    roc = roc_obj,
    caret_output = cm
  )
}