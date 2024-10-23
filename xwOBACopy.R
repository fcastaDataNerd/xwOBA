---
  title: "xwOBA"
author: "Franco"
date: "2024-06-09"
output: html_document
---
  
  
  ```{r}
# Load necessary libraries
library(readxl)
library(dplyr)
library(nnet)
library(caret)
library(glmnet)
library(writexl)
library(pROC)

# Load the bip dataset
bip <- read_excel("C:\\Users\\Franco Castagliuolo\\OneDrive - Bentley University\\NECBL\\NECBLBIP.xlsx")

# Define the angle and direction constants from the bip dataset
angle_constant <- abs(min(bip$Angle, na.rm = TRUE)) + 1
direction_constant <- abs(min(bip$Direction, na.rm = TRUE)) + 1

# Apply transformations to the bip dataset to ensure positivity
bip <- bip %>%
  mutate(
    LogExitSpeed = log(ExitSpeed),
    LogExitSpeed2 = log(ExitSpeed)^2,
    AdjustedAngle = Angle + angle_constant,
    AdjustedAngle2 = AdjustedAngle^2,
    AdjustedAngle3 = AdjustedAngle^3,
    LogExitSpeed_AdjustedAngle = LogExitSpeed * AdjustedAngle,
    LogExitSpeed_AdjustedAngle2 = LogExitSpeed * AdjustedAngle2
  )

# Ensure the OutcomeCategory variable is correctly set up
bip$OutcomeCategory <- factor(bip$Result, levels = c("Out", "Single", "Double", "Triple", "HomeRun"))

# Balance the data using upsampling
set.seed(123) # For reproducibility
balanced_data <- upSample(
  x = bip[, c("LogExitSpeed", "LogExitSpeed2", "AdjustedAngle", "AdjustedAngle2", "AdjustedAngle3", 
              "LogExitSpeed_AdjustedAngle", "LogExitSpeed_AdjustedAngle2")], 
  y = bip$OutcomeCategory
)
colnames(balanced_data)[ncol(balanced_data)] <- "OutcomeCategory"

print(paste("Angle Constant:", angle_constant))
summary(bip$OutcomeCategory)
table(bip$OutcomeCategory)
table(balanced_data$OutcomeCategory)

```
```{r}
# Prepare data for glmnet
x <- model.matrix(OutcomeCategory ~ LogExitSpeed + LogExitSpeed2 + AdjustedAngle + AdjustedAngle2 + AdjustedAngle3 + 
                    LogExitSpeed_AdjustedAngle + LogExitSpeed_AdjustedAngle2, data = balanced_data)[,-1]
y <- balanced_data$OutcomeCategory 

# Define class weights
class_weights <- c("Out" = 2.6, "Single" = 1.3, "Double" = 1.0, "Triple" = 0.7, "HomeRun" = 0.5)

# Assign weights to the data
weights <- ifelse(y == "Out", class_weights["Out"],
                  ifelse(y == "Single", class_weights["Single"],
                         ifelse(y == "Double", class_weights["Double"],
                                ifelse(y == "Triple", class_weights["Triple"],
                                       class_weights["HomeRun"]))))

# Fit Elastic Net regression model with cross-validation to find the best lambda
set.seed(123)
cv_fit <- cv.glmnet(x, y, family = "multinomial", alpha = 0.5, weights = weights, maxit = 200000) # alpha = 0.5 for Elastic Net
best_lambda <- cv_fit$lambda.min

# Fit the final model with the best lambda
final_model <- glmnet(x, y, family = "multinomial", alpha = 0.5, lambda = best_lambda, weights = weights, maxit = 200000)

```

```{r}
train_control <- trainControl(method = "cv", number = 10)
cv_model <- train(
  OutcomeCategory ~ LogExitSpeed + LogExitSpeed2 + AdjustedAngle + AdjustedAngle2 + AdjustedAngle3 + 
    LogExitSpeed_AdjustedAngle + LogExitSpeed_AdjustedAngle2, 
  data = balanced_data, 
  method = "glmnet", 
  trControl = train_control, 
  family = "multinomial", 
  tuneGrid = expand.grid(alpha = 0.5, lambda = best_lambda),
  weights = weights
)

# Print cross-validation results
print(cv_model)
```


```{r}
# Predict probabilities on the balanced dataset
predicted_probs <- predict(final_model, newx = x, type = "response")

# Convert predicted probabilities to predicted classes
predicted_classes <- apply(predicted_probs, 1, function(row) colnames(predicted_probs)[which.max(row)])
predicted_classes <- factor(predicted_classes, levels = levels(y))

# Calculate confusion matrix
confusion_matrix <- confusionMatrix(predicted_classes, y)
print(confusion_matrix)

# Additional metrics
accuracy <- confusion_matrix$overall['Accuracy']
sensitivity <- confusion_matrix$byClass['Sensitivity']
specificity <- confusion_matrix$byClass['Specificity']

# Print metrics
cat("Accuracy: ", accuracy, "\n")
cat("Sensitivity: ", sensitivity, "\n")
cat("Specificity: ", specificity, "\n")

# Calculate ROC AUC for each class
predicted_probs_matrix <- predicted_probs[,,1]

roc_results <- lapply(levels(y), function(class) {
  class_index <- which(levels(y) == class)
  roc(response = as.numeric(y == class), predictor = predicted_probs_matrix[, class_index])
})

cat("ROC AUC for each class:\n")
roc_auc_values <- sapply(roc_results, function(roc_result) roc_result$auc)
names(roc_auc_values) <- levels(y)
print(roc_auc_values)

# Fit a simplified null model with no predictors
null_model <- multinom(OutcomeCategory ~ 1, data = balanced_data)
log_lik_null <- logLik(null_model)

# Calculate log-likelihood for the full model
log_lik_full <- sum(sapply(1:nrow(x), function(i) {
  class_index <- which(levels(y) == y[i])
  log(predicted_probs[i, class_index, 1])
}))

cat("Log-Likelihood (Full Model):", log_lik_full, "\n")
cat("Log-Likelihood (Null Model):", log_lik_null, "\n")
pseudo_r2 <- 1 - (log_lik_full / as.numeric(log_lik_null))
cat("Pseudo R-squared:", pseudo_r2, "\n")

```

```{r}
# Prepare data for glmnet prediction on original bip dataset
x_bip <- model.matrix(~ LogExitSpeed + LogExitSpeed2 + AdjustedAngle + AdjustedAngle2 + AdjustedAngle3 + 
                        LogExitSpeed_AdjustedAngle + LogExitSpeed_AdjustedAngle2, data = bip)[,-1]

# Predict the outcomes using the final model on the bip dataset
predicted_probs_bip <- predict(final_model, newx = x_bip, type = "response")
predicted_classes_bip <- apply(predicted_probs_bip, 1, function(row) colnames(predicted_probs_bip)[which.max(row)])
predicted_classes_bip <- factor(predicted_classes_bip, levels = levels(bip$OutcomeCategory))

# Add Predicted Results to the original bip dataset
bip <- bip %>%
  mutate(PredictedResult = predicted_classes_bip) %>%
  relocate(PredictedResult, .after = Result)

# View the updated bip dataset
View(bip)

```

```{r}
# Load necessary libraries
library(dplyr)
library(readxl)
library(writexl)

# Load the VT_games dataset
vt_games <- read_excel("C:\\Users\\Franco Castagliuolo\\OneDrive - Bentley University\\Neers 24\\Pitchers\\AllGames\\VT_Games.xlsx")

# Filter for Vermont hitters
VTBIP <- vt_games %>%
  filter(PitcherTeam != "VER_MOU", PlayResult %in% c("Out", "Single", "Double", "Triple", "HomeRun")) %>%
  dplyr::select(Batter, PlayResult, ExitSpeed, Angle)

# Apply transformations using the constants derived from the bip dataset
VTBIP_clean <- VTBIP %>%
  mutate(
    LogExitSpeed = log(ExitSpeed),
    LogExitSpeed2 = log(ExitSpeed)^2,
    AdjustedAngle = Angle + angle_constant,
    AdjustedAngle2 = AdjustedAngle^2,
    AdjustedAngle3 = AdjustedAngle^3,
    LogExitSpeed_AdjustedAngle = LogExitSpeed * AdjustedAngle,
    LogExitSpeed_AdjustedAngle2 = LogExitSpeed * AdjustedAngle2,
    row_index = row_number()
  )

# Remove rows with NA values in the columns used for predictions
VTBIP_clean_no_na <- VTBIP_clean %>%
  filter(!is.na(LogExitSpeed) & !is.na(AdjustedAngle) & !is.na(AdjustedAngle2) & !is.na(AdjustedAngle3) & 
           !is.na(LogExitSpeed_AdjustedAngle) & !is.na(LogExitSpeed_AdjustedAngle2))

# Prepare data for glmnet prediction
x_vt_no_na <- model.matrix(~ LogExitSpeed + LogExitSpeed2 + AdjustedAngle + AdjustedAngle2 + AdjustedAngle3 + 
                             LogExitSpeed_AdjustedAngle + LogExitSpeed_AdjustedAngle2, data = VTBIP_clean_no_na)[,-1]

# Predict the outcomes using the final model on the VTBIP_clean_no_na dataset
predicted_probs_vt_no_na <- predict(final_model, newx = x_vt_no_na, type = "response")
predicted_classes_vt_no_na <- apply(predicted_probs_vt_no_na, 1, function(row) colnames(predicted_probs_vt_no_na)[which.max(row)])
predicted_classes_vt_no_na <- factor(predicted_classes_vt_no_na, levels = levels(bip$OutcomeCategory))

# Add Predicted Results to the VTBIP_clean_no_na dataset
VTBIP_clean_no_na <- VTBIP_clean_no_na %>%
  mutate(PredictedOutcome = as.character(predicted_classes_vt_no_na))

# Create a copy of VTBIP_clean for merging
VTBIP_clean_with_index <- VTBIP_clean %>% mutate(row_index = row_number())

# Merge predictions back into VTBIP_clean
VTBIP_clean <- VTBIP_clean_with_index %>%
  left_join(VTBIP_clean_no_na %>% select(row_index, PredictedOutcome), by = "row_index") %>%
  mutate(PredictedOutcome = ifelse(is.na(PredictedOutcome), as.character(PlayResult), PredictedOutcome)) %>%
  relocate(PredictedOutcome, .after = PlayResult) %>%
  select(-row_index)

# Check the updated VTBIP_clean dataset
View(VTBIP_clean)

# Create a summary of predicted outcomes for each batter
PredictedOutcomes <- VTBIP_clean %>%
  group_by(Batter) %>%
  summarise(
    TotalSingles = sum(PredictedOutcome == "Single"),
    TotalDoubles = sum(PredictedOutcome == "Double"),
    TotalTriples = sum(PredictedOutcome == "Triple"),
    TotalHomeRuns = sum(PredictedOutcome == "HomeRun")
  )

# Print the summary to verify
View(PredictedOutcomes)

# Export the PredictedOutcomes dataset to an Excel file
write_xlsx(PredictedOutcomes, "C:\\Users\\Franco Castagliuolo\\OneDrive - Bentley University\\PredictedOutcomes.xlsx")

```



