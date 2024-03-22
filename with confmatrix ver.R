# Avery Tran
# STA5737 - Applied Analytics
# Individual Project
# Due: 3/22/24

# Dataset: https://www.kaggle.com/datasets/davinwijaya/employee-turnover/data

library(dplyr)
library(corrplot)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(xgboost)
library(pROC)

setwd("C:/Users/roddy/OneDrive/Documents/STA5737 - Applied Analytics/Individual Project")

# Load the dataset
data = read.csv("turnover.csv", header = TRUE, stringsAsFactors = TRUE)

# *=========================================*
# *         Understanding the Data          *
# *=========================================*

# View the first and last few rows of the dataset
head(data)
tail(data)

# Examine the structure of the dataset
str(data)

# Check the dimensions of the dataset
dim(data)

# Summarize NA values across the dataset
sum(is.na(data))

# List all column names
names(data)

# Rename variables based on the 'Big 5' personality traits
# Note: "extraversion" is already correctly named
names(data)[names(data) == "stag"] = "experience"
names(data)[names(data) == "independ"] = "agreeableness"
names(data)[names(data) == "selfcontrol"] = "conscientiousness"
names(data)[names(data) == "anxiety"] = "neuroticism"
names(data)[names(data) == "novator"] = "openness"

# Explore unique values in categorical variables
# Note: HoReCa means Hotel/Restaurant/Cafe
sapply(data[c("gender", "industry", "profession", "traffic", "coach", "head_gender", "greywage", "way")], unique)


# *===================================*
# *         Data Preparation          *
# *===================================*

# Standardize categorical variables
data$industry = factor(recode(data$industry, 'etc' = 'Other'))
data$profession = factor(recode(data$profession, 'Finan\xf1e' = 'Finance', 'etc' = 'Other', 'manage' = 'Management'))
data$coach = factor(recode(data$coach, 'my head' = 'Direct Supervisor'))
data$traffic = factor(recode(data$traffic, 'KA' = 'RA', 'referal' = 'Referral', 'recNErab' = 'RecNE_SI', 'rabrecNErab' = 'RecNE_EI'))


# *================================================*
# *         Exploratory Data Analysis (EDA)        *
# *================================================*

# Summary statistics
summary(data)

# Histograms for continuous variables
continuous_vars = c("age", "experience", "extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness")
colors = c("lightblue", "lightgreen", "lightcoral", "lightcyan", "lightgoldenrod", "lightsalmon", "lavender")
par(mfrow = c(3, 3))
for (i in seq_along(continuous_vars)) {
  hist(data[[continuous_vars[i]]], main = continuous_vars[i], xlab = continuous_vars[i], col = colors[i])
}
par(mfrow = c(1, 1))

# Log transformation of 'experience'
data$log_experience = log(data$experience)

hist(data$log_experience, 
     main = "Log-Transformed Experience Time", 
     xlab = "Log(Experience Time)", col = "lightgreen")

# The distribution looks a bit more normal now.

par(mfrow = c(1, 2))
qqnorm(data$experience, main = "QQ-plot for Experience Time"); qqline(data$experience, col = "red")
qqnorm(data$log_experience, main = "QQ-plot for Log(Experience Time)"); qqline(data$log_experience, col = "red")
par(mfrow = c(1, 1))

# A bit more aligned with the QQ-line, though its tails deviate greatly.

# QQ-plots for the other continuous variables (not including experience time/stag)
vars_minus_stag = c("age", "extraversion", "agreeableness", 
                    "conscientiousness", "neuroticism", "openness")

# Loop over each variable name and create a QQ-plot
par(mfrow = c(2, 3))
for (var_name in vars_minus_stag) {
  qqnorm(data[[var_name]], main = paste("QQ-plot for", var_name))
  qqline(data[[var_name]], col = "red", lwd = 2)
  title(main = paste("QQ-plot for", var_name), col.main = "blue")
}

par(mfrow = c(1, 1))

# Shapiro-Wilk tests show low p-values less than 0.05, indicating the data is not normally distributed.

for (var in continuous_vars) {
  result <- shapiro.test(data[[var]])
  cat("Shapiro-Wilk test for", var, ":\n")
  cat("W =", result$statistic, ", p-value =", result$p.value, "\n\n")
}

# Continuous variables including dependent 'event' as well as log_experience' for cormatrix
continuous_plus_event = c("age", "experience", "log_experience", "extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness", "event")
temp_data <- data[, continuous_plus_event]

# Calculate the correlation matrix for this temporary data frame
cor_matrix = cor(temp_data, use = "complete.obs")  # Handling NAs

# Plot the correlation matrix
corrplot(cor_matrix, method = "circle", type = "upper", tl.col = "black", tl.srt = 45, addCoef.col = "black", diag = FALSE)


# Boxplots for 'age' and 'experience' by 'event' (stay/leave)
data$event = factor(data$event, levels = c(0, 1), labels = c("Stayed", "Left"))
boxplot(age ~ event, data = data, main = "Age by Employee Turnover", xlab = "Employee Turnover", ylab = "Age", col = c("lightblue", "salmon"))
boxplot(experience ~ event, data = data, main = "Experience Time by Employee Turnover", xlab = "Employee Turnover", ylab = "Experience Time", col = c("lightgreen", "orange"))

# Contingency tables for chi-square testing
categorical_vars = c("gender", "industry", "profession", "traffic", "coach", "head_gender", "greywage", "way")
for (cat_var in categorical_vars) {
  cat("\n---", cat_var, "---\n")
  table_event = table(data[[cat_var]], data$event)
  print(table_event)
  chi_test_result = chisq.test(table_event)
  print(chi_test_result)
}

# *===========================*
# *         Modeling          *
# *===========================*


# Set the seed for reproducibility
set.seed(123)

# Split the data into 70% for training and 30% for testing
index <- createDataPartition(data$event, p = 0.70, list = FALSE)
train_data <- data[index, ]
test_data <- data[-index, ]

# Train a Logistic Regression Model
log_model <- glm(event ~ ., data = train_data, family = binomial(link = "logit"))
# Predict probabilities for the test set
pred_prob_log <- predict(log_model, newdata = test_data, type = "response")
# Generate ROC curve for the Logistic Regression Model
roc_curve_log <- roc(response = test_data$event, predictor = pred_prob_log)
# Calculate AUC for Logistic Regression Model
auc_log <- auc(roc_curve_log)

# Train a Decision Tree Model
dt_model <- rpart(event ~ ., data = train_data, method = "class")
# Predict probabilities for the test set
dt_probs <- predict(dt_model, newdata = test_data, type = "prob")[,2]
# Generate ROC curve for the Decision Tree Model
roc_curve_dt <- roc(response = test_data$event, predictor = dt_probs)
# Calculate AUC for Decision Tree Model
auc_dt <- auc(roc_curve_dt)

# Plot the Decision Tree Model
rpart.plot(dt_model, main="Decision Tree Model")

# Train a Random Forest Model
rf_model <- randomForest(event ~ ., data = train_data)
# Predict probabilities for the test set
rf_probs <- predict(rf_model, newdata = test_data, type = "prob")[,2]
# Generate ROC curve for the Random Forest Model
roc_curve_rf <- roc(response = test_data$event, predictor = rf_probs)
# Calculate AUC for Random Forest Model
auc_rf <- auc(roc_curve_rf)

# Convert factors to numeric for XGBoost
convert_factors_to_numeric <- function(df) {
  data.frame(lapply(df, function(x) if(is.factor(x)) as.numeric(as.factor(x)) else x))
}

# Prepare data for XGBoost
train_data_prep_for_xgb <- convert_factors_to_numeric(train_data)
test_data_prep_for_xgb <- convert_factors_to_numeric(test_data)

# Train an XGBoost Model
dtrain_xgb <- xgb.DMatrix(data = as.matrix(train_data_prep_for_xgb[,-which(names(train_data_prep_for_xgb) == "event")]), label = train_data_prep_for_xgb$event - 1)
dtest_xgb <- xgb.DMatrix(data = as.matrix(test_data_prep_for_xgb[,-which(names(test_data_prep_for_xgb) == "event")]))
xgb_params <- list(objective = "binary:logistic")
xgb_model <- xgb.train(params = xgb_params, data = dtrain_xgb, nrounds = 100)
xgb_preds <- predict(xgb_model, newdata = dtest_xgb)
# Generate ROC curve for XGBoost Model
roc_curve_xgb <- roc(response = as.numeric(test_data_prep_for_xgb$event) - 1, predictor = xgb_preds)
# Calculate AUC for XGBoost Model
auc_xgb <- auc(roc_curve_xgb)

# Make predictions for each model
predicted_log <- ifelse(pred_prob_log > 0.5, "Left", "Stayed")
predicted_dt <- ifelse(dt_probs > 0.5, "Left", "Stayed")
predicted_rf <- ifelse(rf_probs > 0.5, "Left", "Stayed")

# Set factor levels for predictions
predicted_log <- factor(predicted_log, levels = c("Stayed", "Left"))
predicted_dt <- factor(predicted_dt, levels = c("Stayed", "Left"))
predicted_rf <- factor(predicted_rf, levels = c("Stayed", "Left"))

# Generate confusion matrices for each model
conf_matrix_log <- confusionMatrix(predicted_log, test_data$event)
conf_matrix_dt <- confusionMatrix(predicted_dt, test_data$event)
conf_matrix_rf <- confusionMatrix(predicted_rf, test_data$event)

# Ensure XGBoost's predictions match the same factor levels
predicted_xgb <- ifelse(xgb_preds > 0.5, "Left", "Stayed")
predicted_xgb <- factor(predicted_xgb, levels = c("Stayed", "Left"))

# Adjust the 'event' variable for XGBoost test data to match these levels
test_data_prep_for_xgb$event <- factor(test_data_prep_for_xgb$event, levels = c("1", "2"), labels = c("Stayed", "Left"))

# Generate confusion matrix for XGBoost
conf_matrix_xgb <- confusionMatrix(predicted_xgb, test_data_prep_for_xgb$event)

# Print confusion matrices
print("Logistic Regression Confusion Matrix:")
print(conf_matrix_log)

print("Decision Tree Confusion Matrix:")
print(conf_matrix_dt)

print("Random Forest Confusion Matrix:")
print(conf_matrix_rf)

print("XGBoost Confusion Matrix:")
print(conf_matrix_xgb)

# Plot ROC Curves of all models
plot(roc_curve_log, main="ROC Curves", col="red")
legend("bottomright", legend=c("Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"), col=c("red", "blue", "green", "purple"), lwd=2)
lines(roc_curve_dt, col="blue")
lines(roc_curve_rf, col="green")
lines(roc_curve_xgb, col="purple")

# Report AUC Values
cat("AUC for Logistic Regression:", auc_log, "\n")
cat("AUC for Decision Tree:", auc_dt, "\n")
cat("AUC for Random Forest:", auc_rf, "\n")
cat("AUC for XGBoost:", auc_xgb, "\n")

# *============================================================*
# *        Optimal Feature Selection for Model Reduction       *
# *============================================================*

# Random Forest feature importance
rf_importance = importance(rf_model)
varImpPlot(rf_model, main="Random Forest Feature Importance")

# XGBoost feature importance
xgb_importance = xgb.importance(feature_names = colnames(train_data[, -which(names(train_data) == "event")]), model = xgb_model)
xgb.plot.importance(xgb_importance, main="XGBoost Feature Importance")

# Determine optimal number of features for Random Forest model
feature_importances = sort(rf_importance[, "MeanDecreaseGini"], decreasing = TRUE)
sorted_feature_names = names(feature_importances)

# Test different subsets of features for optimal model performance
N_values = seq(5, min(length(sorted_feature_names), 30), by = 5)
auc_values = numeric(length(N_values))

for (i in seq_along(N_values)) {
  N = N_values[i]
  selected_features = sorted_feature_names[1:N]
  train_data_reduced = train_data[, c(selected_features, "event"), drop = FALSE]
  test_data_reduced = test_data[, c(selected_features, "event"), drop = FALSE]
  
  # Retrain and evaluate the model with N features
  rf_model_reduced = randomForest(event ~ ., data = train_data_reduced)
  test_preds = predict(rf_model_reduced, newdata = test_data_reduced, type = "prob")[,2]
  roc_result = roc(response = test_data_reduced$event, predictor = test_preds)
  auc_values[i] = auc(roc_result)
  
  cat(sprintf("N = %d, AUC = %.4f\n", N, auc_values[i]))
}

best_N_index = which.max(auc_values)
best_N = N_values[best_N_index]
cat("Optimal number of features (Best N):", best_N, "\nAUC for Best N:", auc_values[best_N_index], "\n")
