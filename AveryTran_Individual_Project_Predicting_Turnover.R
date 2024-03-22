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

# First and last few rows of the data
head(data)
tail(data)

# Structure of the data
str(data)

# Dimensions of the data
dim(data)

# Identify NAs to potentially omit from the data
sum(is.na(data))

# 0 NAs found

# List all column names
names(data)

# Rename variables based on the 'Big 5' personality traits
# "extraversion" is already correctly named
names(data)[names(data) == "stag"] = "experience"
names(data)[names(data) == "independ"] = "agreeableness"
names(data)[names(data) == "selfcontrol"] = "conscientiousness"
names(data)[names(data) == "anxiety"] = "neuroticism"
names(data)[names(data) == "novator"] = "openness"

# Explore unique values in categorical variables
sapply(data[c("gender", "industry", "profession", "traffic", "coach", "head_gender", "greywage", "way")], unique)

# Traffic refers to pipeline employee came through to the company
# Coach refers to presence of a dedicated trainer
# Greywage is simply wage
# Way refers to mode of commute to work

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

# Age and Experience are fairly right-skewed
# Age seems natural, but will take the log of Experience to see if it improves

# Log transformation of 'experience'
data$log_experience = log(data$experience)

# Updated histogram with the new log(experience) var
hist(data$log_experience, 
     main = "Log-Transformed Experience Time", 
     xlab = "Log(Experience Time)", col = "lightgreen")

# The distribution looks a bit more normal now, but will compare the QQ-plots
par(mfrow = c(1, 2))
qqnorm(data$experience, main = "QQ-plot for Experience Time"); qqline(data$experience, col = "red")
qqnorm(data$log_experience, main = "QQ-plot for Log(Experience Time)"); qqline(data$log_experience, col = "red")
par(mfrow = c(1, 1))

# Latter is A bit more aligned with the QQ-line, though its tails deviate greatly.

# QQ-plots for the other continuous variables (not including experience time/stag)
vars_minus_stag = c("age", "extraversion", "agreeableness", 
                    "conscientiousness", "neuroticism", "openness")

# For loop to make QQ-plot for each var
par(mfrow = c(2, 3))
for (var_name in vars_minus_stag) {
  qqnorm(data[[var_name]], main = paste("QQ-plot for", var_name))
  qqline(data[[var_name]], col = "red", lwd = 2)
  title(main = paste("QQ-plot for", var_name), col.main = "blue")
}

par(mfrow = c(1, 1))

# For loop of Shapiro-Wilk tests to test for normality
for (var in continuous_vars) {
  result = shapiro.test(data[[var]])
  cat("Shapiro-Wilk test for", var, ":\n")
  cat("W =", result$statistic, ", p-value =", result$p.value, "\n\n")
}

# Each show low p-values less than 0.05, indicating the data is not normally distributed.

#####
# Correlation Matrix of the Continuous Variables

# List of continuous vasr as well as the dependent 'event' and new 'log_experience'
continuous_plus_event = c("age", "experience", "log_experience", "extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness", "event")
temp_data = data[, continuous_plus_event]

# Correlation matrix for temporary df
cor_matrix = cor(temp_data, use = "complete.obs")

# Plot the correlation matrix
corrplot(cor_matrix, method = "circle", type = "upper", tl.col = "black", tl.srt = 45, addCoef.col = "black", diag = FALSE)

# Boxplots for 'Age' and 'Experience' by 'Event' (Stay/Leave)

# Adding labels to the factor levels of event
data$event = factor(data$event, levels = c(0, 1), labels = c("Stayed", "Left"))

# Boxplot for age by event
boxplot(age ~ event, data = data, main = "Age by Employee Turnover", xlab = "Employee Turnover", ylab = "Age", col = c("lightblue", "salmon"))

# Boxplot for experience by event
boxplot(experience ~ event, data = data, main = "Experience Time by Employee Turnover", xlab = "Employee Turnover", ylab = "Experience Time", col = c("lightgreen", "orange"))

#####

# Contingency Tables for Chi-Square Testing

# List of categorical vars
categorical_vars = c("gender", "industry", "profession", "traffic", "coach", "head_gender", "greywage", "way")

# For loop to make a table for each var
for (cat_var in categorical_vars) {
  
  # Print name of var being tested
  cat("\n---", cat_var, "---\n")
  table_event = table(data[[cat_var]], data$event)
  # Print table
  print(table_event)
  
  # Perform chi-squared test on the table
  chi_test_result = chisq.test(table_event)
  
  # Print the chi-square test result
  print(chi_test_result)
}

# *===========================*
# *         Modeling          *
# *===========================*

# Set the seed for reproducibility
set.seed(123)

# Split data into 70% training and 30% testing
index = createDataPartition(data$event, p = 0.70, list = FALSE)
train_data = data[index, ]
test_data = data[-index, ]



# Train Logistic Regression Model
log_model = glm(event ~ ., data = train_data, family = binomial(link = "logit"))

# Predict probabilities for the test set
pred_prob_log = predict(log_model, newdata = test_data, type = "response")

# Create ROC curve
roc_curve_log = roc(response = test_data$event, predictor = pred_prob_log)

# Calculate AUC
auc_log = auc(roc_curve_log)



# Train Decision Tree Model
dt_model = rpart(event ~ ., data = train_data, method = "class")

# Predict probabilities for the test set
dt_probs = predict(dt_model, newdata = test_data, type = "prob")[,2]

# Create ROC curve
roc_curve_dt = roc(response = test_data$event, predictor = dt_probs)

# Calculate AUC
auc_dt = auc(roc_curve_dt)

# Plot the Decision Tree Model
rpart.plot(dt_model, main="Decision Tree Model")



# Train Random Forest Model
rf_model = randomForest(event ~ ., data = train_data)

# Predict probabilities for the test set
rf_probs = predict(rf_model, newdata = test_data, type = "prob")[,2]

# Create ROC curve
roc_curve_rf = roc(response = test_data$event, predictor = rf_probs)

# Calculate AUC
auc_rf = auc(roc_curve_rf)



# Convert factors to numeric for XGBoost
convert_factors_to_numeric = function(df) {
  data.frame(lapply(df, function(x) if(is.factor(x)) as.numeric(as.factor(x)) else x))
}

# Prepare data for XGBoost
train_data_prep_for_xgb = convert_factors_to_numeric(train_data)
test_data_prep_for_xgb = convert_factors_to_numeric(test_data)

# Train XGBoost Model
dtrain_xgb = xgb.DMatrix(data = as.matrix(train_data_prep_for_xgb[,-which(names(train_data_prep_for_xgb) == "event")]), label = train_data_prep_for_xgb$event - 1)
dtest_xgb = xgb.DMatrix(data = as.matrix(test_data_prep_for_xgb[,-which(names(test_data_prep_for_xgb) == "event")]))
xgb_params = list(objective = "binary:logistic")
xgb_model = xgb.train(params = xgb_params, data = dtrain_xgb, nrounds = 100)

# Predict probabilities for the test set
xgb_probs = predict(xgb_model, newdata = dtest_xgb)

# Generate ROC curve for XGBoost Model
roc_curve_xgb = roc(response = as.numeric(test_data_prep_for_xgb$event) - 1, predictor = xgb_probs)

# Calculate AUC for XGBoost Model
auc_xgb = auc(roc_curve_xgb)


#####
# Confusion Matrices

# Make predictions for each model
predicted_log = ifelse(pred_prob_log > 0.5, "Left", "Stayed")
predicted_dt = ifelse(dt_probs > 0.5, "Left", "Stayed")
predicted_rf = ifelse(rf_probs > 0.5, "Left", "Stayed")

# Set factor levels for predictions
predicted_log = factor(predicted_log, levels = c("Stayed", "Left"))
predicted_dt = factor(predicted_dt, levels = c("Stayed", "Left"))
predicted_rf = factor(predicted_rf, levels = c("Stayed", "Left"))

# Create confusion matrices
conf_matrix_log = confusionMatrix(predicted_log, test_data$event)
conf_matrix_dt = confusionMatrix(predicted_dt, test_data$event)
conf_matrix_rf = confusionMatrix(predicted_rf, test_data$event)

# Get XGBoost preds and event back to factors to generate the matrix
predicted_xgb = ifelse(xgb_probs > 0.5, "Left", "Stayed")
predicted_xgb = factor(predicted_xgb, levels = c("Stayed", "Left"))
test_data_prep_for_xgb$event = factor(test_data_prep_for_xgb$event, levels = c("1", "2"), labels = c("Stayed", "Left"))

# Create confusion matrix
conf_matrix_xgb = confusionMatrix(predicted_xgb, test_data_prep_for_xgb$event)

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

# Print AUC Values
cat("AUC for Logistic Regression:", auc_log, "\n")
cat("AUC for Decision Tree:", auc_dt, "\n")
cat("AUC for Random Forest:", auc_rf, "\n")
cat("AUC for XGBoost:", auc_xgb, "\n")

# Random Forest shows best AUC score.

# *============================================================*
# *        Optimal Feature Selection for Model Reduction       *
# *============================================================*

# Will use Feature Importance on the random forest and XGBoost models to try for model reductions.

# Random Forest feature importance
rf_importance = importance(rf_model)

# Print variable importance plot
varImpPlot(rf_model, main="Random Forest Feature Importance")

# XGBoost feature importance
xgb_importance = xgb.importance(feature_names = colnames(train_data[, -which(names(train_data) == "event")]), model = xgb_model)

# Print feature importance plot
xgb.plot.importance(xgb_importance, main="XGBoost Feature Importance")


# Determine optimal number of features for Random Forest model
# Can sort with the Gini metric
feature_importances = sort(rf_importance[, "MeanDecreaseGini"], decreasing = TRUE)
sorted_feature_names = names(feature_importances)


# Test different subsets of features for optimal model performance
N_values = seq(5, min(length(sorted_feature_names), 30), by = 5)
auc_values = numeric(length(N_values))

# For loop to update the training and testing data with the different N subsets
for (i in seq_along(N_values)) {
  N = N_values[i]
  selected_features = sorted_feature_names[1:N]
  train_data_reduced = train_data[, c(selected_features, "event"), drop = FALSE]
  test_data_reduced = test_data[, c(selected_features, "event"), drop = FALSE]
  
  # Retrain and evaluate the model with N feature subset
  rf_model_reduced = randomForest(event ~ ., data = train_data_reduced)
  test_preds = predict(rf_model_reduced, newdata = test_data_reduced, type = "prob")[,2]
  
  # Get ROC score to get AUC value
  roc_result = roc(response = test_data_reduced$event, predictor = test_preds)
  auc_values[i] = auc(roc_result)
  
  # Print each N subset's AUC score
  cat(sprintf("N = %d, AUC = %.4f\n", N, auc_values[i]))
}

# N = 15 shows best AUC, however it is only marginally better than the model with N = 5
# Potentially, the more simple model can be used to get approximately the same AUC score as the full model

# Here are the top 5 features in each
print(sorted_feature_names[1:5])
print(xgb_importance[1:5])

#####

# Refitting the Simplified Model

# Store the top 5 into a variable which will be used to make the new datasets
top_5_features = sorted_feature_names[1:5]

# Create datasets that include only these top 5 features and the outcome
train_data_reduced = train_data[, c(top_5_features, "event")]
test_data_reduced = test_data[, c(top_5_features, "event")]

# Train a new Random Forest model with these selected features
rf_model_reduced = randomForest(event ~ ., data = train_data_reduced)

# Predict the outcome on the test dataset
predictions_rf = predict(rf_model_reduced, newdata = test_data_reduced, type = "prob")[, 2]

# Evaluate the model using AUC
roc_result_rf = roc(test_data_reduced$event, predictions_rf)
auc_rf = round(auc(roc_result_rf), 4) # Rounding to 4 decimal places

# Print the AUC result
print(paste("AUC for Reduced Random Forest Model:", auc_rf))


# Since it is XGBoost, convert the categorical back from factors to numeric
train_data_reduced_xgb = convert_factors_to_numeric(train_data_reduced)
test_data_reduced_xgb = convert_factors_to_numeric(test_data_reduced)

# Prepare data again into the matrix for XGBoost
dtrain_xgb_reduced = xgb.DMatrix(data = as.matrix(train_data_reduced_xgb[, -which(names(train_data_reduced_xgb) == "event")]), label = train_data_reduced_xgb$event - 1)
dtest_xgb_reduced = xgb.DMatrix(data = as.matrix(test_data_reduced_xgb[, -which(names(test_data_reduced_xgb) == "event")]))

# Train XGBoost model
xgb_model_reduced = xgb.train(params = list(objective = "binary:logistic"), data = dtrain_xgb_reduced, nrounds = 100)

# Predict with XGBoost model
predictions_xgb = predict(xgb_model_reduced, newdata = dtest_xgb_reduced)

# Evaluate XGBoost model using AUC
roc_result_xgb = roc(as.numeric(test_data_reduced_xgb$event) - 1, predictions_xgb)
auc_xgb = round(auc(roc_result_xgb), 4) # Rounding to 4 decimal places

# Print the AUC result
print(paste("AUC for Reduced XGBoost Model:", auc_xgb))

# AUC for Reduced Model with Random Forest about 1.5% worse than the previous full model, as anticipated