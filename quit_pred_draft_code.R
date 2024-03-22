# Dataset: https://www.kaggle.com/datasets/davinwijaya/employee-turnover/data

library(dplyr)
library(corrplot)
library(caret)
library(rpart)
library(randomForest)
library(xgboost)
library(pROC)

# Read in the dataset
data = read.csv("C:/Users/roddy/OneDrive/Documents/STA5737 - Applied Analytics/Individual Project/turnover.csv", header=TRUE, stringsAsFactors = TRUE)

# *=========================================*
# *         Understanding the Data          *
# *=========================================*

# Check first 5 rows of the data
head(data)

# Check variable types
str(data)

# Check dimensions
dim(data)

# Check for NA values
sum(is.na(data))

# Check names of columns
names(data)

# Rename last 5 vars to the 'Big 5' names they are based off of
names(data)[names(data) == "stag"] = "experience"
names(data)[names(data) == "extraversion"] = "extraversion"
names(data)[names(data) == "independ"] = "agreeableness" 
names(data)[names(data) == "selfcontrol"] = "conscientiousness"
names(data)[names(data) == "anxiety"] = "neuroticism"
names(data)[names(data) == "novator"] = "openness"

# Check unique values
unique(data$gender)
unique(data$industry)
unique(data$profession)
unique(data$traffic)
unique(data$coach)
unique(data$head_gender)
unique(data$greywage)
unique(data$way)

# HoReCa means Hotel/Restaurant/Cafe

# *===================================*
# *         Data Preparation          *
# *===================================*

# Clean data entries to have consistent, understandable labels
data$industry = as.character(data$industry)
data$industry[data$industry == 'etc'] = 'Other'
data$industry = as.factor(data$industry)

data$profession = as.character(data$profession)
data$profession[data$profession == 'Finan\xf1e'] = 'Finance'
data$profession[data$profession == 'etc'] = 'Other'
data$profession[data$profession == 'manage'] = 'Management'
data$profession = as.factor(data$profession)

data$coach = as.character(data$coach)
data$coach[data$coach == 'my head'] = 'Direct Supervisor'
data$coach = as.factor(data$coach)

data$traffic = as.character(data$traffic)
data$traffic[data$traffic == 'KA'] = 'RA' # Recruiting Agency
data$traffic[data$traffic == 'referal'] = 'Referral'  # Correcting typo
data$traffic[data$traffic == 'recNErab'] = 'RecNE_SI' # Recommendation External - self Initiated
data$traffic[data$traffic == 'rabrecNErab'] = 'RecNE_EI' # Recommendation External - Employer Initiated
data$traffic = as.factor(data$traffic) 

# *================================================*
# *         Exploratory Data Analysis (EDA)        *
# *================================================*

# Descriptive statistics for the data
summary(data)

# List of continuous variable names
continuous_var_names = c("age", "experience", "extraversion", "agreeableness", 
                          "conscientiousness", "neuroticism", "openness")

# Colors for each histogram
colors = c("lightblue", "lightgreen", "lightcoral", "lightcyan", 
            "lightgoldenrod", "lightsalmon", "lavender")

# Set up 3x3 plot grid
par(mfrow = c(3, 3))

# Loop over each variable name and create a histogram
for (var_name in 1:length(continuous_var_names)) {
  hist(data[[continuous_var_names[var_name]]], main = continuous_var_names[var_name],
       xlab = continuous_var_names[var_name], col = colors[var_name])
}

par(mfrow = c(1, 1))

#####

# Take log-transformation of experience to see if this normalizes the data
data$log_experience = log(data$experience)
hist(data$log_experience, main = "Log-Transformed Experience Time", xlab = "Log(Experience Time)", col = "lightgreen")

# The distribution looks a bit more normal now.

# Set up 1x2 plot grid
par(mfrow = c(1,2))

qqnorm(data$experience, main = "QQ-plot for Experience Time")
qqline(data$experience, col = "red")

# QQ-plot for Experience Time
qqnorm(data$log_experience, main = "QQ-plot for Log(Experience Time)")
qqline(data$log_experience, col = "red")

# A bit more aligned with the QQ-line, though its tails deviate greatly.

# Verify non-normal with a Shapiro-Wilk test
shapiro.test(data$experience)

# QQ-plots for the other continuous variables (not including experience time/stag)
par(mfrow = c(2, 3))

vars_minus_stag = c("age", "extraversion", "agreeableness", 
                     "conscientiousness", "neuroticism", "openness")

# Loop over each variable name and create a QQ-plot
for (var_name in vars_minus_stag) {
  qqnorm(data[[var_name]], main = paste("QQ-plot for", var_name))
  qqline(data[[var_name]], col = "red", lwd = 2)
  title(main = paste("QQ-plot for", var_name), col.main = "blue")
}

par(mfrow = c(1, 1))

#####

# Boxplot for Experience Time by Age
boxplot(age ~ event, data = data, 
        main = "Boxplot of Age by Employee Turnover",
        xlab = "Employee Turnover", ylab = "Age (years)", col = c("lightblue", "salmon"),
        names = c("Stayed", "Left"))

# Boxplot for Experience Time by Employee Turnover
boxplot(experience ~ event, data = data,
        main = "Boxplot of Experience Time by Employee Turnover",
        xlab = "Employee Turnover", ylab = "Experience Time (months)",
        col = c("lightgreen", "orange"),
        names = c("Stayed", "Left"))

# Continuous variables for correlation analysis
continuous_vars = data[, c("age", "experience", "extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness")]

# Calculate correlation matrix
cor_matrix = cor(continuous_vars)

# Display the correlation matrix
print(cor_matrix)

# Correlation matrix plot
corrplot(cor_matrix, method = "circle", type = "upper",
         tl.col = "black", tl.srt = 45, addCoef.col = "black",
         diag = FALSE)

#####

# List of categorical variables
categorical_vars = c("gender", "industry", "profession", "traffic", "coach", "head_gender", "greywage", "way")

# Adding 0 = stayed 1 = left labels to the factor levels
data$event = factor(data$event, levels = c(0, 1), labels = c("Stayed", "Left"))

# Loop through each categorical var to create a contingency table and perform a Chi-squared test

for (cat_var in categorical_vars) {
  # Print the name of the variable being tested
  cat("\n---Table of:", cat_var, "---\n")
  table_event = table(data[[cat_var]], data$event)
  
  # Rename factor levels for 0 = stayed 1 = left
  dimnames(table_event) = list(Category = levels(data[[cat_var]]), Event = c("Stayed", "Left"))
  
  print(table_event)
  
  # Conduct chi-squared test
  test_result = chisq.test(table_event)
  
  print(test_result)
}

# *===========================*
# *         Modeling          *
# *===========================*

set.seed(123)
index = createDataPartition(data$event, p = 0.7, list = FALSE)
train_data = data[index, ]
test_data = data[-index, ]

# Function to convert factors to numeric
convert_factors = function(df) {
  data.frame(lapply(df, function(x) if(is.factor(x)) as.numeric(as.factor(x)) else x))
}

train_data_prep = convert_factors(train_data)
test_data_prep = convert_factors(test_data)


# Logistic Regression
log_model = glm(event ~ ., data = train_data, family = binomial(link = "logit"))
pred_prob_log = predict(log_model, newdata = test_data, type = "response")
roc_curve_log = roc(response = test_data$event, predictor = pred_prob_log)
auc_log = auc(roc_curve_log)


# Decision Tree
dt_model = rpart(event ~ ., data = train_data, method = "class")
dt_probs = predict(dt_model, newdata = test_data, type = "prob")[,2]
roc_curve_dt = roc(response = test_data$event, predictor = dt_probs)
auc_dt = auc(roc_curve_dt)


# Random Forest
rf_model = randomForest(event ~ ., data = train_data)
rf_probs = predict(rf_model, newdata = test_data, type = "prob")[,2]
roc_curve_rf = roc(response = test_data$event, predictor = rf_probs)
auc_rf = auc(roc_curve_rf)


# XGBoost
dtrain_xgb = xgb.DMatrix(data = as.matrix(train_data_prep[,-which(names(train_data_prep) == "event")]), label = train_data_prep$event - 1)
dtest_xgb = xgb.DMatrix(data = as.matrix(test_data_prep[,-which(names(test_data_prep) == "event")]))
xgb_params = list(objective = "binary:logistic")
xgb_model = xgb.train(params = xgb_params, data = dtrain_xgb, nrounds = 100)
xgb_preds = predict(xgb_model, newdata = dtest_xgb)
roc_curve_xgb = roc(response = as.numeric(test_data_prep$event) - 1, predictor = xgb_preds)
auc_xgb = auc(roc_curve_xgb)


plot(roc_curve_log, main="ROC Curves", col="red")
legend("bottomright", legend=c("Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"), col=c("red", "blue", "green", "purple"), lwd=2)
lines(roc_curve_dt, col="blue")
lines(roc_curve_rf, col="green")
lines(roc_curve_xgb, col="purple")

# Reporting AUCs
cat("AUC for Logistic Regression:", auc_log, "\n")
cat("AUC for Decision Tree:", auc_dt, "\n")
cat("AUC for Random Forest:", auc_rf, "\n")
cat("AUC for XGBoost:", auc_xgb, "\n")