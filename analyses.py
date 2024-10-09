import pandas as pd
import numpy as np
import forestplot as fp # pip install forestplot; https://github.com/LSYS/forestplot?tab=readme-ov-file
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import shap
from helper_functions import * # functions used in analyses below
from figure_functions import * # functions to plot figures

df = pd.read_pickle("cohort3.pkl")  # load DataFrame from the pickle file

include_interaction_term = 0 # 1 if including interaction term, relevant for the LR analysis

# Identify and remove features which are excluded from the analysis i.e. which data the analysis is blind to
factors_to_exclude = [] # 'Z' if wanting to exclude 'Z' variable for data deficiency analysis
df = df.drop(columns=factors_to_exclude) # remove them from the df if modelling that the trial didn't collect that data

# Set values in 'Trial outcome' based upon whether they received treatment or placebo
df.loc[df['Treatment arm'] == 1, 'Trial outcome'] = df.loc[df['Treatment arm'] == 1, 'Rx 3 outcome'] # Rx 3 = the treatment with 3 variables defining treatment response
df.loc[df['Treatment arm'] == 0, 'Trial outcome'] = df.loc[df['Treatment arm'] == 0, 'Placebo outcome']

# RCT simulation
# Overall group analysis
outcome_table = pd.DataFrame()
overall_treatment_outcome = df[df['Treatment arm'] == 1]['Trial outcome']
overall_placebo_outcome = df[df['Treatment arm'] == 0]['Trial outcome']
(overall_treatment_mean, overall_treatment_lower_ci, overall_treatment_upper_ci,
 overall_placebo_mean, overall_placebo_lower_ci, overall_placebo_upper_ci,
 overall_mean_difference, overall_lower_ci_difference, overall_upper_ci_difference,
 overall_p_value) = mean_confidence_interval_and_p_value(overall_treatment_outcome, overall_placebo_outcome)
overall_data = {'Group': 'Overall', 
                'Treatment Mean': overall_treatment_mean, 'Treatment Lower CI': overall_treatment_lower_ci, 'Treatment Upper CI': overall_treatment_upper_ci, 
                'Placebo Mean': overall_placebo_mean, 'Placebo Lower CI': overall_placebo_lower_ci, 'Placebo Upper CI': overall_placebo_upper_ci, 
                'Mean Difference': overall_mean_difference, 'Difference Lower CI': overall_lower_ci_difference, 'Difference Upper CI': overall_upper_ci_difference,
                'P-value': overall_p_value, 'Group_variable': 'Overall', 'Label': 'All'} 
outcome_table = pd.concat([outcome_table, pd.DataFrame([overall_data])], ignore_index=True)

# Subgroup analyses
subgroup_columns = [col for col in df.columns.tolist() if col.startswith(("Age","Sex","V","X","Y","Z"))]
subgroup_columns_present = [replace_words_present(col) for col in subgroup_columns]    
subgroup_columns_not = [replace_words_absent(col) for col in subgroup_columns]

col_count = 0
for col in subgroup_columns:
    if set(df[col].unique()) == {0, 1}: # if it is a binary variable
        for val in [1,0]:
            # for treatment in [0, 1]:
            treatment_outcome = df[(df['Treatment arm'] == 1) & (df[col] == val)]['Trial outcome']
            placebo_outcome = df[(df['Treatment arm'] == 0) & (df[col] == val)]['Trial outcome']
            (treatment_mean, treatment_lower_ci, treatment_upper_ci,
             placebo_mean, placebo_lower_ci, placebo_upper_ci,
             mean_difference, lower_ci_difference, upper_ci_difference,
             p_value) = mean_confidence_interval_and_p_value(treatment_outcome, placebo_outcome)
            if val == 1:
                label = subgroup_columns_present[col_count] # how to refer to it being present
            elif val == 0:
                label = subgroup_columns_not[col_count] # how to refer to it not being present
            subgroup_data = {'Group': f'{col}: {val}', 
                            'Treatment Mean': treatment_mean, 'Treatment Lower CI': treatment_lower_ci, 'Treatment Upper CI': treatment_upper_ci, 
                            'Placebo Mean': placebo_mean, 'Placebo Lower CI': placebo_lower_ci, 'Placebo Upper CI': placebo_upper_ci, 
                            'Mean Difference': mean_difference, 'Difference Lower CI': lower_ci_difference, 'Difference Upper CI': upper_ci_difference,
                            'P-value': p_value, 'Group_variable': col, 'Label': label}
            outcome_table = pd.concat([outcome_table, pd.DataFrame([subgroup_data])], ignore_index=True)
    else:  # if it is a continuous variable
        if len(df[col].unique()) > 2:
            # Split the column into three subgroups based on quantiles
            quantiles = df[col].quantile([1/3, 2/3])
            for i, (low, high) in enumerate(zip([df[col].min(), quantiles.iloc[0], quantiles.iloc[1]],
                                                [quantiles.iloc[0], quantiles.iloc[1], df[col].max()])):
                treatment_outcome = df[(df['Treatment arm'] == 1) & (df[col] >= low) & (df[col] <= high)]['Trial outcome']
                placebo_outcome = df[(df['Treatment arm'] == 0) & (df[col] >= low) & (df[col] <= high)]['Trial outcome']
                (treatment_mean, treatment_lower_ci, treatment_upper_ci,
                 placebo_mean, placebo_lower_ci, placebo_upper_ci,
                 mean_difference, lower_ci_difference, upper_ci_difference,
                 p_value) = mean_confidence_interval_and_p_value(treatment_outcome, placebo_outcome)
                label = str(int(low)) + ' - ' + str(int(high)) # range that those values take
                subgroup_data = {'Group': f'{col}: {i+1}', 
                                'Treatment Mean': treatment_mean, 'Treatment Lower CI': treatment_lower_ci, 'Treatment Upper CI': treatment_upper_ci, 
                                'Placebo Mean': placebo_mean, 'Placebo Lower CI': placebo_lower_ci, 'Placebo Upper CI': placebo_upper_ci, 
                                'Mean Difference': mean_difference, 'Difference Lower CI': lower_ci_difference, 'Difference Upper CI': upper_ci_difference,
                                'P-value': p_value, 'Group_variable': col, 'Label': label}
                outcome_table = pd.concat([outcome_table, pd.DataFrame([subgroup_data])], ignore_index=True)
    col_count+=1

# Sort the outcome_table dataframe alphabetically
overall_row = outcome_table[outcome_table['Group'] == 'Overall'] # Remove the 'Overall' row and save it
outcome_table = outcome_table[outcome_table['Group'] != 'Overall']
outcome_table = outcome_table.sort_values(by='Group') # Sort the DataFrame by the 'Group' column
outcome_table = pd.concat([outcome_table, overall_row])

figure_2(subgroup_columns,outcome_table) # plot figure 2 - forest plots

plt.rcParams['font.family'] = 'Arial' # make all plots Arial font

# ML classification on trial outcome
df_trial = df[df['Treatment arm'] == 1] # Just data for people who received treatment, to try and predict i.e. don't use placebo which offers no information about responses to people receiving treatment
X_trial_df = df_trial[subgroup_columns]   

# Identify binary columns (those with only 2 unique values)
binary_columns = [col for col in X_trial_df.columns if X_trial_df[col].nunique() == 2]
non_binary_columns = list(set(X_trial_df.columns) - set(binary_columns)) # Get non-binary columns by difference

scaler = StandardScaler() # Create a scaler object
X_trial_before_scaled = X_trial_df.values
X_trial_df[non_binary_columns] = scaler.fit_transform(X_trial_df[non_binary_columns]) # Fit the scaler to the non-binary columns and transform them
X_trial_scaled = X_trial_df.values

# Add interaction terms
if include_interaction_term == 1:
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False) # PolynomialFeatures(degree=2, include_bias=False) will generate interaction terms
    X_trial_scaled = poly.fit_transform(X_trial_scaled) # with added interaction terms
    interaction_names = poly.get_feature_names_out(input_features=X_trial_df.columns)
    feature_names = interaction_names
else:
    feature_names = X_trial_df.columns

# Trial outcome labels
y_Rx_trial = df_trial['Trial outcome'].values # Extract values of target variable (y_Rx_trial) from the filtered DataFrame
y_trial = np.where(y_Rx_trial > 5, 1, 0) # if above 5 deem it a treatment response, if below 5 deem it non-responsive
y_truth = df_trial['Rx 3 responsive'].values # Ground truth labels (Rx 3 referring to treatment with 3 phenotype variables defining the treatment response)

lr_model = LogisticRegression() # initialize logistic regression model

# Define XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0,
    'reg_lambda': 1,
}

xgb_model = xgb.XGBClassifier(**params)  # initialize XGB model

n_splits = 5 # number of folds for cross-validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Setup empty variables to be filled in ML analyses
shap_values = []; y_pred = np.zeros_like(y_trial, dtype=float); shap_values = np.zeros(((len(X_trial_scaled),len(feature_names))), dtype=float)
lr_coefficients = []; y_pred_lr = np.zeros_like(y_trial, dtype=float)
 
# Perform cross-validation
for train_index, test_index in kf.split(X_trial_scaled):
    X_train, X_test = X_trial_scaled[train_index], X_trial_scaled[test_index]
    y_train, y_test = y_trial[train_index], y_trial[test_index]
    
    # XGB model fit/prediction
    xgb_model.fit(X_train, y_train) # Fit the model on the training data
    y_pred[test_index] = xgb_model.predict(X_test) # Predict on the test data
    
    # Compute SHAP values for XGB
    explainer = shap.Explainer(xgb_model, feature_names=feature_names)
    shap_values[test_index,:] = explainer.shap_values(X_test)
    
    # Logistic regression model fit/coefficient estimation/prediction
    lr_model.fit(X_train, y_train) # Train the model
    lr_coefficients.append(lr_model.coef_[0]) # Get the feature coefficients
    y_pred_lr[test_index] = lr_model.predict(X_test) # get the logistic regression model predictions

# Print LR and XGB ML analyses classification metrics
print("\nLR vs trial outcome classification report:")
class_report_lr_trial = classification_report(y_trial, y_pred_lr)
conf_matrix_lr_trial = confusion_matrix(y_trial, y_pred_lr)
print(class_report_lr_trial)
accuracy_and_ci(conf_matrix_lr_trial)

print("\nLR vs ground truth classification report:")
class_report_lr_truth = classification_report(y_truth, y_pred_lr)
conf_matrix_lr_truth = confusion_matrix(y_truth, y_pred_lr)
print(class_report_lr_truth)
accuracy_and_ci(conf_matrix_lr_truth)

print("\nXGB vs trial outcome measure classification report:")
class_report_xgb_trial = classification_report(y_trial, y_pred)
conf_matrix_xgb_trial = confusion_matrix(y_trial, y_pred)
print(class_report_xgb_trial)
accuracy_and_ci(conf_matrix_xgb_trial)

print("\nXGB vs ground truth classification report:")
class_report_xgb_truth = classification_report(y_truth, y_pred)
conf_matrix_xgb_truth = confusion_matrix(y_truth, y_pred)
print(class_report_xgb_truth)
accuracy_and_ci(conf_matrix_xgb_truth)

figure_3(conf_matrix_xgb_trial,conf_matrix_xgb_truth) # plot figure 3 - confusion matrices
figure_4(shap_values,X_trial_scaled,feature_names) # plot figure 4 - SHAP values
figure_5(y_trial,X_trial_before_scaled,shap_values,factors_to_exclude) # plot figure 5