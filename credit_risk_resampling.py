#!/usr/bin/env python
# coding: utf-8

# # Credit Risk Resampling Techniques

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


# # Read the CSV and Perform Basic Data Cleaning

# In[3]:


columns = [
    "loan_amnt", "int_rate", "installment", "home_ownership",
    "annual_inc", "verification_status", "issue_d", "loan_status",
    "pymnt_plan", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "total_acc",
    "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt", "next_pymnt_d",
    "collections_12_mths_ex_med", "policy_code", "application_type", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_act_il",
    "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il",
    "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
    "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl",
    "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "hardship_flag", "debt_settlement_flag"
]

target = ["loan_status"]


# In[4]:


# Load the data
file_path = Path('LoanStats_2019Q1.csv')
df = pd.read_csv(file_path, skiprows=1)[:-2]
df = df.loc[:, columns].copy()

# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')

# Drop the null rows
df = df.dropna()

# Remove the `Issued` loan status
issued_mask = df['loan_status'] != 'Issued'
df = df.loc[issued_mask]

# convert interest rate to numerical
df['int_rate'] = df['int_rate'].str.replace('%', '')
df['int_rate'] = df['int_rate'].astype('float') / 100


# Convert the target column values to low_risk and high_risk based on their values
x = {'Current': 'low_risk'}   
df = df.replace(x)

x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk')    
df = df.replace(x)

df.reset_index(inplace=True, drop=True)

df.head()


# # Split the Data into Training and Testing

# In[5]:


# Create our features
X = df.drop(columns="loan_status", axis=1)
X = pd.get_dummies(X)


# Create our target
y = df["loan_status"]


# In[6]:


X.describe()


# In[7]:


# Check the balance of our target values
y.value_counts()


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
X_train.shape


# # Oversampling
# 
# In this section, you will compare two oversampling algorithms to determine which algorithm results in the best performance. You will oversample the data using the naive random oversampling algorithm and the SMOTE algorithm. For each algorithm, be sure to complete the folliowing steps:
# 
# 1. View the count of the target classes using `Counter` from the collections library. 
# 3. Use the resampled data to train a logistic regression model.
# 3. Calculate the balanced accuracy score from sklearn.metrics.
# 4. Print the confusion matrix from sklearn.metrics.
# 5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.
# 
# Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests

# ### Naive Random Oversampling

# In[9]:


# Resample the training data with the RandomOversampler
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

Counter(y_resampled)


# In[10]:


# Train the Logistic Regression model using the resampled data
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=1)
model.fit(X_resampled, y_resampled)


# In[11]:


y_pred = model.predict(X_test)

results = pd.DataFrame({"Prediction": y_pred, "Actual": y_test}).reset_index(drop=True)
results.head(20)


# In[12]:


# Calculated the balanced accuracy score
from sklearn.metrics import accuracy_score

acc_score = (accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[13]:


# Display the confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

matrix = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    matrix, index=["Actual High Risk", "Actual Low-Risk"], columns=["Predicted High Risk", "Predicted Low_Risk"])
cm_df


# In[14]:


# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))


# ### SMOTE Oversampling

# In[15]:


# Resample the training data with SMOTE
from imblearn.over_sampling import SMOTE

X_resample2, y_resample2 = SMOTE(random_state=1, sampling_strategy='auto').fit_resample(X_train, y_train)


# In[16]:


# Train the Logistic Regression model using the resampled data
model = LogisticRegression(random_state=1)

model.fit(X_resample2, y_resample2)
y_pred_sm = model.predict(X_test)


# In[17]:


# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score

acc_score2 = balanced_accuracy_score(y_test, y_pred_sm)
acc_score2


# In[18]:


# Display the confusion matrix
matrix_sm = confusion_matrix(y_test, y_pred_sm)

cm2_df = pd.DataFrame(
    matrix_sm, index=["Actual High-Risk", "Actual Low-Risk"], columns=["Predicted High_Risk", "Predicted Low_Risk"])
cm2_df


# In[19]:


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred_sm))


# # Undersampling
# 
# In this section, you will test an undersampling algorithms to determine which algorithm results in the best performance compared to the oversampling algorithms above. You will undersample the data using the Cluster Centroids algorithm and complete the folliowing steps:
# 
# 1. View the count of the target classes using `Counter` from the collections library. 
# 3. Use the resampled data to train a logistic regression model.
# 3. Calculate the balanced accuracy score from sklearn.metrics.
# 4. Print the confusion matrix from sklearn.metrics.
# 5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.
# 
# Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests

# In[20]:


# Resample the data using the ClusterCentroids resampler
# Warning: This is a large dataset, and this step may take some time to complete
from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=1)
X_resample3, y_resample3 = cc.fit_resample(X_train, y_train)
Counter(y_resample3)


# In[21]:


# Train the Logistic Regression model using the resampled data
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=1)
model.fit(X_resample3, y_resample3)
y_pred_cc = model.predict(X_test)


# In[22]:


# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score

acc_score3 = balanced_accuracy_score(y_test, y_pred_cc)
acc_score3


# In[23]:


# Display the confusion matrix
from sklearn.metrics import confusion_matrix
matrix_cc = confusion_matrix(y_test, y_pred_cc)

cm3_df = pd.DataFrame(
    matrix_cc, index=["Actual High-Risk", "Actual Low-Risk"], columns=["Predicted High_Risk", "Predicted Low-Risk"])
cm3_df


# In[24]:


# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred_cc))


# # Combination (Over and Under) Sampling
# 
# In this section, you will test a combination over- and under-sampling algorithm to determine if the algorithm results in the best performance compared to the other sampling algorithms above. You will resample the data using the SMOTEENN algorithm and complete the folliowing steps:
# 
# 1. View the count of the target classes using `Counter` from the collections library. 
# 3. Use the resampled data to train a logistic regression model.
# 3. Calculate the balanced accuracy score from sklearn.metrics.
# 4. Print the confusion matrix from sklearn.metrics.
# 5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.
# 
# Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests

# In[25]:


# Resample the training data with SMOTEENN
# Warning: This is a large dataset, and this step may take some time to complete
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=1)
X_resample4, y_resample4 = smote_enn.fit_resample(X, y)


# In[26]:


# Train the Logistic Regression model using the resampled data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=1)

model.fit(X_resample4, y_resample4)
from sklearn.metrics import confusion_matrix
y_pred_st = model.predict(X_test)


# In[27]:


# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
acc_score4 = balanced_accuracy_score(y_test, y_pred_st)
acc_score4


# In[28]:


# Display the confusion matrix
from sklearn.metrics import confusion_matrix
matrix_st = confusion_matrix(y_test, y_pred_st)

cm4_df = pd.DataFrame(matrix_st, index=["Actual High-Risk", "Actual Low-Risk"], columns=["Predicted High_Risk", "Predicted Low-Risk"])
cm4_df


# In[29]:


# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred_st))


# In[ ]:




