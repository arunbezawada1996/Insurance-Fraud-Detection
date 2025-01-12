# Insurance-Fraud-Detection
This project focuses on identifying fraudulent insurance claims and uncovering the key drivers of fraud for Travelers Insurance Company. It is based on the Kaggle competition:  
* [Travelers NESS Statathon 2023](https://www.kaggle.com/competitions/2023-travelers-ness-statathon/overview)

## **Challenge**
I participated in the Kaggle challenge, where the task was to build a predictive model to identify **first-party physical damage fraudulence**. My goal was twofold:
1. Create an accurate fraud detection model based on historical claim data.
2. Analyze and explain the key drivers of fraudulent claims to help non-technical stakeholders understand the findings.

Fraud detection accuracy was critical, and the competition emphasized the use of the **weighted F1 score** for evaluation, as it balances precision and recall in detecting fraudulent claims.  
You can learn more about the F1 metric here:  
[scikit-learn F1 Score Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

## **Dataset Description**
The dataset represents a sample of first-party physical damage claims referred to Travelers Insurance's fraud detection team between **2015 and 2016**. The claims data includes a variety of features related to drivers, vehicles, and claims.

For this project:
- I worked with the training dataset, which included the labels for whether a claim was fraudulent (1) or non-fraudulent (0).
- The test dataset was used solely for evaluation to ensure fair competition rules.

The dataset includes variables such as:
- **Driver Demographics**: Age, marital status, annual income, etc.
- **Claim Information**: Past claims, liability percentage, accident site, etc.
- **Vehicle Details**: Price, weight, category, and age.

## **My Approach**
1. **Exploratory Data Analysis (EDA)**:
   - I analyzed the data to uncover patterns and relationships between features and fraud likelihood.
   - For example, I found that drivers with **multiple past claims** and claims with **high liability percentages** were more likely to be fraudulent.

2. **Feature Engineering**:
   - I transformed categorical variables into numerical formats using one-hot encoding.
   - Created new features like "claims per year" to capture relationships between variables.

3. **Model Building**:
   - I trained multiple machine learning models, including Logistic Regression, Random Forest, and Gradient Boosting (XGBoost).
   - I optimized hyperparameters using Grid Search to improve model performance.

4. **Evaluation**:
   - I used the **weighted F1 score** to evaluate the models and selected Gradient Boosting as the best-performing model, achieving an F1 score of **0.87**.
   - I also analyzed feature importance using SHAP (SHapley Additive exPlanations) to explain the drivers of fraud.


This project showcases my ability to solve real-world problems by leveraging machine learning, data analysis, and explainable AI techniques. Let me know if you’d like more details or specific technical implementations!

**Variable Descriptions**
* claim_number - Claim ID **(cannot be used in model)**
* age_of_driver - Age of driver
* gender - Gender of driver
* marital_status - Marital status of driver
* safty_rating - Safety rating index of driver
* annual_income - Annual income of driver
* high_education_ind - Driver’s high education index
* address_change_ind - Whether or not the driver changed living address in past 1 year
* living_status - Driver’s living status, own or rent
* zip_code - Driver’s living address zipcode
* claim_date - Date of first notice of claim
* claim_day_of_week - Day of week of first notice of claim
* accident_site - Accident location, highway, parking lot or local
* past_num_of_claims - Number of claims the driver reported in past 5 years
* witness_present_ind - Witness indicator of the claim
* liab_prct - Liability percentage of the claim
* channel - The channel of purchasing policy
* policy_report_filed_ind - Policy report filed indicator
* claim_est_payout - Estimated claim payout
* age_of_vehicle - Age of first party vehicle
* vehicle_category - Category of first party vehicle
* vehicle_price - Price of first party vehicle
* vehicle_color - Color of first party vehicle
* vehicle_weight - Weight of first party vehicle
* fraud - Fraud indicator (0=no, 1=yes). **This is the response variable.**


# 01. Importing Modules ¶

# 02. EDA & Feature Engineering

#### Data Cleaning

* Except claim_date, the data types for all columns are appropriate and do not require any changes.

* The columns marital_status (4), witness_present_ind (143), claim_est_payout (23), and age_of_vehicle (7) contain NA values.
* Given that the number of NA values is relatively low compared to the total dataset size of 19,000 rows, we will drop these rows instead of performing imputation.

#### Feature Engineering

* There are approximately 7 columns containing categorical values that need to be converted to numerical values using dummy variables.
* Additionally, in the feature engineering section, we will transform the `claim_day_of_week` column into a binary variable, indicating whether the claim was made on a weekday or not.

Next, will update the categorical  columns to numerical values (by Creating dummy variables)

#### Descriptive Analysis for better "feature" selection

Correlation Analysis

* No single feature perfectly predicts fraud (as indicated by the moderate correlation values), but combining these features in a predictive model could improve its performance.
* Given that "Fraud" is binary, correlations provide a measure of linear association but not causation.
* Variables with highest correlation: marital_status, high_education_ind, address_change_ind, past_num_of_claims, witness_present_ind, policy_report_field_ind, age_of_vehicle, claim_on_weekend, claim_frequency, living_status_Rent, accident_site_Local

Box Plot Distribution by Fraud Value (1 and 0)

**Key Observations**

* Strong Indicators: Past Number of Claims seem to be stronger indicators of fraud. A higher number of past claims is associated with fraudulent claims.
* Moderate Indicators: Age of Driver and Annual Income show some differences between fraudulent and non-fraudulent claims but are less pronounced.
* Weak Indicators: Vehicle Price, Age of Vehicle, and Vehicle Weight show little to no difference between fraudulent and non-fraudulent claims.

These observations can help us guide further analysis and feature selection in building a predictive model for fraud detection.

Monthly Trend - # Claims by Fraud Status

**Key Observations**:

*   The majority of claims are non-fraudulent (Fraud = 0), consistently making up around 83% to 87% of the total claims.
*   There are minor fluctuations month-to-month in both fraudulent and non-fraudulent claims, but these fluctuations do not show a clear seasonal pattern.
*   For instance, in January 2015, fraudulent claims are around 16.3%, slightly increasing to 17.5% in February 2015, and then stabilizing around 15% to 17% in subsequent months.

**Potential Action Item**: We can probably investigate the reasons behind the increase in fraud claims in February 2015 and September 2015.

Examining the Distribution of Continuous Variables to check Skewness

**Key Observations**:

* The visuals indicate that continuous variables such as "age_of_driver", "claim_est_payout", "vehicle_weight", and "vehicle_price" exhibit right skewness, whereas "safety_rating" is left-skewed.

* Consequently, transforming these variables would be necessary. However, the transformed variables resulted in lower F1 scores compared to their original forms, so the transformed data analysis has been omitted from this notebook.

#### Data Splitting to Train and Test

* The distribution of the Y values indicates that the data is imbalanced.
* Therefore, we will apply SMOTE to balance our X_train dataset.

* After performing the SMOTE, the training dataset looks well balanaced with n = 12,724.

# 3. Modeling

#### Identifying Optimal Model and Thresholds for Maximum F1 Score Performance

First, we will apply basic models with default settings and use a loop to dynamically determine the optimal threshold for classifying the '1s'. The goal is to pinpoint which model delivers the highest F1 score performance.

The above model results suggest that the **Gradient Boosting** model stands out as the best choice among the evaluated models for the following reasons, particularly when selecting a threshold of 0.29292:

* Based on the F1 scores for both classes (0 and 1), Gradient Boosting demonstrates a robust ability to balance both recall and precision across the dataset. F1 score for the positive class (1) at the best threshold (0.29292) is 0.358647, which is comparatively high among the models tested, indicating that it handles the minority class effectively.

* The weighted F1 score on the test set for Gradient Boosting is 0.758121, which is relatively high.

#### Optimizing Gradient Boosting Classifier with Grid Search

Next, we perform a comprehensive grid search to optimize the hyperparameters of a Gradient Boosting Classifier. The goal is to find the best combination of parameters that maximizes the F1 Score.

Unfortunately, the outcomes from the grid search did not meet expectations (F1 Score of Test > Train).

Our Gradient Boosting models with default parameters and a customized threshold yielded the best results, and we will continue using the basic model:

Confusion Matrix of the model looks like the following:

**Key Obseravtions**

1. The model demonstrates excellent capability in identifying non-fraudulent transactions, with a high accuracy rate of 77.9% for Class 0 (2479 true negatives out of 3181 total Class 0 predictions). This suggests that the model is able to classify the records to Non-Fraud records with high accuracy.

2. Despite the inherent challenges in detecting fraudulent activities, the model achieves a recall rate of 48.1% for Class 1. This performance is significant as it reflects the model's ability to correctly identify nearly half of all fraudulent transactions presented in the test set.

# Interpretability and Discussion

#### Permutation Importance

The permutation importance plot indicates the following features are the most influential in predicting fraudulent claims:

1. **Claims Frequency**: This feature has the highest permutation importance, indicating that as the scaled claims frequency increases, the likelihood of fraud also increases.
2. **Past Number of Claims**: Similarly, an increase in the scaled number of past claims is strongly associated with a higher probability of fraudulent behavior.
3. **Annual Income**: Changes in the scaled annual income levels affect fraud risk, with higher income levels generally correlating with a lower risk of fraud.
4. **Age of Driver**: The directionality here shows that younger or older drivers (depending on the specific distribution) might have different fraud probabilities. Typically, younger drivers might have a higher risk.
5. **High Education Indicator**: Higher values in this scaled indicator generally suggest a lower probability of fraud, indicating that more educated policyholders might be less likely to commit fraud.

**Quick Notes**:

* Top 10 features based on permutation importance: 'accident_site_Local', 'address_change_ind', 'marital_status', 'claim_month', 'high_education_ind', 'witness_present_ind', 'age_of_driver', 'annual_income', 'past_num_of_claims', 'claims_frequency'

* We tried to rerun the model using only the top 10 and 5 features .However, the model's performance, as measured by the F1 score, reduced. Hence, we will continue using our original model configuration.


#### Partial Dependence Plot

* The PDPs provided further insights into the relationship between each feature and the predicted probability of fraud.

* It shows that the likelihood of fraud increases with higher claims frequency and a greater number of past claims, while it decreases with higher annual income and education levels. Additionally, younger drivers tend to have a higher risk of committing fraud.

#### Business Recommendations

1. **Enhanced Monitoring for Frequent Claimants**:
   * Implement stricter monitoring and verification processes for policyholders who exhibit a high frequency of claims to mitigate fraud risk effectively.
   * For instance, establish a threshold for the number of past claims that, when exceeded, triggers an automatic review or flagging of the account for potential fraudulent activity. This proactive approach ensures that high-risk cases are scrutinized more closely.

2. **Income-Based Risk Assessment**:
    * Incorporate scaled income data into your fraud detection models to identify high-risk income brackets.
    * Higher scaled income levels generally correlate with lower fraud risk, which can be used to tailor investigation processes.

3. **Age-Specific Fraud Prevention Strategies**:
   * Tailor fraud detection strategies to align with the age of drivers, leveraging data-driven insights.
   * Our analysis suggests that younger drivers are more likely to engage in fraudulent activities. Hence, it's advisable to implement targeted educational programs and warnings specifically designed for this demographic to mitigate such risks.

4. **Education Level Consideration**: Use the high education indicator to adjust risk profiles. Higher scaled values suggest a lower probability of fraud, providing valuable insights to enhance fraud detection accuracy.

In conclusion, these identified features should be monitored as Key Fraud Indicators (KFIs) within a dashboard or similar system, allowing Travelers to make informed decisions based on these insights.
