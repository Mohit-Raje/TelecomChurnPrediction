<h1>Telecom Churn Prediction using Gradient Boosting Algorithm</h1>

<h2>Step 1: Data Collection</h2>
<ul>
  <li>Data Source – Kaggle</li>
</ul>

<h2>Step 2: EDA and Data Cleaning</h2>
<ul>
  <li>Conducted univariate analysis of numeric and categorical features.</li>
  <li>Performed bivariate analysis of numeric and categorical features with respect to the target column “Churn” using histograms and count plots (with <code>hue="Churn"</code>).</li>
  <li>Identified outliers using boxplots and the IQR method.</li>
  <li>Analyzed skewness and kurtosis of numerical distributions.</li>
  <li>Explored correlations among independent numeric features.</li>
  <li>Performed bivariate analysis of numeric features using the <code>groupby</code> method to derive insights.</li>
  <li>Evaluated the correlation of numeric features with the categorical target column “Churn” using the <code>pointbiserialr</code> method from <code>scipy.stats</code>.</li>
  <li>Analyzed categorical feature distributions by “Churn” using the <code>value_counts</code> method.</li>
  <li>Conducted a detailed analysis of categorical features with respect to the target “Churn” using <code>groupby</code> and <code>value_counts()</code>.</li>
  <li>Identified and removed less important features based on their impact on the model using <code>feature_importances_</code> from a trained Random Forest classifier.</li>
</ul>

<h2>Step 3: Feature Engineering</h2>
<ul>
  <li>Combined similar categories where appropriate (e.g., merged “No internet service” into the “No” category).</li>
  <li>Identified ordinal and nominal features for tailored preprocessing.</li>
  <li>Created separate pipelines:
    <ul>
      <li><strong>Numeric Pipeline</strong> – Applied imputation and scaling.</li>
      <li><strong>Ordinal Pipeline</strong> – Determined feature hierarchy, applied imputation, and used an Ordinal Encoder.</li>
      <li><strong>Nominal Pipeline</strong> – Applied imputation and One-Hot Encoding.</li>
    </ul>
  </li>
  <li>Combined all pipelines using a <code>ColumnTransformer</code>.</li>
  <li>Split the dataset into training and testing sets.</li>
  <li>Applied the preprocessing and feature engineering pipeline to the data.</li>
</ul>

<h2>Step 4: Machine Learning Model Training and Evaluation</h2>
<ul>
  <li>Tested multiple classification algorithms and evaluated them based on precision, recall, and F1-score.</li>
  <li>Selected the Gradient Boosting algorithm for its balanced performance.</li>
  <li>Prioritized recall to minimize false negatives in identifying potential churners.</li>
</ul>

<h2>Step 5: Hyperparameter Tuning of Gradient Boosting Algorithm</h2>
<ul>
  <li>Used <code>GridSearchCV</code> to explore various combinations of parameters.</li>
  <li>Applied cross-validation to assess generalization performance.</li>
  <li>Trained the final model using the best parameters from the grid search.</li>
</ul>

<h2>Step 6: Improving Accuracy</h2>
<ul>
  <li><strong>PCA (Principal Component Analysis):</strong>
    <ul>
      <li>Experimented with different numbers of principal components.</li>
      <li>Selected 10 components capturing over 90% of the total variance.</li>
      <li>Retrained the model using PCA-transformed data.</li>
    </ul>
  </li>
  <li><strong>Feature Importance:</strong>
    <ul>
      <li>Assessed each feature’s contribution using <code>feature_importances_</code>.</li>
      <li>Created a DataFrame to rank features by importance.</li>
      <li>Sorted features and identified the least impactful ones.</li>
    </ul>
  </li>
  <li><strong>SHAP Analysis:</strong>
    <ul>
      <li>Used SHAP (SHapley Additive exPlanations) with <code>TreeExplainer</code>.</li>
      <li>Visualized features with the greatest impact on predictions.</li>
      <li>Sorted features by SHAP impact and focused on those contributing 1–2%.</li>
    </ul>
  </li>
</ul>

<h2>Step 6 (Continued): Model Refinement</h2>
<ul>
  <li>Dropped low-importance features and retrained the model.</li>
  <li>Re-evaluated precision, recall, and F1-score after feature removal.</li>
  <li>Finalized the reduced feature set if performance remained stable.</li>
</ul>

<h2>Step 7: Model and Pipeline Serialization</h2>
<ul>
  <li>Saved the trained model using <code>Pickle</code>.</li>
  <li>Serialized the preprocessing pipeline using <code>Pickle</code> for reuse during inference.</li>
</ul>

<h2>Step 8: Custom Predictions</h2>
<ul>
  <li>Loaded the pickled model and preprocessor pipeline.</li>
  <li>Collected input data from users.</li>
  <li>Converted input into a dictionary (key-value pairs).</li>
  <li>Transformed the dictionary into a <code>DataFrame</code>.</li>
  <li>Applied preprocessing pipeline to user input.</li>
  <li>Passed processed data to the model for prediction.</li>
</ul>

<h2>Step 9: End-to-End Application Development</h2>
<ul>
  <li>Built an interactive web application using <strong>Streamlit</strong>.</li>
  <li>Integrated the model and preprocessor for real-time predictions.</li>
</ul>

<h2>Step 10: Deployment</h2>
<ul>
  <li>Deployed the Streamlit app using <strong>GitHub</strong> for version control and hosting.</li>
</ul>
