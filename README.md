# ML_Project31-BreastCancerDetection

### Breast Cancer Detection - Using Machine Learning Classifiers
This project explores building machine learning models to classify breast cancer using the Wisconsin Diagnostic Breast Cancer (WBC) dataset. The code utilizes:

Libraries: pandas, numpy, scikit-learn (GaussianNB, LogisticRegression), seaborn, matplotlib
Dataset: WBC dataset (loaded using sklearn.datasets.load_breast_cancer())


### Key Steps:

### Data Loading and Preprocessing:

Load the WBC dataset.

Create a pandas DataFrame for easier data manipulation.

Explore data (head, value counts, correlations).

Split the data into features (X) and target variable (y).

Convert features and target to NumPy arrays for model compatibility.

Split the data into training and testing sets (using train_test_split).

### Model Building and Evaluation:

##### Gaussian Naive Bayes (GNB):
Train a GNB model on the training data.

Make predictions on the testing data.

Evaluate model performance using:

i.Accuracy score

ii.Classification report (precision, recall, F1-score)

iii.Confusion matrix (using seaborn heatmap)

##### Logistic Regression:
Train a logistic regression model on the training data (handling convergence warnings).

Make predictions on the testing data.

Evaluate model performance using the same metrics as GNB.


### Results:

The code demonstrates how to build and evaluate two machine learning models for breast cancer classification. Both GNB and Logistic Regression achieve high accuracy (above 95%). You can compare the performance metrics to determine which model might be more suitable for this specific task.

### Additional Considerations:

This is a basic example. Real-world applications might involve more sophisticated techniques like feature engineering, hyperparameter tuning, and ensemble methods.

Medical diagnosis should never rely solely on machine learning models. Consult a healthcare professional for any medical concerns.

### Future Enhancements:

Explore other machine learning algorithms (e.g., Support Vector Machines, Random Forests).

Implement hyperparameter tuning to optimize model performance.

Visualize feature distributions for potential insights.

Consider deploying the model as a web application for easier prediction.
