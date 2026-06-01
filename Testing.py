import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score
from joblib import load
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the diabetic dataset again (assuming the same CSV file)
df = pd.read_csv('diabetes.csv')
# Split features and target
X = df.drop('Outcome', axis=1)  # Assuming the label column is named 'Outcome'
y = df['Outcome']

# Split data into 80% training and 20% testing (should be consistent with previous split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data (for KNN and ANN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the previously saved models
rf_model = load('random_forest_model.joblib')
knn_model = load('knn_model.joblib')
ann_model = load_model('ann_model.h5')

# Test Random Forest
rf_predictions = rf_model.predict(X_test)

# Test KNN
knn_predictions = knn_model.predict(X_test_scaled)

# Test ANN
ann_predictions = (ann_model.predict(X_test_scaled) > 0.5).astype(int).flatten()

# Read CSV file
dff = pd.read_csv('res.csv')

# Define actual and predicted columns for the models
y_true = dff['Actual']
y_pred_ann = dff['ANN']
y_pred_rf = dff['RF']
y_pred_knn = dff['KNN']

# Function to calculate all performance metrics
def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = recall_score(y_true, y_pred)  # Sensitivity = recall = TP / (TP + FN)
    specificity = tn / (tn + fp)
    ppv = precision_score(y_true, y_pred)  # PPV = precision = TP / (TP + FP)
    npv = tn / (tn + fn)
    fpr = fp / (fp + tn)  # FPR = 1 - specificity
    fnr = fn / (fn + tp)  # FNR = 1 - sensitivity
    fdr = fp / (fp + tp)  # FDR = 1 - PPV

    return accuracy, sensitivity, specificity, ppv, npv, fpr, fnr, fdr

# Calculate metrics for each model
metrics_ann = calculate_metrics(y_true, y_pred_ann)
metrics_rf = calculate_metrics(y_true, y_pred_rf)
metrics_knn = calculate_metrics(y_true, y_pred_knn)

# Create a DataFrame to store the metrics for each model
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'FPR', 'FNR', 'FDR'],
    'ANN': metrics_ann,
    'RF': metrics_rf,
    'KNN': metrics_knn
})

# Print the metrics
print(metrics_df)

# Function to plot each metric individually
def plot_individual_metric(metric_name):
    plt.figure(figsize=(7, 5))
    sns.barplot(x=['ANN', 'RF', 'KNN'], y=metrics_df.set_index('Metric').loc[metric_name])
    plt.title(f'Comparison of {metric_name}')
    plt.ylabel(metric_name)
    plt.xlabel('Model')
    plt.tight_layout()
    plt.show()
print("\033[1mThis is bold text\033[0m")
# Loop through each metric and plot it individually
for metric in metrics_df['Metric']:
    plot_individual_metric(metric)


# ROC Curve plot
def plot_roc_curve(y_true, y_pred, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Plot ROC curve for each model
plt.figure(figsize=(10, 7))
plot_roc_curve(y_true, y_pred_ann, 'ANN')
plot_roc_curve(y_true, y_pred_rf, 'RF')
plot_roc_curve(y_true, y_pred_knn, 'KNN')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
print("\033[1mThis is bold text\033[0m")