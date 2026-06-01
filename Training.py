# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from joblib import dump


df = pd.read_csv('diabetes.csv')

# Split features and target
X = df.drop('Outcome', axis=1) 
y = df['Outcome']

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5) 
knn_model.fit(X_train_scaled, y_train)

# ANN (Levenberg-Marquardt Algorithm using Keras/TensorFlow)
# Defining the ANN structure
ann_model = Sequential()
ann_model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
ann_model.add(Dense(8, activation='relu'))
ann_model.add(Dense(1, activation='sigmoid'))

# Compile the model using an optimizer that approximates Levenberg-Marquardt (can use 'adam' as a proxy)
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN model
ann_model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=0)

# Saving models
dump(rf_model, 'random_forest_model.joblib')
dump(knn_model, 'knn_model.joblib')
ann_model.save('ann_model.h5')

# Evaluating models
rf_predictions = rf_model.predict(X_test)
knn_predictions = knn_model.predict(X_test_scaled)
ann_predictions = (ann_model.predict(X_test_scaled) > 0.5).astype(int)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("KNN Accuracy:", accuracy_score(y_test, knn_predictions))
print("ANN Accuracy:", accuracy_score(y_test, ann_predictions))
