# %% [markdown]
# # Framingham Heart Study
# 
# This script initializes and trains multiple machine learning models on the Framingham Heart Study dataset,
# evaluates their performance, and visualizes the results using confusion matrices.
# Models used:
# - Random Forest
# - Decision Tree
# - XGBoost
# - Gradient Boost
# - Multi-Layer Perceptron (MLP)
# The script performs the following steps:
# 1. Initializes the models with specified parameters.
# 2. Trains each model on the training dataset (X_train, y_train).
# 3. Makes predictions on the test dataset (X_test).
# 4. Calculates the accuracy of each model.
# 5. Prints the classification report for each model.
# 6. Plots and displays the confusion matrix for each model.
# Variables:
# - models: Dictionary containing the initialized models.
# - model_acc: Dictionary to store the accuracy of each model.
# - X_train: Training feature set.
# - y_train: Training labels.
# - X_test: Test feature set.
# - y_test: Test labels.
# Libraries required:
# - matplotlib
# - seaborn
# - scikit-learn
# - xgboost

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


# %%
# Load the dataset 
df = pd.read_csv('framingham.csv')

# %%
# Display basic info about the dataset
print(df.info())
print(df.head())

# %%
# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Replace missing values with the mean for numeric columns
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Feature selection (we will drop 'TenYearCHD' from the features since it is the target)
X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Standardize the data using StandardScaler (important for some models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
# Correlation Map
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr() # Correlation matrix for the dataset
sns.heatmap(correlation_matrix, annot=True, fmt=".1f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Map")
plt.show()

# %%

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "Gradient Boost" : GradientBoostingClassifier(random_state=42),
    "MLP" :MLPClassifier((256, ), max_iter=500, random_state=42),
}

# Train and evaluate each model
model_acc = {}
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_acc[model_name] = accuracy
    
    # Evaluate the model
    print(f"\n{model_name} Model:")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", 
                cmap="Blues", 
                xticklabels=["No CHD", "CHD"], 
                yticklabels=["No CHD", "CHD"])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    


# %%
# Report model's accuracy
print('Accuracy report:')
for model_name, accuracy in model_acc.items():
    print(model_name, ' : ', accuracy)


# %%
# param_grid_rf is a dictionary containing the hyperparameters for a Random Forest classifier.
# The keys represent the hyperparameter names and the values are lists of possible values for each hyperparameter.
# 
# Hyperparameters:
# - 'n_estimators': Number of trees in the forest. Possible values: [100, 200, 300]
# - 'criterion': Function to measure the quality of a split. Possible values: ['gini', 'entropy']
# - 'max_depth': Maximum depth of the tree. Possible values: [None, 10, 20, 30]
# - 'min_samples_split': Minimum number of samples required to split an internal node. Possible values: [2, 5, 10]
# - 'min_samples_leaf': Minimum number of samples required to be at a leaf node. Possible values: [1, 2, 4]

# Define the parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, ],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the model
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV for Random Forest
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf,
                              cv=3, n_jobs=-1, verbose=3)

# Fit the grid search to the data for Random Forest
grid_search_rf.fit(X_train, y_train)

# Print the best parameters and best score for Random Forest
print("Best accuracy found for Random Forest: ", grid_search_rf.best_score_)
print("Best parameters found for Random Forest: ", grid_search_rf.best_params_)


# %%



