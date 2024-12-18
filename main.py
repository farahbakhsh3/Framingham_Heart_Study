# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer

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
correlation_matrix = df.corr()  # Correlation matrix for the dataset
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
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



