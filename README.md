# Framingham Heart Study
Using Machine Learning for Analysis

This script initializes and trains multiple machine learning models on the Framingham Heart Study dataset,
evaluates their performance, and visualizes the results using confusion matrices.
Models used:
- Random Forest
- Decision Tree
- XGBoost
- Gradient Boost
- Multi-Layer Perceptron (MLP)
The script performs the following steps:
1. Initializes the models with specified parameters.
2. Trains each model on the training dataset (X_train, y_train).
3. Makes predictions on the test dataset (X_test).
4. Calculates the accuracy of each model.
5. Prints the classification report for each model.
6. Plots and displays the confusion matrix for each model.
Variables:
- models: Dictionary containing the initialized models.
- model_acc: Dictionary to store the accuracy of each model.
- X_train: Training feature set.
- y_train: Training labels.
- X_test: Test feature set.
- y_test: Test labels.
Libraries required:
- matplotlib
- seaborn
- scikit-learn
- xgboost