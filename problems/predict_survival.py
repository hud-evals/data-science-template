#!/usr/bin/env python3
"""
Golden script for the Titanic predict_survival problem.

This script trains a logistic regression model on the training data
and outputs survival probability predictions for the test set.

Expected to be run from the workspace directory where:
- Titanic-Dataset-train.csv exists (training data with labels)
- Titanic-Dataset-test_x.csv exists (test features without labels)
"""

import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess(df: pd.DataFrame, label_encoders: dict = None, fit: bool = False):
    """
    Preprocess the Titanic dataset for modeling.
    
    Args:
        df: Input dataframe
        label_encoders: Dict of fitted LabelEncoders (for transform-only mode)
        fit: If True, fit new encoders; if False, use provided encoders
    
    Returns:
        Tuple of (processed features dataframe, label_encoders dict)
    """
    # Select features for the model
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df_features = df[features].copy()
    
    # Fill missing values
    df_features['Age'] = df_features['Age'].fillna(df_features['Age'].median())
    df_features['Fare'] = df_features['Fare'].fillna(df_features['Fare'].median())
    df_features['Embarked'] = df_features['Embarked'].fillna('S')  # Most common
    
    # Encode categorical variables
    if label_encoders is None:
        label_encoders = {}
    
    for col in ['Sex', 'Embarked']:
        if fit:
            le = LabelEncoder()
            df_features[col] = le.fit_transform(df_features[col])
            label_encoders[col] = le
        else:
            df_features[col] = label_encoders[col].transform(df_features[col])
    
    return df_features, label_encoders


def main():
    # Load training data
    train_path = Path('Titanic-Dataset-train.csv')
    test_path = Path('Titanic-Dataset-test_x.csv')
    
    if not train_path.exists():
        print(f"Error: Training data not found at {train_path}")
        exit(1)
    
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        exit(1)
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Preprocess training data
    X_train, label_encoders = load_and_preprocess(train_df, fit=True)
    y_train = train_df['Survived']
    
    # Preprocess test data using same encoders
    X_test, _ = load_and_preprocess(test_df, label_encoders=label_encoders, fit=False)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predict probabilities for test set
    probabilities = model.predict_proba(X_test_scaled)[:, 1]  # Probability of survival
    
    # Create output dataframe
    predictions_df = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': probabilities
    })
    
    # Save predictions
    output_path = Path('predictions.csv')
    predictions_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")
    print(f"Number of predictions: {len(predictions_df)}")
    print(f"Mean predicted probability: {probabilities.mean():.4f}")


if __name__ == '__main__':
    main()
