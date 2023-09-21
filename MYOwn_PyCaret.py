import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier  # Uncomment if XGBoost is available
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data from various file formats
def load_data(file_path):
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        elif file_path.endswith('.db'):
            conn = sqlite3.connect(file_path)
            data = pd.read_sql_query('SELECT * FROM your_table', conn)
            conn.close()
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        print(e)
        return None

    return data

# Function to preprocess the loaded data
def preprocess_data(data):
    summary = data.describe()
    print(summary)
    # droping unnecessary columns
    data=data.drop([input("Enter the undesired column to drop")],axis=1)

    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    numeric_imputer = SimpleImputer(strategy='mean')
    data[numeric_features] = numeric_imputer.fit_transform(data[numeric_features])

    categorical_imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_features] = categorical_imputer.fit_transform(data[categorical_features])
    
#     data = automate_encoding(data)


    return data

# Function to automate encoding of categorical variables
def automate_encoding(data):
    encoded_data = data.copy()
    
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    for column in categorical_columns:
        unique_count = len(data[column].unique())
        
        if unique_count <= 2:
            le = LabelEncoder()
            encoded_data[column] = le.fit_transform(encoded_data[column])
        else:
            encoded_data = pd.get_dummies(encoded_data, columns=[column], prefix=[column])
            
    return encoded_data

# Function to train a classification model with hyperparameter tuning
def train_classification_model(encoded_data, target_variable, model_type):
    X = encoded_data.drop(target_variable, axis=1)
    
    # Apply Standard Scaling to numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    y = encoded_data[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    
    if model_type == "logistic_regression":
        # Define hyperparameters and their possible values for tuning
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2']
        }

        # Create a Logistic Regression model
        model = LogisticRegression()

        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        print(f"Best Hyperparameters for Logistic Regression: {best_params}")

        # Use the best model for prediction
        model = grid_search.best_estimator_
        
    elif model_type == "random_forest":
        # Define hyperparameters and their possible values for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Create a Random Forest Classifier model
        model = RandomForestClassifier()

        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        print(f"Best Hyperparameters for Random Forest: {best_params}")

        # Use the best model for prediction
        model = grid_search.best_estimator_
        
    elif model_type == "decision_tree":
        # Define hyperparameters and their possible values for tuning
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Create a Decision Tree Classifier model
        model = DecisionTreeClassifier()

        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        print(f"Best Hyperparameters for Decision Tree: {best_params}")

        # Use the best model for prediction
        model = grid_search.best_estimator_
        
    elif model_type == "k_nearest_neighbors":
        # Define hyperparameters and their possible values for tuning
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }

        # Create a K-Nearest Neighbors Classifier model
        model = KNeighborsClassifier()

        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        print(f"Best Hyperparameters for K-Nearest Neighbors: {best_params}")

        # Use the best model for prediction
        model = grid_search.best_estimator_
        
    elif model_type == "naive_bayes":
        # Naive Bayes does not have many hyperparameters to tune, so we skip hyperparameter tuning for it.
        model = GaussianNB()
        
    elif model_type == "neural_network":
        # Define hyperparameters and their possible values for tuning
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'activation': ['logistic', 'tanh', 'relu'],
            'alpha': [0.0001, 0.001, 0.01, 0.1]
        }

        # Create a Multi-layer Perceptron (Neural Network) Classifier model
        model = MLPClassifier()

        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        print(f"Best Hyperparameters for Neural Network: {best_params}")

        # Use the best model for prediction
        model = grid_search.best_estimator_
        
    elif model_type == "xgboost":
        # Define hyperparameters and their possible values for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        # Create an XGBoost Classifier model
        model = XGBClassifier()

        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        print(f"Best Hyperparameters for XGBoost: {best_params}")

        # Use the best model for prediction
        model = grid_search.best_estimator_
        
    else:
        raise ValueError("Invalid model_type")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the {model_type} model: {accuracy}")


# Function to train a classification model with hyperparameter tuning

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the {model_type} model: {accuracy}")

    # Plot a confusion matrix
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix with numbers
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    # Plot ROC curve and calculate AUC
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    
        # Pairwise scatter plots
    sns.pairplot(processed_data, hue=target_variable)
    plt.show()

    # Histograms for numerical features
    numerical_features = processed_data.select_dtypes(include=['int64', 'float64']).columns
    for feature in numerical_features:
        sns.histplot(processed_data[feature], kde=True)
        plt.xlabel(feature)
        plt.show()

    # Box plots by target variable
    for feature in numerical_features:
        sns.boxplot(x=target_variable, y=feature, data=processed_data)
        plt.xlabel(target_variable)
        plt.ylabel(feature)
        plt.show()

    # Count plots for categorical features
    categorical_features = processed_data.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        sns.countplot(x=feature, data=processed_data)
        plt.xlabel(feature)
        plt.xticks(rotation=45)
        plt.show()

    # Correlation heatmap
    correlation_matrix = processed_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    

file_path = input("Please enter the path of the file you want to upload: ")
loaded_data = load_data(file_path)

# Specify target variable and model type (classification or regression):
target_variable = input("Please enter the column you want to Predict: ")
model_type = input("Please enter  the desired model type as KNeighborsClassifier ,decision_tree,random_forest, Logistic Regression,xgboost,neural_network: ")  # Change this to the desired model type


if loaded_data is not None:
    print("Data loaded successfully.")
    print(loaded_data.head())
    print("\n + *************************************************************** +")
    processed_data = preprocess_data(loaded_data)
    print("\n + *******************THE ENCODED DATA******************************************** +")
    automated_encoding = automate_encoding(processed_data)
    print(automated_encoding)
    
    print("\n + *************************************************************** +")

    train_classification_model(automated_encoding, target_variable, model_type)
    print("\n + *************************************************************** +")


