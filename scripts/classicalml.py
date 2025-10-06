import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


MODELS = {
    "ET": {
        "function": ExtraTreesClassifier(),
        "param_grid": [
            {
                "ET__n_estimators": [2, 5, 10, 20],
                "ET__criterion": ["gini", "entropy", "log_loss"],
                "ET__max_depth": [i for i in range (2,16)], #, 15, 20, 25
                "ET__max_features": [0.1, 0.2, "sqrt", "log2"],
            }
        ],
    },
    "RF": {
        "function": RandomForestClassifier(),
        "param_grid": [
            {
                "RF__n_estimators": [2, 5, 10, 20],
                "RF__criterion": ["gini", "entropy", "log_loss"],
                "RF__max_depth": [i for i in range (2,16)], #, 15, 20, 25
                "RF__max_features": [0.1, 0.2, "sqrt", "log2"],
            }
        ],
    },
    "SVM": {
        "function": SVC(),
        "param_grid": [
            {"SVM__kernel": ["poly"], "SVM__degree": [2, 3, 4, 5, 6, 7, 8]}, #, 6, 7, 8, 9, 10, 15
            {"SVM__kernel": ["linear", "rbf", "sigmoid"]},
        ],
    },
    "MLP": {
        "function": MLPClassifier(),
        "param_grid": [{
                'MLP__hidden_layer_sizes': [(10,10), (5,10,5), (10,50,10)],
                'MLP__activation': ['relu'],
                'MLP__solver': ['sgd', 'adam'],
                'MLP__learning_rate_init': [0.1, 0.01, 0.001],
                'MLP__learning_rate': ['constant', 'adaptive'],
                'MLP__max_iter': [100,250]
            }], # , 9, 11, 13, 15
    }
}

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with metrics
    """
    return {
        'accuracy' : accuracy_score(y_true, y_pred),
        'balanced accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='binary')
    }

def train_and_cross_validate(X, Y, cl_name, seed, cv_folds=5) -> tuple:
    '''
    Train and cross-validate a model with hyperparameter tuning.
    
    Args:
        X: Feature matrix
        Y: Labels
        cl_name: Classifier name (key in MODELS)
        seed: Random seed
        cv_folds: Number of cross-validation folds
    
    Returns:
        DataFrame with validation metrics and the best model
    '''
    X_train, X_val, y_train, y_val = train_test_split(
                    X, Y, test_size=0.2, random_state=seed, shuffle=True, stratify=Y
                )
    
    # Create pipeline with scaler and classifier
    pipe = Pipeline([
        ('sc', StandardScaler()),
        (cl_name, MODELS[cl_name]['function'])
    ])

    param_grid = MODELS[cl_name]['param_grid']
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
                pipe, param_grid, cv=cv_folds, 
                scoring='f1', n_jobs=-1, verbose=1
            )
            
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Evaluate on validation set
    y_pred = best_model.predict(X_val)

    metrics = calculate_metrics(y_val, y_pred)
    metrics['model'] = cl_name
    metrics['split'] = 'val'

    metrics_df = pd.DataFrame(metrics, index=[0])
    
    return metrics_df, best_model

def initiate_cross_validation(X, Y, seed, cv_folds=5) -> pd.DataFrame:
    '''
    Initiate cross-validation for all models.

    Args:
        X: Feature matrix
        Y: Labels
        seed: Random seed
        cv_folds: Number of cross-validation folds
    
    Returns:
        DataFrame with all results
    '''
    X_train, X_test, y_train, y_test = train_test_split(
                        X, Y, test_size=0.2, random_state=seed, shuffle=True, stratify=Y
                    )

    all_results_list = []

    # Iterate over all models
    for name, _ in MODELS.items():

        print(f"Model: {name} ")

        # Standard scaler ensures that all the features are in the same range (which are key for regression & distance based algorithms)
        
        val_df, best_model = train_and_cross_validate(X_train, y_train, name, cv_folds)

        y_pred = best_model.predict(X_test)

        test_metrics = calculate_metrics(y_pred, y_test)
        test_metrics['model'] = name
        test_metrics['split'] = 'test'

        test_df = pd.DataFrame(test_metrics, index=[0])
        
        results = pd.concat([val_df, test_df], ignore_index=True)
        all_results_list.append(results)

        print(f'Train & evaluation completed for {name}!')

    all_results_df = pd.concat(all_results_list, ignore_index=True)

    return all_results_df