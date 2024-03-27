import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from nltk.tokenize import word_tokenize
import numpy as np

def build_dataset(path, num_samples=-1, rnd_state=42):
    df1 = pd.read_json(path + "/fevrier.json")
    df2 = pd.read_json(path + "/janvier.json")
    df3 = pd.read_json(path + "/mars.json")
    df4 = pd.read_json(path + "/decembre.json")
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    df['section_label'], _ = pd.factorize(df['section_1'])
    if num_samples != -1:
        df = df.sample(n=min(len(df), num_samples), replace=False, random_state=rnd_state)
    return df.T.to_dict()

def tune_logistic_regression(X_train, Y_train):
    model = LogisticRegression(multi_class='multinomial', max_iter=5000)
    param_grid = {
        'solver': ['newton-cg', 'lbfgs', 'saga'],  # Removed 'liblinear' as it's not the best for multiclass.
        'penalty': ['l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    return best_model

def tune_svm(X_train, Y_train):
    model = SVC()
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4, 5, 6, 7, 8, 9],
        'gamma': ['scale', 'auto'],
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    return best_model

def tune_mlp(X_train, Y_train):
    model = MLPClassifier(max_iter=5000) 
    param_grid = {
        'hidden_layer_sizes': [(25,),(50,)],  
        'activation': ['tanh', 'relu'],  
        'solver': ['sgd', 'adam'],  
        'alpha': [0.0001, 0.001, 0.01, 0.1],  
        'learning_rate': ['constant', 'adaptive'],  
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    return best_model

def tune_naive_bayes(X_train, Y_train):
    model = MultinomialNB()
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
        'fit_prior': [True, False] 
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    return best_model

def evaluate(Y_test, y_pred):
    print('Precision: ', precision_score(Y_test, y_pred, average='weighted',zero_division=0))
    print('Recall: ', recall_score(Y_test, y_pred, average='weighted',zero_division=0))
    print('F1_score: ', f1_score(Y_test, y_pred, average='weighted',zero_division=0))
    print('accuracy: ', accuracy_score(Y_test, y_pred))

def preprocess_text(text, language='french'):
    return word_tokenize(text.lower(), language=language)

def text_to_word2vec(text, model):
    words = preprocess_text(text)
    vectors = [model[word] for word in words if word in model]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

