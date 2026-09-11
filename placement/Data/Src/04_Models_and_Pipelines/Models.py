import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, silhouette_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore")

def load_and_preprocess_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "01_Datasets", "placement_predict_50k Dataset (2).csv")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    # Drop rows where target is missing if any
    if 'PlacementStatus' in df.columns:
        df = df.dropna(subset=['PlacementStatus'])
    
    # Impute missing values for numericals
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Drop ID and target columns from features
    drop_cols = ['StudentID', 'PlacementStatus', 'Salary Package', 'IsAnomaly']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Encode categoricals
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # Target for Classification
    le = LabelEncoder()
    y_class = le.fit_transform(df['PlacementStatus'])
    
    # Target for Regression (Salary Package) - only for Placed students where salary is > 0
    df['Salary_Numeric'] = pd.to_numeric(df['Salary Package'], errors='coerce')
    
    reg_mask = df['Salary_Numeric'].notna() & (df['Salary_Numeric'] > 0)
    X_reg = X[reg_mask]
    y_reg = df.loc[reg_mask, 'Salary_Numeric']
    
    return X, y_class, X_reg, y_reg

def evaluate_models():
    X_cls, y_cls, X_reg, y_reg = load_and_preprocess_data()
    
    # Split for Classification
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    
    # Split for Regression
    if len(X_reg) > 0:
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    else:
        # Fallback if no valid salary data
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = X_train_cls, X_test_cls, y_train_cls, y_test_cls
    
    # Scaling
    scaler = StandardScaler()
    X_train_cls_scaled = scaler.fit_transform(X_train_cls)
    X_test_cls_scaled = scaler.transform(X_test_cls)
    
    X_train_reg_scaled = scaler.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler.transform(X_test_reg)
    
    results = []
    
    def train_and_eval(model, name, family, X_train, y_train, X_test, y_test, is_scaled, task="Classification", is_cluster=False):
        start_time = time.time()
        
        try:
            if is_cluster:
                model.fit(X_train)
                train_time = time.time() - start_time
                # Silhouette score can be slow on large data, so we sample
                sample_size = min(3000, X_train.shape[0])
                indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
                # Convert to numpy array for indexing if it's a pandas dataframe
                X_train_sample = X_train.iloc[indices].values if hasattr(X_train, 'iloc') else X_train[indices]
                labels_sample = model.labels_[indices]
                # If only 1 cluster is found, silhouette score will fail
                if len(set(labels_sample)) > 1:
                    score = silhouette_score(X_train_sample, labels_sample)
                else:
                    score = 0.0
                metric_name = "Silhouette Score"
            else:
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                preds = model.predict(X_test)
                if task == "Classification":
                    score = accuracy_score(y_test, preds)
                    metric_name = "Accuracy"
                else:
                    score = r2_score(y_test, preds)
                    metric_name = "R2 Score"
                    
            results.append({
                "Family": family,
                "Model Name": name,
                "Task": task,
                "Scaled": "Yes" if is_scaled else "No",
                "Metric": metric_name,
                "Score": round(score, 4),
                "Train Time (s)": round(train_time, 4)
            })
        except Exception as e:
            print(f"Error training {name} ({'Scaled' if is_scaled else 'Unscaled'}): {e}")

    # To keep training time reasonable for SVM and Neural Nets, we subsample the classification data
    sample_size = min(5000, X_train_cls.shape[0])
    X_tr_cls_sub = X_train_cls.iloc[:sample_size] if hasattr(X_train_cls, 'iloc') else X_train_cls[:sample_size]
    y_tr_cls_sub = y_train_cls[:sample_size]
    X_ts_cls_sub = X_test_cls.iloc[:1000] if hasattr(X_test_cls, 'iloc') else X_test_cls[:1000]
    y_ts_cls_sub = y_test_cls[:1000]
    
    X_tr_cls_scaled_sub = X_train_cls_scaled[:sample_size]
    X_ts_cls_scaled_sub = X_test_cls_scaled[:1000]

    # Classification Models
    models_cls = [
        ("Distance-based", "KNN", KNeighborsClassifier(n_neighbors=5)),
        ("Distance-based", "SVM (RBF)", SVC(kernel='rbf', max_iter=2000)),
        ("Gradient-descent", "Logistic Regression", LogisticRegression(max_iter=500)),
        ("Gradient-descent", "Neural Net", MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, early_stopping=True))
    ]
    
    for family, name, model in models_cls:
        # Unscaled
        train_and_eval(model, name, family, X_tr_cls_sub, y_tr_cls_sub, X_ts_cls_sub, y_ts_cls_sub, False, "Classification")
        
        # Re-initialize to reset weights
        if name == "KNN": model = KNeighborsClassifier(n_neighbors=5)
        elif name == "SVM (RBF)": model = SVC(kernel='rbf', max_iter=2000)
        elif name == "Logistic Regression": model = LogisticRegression(max_iter=500)
        elif name == "Neural Net": model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, early_stopping=True)
        
        # Scaled
        train_and_eval(model, name, family, X_tr_cls_scaled_sub, y_tr_cls_sub, X_ts_cls_scaled_sub, y_ts_cls_sub, True, "Classification")
        
    # K-Means (Unsupervised Distance-based)
    family = "Distance-based"
    name = "K-Means (k=2)"
    kmeans_unscaled = KMeans(n_clusters=2, random_state=42, n_init=10)
    train_and_eval(kmeans_unscaled, name, family, X_tr_cls_sub, None, None, None, False, "Clustering", is_cluster=True)
    
    kmeans_scaled = KMeans(n_clusters=2, random_state=42, n_init=10)
    train_and_eval(kmeans_scaled, name, family, X_tr_cls_scaled_sub, None, None, None, True, "Clustering", is_cluster=True)

    # Regression Models
    models_reg = [
        ("Regularized", "Ridge", Ridge(alpha=1.0)),
        ("Regularized", "Lasso", Lasso(alpha=0.1))
    ]
    
    for family, name, model in models_reg:
        train_and_eval(model, name, family, X_train_reg, y_train_reg, X_test_reg, y_test_reg, False, "Regression")
        if name == "Ridge": model = Ridge(alpha=1.0)
        elif name == "Lasso": model = Lasso(alpha=0.1)
        train_and_eval(model, name, family, X_train_reg_scaled, y_train_reg, X_test_reg_scaled, y_test_reg, True, "Regression")
        
    return results

if __name__ == "__main__":
    print("Evaluating models with and without scaling. This may take a moment...")
    res = evaluate_models()
    df_res = pd.DataFrame(res)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Outputs", "model_scaling_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_res.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")
    print(df_res)
