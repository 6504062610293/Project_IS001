from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_ensemble_regression_model(X_train, y_train, X_test, y_test, model_path='models/ensemble_reg_model.pkl'):
    """
    ฟังก์ชันสำหรับฝึกสอนโมเดล Machine Learning แบบ Ensemble สำหรับ Regression
    พร้อมกับประเมินผล 3 โมเดลย่อย (Random Forest, SVM, KNN)
    """
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    svm_model = SVR(kernel='rbf')
    knn_model = KNeighborsRegressor(n_neighbors=5)
    
    ensemble_model = VotingRegressor(
        estimators=[
            ('rf', rf_model),
            ('svm', svm_model),
            ('knn', knn_model)
        ]
    )
    
    # Train all individual models for reporting
    rf_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)
    
    # Train ensemble
    ensemble_model.fit(X_train, y_train)
    
    y_pred = ensemble_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'rf_r2': r2_score(y_test, rf_model.predict(X_test)),
        'svm_r2': r2_score(y_test, svm_model.predict(X_test)),
        'knn_r2': r2_score(y_test, knn_model.predict(X_test)),
        'ensemble_r2': r2,
        'ensemble_mse': mse
    }
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(ensemble_model, model_path)
    
    return {
        'model_name': 'Ensemble Regressor (RF, SVR, KNN)',
        'metrics': metrics,
        'path': model_path
    }
