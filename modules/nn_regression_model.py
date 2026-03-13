from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_neural_network_regression(X_train, y_train, X_test, y_test, input_dim, model_path='models/nn_reg_model.pkl'):
    """
    ฟังก์ชันสำหรับฝึกสอน Neural Network Model สำหรับ Regression (ค่าใช้จ่ายประกัน)
    ปรับแต่ง Architecture (64, 32) เหมาะสำหรับข้อมูลที่มี 6 Features 
    พฤติกรรมมีความซับซ้อนปานกลาง (มี One-Hot/Label Encoding ปนด้วย)
    """
    
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32), 
        activation='relu',
        solver='adam',
        max_iter=1000,          # เพิ่ม max_iter ป้องกันไม่ลู่เข้า (not converged)
        batch_size=32,          # ปรับ batch_size ให้เหมาะกับจำนวนแถว ~1338 แถว
        learning_rate_init=0.01,
        early_stopping=True,    # ป้องกัน Overfitting
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    loss = getattr(model, 'loss_', 0)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    return {
        'model_name': 'Neural Network Regressor (64-32 Arch)',
        'r2_score': r2,
        'mse': mse,
        'loss': loss,
        'path': model_path
    }
