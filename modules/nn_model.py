from sklearn.neural_network import MLPClassifier
import joblib
import os
import numpy as np

def train_neural_network(X_train, y_train, X_test, y_test, input_dim, model_path='models/nn_model.pkl'):
    """
    ฟังก์ชันสำหรับฝึกสอน Neural Network Model สำหรับการจำแนกประเภท Binary
    โดยใช้แบบจำลอง Multi-Layer Perceptron (MLP) ของ Scikit-Learn แทน Keras/TF
    เพื่อให้รองรับระบบปฏิบัติการที่ใช้ Python 3.14 ได้โดยไม่มีปัญหา Dependency
    """
    
    # สร้างโครงสร้างแบบจำลอง Multi-Layer Perceptron (MLP) สำหรับ Binary Classification
    # โครงสร้างเดิมคือ 32 -> 16
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16), 
        activation='relu',
        solver='adam',
        max_iter=500, # เพิ่มจำนวนรอบ (Epochs) ให้เพียงพอ
        batch_size=16,
        random_state=42
    )
    
    # Train โมเดล 
    model.fit(X_train, y_train)
    
    # ดึงค่าความแม่นยำจากการประเมินผล Test Set
    accuracy = model.score(X_test, y_test)
    loss = getattr(model, 'loss_', 0)
    
    # บันทึก Model เป็นไฟล์ .pkl เพื่อง่ายต่อการข้ามแพลตฟอร์ม
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    return {
        'model_name': 'Multi-Layer Perceptron Neural Network',
        'accuracy': accuracy,
        'loss': loss,
        'history': {}, # ข้ามประวัติ History ไปก่อนเพราะ sklearn ไม่คืนค่า history แบบ Keras
        'path': model_path
    }

def predict_nn(model_path, X_new):
    """
    โหลด Neural Network โมเดลและทำการ Prediction
    """
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        # ทำนายคลาส
        prediction = model.predict(X_new)
        # ทำนายความน่าจะเป็น
        proba = model.predict_proba(X_new)
        
        # จัดรูปแบบผลลัพธ์ให้เป็น array 2D เพื่อให้เข้ากับโค้ดฝั่ง app.py (เหมือนโครงสร้าง output ของ Keras เดิม)
        prediction_class = np.array([[prediction[0]]])
        
        # predict_proba คืนค่า array 2 ค่า [ความน่าจะเป็นคลาส0, ความน่าจะเป็นคลาส1] 
        # เรานำค่าความน่าจะเป็นที่มันจะเป็นมะเร็ง (คลาส 1) มาใช้
        prob_positive = proba[0][1] if len(proba[0]) > 1 else proba[0][0]
        probabilities = np.array([[prob_positive]])
        
        return prediction_class, probabilities
    return None, None
