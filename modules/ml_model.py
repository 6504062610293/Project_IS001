from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_ensemble_model(X_train, y_train, X_test, y_test, model_path='models/ensemble_model.pkl'):
    """
    ฟังก์ชันสำหรับฝึกสอนโมเดล Machine Learning แบบ Ensemble
    ประกอบด้วย Random Forest, SVM และ K-Nearest Neighbors
    """
    
    # กำหนดโมเดลย่อย (Base Estimators)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # probability=True เพื่อให้ระบบ Voting แบบ Soft Voting ทำงานได้
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    
    # สร้างโมเดล Ensemble ด้วย Voting Classifier (นำ 3 โมเดลย่อยมาโหวตคำตอบร่วมกัน)
    # ใช้ 'soft' voting นำค่าความน่าจะเป็นมาเฉลี่ยรวมกันก่อนเลือก Class ที่มีโอกาสสูงสุด
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('svm', svm_model),
            ('knn', knn_model)
        ],
        voting='soft'
    )
    
    # ทำการ Train
    ensemble_model.fit(X_train, y_train)
    
    # ทดสอบความแม่นยำ
    y_pred = ensemble_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # บันทึกโมเดลเก็บไว้เพื่อให้หน้าเว็บสามารถเรียกไปทำนายข้อมูลชุดใหม่ได้
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(ensemble_model, model_path)
    
    return {
        'model_name': 'Ensemble (Voting: RF, SVM, KNN)',
        'accuracy': accuracy,
        'report': report,
        'path': model_path
    }

def predict_ensemble(model_path, X_new):
    """
    โหลดโมเดลแล้วนำมาทำนายชุดข้อมูลใหม่
    """
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        prediction = model.predict(X_new)
        return prediction
    return None
