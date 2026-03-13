import os
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split

# นำเข้าฟังก์ชันจาก module ที่เราสร้าง
from modules.preprocessing import process_data, process_insurance_data
from modules.ml_model import train_ensemble_model, predict_ensemble
from modules.nn_model import train_neural_network, predict_nn
from modules.ml_regression_model import train_ensemble_regression_model
from modules.nn_regression_model import train_neural_network_regression

# สร้างอินสแตนซ์ของแพลตฟอร์ม Flask
app = Flask(__name__)
# ตั้งค่าคีย์สำหรับการใช้งาน Session/Flash messages
app.secret_key = 'super_secret_key_for_flash_messages'

# กำหนดโฟลเดอร์
UPLOAD_FOLDER = 'uploads'
DATASET_FOLDER = 'Dataset'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATASET_FOLDER'] = DATASET_FOLDER

# ตรวจสอบว่ามีโฟลเดอร์ หรือยัง ถ้ายังไม่มีก็สร้างใหม่
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """
    Route เริ่มต้น (Homepage) แสดงหน้ารายการ Dataset
    """
    datasets = [
        {
            'filename': 'breast-cancer.csv',
            'title': 'Breast Cancer Dataset',
            'desc': 'ชุดข้อมูลการวินิจฉัยมะเร็งเต้านม',
            'link': 'https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset'
        },
        {
            'filename': 'insurance.csv',
            'title': 'Healthcare Insurance Dataset',
            'desc': 'ชุดข้อมูลค่าใช้จ่ายประกันสุขภาพ',
            'link': 'https://www.kaggle.com/datasets/muqaddasejaz/healthcare-insurance-datasets'
        }
    ]
    return render_template('datasets.html', datasets=datasets)

@app.route('/preprocess/<filename>')
def preprocess(filename):
    """
    Route สำหรับเข้ากระบวนการการเตรียมข้อมูล
    """
    filepath = os.path.join(app.config['DATASET_FOLDER'], filename)
    
    # ถ้าไฟล์ไม่มีอยู่จริง ให้กลับไปหน้าแรก
    if not os.path.exists(filepath):
        flash('ไม่พบไฟล์ Dataset ที่ระบุ')
        return redirect(url_for('index'))
        
    try:
        # เรียกใช้ฟังก์ชันตาม Dataset
        if filename == 'insurance.csv':
            steps_info, df_scaled = process_insurance_data(filepath)
            template_name = 'eda_insurance.html'
        else:
            steps_info, df_scaled = process_data(filepath)
            template_name = 'preprocessing.html'
        
        # บันทึก Dataset ที่เตรียมข้อมูลเสร็จแล้ว
        ready_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'ready_' + filename)
        df_scaled.to_csv(ready_filepath, index=False)
        
        # ส่งต่อตัวแปรไปยังหน้า Template
        return render_template(template_name, steps=steps_info, filename=filename)
    except Exception as e:
        flash(f'เกิดข้อผิดพลาดในการประมวลผลข้อมูล: {str(e)}')
        return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train_models():
    """
    Route สำหรับการ Train 2 โมเดลหลังจากปรับแต่งข้อมูลเสร็จแล้ว
    """
    filename = request.form.get('filename')
    ready_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'ready_' + filename)
    
    try:
        df = pd.read_csv(ready_filepath)
        
        # สมมติฐานว่าคอลัมน์เป้าหมายคือ diagnosis หรือ charges ตามแต่ Dataset
        target_col = None
        if 'diagnosis' in df.columns:
            target_col = 'diagnosis'
        elif 'charges' in df.columns:
            target_col = 'charges'
            
        if target_col is None:
            flash("ไม่พบคอลัมน์วินิจฉัยโรค ('diagnosis') หรือค่าใช้จ่าย ('charges') ในชุดข้อมูลนี้ ไม่สามารถจำแนกประเภทได้")
            return redirect(url_for('index'))

        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # แบ่งชุดข้อมูลเป็น Train Setting (80%) และ Test Setting (20%) สำหรับตรวจสอบความแม่นยำ
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. เทรน Ensemble Machine Learning Model
        # 2. เทรน Neural Network Model 
        # input_dim เท่ากับจำนวนฟีเจอร์ของข้อมูล X
        if target_col == 'charges':
            ml_results = train_ensemble_regression_model(X_train, y_train, X_test, y_test)
            nn_results = train_neural_network_regression(X_train, y_train, X_test, y_test, input_dim=X.shape[1])
            return render_template('training_success_regression.html', ml_results=ml_results, nn_results=nn_results)
        else:
            ml_results = train_ensemble_model(X_train, y_train, X_test, y_test)
            nn_results = train_neural_network(X_train, y_train, X_test, y_test, input_dim=X.shape[1])
            return render_template('training_success.html', ml_results=ml_results, nn_results=nn_results)
        
    except Exception as e:
        flash(f'เกิดข้อผิดพลาดในการTrain โมเดล: {str(e)}')
        return redirect(url_for('index'))

@app.route('/model/ml')
def explain_ml():
    """หน้าเพจอธิบายและทดสอบ Ensemble ML"""
    return render_template('explain_ml.html')

@app.route('/model/nn')
def explain_nn():
    """หน้าเพจอธิบายและทดสอบ Neural Network"""
    return render_template('explain_nn.html')

@app.route('/model/ml_regression')
def explain_ml_regression():
    """หน้าเพจอธิบายและทดสอบ ML สำหรับ Regression"""
    return render_template('explain_ml_regression.html')

@app.route('/model/nn_regression')
def explain_nn_regression():
    """หน้าเพจอธิบายและทดสอบ Neural Network สำหรับ Regression"""
    return render_template('explain_nn_regression.html')

def prepare_input_features(request_form, total_features=30):
    """
    ฟังก์ชันช่วยจำลองข้อมูล เนื่องจากรับค่ามาจากฟอร์มแค่ 5 ฟีเจอร์หลัก 
    แต่โมเดล Train ไว้กับ Breast Cancer dataset ที่มี 30 ฟีเจอร์
    ในของจริง User ควรต้องกรอกข้อมูลให้ครบ หรือระบบต้องดึงค่ากลาง (Mean) มาใส่แทน
    """
    # ดึงค่า 5 ค่าที่ได้กรอกมาจากฟอร์ม
    try:
        f1 = float(request_form.get('radius_mean', 0))
        f2 = float(request_form.get('texture_mean', 0))
        f3 = float(request_form.get('area_mean', 0))
        f4 = float(request_form.get('smoothness_mean', 0))
        f5 = float(request_form.get('concavity_mean', 0))
        
        # สร้าง array ที่ใส่ข้อมูลที่เรามี 5 ตัวแรก
        features = [f1, f2, f3, f4, f5]
        
        # เติม 0 (หรือจริงๆ ควรเป็นค่า mean หลัง scale) สำหรับฟีเจอร์ที่เหลือ 
        # เพื่อให้ Array มีขนาด = 30 เท่ากับตอนที่เทรนมา
        remaining = total_features - len(features)
        features.extend([0.0] * remaining)
        
        return np.array([features])
    except ValueError:
        return None

@app.route('/predict/ml', methods=['POST'])
def predict_ml_route():
    """รับค่าฟอร์มแล้วทำนายผลลัพธ์ผ่าน Ensemble Model"""
    features = prepare_input_features(request.form)
    
    if features is None:
        return render_template('explain_ml.html', message="กรุณากรอกข้อมูลเป็นตัวเลขที่ถูกต้อง")
        
    model_path = 'models/ensemble_model.pkl'
    if not os.path.exists(model_path):
        return render_template('explain_ml.html', message="ไม่พบไฟล์โมเดล กรุณาอัปโหลดชุดข้อมูลเพื่อ Train ก่อนทดสอบ")
        
    try:
        prediction = predict_ensemble(model_path, features)
        # prediction เป็น array เช่น [1] หรือ [0]
        result = "เนื้อร้าย (Malignant)" if prediction[0] == 1 else "เนื้อดี (Benign)"
        
        # ส่งค่ากลับไปหาหน้าจอเพื่อแสดงผล (ในที่นี้เราใช้ flash message แบบง่ายแทน)
        # ของจริงอาจใช้ Modal หรือหน้า Result
        return render_template('explain_ml.html', 
            message=f"ผลการประเมินจากชุดข้อมูล: พบว่าเป็น **{result}**")
            
    except Exception as e:
        return render_template('explain_ml.html', message=f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")

@app.route('/predict/nn', methods=['POST'])
def predict_nn_route():
    """รับค่าฟอร์มแล้วทำนายผลลัพธ์ผ่าน Neural Network Model"""
    features = prepare_input_features(request.form)
    
    if features is None:
        return render_template('explain_nn.html', message="กรุณากรอกข้อมูลเป็นตัวเลขที่ถูกต้อง")
        
    model_path = 'models/nn_model.pkl'
    if not os.path.exists(model_path):
        return render_template('explain_nn.html', message="ไม่พบไฟล์โมเดล กรุณาอัปโหลดชุดข้อมูลเพื่อ Train ก่อนทดสอบ")
        
    try:
        prediction_class, probabilities = predict_nn(model_path, features)
        
        # คลาสที่ได้และเปอร์เซนต์ความมั่นใจ
        p_class = prediction_class[0][0] # แปลงจาก array 2D
        prob_val = probabilities[0][0]
        
        result = "เนื้อร้าย (Malignant)" if p_class == 1 else "เนื้อดี (Benign)"
        confidence = f"ความมั่นใจของโครงข่าย: {(prob_val * 100):.2f}%" if p_class == 1 else f"ความมั่นใจของโครงข่าย: {((1 - prob_val) * 100):.2f}%"
        
        return render_template('explain_nn.html', 
            message=f"ผลการประเมินจากชุดข้อมูล: พบว่าเป็น **{result}** ({confidence})")
            
    except Exception as e:
        return render_template('explain_nn.html', message=f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")

# =========================================================
# Regression Testing (Healthcare Insurance)
# =========================================================

def prepare_insurance_features(request_form):
    """
    ฟังก์ชันช่วยจำลองข้อมูล 6 คอลัมน์ (Age, Sex, BMI, Children, Smoker, Region) 
    ของช้อมูล Insurance และผ่านเข้า Standard Scaler สมมติ
    *หมายเหตุ: ในระบบจริงควร Save StandardScaler object (joblib) ตอน Train มาใช้งานเพื่อให้สเกลตรงกัน*
    ที่นี่ใช้การสเกลอย่างง่ายเพื่อให้รันได้
    """
    try:
        age_in = float(request_form.get('age', 0))
        sex_in = float(request_form.get('sex', 0))
        bmi_in = float(request_form.get('bmi', 0))
        children_in = float(request_form.get('children', 0))
        smoker_in = float(request_form.get('smoker', 0))
        region_in = float(request_form.get('region', 0))
        
        features = [age_in, sex_in, bmi_in, children_in, smoker_in, region_in]
        
        # สมมติ Mean & Std แบบคร่าวๆ จาก Dataset (เฉพาะ Age, BMI, Children)
        # ของจริงต้องโหลดมาจาก scaler_params.pkl 
        age_mean, age_std = 39.2, 14.0
        bmi_mean, bmi_std = 30.66, 6.09
        child_mean, child_std = 1.09, 1.2
        
        features[0] = (features[0] - age_mean) / age_std
        features[2] = (features[2] - bmi_mean) / bmi_std
        features[3] = (features[3] - child_mean) / child_std
        
        return np.array([features])
    except ValueError:
        return None

@app.route('/predict/ml_regression', methods=['POST'])
def predict_ml_regression_route():
    """ทำนายค่าใช้จ่ายด้วย Ensemble Regression"""
    features = prepare_insurance_features(request.form)
    
    if features is None:
        return render_template('explain_ml_regression.html', message="กรุณากรอกข้อมูลเป็นตัวเลขที่ถูกต้อง")
        
    model_path = 'models/ensemble_reg_model.pkl'
    if not os.path.exists(model_path):
        return render_template('explain_ml_regression.html', message="ไม่พบไฟล์โมเดล กรุณา Train ข้อมูล Insurance ก่อนทดสอบ")
        
    try:
        import joblib
        model = joblib.load(model_path)
        prediction = model.predict(features)
        
        # prediction เป็น array เช่น [12450.5]
        predicted_cost = f"${prediction[0]:,.2f} USD"
        
        return render_template('explain_ml_regression.html', 
            message=f"💰 การทำนายค่าใช้จ่ายโดยประมาณ: <strong>{predicted_cost}</strong>")
            
    except Exception as e:
        return render_template('explain_ml_regression.html', message=f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")

@app.route('/predict/nn_regression', methods=['POST'])
def predict_nn_regression_route():
    """ทำนายค่าใช้จ่ายด้วย Neural Network Regression"""
    features = prepare_insurance_features(request.form)
    
    if features is None:
        return render_template('explain_nn_regression.html', message="กรุณากรอกข้อมูลเป็นตัวเลขที่ถูกต้อง")
        
    model_path = 'models/nn_reg_model.pkl'
    if not os.path.exists(model_path):
        return render_template('explain_nn_regression.html', message="ไม่พบไฟล์โมเดล กรุณา Train ข้อมูล Insurance ก่อนทดสอบ")
        
    try:
        import joblib
        model = joblib.load(model_path)
        prediction = model.predict(features)
        
        predicted_cost = f"${prediction[0]:,.2f} USD"
        
        return render_template('explain_nn_regression.html', 
            message=f"💰 การทำนายค่าใช้จ่ายโดยประมาณ: <strong>{predicted_cost}</strong>")
            
    except Exception as e:
        return render_template('explain_nn_regression.html', message=f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")

if __name__ == '__main__':
    # รันเซิร์ฟเวอร์แบบ debug เพื่อการพัฒนา
    app.run(debug=True, port=5000)
