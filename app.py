import os
import datetime
import numpy as np
import gc  # Garbage Collector bach n-ms7o l-RAM
from flask import Flask, render_template, request, url_for, redirect, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash 
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func 
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from PIL import Image

# --- OPTIMIZATION EXTRÊME ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config['SECRET_KEY'] = 'votre_cle_secrete_nadi_2026'

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///' + os.path.join(BASE_DIR, 'predictions.db'))
if app.config['SQLALCHEMY_DATABASE_URI'] and app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- MODÈLES ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    predictions = db.relationship('Prediction', backref='author', lazy=True)
    def set_password(self, password): self.password_hash = generate_password_hash(password)
    def check_password(self, password): return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    result = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float)
    image_name = db.Column(db.String(100))
    timestamp = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id): return User.query.get(int(user_id))

classes = {0: "Limite 20", 1: "Limite 30", 14: "Stop", 13: "Cédez le passage", 17: "Sens interdit", 27: "Passage piétons", 12: "Route prioritaire"}

# --- ROUTES ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            login_user(user)
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        new_user = User(username=request.form['username']); new_user.set_password(request.form['password'])
        db.session.add(new_user); db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout(): logout_user(); return redirect(url_for('login'))

@app.route('/')
@login_required
def index(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    f = request.files['imagefile']
    filename = secure_filename(f.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(file_path)
    
    # LAZY LOADING + GARBAGE COLLECTION
    from tensorflow.keras.models import load_model
    model = load_model(os.path.join(BASE_DIR, 'model_traffic_signs.h5'))
    
    img = Image.open(file_path).convert("RGB").resize((32, 32))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    
    pred = model.predict(img_array)
    res, conf = classes.get(np.argmax(pred), "Inconnu"), round(float(np.max(pred)) * 100, 2)
    
    # Cleaning RAM deghya
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    new_p = Prediction(result=res, confidence=conf, image_name=filename, timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), author=current_user)
    db.session.add(new_p); db.session.commit()
    return render_template('result.html', prediction=res, confidence=conf, image_path=url_for('static', filename='uploads/'+filename))

@app.route('/dashboard')
@login_required
def dashboard():
    u_id = current_user.id
    user_total = Prediction.query.filter_by(user_id=u_id).count()
    global_total = Prediction.query.count()
    top_signs = db.session.query(Prediction.result, func.count(Prediction.result)).filter(Prediction.user_id == u_id).group_by(Prediction.result).limit(5).all()
    date_label = func.substr(Prediction.timestamp, 1, 10)
    evo_data = db.session.query(date_label, func.count(Prediction.id)).filter(Prediction.user_id == u_id).group_by(date_label).all()
    return render_template('dashboard.html', username=current_user.username, user_total=user_total, global_total=global_total, labels=[r[0] for r in top_signs], values=[r[1] for r in top_signs], evo_labels=[r[0] for r in evo_data], evo_values=[r[1] for r in evo_data])

if __name__ == '__main__':
    with app.app_context(): db.create_all()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
