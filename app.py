import os
import datetime
import numpy as np
from flask import Flask, render_template, request, url_for, redirect, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash 
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func 
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from PIL import Image

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'votre_cle_secrete_nadi_2026')

# Configuration dial l-base de données (Render & Local)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///' + os.path.join(BASE_DIR, 'predictions.db'))
if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' 

# --- MODÈLES (BDD) ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    predictions = db.relationship('Prediction', backref='author', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    result = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float)
    image_name = db.Column(db.String(100))
    timestamp = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- IA MODEL (LAZY LOADING) ---
MODEL_PATH = os.path.join(BASE_DIR, 'model_traffic_signs.h5')
_model = None
_interpreter = None

def get_model():
    """Charge le modèle seulement quand nécessaire (lazy loading)"""
    global _model, _interpreter
    
    if _interpreter is None:
        # Priorité: TFLite si disponible (plus léger)
        tflite_path = os.path.join(BASE_DIR, 'model_traffic_signs.tflite')
        if os.path.exists(tflite_path):
            import tensorflow as tf
            _interpreter = tf.lite.Interpreter(model_path=tflite_path)
            _interpreter.allocate_tensors()
            print("✓ Modèle TFLite chargé avec succès")
        elif os.path.exists(MODEL_PATH):
            # Fallback: .h5 model
            from tensorflow.keras.models import load_model
            _model = load_model(MODEL_PATH)
            print("✓ Modèle .h5 chargé avec succès")
        else:
            raise FileNotFoundError("Aucun modèle trouvé (.h5 ou .tflite)")
    
    return _interpreter if _interpreter else _model

# Liste complète des 43 classes 
classes = {
    0: "Limite 20", 1: "Limite 30", 2: "Limite 50", 3: "Limite 60", 
    4: "Limite 70", 5: "Limite 80", 6: "Fin limite 80", 7: "Limite 100", 
    8: "Limite 120", 9: "Interdiction dépasser", 10: "Interdiction dépasser (>3.5t)", 
    11: "Priorité intersection", 12: "Route prioritaire", 13: "Cédez le passage", 
    14: "Stop", 15: "Interdiction totale", 16: "Camions interdits", 
    17: "Sens interdit", 18: "Danger", 19: "Virage gauche", 
    20: "Virage droite", 21: "Double virage", 22: "Cassis ou dos-d'âne", 
    23: "Chaussée glissante", 24: "Rétrécissement droite", 25: "Travaux", 
    26: "Feux tricolores", 27: "Passage piétons", 28: "Enfants", 
    29: "Vélos", 30: "Neige/Verglas", 31: "Animaux sauvages", 
    32: "Fin de toutes interdictions", 33: "Direction droite", 34: "Direction gauche", 
    35: "Tout droit", 36: "Droit ou droite", 37: "Droit ou gauche", 
    38: "Contournement droite", 39: "Contournement gauche", 40: "Rond-point", 
    41: "Fin interdiction dépasser", 42: "Fin interdiction dépasser (>3.5t)"
} 

def preprocess(img_path):
    """Prétraite l'image pour la prédiction"""
    img = Image.open(img_path).convert("RGB").resize((32, 32)) 
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(img_array):
    """Effectue la prédiction avec TFLite ou Keras model"""
    model = get_model()
    
    if _interpreter:  # TFLite
        input_details = _interpreter.get_input_details()
        output_details = _interpreter.get_output_details()
        _interpreter.set_tensor(input_details[0]['index'], img_array)
        _interpreter.invoke()
        prediction = _interpreter.get_tensor(output_details[0]['index'])
    else:  # Keras .h5
        prediction = model.predict(img_array, verbose=0)
    
    return prediction

# --- ROUTES D'AUTHENTIFICATION ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            login_user(user)
            flash('Connexion réussie !', 'success')
            return redirect(url_for('index'))
        flash("Nom d'utilisateur ou mot de passe incorrect.", 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if User.query.filter_by(username=request.form['username']).first():
            flash("Cet utilisateur existe déjà.", "danger")
            return redirect(url_for('register'))
        new_user = User(username=request.form['username'])
        new_user.set_password(request.form['password'])
        db.session.add(new_user)
        db.session.commit()
        flash("Compte créé avec succès !", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Déconnexion réussie.", "info")
    return redirect(url_for('login'))

# --- ROUTES PRINCIPALES ---
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'imagefile' not in request.files:
        flash("Aucune image fournie.", "danger")
        return redirect(url_for('index'))
    
    f = request.files['imagefile']
    if f.filename == '':
        flash("Aucune image sélectionnée.", "danger")
        return redirect(url_for('index'))
    
    filename = secure_filename(f.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(file_path)
    
    try:
        # Prétraitement et prédiction
        img_array = preprocess(file_path)
        pred = predict_image(img_array)
        
        # Résultats
        class_idx = np.argmax(pred)
        res = classes.get(class_idx, "Inconnu")
        conf = round(float(np.max(pred)) * 100, 2)
        
        # Sauvegarde dans la BDD
        new_p = Prediction(
            result=res, 
            confidence=conf, 
            image_name=filename, 
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            author=current_user
        )
        db.session.add(new_p)
        db.session.commit()
        
        return render_template(
            'result.html', 
            prediction=res, 
            confidence=conf, 
            image_path=url_for('static', filename='uploads/'+filename)
        )
    
    except Exception as e:
        flash(f"Erreur lors de la prédiction: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    u_id = current_user.id
    user_total = Prediction.query.filter_by(user_id=u_id).count()
    global_total = Prediction.query.count()
    
    # Top signs o evo_data bach d-dir les graphiques
    top_signs = db.session.query(
        Prediction.result, 
        func.count(Prediction.result)
    ).filter(
        Prediction.user_id == u_id
    ).group_by(Prediction.result).order_by(
        func.count(Prediction.result).desc()
    ).limit(5).all()
    
    date_label = func.substr(Prediction.timestamp, 1, 10)
    evo_data = db.session.query(
        date_label, 
        func.count(Prediction.id)
    ).filter(
        Prediction.user_id == u_id
    ).group_by(date_label).order_by(date_label).all()
    
    return render_template(
        'dashboard.html', 
        username=current_user.username, 
        user_total=user_total, 
        global_total=global_total, 
        labels=[r[0] for r in top_signs], 
        values=[r[1] for r in top_signs],
        evo_labels=[r[0] for r in evo_data], 
        evo_values=[r[1] for r in evo_data]
    )

@app.route('/health')
def health():
    """Health check endpoint for Render"""
    return {"status": "healthy", "model_loaded": _model is not None or _interpreter is not None}, 200

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
