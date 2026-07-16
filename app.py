import os
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
from werkzeug.security import generate_password_hash, check_password_hash
import io

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
# Use Flask's normal folders: templates/ and static/ sit next to this file.
# (No more '../Final' — that only worked on one machine's exact folder layout.)
app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='')

# Secret key: read from the environment in real use. Falls back to a random
# key so the app still runs for a demo, but that means sessions won't persist
# across restarts — set FLASK_SECRET_KEY yourself for anything beyond a demo.
app.secret_key = os.environ.get('FLASK_SECRET_KEY') or os.urandom(24)

# Session cookie hardening
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.environ.get('FLASK_ENV') == 'production',
    MAX_CONTENT_LENGTH=8 * 1024 * 1024,  # 8 MB upload cap
)

# ---------------------------------------------------------------------------
# Doctor credentials
# ---------------------------------------------------------------------------
# Never keep real passwords in plaintext in source code. Read them from the
# environment; only fall back to a demo password locally, and warn loudly.
DOCTOR_USER = os.environ.get('DOCTOR_USER', 'doctor@dial.org')
DOCTOR_PASS_HASH = os.environ.get('DOCTOR_PASS_HASH')

if not DOCTOR_PASS_HASH:
    DEMO_PASSWORD = 'medical2026'
    DOCTOR_PASS_HASH = generate_password_hash(DEMO_PASSWORD)
    print(
        "\n[WARNING] DOCTOR_PASS_HASH is not set — using a built-in demo "
        f"password ('{DEMO_PASSWORD}'). Set DOCTOR_USER and DOCTOR_PASS_HASH "
        "environment variables before this app is used with real patient data.\n"
    )

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# Configurable via env var so this runs on any machine, not just yours.
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(__file__), 'model', 'leukemia_model.h5'))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAP = {0: "Leukemia (ALL)", 1: "Healthy (HEM)"}

model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"\n[WARNING] Could not load model from {MODEL_PATH}: {e}")
    print("The site will still run, but /predict will return an error until a model is available.\n")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Public routes
# ---------------------------------------------------------------------------
@app.route('/')
def shfaq_home():
    return render_template('home.html')

@app.route('/info')
def shfaq_info():
    return render_template('info.html')

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        username = (data.get('username') or '').strip()
        password = data.get('password') or ''

        if username == DOCTOR_USER and check_password_hash(DOCTOR_PASS_HASH, password):
            session.clear()
            session['logged_in'] = True
            return jsonify({"success": True})
        else:
            # Generic message — don't reveal whether the username or the
            # password was the wrong part.
            return jsonify({"success": False, "message": "Invalid medical credentials."})

    if 'logged_in' in session:
        return redirect(url_for('shfaq_ai'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login_page'))

# ---------------------------------------------------------------------------
# Protected routes — doctors only
# ---------------------------------------------------------------------------
@app.route('/ai-tool')
def shfaq_ai():
    if 'logged_in' not in session:
        return redirect(url_for('login_page'))
    return render_template('AI.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'logged_in' not in session:
        return jsonify({"error": "Unauthorized access. Please log in."}), 401

    if model is None:
        return jsonify({"error": "Model is not loaded on the server. Contact the administrator."}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Please upload a PNG or JPEG image.'}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
    except UnidentifiedImageError:
        return jsonify({'error': 'Could not read the uploaded file as an image.'}), 400

    img = img.resize((180, 180))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_idx = int(np.argmax(score))

    return jsonify({
        'prediction': MAP.get(class_idx, 'Unknown'),
        'confidence': f"{100 * np.max(score):.2f}%"
    })


if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', '1') == '1'
    # Port 5001 avoids clashing with macOS AirPlay on port 5000.
    app.run(debug=debug_mode, port=5001)
