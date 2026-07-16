# DIAL — setup

```
DIAL-App/
├── app.py
├── requirements.txt
├── templates/
│   ├── home.html
│   ├── info.html
│   ├── AI.html
│   └── login.html
├── static/
│   ├── style.css
│   ├── webpic2.png     ← add your own image here
│   └── web1pic.jpeg    ← add your own image here
├── model/
│   └── leukemia_model.h5   ← put your trained model here
├── train_ai.py
├── predict.py
└── test.py
```

## Run it

```bash
pip install -r requirements.txt

export DOCTOR_USER="doctor@dial.org"
export DOCTOR_PASS_HASH=$(python3 -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('YOUR-REAL-PASSWORD'))")
export FLASK_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
export MODEL_PATH="./model/leukemia_model.h5"

python3 app.py
```

Then open `http://127.0.0.1:5001`.

If you skip the `DOCTOR_*` env vars, the app still runs with a demo login
(`doctor@dial.org` / `medical2026`) and prints a warning — fine for testing,
not fine once real patients' images touch this.

## Training

```bash
export DATASET_DIR="/path/to/C-NMC_Leukemia/training_data/fold_2"
python3 train_ai.py
```

Saves to `./model/leukemia_model.h5` by default (override with `MODEL_PATH`).
