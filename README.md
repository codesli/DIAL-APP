# DIAL — AI-Powered Leukemia Cell Detection

**Student project · Intel® AI Global Impact Festival 2026**

DIAL is a web application that uses a convolutional neural network to help identify Acute Lymphoblastic Leukemia (ALL) cells from microscopic blood smear images — built as an assistive tool for medical professionals.

---

## The Problem

Acute Lymphoblastic Leukemia is the most common cancer in children, and it progresses
quickly — early detection has a major impact on treatment outcomes and survival rates.
Diagnosis today relies on trained specialists manually examining blood and bone marrow
samples under a microscope, a process that is accurate but time-consuming and dependent
on specialist availability, especially outside major hospitals.

## Solution

DIAL is a prototype clinical support tool: a doctor logs into a secure portal, uploads a
microscopic image of a blood cell, and within seconds receives:
- A classification (**Healthy** or **Leukemia (ALL)**)
- A confidence score from the model
- An explainability (XAI) report highlighting the morphological indicators behind the
  prediction — so the result is a starting point for review, not a black-box answer

The tool is designed to sit **alongside** a hematologist, speeding up preliminary
screening rather than replacing clinical judgment.

## How It Works

1. A Convolutional Neural Network (CNN), built with TensorFlow/Keras, is trained on the
   [C-NMC Leukemia dataset](https://competitions.codalab.org/competitions/20395) —
   labeled microscopic images of healthy and leukemic (ALL) lymphocytes.
2. The model learns to recognize morphological differences between healthy lymphocytes
   and lymphoblasts (enlarged nucleus, irregular cell contour, altered cytoplasm ratio).
3. A Flask web app serves the trained model behind a login-protected interface, so only
   authorized medical staff can access the diagnostic tool.

**Tech stack:** Python, TensorFlow/Keras, Flask, HTML/CSS/JS

## Project Structure# DIAL — setup

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
