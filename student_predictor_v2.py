import os
import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------- helper: normalize & map columns ----------
def normalize_name(s: str) -> str:
    return ''.join(ch.lower() for ch in str(s) if ch.isalnum())

ALIASES = {
    "Study Hours": ["study_hours", "study hours", "studyhours", "study"],
    "Sleep Hours": ["sleep_hours", "sleep hours", "sleephours", "sleep"],
    "Last Exam": ["last_exam", "last exam", "lastpercentage", "last_percentage", "last_exam_percentage", "last_percent"],
    "Internet Access": ["internet", "internet_access", "internet access", "hasinternet", "has_internet"],
    "Academy": ["academy", "academy_attendance", "academy attendance", "goestoacademy", "goes_to_academy"],
    "Family Income": ["family_income", "family income", "income", "familyincome"],
    "CoCurricular": ["cocurricular", "co-curricular", "co curricular", "cocurricularactivities", "co_curricular"],
    "Stress Level": ["stress_level", "stress level", "stresslevel", "stress"],
    "Health": ["health", "health_condition", "health condition"],
    "Percentage": ["final_percentage", "final percentage", "final%", "percentage", "percent", "finalpercent", "final_percent"]
}

CANONICAL_ORDER_NUMERIC = ["Study Hours", "Sleep Hours", "Last Exam"]
CANONICAL_ORDER_CATEGORICAL = ["Internet Access", "Academy", "Family Income", "CoCurricular", "Stress Level", "Health"]
ALL_REQUIRED = CANONICAL_ORDER_NUMERIC + CANONICAL_ORDER_CATEGORICAL + ["Percentage"]

def map_columns(df: pd.DataFrame):
    normalized_to_col = {normalize_name(col): col for col in df.columns}
    rename_map = {}
    found = set()

    for canonical, aliases in ALIASES.items():
        matched = None
        if normalize_name(canonical) in normalized_to_col:
            matched = normalized_to_col[normalize_name(canonical)]
        else:
            for alias in aliases:
                key = normalize_name(alias)
                if key in normalized_to_col:
                    matched = normalized_to_col[key]
                    break
        if matched:
            rename_map[matched] = canonical
            found.add(canonical)

    if rename_map:
        df = df.rename(columns=rename_map)

    return df, found

# ---------- TRAIN MODEL ----------
def train_model():
    try:
        candidates = ["student_data_complete.csv", "student_data_large.csv", "student_data.csv", "student_synthetic_12000.csv"]
        file_name = next((c for c in candidates if os.path.exists(c)), None)
        if file_name is None:
            messagebox.showerror("CSV not found", "Place your training CSV (e.g., 'student_synthetic_12000.csv') in this folder.")
            return

        df = pd.read_csv(file_name)
        df, found = map_columns(df)

        missing = [c for c in ALL_REQUIRED if c not in df.columns]
        if missing:
            messagebox.showerror("Missing columns", f"Missing required columns: {missing}\n\nFound: {list(df.columns)}")
            return

        for col in CANONICAL_ORDER_NUMERIC + ["Percentage"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=CANONICAL_ORDER_NUMERIC + ["Percentage"])

        X = df.drop(columns=["Percentage"])
        y = df["Percentage"]

        categorical_features = CANONICAL_ORDER_CATEGORICAL
        numeric_features = CANONICAL_ORDER_NUMERIC

        # ✅ Handle both old and new sklearn versions
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", encoder, categorical_features),
                ("num", StandardScaler(), numeric_features),
            ],
            remainder='drop'
        )

        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", GradientBoostingRegressor(n_estimators=350, learning_rate=0.05, random_state=42))
        ])

        model.fit(X, y)
        joblib.dump(model, "smart_student_model.pkl")
        messagebox.showinfo("Training complete", f"✅ Model trained successfully using '{file_name}'.")

    except Exception as e:
        messagebox.showerror("Training error", f"Training failed:\n{e}")

# ---------- PREDICT ----------
def predict_percentage():
    try:
        if not os.path.exists("smart_student_model.pkl"):
            messagebox.showerror("Model not found", "Train the model first by clicking 'Train AI Model'.")
            return

        model = joblib.load("smart_student_model.pkl")

        input_df = pd.DataFrame([{
            "Study Hours": float(entry_study.get()),
            "Sleep Hours": float(entry_sleep.get()),
            "Last Exam": float(entry_last.get()),
            "Internet Access": internet_var.get(),
            "Academy": academy_var.get(),
            "Family Income": income_var.get(),
            "CoCurricular": cocurricular_var.get(),
            "Stress Level": stress_var.get(),
            "Health": health_var.get()
        }])

        pred = model.predict(input_df)[0]
        pred = float(np.clip(pred, 0, 100))
        messagebox.showinfo("Prediction", f"🎯 Predicted Percentage: {pred:.2f}%")

    except Exception as e:
        messagebox.showerror("Prediction error", f"Prediction failed:\n{e}")

# ---------- GUI ----------
root = tk.Tk()
root.title("Smart Student Predictor — Fixed")

# Optional .ico support
ico_name = next((f for f in os.listdir() if f.endswith(".ico")), None)
if ico_name:
    try:
        root.iconbitmap(ico_name)
    except Exception:
        pass

root.geometry("460x630")
root.configure(bg="#f6fafc")

tk.Label(root, text="🤖 Smart Student Predictor (Stable Build)", font=("Segoe UI", 14, "bold"), bg="#f6fafc").pack(pady=10)

frm = tk.Frame(root, bg="#f6fafc")
frm.pack(pady=5)

def add_field(label):
    tk.Label(frm, text=label, bg="#f6fafc", font=("Segoe UI", 10)).pack(anchor="w", pady=(6, 0))
    e = tk.Entry(frm, width=30)
    e.pack()
    return e

entry_study = add_field("Study Hours (e.g. 20)")
entry_sleep = add_field("Sleep Hours (e.g. 7)")
entry_last = add_field("Last Exam % (e.g. 75)")

def add_combo(label, var, options):
    tk.Label(frm, text=label, bg="#f6fafc", font=("Segoe UI", 10)).pack(anchor="w", pady=(6, 0))
    cb = ttk.Combobox(frm, values=options, textvariable=var, state="readonly")
    cb.pack()
    cb.current(0)

internet_var = tk.StringVar(value="Yes")
academy_var = tk.StringVar(value="Yes")
income_var = tk.StringVar(value="Moderate")
cocurricular_var = tk.StringVar(value="Moderate")
stress_var = tk.StringVar(value="Moderate")
health_var = tk.StringVar(value="Good")

add_combo("Internet Access", internet_var, ["Yes", "No"])
add_combo("Academy", academy_var, ["Yes", "No"])
add_combo("Family Income", income_var, ["Low", "Moderate", "High"])
add_combo("CoCurricular", cocurricular_var, ["Low", "Moderate", "High"])
add_combo("Stress Level", stress_var, ["Low", "Moderate", "High"])
add_combo("Health", health_var, ["Poor", "Average", "Good"])

btn_frame = tk.Frame(root, bg="#f6fafc")
btn_frame.pack(pady=14)

tk.Button(btn_frame, text="🧠 Train AI Model", command=train_model, bg="#2ea44f", fg="white", width=16, height=1).grid(row=0, column=0, padx=6)
tk.Button(btn_frame, text="🔮 Predict", command=predict_percentage, bg="#0366d6", fg="white", width=16, height=1).grid(row=0, column=1, padx=6)

tk.Label(root, text="Supported CSV names:\nstudent_data_complete.csv / student_data_large.csv / student_data.csv / student_synthetic_12000.csv\n\n"
                   "Place the CSV and optional .ico file in the SAME folder as this script.",
         bg="#f6fafc", font=("Segoe UI", 8), justify="left").pack(pady=8)

# ✨ Credit label
tk.Label(root, text="Made by Ahsan Khan Yousafzai", font=("Segoe UI", 9, "italic"),
         bg="#f6fafc", fg="#555").pack(side="bottom", pady=8)

root.mainloop()
