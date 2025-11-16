# app.py — Real Estate AI (CSV версия, без фотоаналитики)
# Автор: адаптация для Виктора Евтушенко

import os
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from io import BytesIO
import streamlit as st
from google.oauth2.service_account import Credentials
import gspread
import joblib



st.write("SECRETS LOADED:", "gcp_service_account" in st.secrets)
st.write(st.secrets)


# --- Попытка импортировать XGBoost ---
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# --- Авторизация Google Sheets ---
def get_gcp_credentials_from_secrets():
    return Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )


# --- DEV_MODE ---
DEV_MODE = os.environ.get("DEV_MODE", "False").lower() == "true"

if not DEV_MODE:
    creds = get_gcp_credentials_from_secrets()
    client = gspread.authorize(creds)
    SHEET_ID = st.secrets["SHEET_ID"]
    licenses_sheet = client.open_by_key(SHEET_ID).worksheet("Licenses")
    logs_sheet = client.open_by_key(SHEET_ID).worksheet("Logs")


# --- Интернационализация ---
TEXTS = {
    "EN": {
        "title": "🏠 Real Estate AI",
        "auth_title": "🔑 Authorization",
        "auth_prompt": "Enter your license key",
        "email_prompt": "Enter your email",
        "csv_error": "❌ CSV must contain required columns.",
        "upload": "Upload CSV file",
        "data_preview": "📊 Data Preview",
        "xlabel": "Living area (GrLivArea)",
        "ylabel": "Price (€)",
        "download": "⬇️ Download Predictions (CSV)",
        "download_png": "⬇️ Download Plot (PNG)",
        "prediction_result": "Predicted price: {price} €",
        "enter_credentials": "👉 Please enter your email and license key.",
        "error_license": "❌ Invalid or expired license",
        "plan_info": "📌 Plan: {plan}",
        "expiry_info": "⏳ Valid until: {date}",
    },
    "RU": {
        "title": "🏠 ИИ для недвижимости",
        "auth_title": "🔑 Авторизация",
        "auth_prompt": "Введите лицензионный ключ",
        "email_prompt": "Введите email",
        "csv_error": "❌ CSV должен содержать нужные столбцы.",
        "upload": "Загрузите CSV-файл",
        "data_preview": "📊 Предпросмотр данных",
        "xlabel": "Жилая площадь (GrLivArea)",
        "ylabel": "Цена (€)",
        "download": "⬇️ Скачать прогнозы (CSV)",
        "download_png": "⬇️ Скачать график (PNG)",
        "prediction_result": "Прогнозируемая цена: {price} €",
        "enter_credentials": "👉 Введите email и лицензионный ключ.",
        "error_license": "❌ Лицензия недействительна или истекла",
        "plan_info": "📌 План: {plan}",
        "expiry_info": "⏳ Действует до: {date}",
    }
}


# --- Проверка лицензии ---
def check_key_valid(key: str, email: str):
    if DEV_MODE:
        return True, "user", "Pro", "2099-12-31", "✅ Test license active (DEV_MODE)"

    try:
        records = licenses_sheet.get_all_records()

        for row in records:
            if str(row.get("key")).strip() == str(key).strip() and \
               row.get("email", "").lower() == email.lower():

                expiry = datetime.strptime(row["expiry"], "%Y-%m-%d")

                if expiry < datetime.now():
                    return False, None, None, None, "❌ License expired"

                return True, row.get("status", "user"), row.get("plan", "Basic"), row.get("expiry"), "✅ License valid"

        return False, None, None, None, "❌ License not found"

    except Exception as e:
        return False, None, None, None, f"⚠️ Error: {e}"


# --- Логи ---
def log_access(key: str, email: str, role: str, plan: str):
    if DEV_MODE:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        logs_sheet.append_row([key, email, plan, role, now])
    except Exception:
        pass


# --- Загрузка моделей ---
@st.cache_resource
def load_pretrained_model(model_type):
    path = os.path.join("model", f"{model_type}.pkl")
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"⚠️ Ошибка загрузки модели {model_type}: {e}")
    return None


# --- Интерфейс Streamlit ---
st.set_page_config(page_title="Real Estate AI", layout="wide")

lang = st.sidebar.selectbox("🌐 Language / Язык", ["RU", "EN"])
TXT = TEXTS[lang]

st.sidebar.title(TXT["auth_title"])
email = st.sidebar.text_input(TXT["email_prompt"])
key = st.sidebar.text_input(TXT["auth_prompt"], type="password")

if not email or not key:
    st.info(TXT["enter_credentials"])
    st.stop()

valid, role, plan, expiry, message = check_key_valid(key, email)

if not valid:
    st.error(message)
    st.stop()
else:
    st.success(message)
    log_access(key, email, role, plan)

st.sidebar.markdown(f"**{TXT['plan_info'].format(plan=plan)}**")
st.sidebar.markdown(f"**{TXT['expiry_info'].format(date=expiry)}**")

st.title(TXT["title"])

tabs = ["CSV Analysis"]
tab1 = st.tabs(tabs)[0]


# --- Основные признаки ---
REQUIRED_COLUMNS = [
    "GrLivArea", "OverallQual", "GarageCars", "GarageArea",
    "TotalBsmtSF", "FullBath", "YearBuilt", "Price"
]


# --- Анализ CSV ---
with tab1:
    st.header(TXT["upload"])
    uploaded = st.file_uploader("📂 CSV", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Ошибка чтения CSV: {e}")
            st.stop()

        st.subheader(TXT["data_preview"])
        st.dataframe(df.head())

        # Проверка колонок
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            st.error(TXT["csv_error"])
            st.error(f"Отсутствуют столбцы: {missing}")
            st.stop()

        X = df[REQUIRED_COLUMNS[:-1]]
        y = df["Price"]

        # Выбор модели
        model_choice = "linear"
        if plan.lower() in ["pro", "premium"]:
            options = ["Linear Regression", "Random Forest"]
            if XGB_AVAILABLE:
                options.append("XGBoost")

            choice = st.selectbox("Model:", options)

            if choice == "Random Forest":
                model_choice = "rf"
            elif choice == "XGBoost":
                model_choice = "xgb"

        model = load_pretrained_model(model_choice)

        if not model:
            st.error(f"Модель '{model_choice}' не найдена.")
            st.stop()

        preds = model.predict(X)
        df["PredictedPrice"] = preds

        st.success("✅ Прогноз выполнен!")
        st.dataframe(df.head())

        # --- График ---
        fig, ax = plt.subplots()
        ax.scatter(df["GrLivArea"], df["Price"], label="Фактическая цена")
        ax.scatter(df["GrLivArea"], df["PredictedPrice"], label="Прогноз")
        ax.set_xlabel(TXT["xlabel"])
        ax.set_ylabel(TXT["ylabel"])
        ax.legend()
        st.pyplot(fig)

        # --- Скачать CSV ---
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(TXT["download"], csv_data, "predictions.csv", "text/csv")

        # --- Скачать PNG ---
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(TXT["download_png"], buf.getvalue(), "plot.png", "image/png")













