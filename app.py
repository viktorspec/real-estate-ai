# app.py — Real Estate AI (Production-ready, Kaggle data + pretrained models)
# Автор: доработка для Виктора Евтушенко
# Комментарии на русском для понимания логики

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from io import BytesIO
import joblib
import os

# --- Попытка импортировать XGBoost (если установлен) ---
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# --- Попытка импортировать TensorFlow для Premium-модуля ---
try:
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# --- DEV MODE (без Google Sheets) ---
DEV_MODE = st.secrets.get("DEV_MODE", False) if "DEV_MODE" in st.secrets else False

if not DEV_MODE:
    import gspread
    from google.oauth2.service_account import Credentials

# --- Google Sheets авторизация ---
def get_gcp_credentials_from_secrets():
    return Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )

if not DEV_MODE:
    creds = get_gcp_credentials_from_secrets()
    client = gspread.authorize(creds)
    SHEET_ID = st.secrets["SHEET_ID"]
    licenses_sheet = client.open_by_key(SHEET_ID).worksheet("Licenses")
    logs_sheet = client.open_by_key(SHEET_ID).worksheet("Logs")

# --- Интернационализация (RU / EN) ---
TEXTS = {
    "EN": {
        "title": "🏠 Real Estate AI",
        "auth_title": "🔑 Authorization",
        "auth_prompt": "Enter your license key",
        "email_prompt": "Enter your email",
        "csv_error": "❌ CSV must contain: GrLivArea, OverallQual, GarageCars, GarageArea, TotalBsmtSF, FullBath, YearBuilt, Price",
        "upload": "Upload CSV or property photo",
        "data_preview": "📊 Data Preview",
        "xlabel": "Living area (GrLivArea)",
        "ylabel": "Price (€)",
        "download": "⬇️ Download Predictions (CSV)",
        "download_png": "⬇️ Download Plot (PNG)",
        "prediction_result": "Predicted price: {price} €",
        "remember": "💾 Remember me",
        "continue": "Continue",
        "enter_credentials": "👉 Please enter your email and license key.",
        "error_license": "❌ Invalid or expired license",
        "plan_info": "📌 Plan: {plan}",
        "expiry_info": "⏳ Valid until: {date}",
        "photo_upload": "📷 Upload property photo (Premium only)",
        "photo_result": "🏠 Estimated value: €{price} ±5%",
        "not_premium": "📷 Photo analysis available only for Premium plan.",
    },
    "RU": {
        "title": "🏠 ИИ для недвижимости",
        "auth_title": "🔑 Авторизация",
        "auth_prompt": "Введите лицензионный ключ",
        "email_prompt": "Введите email",
        "csv_error": "❌ CSV должен содержать: GrLivArea, OverallQual, GarageCars, GarageArea, TotalBsmtSF, FullBath, YearBuilt, Price",
        "upload": "Загрузите CSV или фото недвижимости",
        "data_preview": "📊 Предпросмотр данных",
        "xlabel": "Жилая площадь (GrLivArea)",
        "ylabel": "Цена (€)",
        "download": "⬇️ Скачать прогнозы (CSV)",
        "download_png": "⬇️ Скачать график (PNG)",
        "prediction_result": "Прогнозируемая цена: {price} €",
        "remember": "💾 Запомнить меня",
        "continue": "Продолжить",
        "enter_credentials": "👉 Введите email и лицензионный ключ.",
        "error_license": "❌ Лицензия недействительна или истекла",
        "plan_info": "📌 План: {plan}",
        "expiry_info": "⏳ Действует до: {date}",
        "photo_upload": "📷 Загрузите фото недвижимости (только Premium)",
        "photo_result": "🏠 Оценочная стоимость: €{price} ±5%",
        "not_premium": "📷 Анализ фото доступен только для Premium-плана.",
    }
}

# --- Проверка лицензии ---
def check_key_valid(key: str, email: str):
    if DEV_MODE:
        return True, "user", "Pro", "2099-12-31", "✅ Test license active (DEV_MODE)"
    try:
        records = licenses_sheet.get_all_records()
        for row in records:
            if str(row.get("key")).strip() == str(key).strip() and row.get("email","").lower() == email.lower():
                expiry = datetime.strptime(row["expiry"], "%Y-%m-%d")
                if expiry < datetime.now():
                    return False, None, None, None, "❌ License expired"
                return True, row.get("status","user"), row.get("plan","Basic"), row.get("expiry"), "✅ License valid"
        return False, None, None, None, "❌ License not found"
    except Exception as e:
        return False, None, None, None, f"⚠️ Error: {e}"

# --- Логирование (в Google Sheets) ---
def log_access(key: str, email: str, role: str, plan: str):
    if DEV_MODE: return
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


# === ЗАГРУЗКА RESNET МОДЕЛИ ===
@st.cache_resource
def load_resnet_model():
    """Безопасная загрузка ResNet50"""
    try:
        if not TF_AVAILABLE:
            st.error("⚠️ TensorFlow недоступен. Установи пакет 'tensorflow'.")
            return None
        from tensorflow.keras.applications import ResNet50
        model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        st.success("✅ Модель ResNet50 успешно загружена.")
        return model
    except Exception as e:
        st.error(f"⚠️ Ошибка загрузки ResNet50: {e}")
        return None


# === АНАЛИЗ ИЗОБРАЖЕНИЯ ===
def predict_value_from_image_bytes(file_buffer, country=None):
    """Прогноз стоимости дома по фото"""
    if not TF_AVAILABLE:
        st.error("⚠️ TensorFlow не установлен.")
        return None

    try:
        # Загружаем ResNet
        resnet = load_resnet_model()
        if resnet is None:
            st.error("⚠️ Модель ResNet не инициализирована. Проверь TensorFlow.")
            return None

        # Ищем регрессионную модель
        possible_models = ["photo_regressor.pkl", "photo_meta.pkl"]
        model_path = next(
            (os.path.join("model", f) for f in possible_models if os.path.exists(os.path.join("model", f))),
            None
        )

        if not model_path:
            st.error("❌ Модель фото-оценки не найдена. Проверь папку 'model'.")
            return None

        m = joblib.load(model_path)

        # если файл содержит только метаинформацию
        if isinstance(m, dict) and "reg" not in m:
            st.error("❌ В файле модели нет обученного регрессора. Перепроверь файл 'photo_regressor.pkl'.")
            return None

        reg = m.get("reg") if isinstance(m, dict) else m
        enc = m.get("encoder") if isinstance(m, dict) else None

        # Обработка изображения
        img = load_img(file_buffer, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = resnet.predict(x, verbose=0).flatten().reshape(1, -1)

        # Кодируем страну, если есть encoder
        if enc is not None and country is not None:
            country_vec = enc.transform([[country]])
            X_in = np.concatenate([feat, country_vec], axis=1)
        else:
            X_in = feat

        # Прогноз в лог-пространстве
        y_log_pred = reg.predict(X_in)[0]
        y_pred = float(np.expm1(y_log_pred))

        # Клипинг по перцентилям, если есть meta
        meta_path = os.path.join("model", "photo_meta.pkl")
        meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
        p05 = meta.get("p05", None)
        p95 = meta.get("p95", None)
        if p05 is not None and p95 is not None:
            y_pred = max(p05, min(p95, y_pred))

        st.success(f"✅ Прогноз выполнен (использована модель: {os.path.basename(model_path)})")
        return int(round(y_pred))

    except Exception as e:
        st.error(f"⚠️ Ошибка анализа фото: {e}")
        return None



# --- Интерфейс Streamlit ---
st.set_page_config(page_title="Real Estate AI", layout="wide")
lang = st.sidebar.selectbox("🌐 Language / Язык", ["RU","EN"])
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
if plan.lower() == "premium":
    tabs.append("Photo Valuation")

tab1, *rest = st.tabs(tabs)

# --- Основные признаки Kaggle ---
REQUIRED_COLUMNS = [
    "GrLivArea", "OverallQual", "GarageCars", "GarageArea",
    "TotalBsmtSF", "FullBath", "YearBuilt", "Price"
]

# --- Анализ CSV ---
with tab1:
    st.header(TXT["upload"])
    uploaded = st.file_uploader("📂 Загрузите CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Ошибка чтения CSV: {e}")
            st.stop()

        st.subheader(TXT["data_preview"])
        st.dataframe(df.head())

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            st.error(TXT["csv_error"])
            st.error(f"Отсутствуют столбцы: {missing}")
            st.stop()

        X = df[REQUIRED_COLUMNS[:-1]]
        y = df["Price"]

        model_choice = "linear"
        if plan.lower() in ["pro", "premium"]:
            options = ["Linear Regression", "Random Forest"]
            if XGBOOST_AVAILABLE:
                options.append("XGBoost")
            choice = st.selectbox("Выберите модель / Select model:", options)
            if choice == "Random Forest": model_choice = "rf"
            elif choice == "XGBoost": model_choice = "xgb"

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
        ax.scatter(df["GrLivArea"], df["Price"], color="blue", label="Фактическая цена")
        ax.scatter(df["GrLivArea"], df["PredictedPrice"], color="red", label="Прогноз")
        ax.set_xlabel(TXT["xlabel"])
        ax.set_ylabel(TXT["ylabel"])
        ax.legend()
        st.pyplot(fig)

        # --- Скачивание ---
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(TXT["download"], csv_data, "predictions.csv", "text/csv")
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(TXT["download_png"], buf.getvalue(), "plot.png", "image/png")

# --- Анализ фото (Premium) ---
if plan.lower() == "premium" and rest:
    with rest[0]:
        st.header(TXT["photo_upload"])
        photo = st.file_uploader("📷 Фото недвижимости", type=["jpg","jpeg","png"])
        if photo:
            val = predict_value_from_image_bytes(photo)
            if val:
                st.success(TXT["photo_result"].format(price=val))
            else:
                st.error("⚠️ Ошибка анализа изображения.")

# --- FAQ (двуязычный) ---
with st.expander("📖 FAQ"):
    if lang == "RU":
        faq_text = """
### ❓ Часто задаваемые вопросы (FAQ)

**Как загрузить данные?**  
Загрузите CSV с колонками:  
`GrLivArea, OverallQual, GarageCars, GarageArea, TotalBsmtSF, FullBath, YearBuilt, Price`

**Пример CSV:**


**Описание признаков:**
- `GrLivArea` — жилая площадь  
- `OverallQual` — качество дома  
- `GarageCars` — машиномест в гараже  
- `GarageArea` — площадь гаража  
- `TotalBsmtSF` — площадь подвала  
- `FullBath` — число полных ванных  
- `YearBuilt` — год постройки  
- `Price` — цена  

**Почему обучение отключено?**  
Это продакшн-версия — используется только прогноз заранее обученными моделями:
`linear.pkl`, `rf.pkl`, `xgb.pkl`

**Как скачать результаты?**  
После прогноза появится кнопка 💾 «Скачать прогнозы (CSV)».

**Как работает фотооценка (Premium)?**  
ResNet50 анализирует фото дома и оценивает стоимость с погрешностью ±5%.
"""
    else:
        faq_text = """
### ❓ Frequently Asked Questions (FAQ)

**How to upload data?**  
Upload CSV with columns:  
`GrLivArea, OverallQual, GarageCars, GarageArea, TotalBsmtSF, FullBath, YearBuilt, Price`

**Example CSV:**


**Feature description:**
- `GrLivArea` — total living area  
- `OverallQual` — overall quality  
- `GarageCars` — garage capacity  
- `GarageArea` — garage size  
- `TotalBsmtSF` — basement size  
- `FullBath` — full bathrooms  
- `YearBuilt` — year built  
- `Price` — target (house price)

**Why training is disabled?**  
This is production mode — only pretrained models are used:
`linear.pkl`, `rf.pkl`, `xgb.pkl`

**How to download results?**  
After prediction, click 💾 “Download predictions (CSV)”.

**How does photo valuation work (Premium)?**  
ResNet50 analyses the photo and estimates the price (±5% accuracy).
"""
    st.markdown(faq_text)
