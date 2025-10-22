# app_premium.py
# 🏠 Real Estate AI Agency — Unified Streamlit app
# Combines old functionality + Premium (image valuation) + DEV_MODE + license checks
# Developed for Viktor Yevtushenko

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime, timedelta
from io import BytesIO

# Optional XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# --- Premium image model (TensorFlow ResNet50) ---
try:
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# --- DEV MODE (offline test mode) ---
# set to True to run without Google Sheets; "test" key or @example.com emails are accepted
DEV_MODE = st.secrets.get("DEV_MODE", False) if "DEV_MODE" in st.secrets else False

if not DEV_MODE:
    import gspread
    from google.oauth2.service_account import Credentials

# --- Google Sheets setup (only if not DEV_MODE) ---
def get_gcp_credentials_from_secrets():
    # expects st.secrets["gcp_service_account"] present in deployed env
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

    def ensure_headers():
        try:
            headers_licenses = ["key", "expiry", "email", "plan", "created_at", "status"]
            cur = licenses_sheet.get_all_values()
            if not cur or cur[0] != headers_licenses:
                licenses_sheet.clear()
                licenses_sheet.append_row(headers_licenses)
            headers_logs = ["key", "email", "plan", "role", "created_at"]
            cur2 = logs_sheet.get_all_values()
            if not cur2 or cur2[0] != headers_logs:
                logs_sheet.clear()
                logs_sheet.append_row(headers_logs)
        except Exception as e:
            st.warning(f"⚠️ Error ensuring sheet headers: {e}")

    ensure_headers()

# --- Localization texts ---
TEXTS = {
    "EN": {
        "title": "🏠 Real Estate AI",
        "auth_title": "🔑 Authorization",
        "auth_prompt": "Enter your license key",
        "email_prompt": "Enter your email",
        "csv_error": "❌ CSV must contain: city, sqft, rooms, bathrooms, price",
        "upload": "Upload CSV or property photo",
        "data_preview": "📊 Data Preview",
        "plot": "📈 Price vs. Sqft",
        "xlabel": "Square footage",
        "ylabel": "Price (€)",
        "download": "⬇️ Download Predictions (Excel)",
        "download_png": "⬇️ Download Plot (PNG)",
        "prediction_input": "Enter square footage for prediction",
        "prediction_result": "Predicted price: {price} €",
        "remember": "💾 Remember me",
        "continue": "Continue",
        "enter_credentials": "👉 Please enter your email and license key to continue.",
        "error_license": "❌ Invalid or expired license",
        "plan_info": "📌 Plan: {plan}",
        "expiry_info": "⏳ Valid until: {date}",
        "photo_upload": "📷 Upload property photo (Premium only)",
        "photo_result": "🏠 Estimated value: €{price} ±5%",
        "not_premium": "📷 Photo analysis is available only for Premium plan.",
    },
    "RU": {
        "title": "🏠 ИИ для недвижимости",
        "auth_title": "🔑 Авторизация",
        "auth_prompt": "Введите лицензионный ключ",
        "email_prompt": "Введите email",
        "csv_error": "❌ CSV должен содержать: city, sqft, rooms, bathrooms, price",
        "upload": "Загрузите CSV или фото недвижимости",
        "data_preview": "📊 Предпросмотр данных",
        "plot": "📈 Цена vs. Площадь",
        "xlabel": "Площадь (кв.м)",
        "ylabel": "Цена (€)",
        "download": "⬇️ Скачать прогнозы (Excel)",
        "download_png": "⬇️ Скачать график (PNG)",
        "prediction_input": "Введите площадь для прогноза",
        "prediction_result": "Прогнозируемая цена: {price} €",
        "remember": "💾 Запомнить меня",
        "continue": "Продолжить",
        "enter_credentials": "👉 Введите email и лицензионный ключ, чтобы продолжить.",
        "error_license": "❌ Лицензия недействительна или истекла",
        "plan_info": "📌 План: {plan}",
        "expiry_info": "⏳ Действует до: {date}",
        "photo_upload": "📷 Загрузите фото недвижимости (только Premium)",
        "photo_result": "🏠 Оценочная стоимость: €{price} ±5%",
        "not_premium": "📷 Анализ фото доступен только для тарифа Premium.",
    }
}

# --- License check & logging ---
def check_key_valid(key: str, email: str):
    """Return (valid:bool, role, plan, expiry_str, message)"""
    if DEV_MODE:
        # In dev mode, accept "test" key or emails at example.com as test users
        if key.strip().lower() == "test" or email.strip().endswith("@example.com"):
            return True, "user", "Pro", "2099-12-31", "✅ Test license active (DEV_MODE)"
        # fallback: allow basic offline
        return True, "user", "Basic", "2099-12-31", "✅ Offline license (DEV_MODE)"
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
        return False, None, None, None, f"⚠️ Error checking license: {e}"

def log_access(key: str, email: str, role: str, plan: str):
    if DEV_MODE:
        return
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logs_sheet.append_row([key, email, plan, role, now])
    except Exception:
        pass

# --- Caching utilities ---
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

@st.cache_data
def train_model(X, y, model_type="linear"):
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        if XGBOOST_AVAILABLE:
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    return model, preds

# --- ResNet loader (Premium) ---
@st.cache_resource
def load_resnet_model():
    if not TF_AVAILABLE:
        return None
    base = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return base

def predict_value_from_image_bytes(file_buffer):
    """Return estimated value (int) or None"""
    if not TF_AVAILABLE:
        return None
    model = load_resnet_model()
    try:
        img = load_img(file_buffer, target_size=(224,224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)  # (1, feat_dim)
        val = float(np.mean(features)) * 1000.0  # simple heuristic
        val = int(max(50000, min(val, 2_000_000)))
        return val
    except Exception as e:
        st.error(f"Image analysis error: {e}")
        return None

# --- Session defaults ---
if "email" not in st.session_state:
    st.session_state.email = ""
if "key" not in st.session_state:
    st.session_state.key = ""
if "remember" not in st.session_state:
    st.session_state.remember = True

# --- UI config ---
st.set_page_config(page_title="Real Estate AI", layout="wide")
lang = st.sidebar.selectbox("🌐 Language / Язык", ["EN","RU"])
TXT = TEXTS[lang]

st.sidebar.title(TXT["auth_title"])

# Query params auto-fill (for Activate button link)
try:
    params = st.query_params

    if "email" in params and params["email"]:
        st.session_state.email = params["email"][0]
    if "key" in params and params["key"]:
        st.session_state.key = params["key"][0]
except Exception:
    pass

email = st.sidebar.text_input(TXT["email_prompt"], value=st.session_state.email)
password = st.sidebar.text_input(TXT["auth_prompt"], value=st.session_state.key, type="password")
remember = st.sidebar.checkbox(TXT["remember"], value=st.session_state.remember)

if st.sidebar.button(TXT["continue"]):
    st.session_state.remember = remember
    if remember:
        st.session_state.email = email
        st.session_state.key = password

# Validate presence
if (not email) or (not password):
    st.info(TXT["enter_credentials"])
    st.stop()

# License check
valid, role, plan, expiry, message = check_key_valid(password.strip(), email.strip())
if not valid:
    st.error(message)
    st.stop()
else:
    st.success(message)
    # show plan / expiry in sidebar
    st.sidebar.markdown(
        f"""
        <div style='padding:12px;border-radius:8px;background:#0b1120;color:#e5e7eb'>
            <strong>{TXT['plan_info'].format(plan=plan)}</strong><br>
            <small>{TXT['expiry_info'].format(date=expiry)}</small>
        </div>
        """,
        unsafe_allow_html=True
    )
    log_access(password.strip(), email.strip(), role, plan)

# --- Main App UI: tabs for CSV analysis and Photo valuation (if Premium) ---
st.title(TXT["title"])
tabs = ["CSV Analysis"]
if str(plan).lower() == "premium":
    tabs.append("Photo Valuation")
elif str(plan).lower() == "pro":
    tabs.append("CSV Analysis (Pro Models)")  # same tab but with model choices

tab1, *rest = st.tabs(tabs)

with tab1:
    st.header(TXT["upload"])
    uploaded_file = st.file_uploader(TXT["upload"], type=["csv","jpg","jpeg","png"])
    if uploaded_file is None:
        st.info("Upload a CSV file for CSV analysis or an image for photo valuation (Premium).")
    else:
        # CSV branch
        if uploaded_file.type == "text/csv" or uploaded_file.name.lower().endswith(".csv"):
            try:
                df = load_csv(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                st.stop()

            st.subheader(TXT["data_preview"])
            st.dataframe(df.head())

            required = {"city","sqft","rooms","bathrooms","price"}
            if not required.issubset(set(df.columns)):
                st.error(TXT["csv_error"])
            else:
                X = df[["sqft","rooms","bathrooms"]].astype(float)
                y = df["price"].astype(float)

                # choose model depending on plan
                default_model = "linear"
                model_type = default_model
                if str(plan).lower() in ["pro","premium"]:
                    st.success("🚀 Pro / Premium — choose model.")
                    options = ["Linear Regression","Random Forest"]
                    if XGBOOST_AVAILABLE:
                        options.append("XGBoost")
                    model_choice = st.selectbox("Select model:", options)
                    if model_choice == "Linear Regression":
                        model_type = "linear"
                    elif model_choice == "Random Forest":
                        model_type = "rf"
                    else:
                        model_type = "xgb"
                else:
                    st.info("🔑 Basic plan — Linear Regression only.")

                model, preds = train_model(X,y, model_type=model_type)
                r2 = r2_score(y, preds)
                mae = mean_absolute_error(y, preds)
                mae_percent = (mae / y.mean())*100 if y.mean() else 0.0

                st.write(f"**R²:** {r2:.3f} | **MAE:** {mae:,.0f} € (~{mae_percent:.2f}%)")
                if mae_percent < 2:
                    st.success("📌 High accuracy (<2%).")
                elif mae_percent < 5:
                    st.info("📌 Reliable (<5%).")
                else:
                    st.warning("📌 Consider adding more data.")

                # Plot
                st.subheader(TXT["plot"])
                fig, ax = plt.subplots(figsize=(8,5))
                for city in df["city"].unique():
                    cd = df[df["city"]==city]
                    ax.scatter(cd["sqft"], cd["price"], label=city, alpha=0.7)
                sqft_vals = np.linspace(df["sqft"].min(), df["sqft"].max(), 200)
                sqft_df = pd.DataFrame({
                    "sqft": sqft_vals,
                    "rooms": np.full_like(sqft_vals, 3),
                    "bathrooms": np.full_like(sqft_vals, 2)
                })
                pred_line = model.predict(sqft_df)
                ax.plot(sqft_vals, pred_line, color="red", linewidth=2, label="Prediction")
                ax.set_xlabel(TXT["xlabel"])
                ax.set_ylabel(TXT["ylabel"])
                ax.legend()
                st.pyplot(fig)

                # Download plot
                png_buf = BytesIO()
                fig.savefig(png_buf, format="png", bbox_inches="tight")
                st.download_button(TXT["download_png"], data=png_buf.getvalue(), file_name="price_vs_sqft.png", mime="image/png")

                # Export Excel with preds
                df_export = df.copy()
                df_export["predicted_price"] = preds.astype(int)
                out = BytesIO()
                df_export.to_excel(out, index=False, engine="openpyxl")
                st.download_button(TXT["download"], data=out.getvalue(), file_name="predictions.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                # Single prediction
                st.subheader("🔮 Predict New Property")
                sqft_input = st.number_input(TXT["prediction_input"], min_value=1, max_value=10000, value=int(np.median(df["sqft"])), step=1)
                rooms_input = st.number_input("Rooms", min_value=1, max_value=10, value=3, step=1)
                baths_input = st.number_input("Bathrooms", min_value=1, max_value=5, value=2, step=1)
                if st.button("Predict Price"):
                    new_X = np.array([[sqft_input, rooms_input, baths_input]])
                    pred_price = model.predict(new_X)[0]
                    st.success(TXT["prediction_result"].format(price=int(pred_price)))
        # Image branch
        elif uploaded_file.type.startswith("image/"):
            if str(plan).lower() != "premium":
                st.warning(TXT["not_premium"])
            else:
                st.image(uploaded_file, caption="Uploaded photo", use_container_width=True)
                with st.spinner("Analyzing image..."):
                    val = predict_value_from_image_bytes(uploaded_file)
                if val:
                    # color logic for range
                    if val < 200_000:
                        bg = "#2ecc71"  # green
                        label = "Low"
                    elif val < 500_000:
                        bg = "#f39c12"  # yellow
                        label = "Medium"
                    else:
                        bg = "#e74c3c"  # red
                        label = "High"

                    st.markdown(
                        f"""
                        <div style="padding:18px;border-radius:12px;background:{bg};color:#0b1120;text-align:center;font-weight:700;font-size:20px;">
                            🏠 {TXT['photo_result'].format(price=val)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# --- FAQ (both langs) ---
st.markdown("---")
st.subheader("❓ FAQ")
faqs = {
    "EN": [
        ("How to upload data?", "Upload a CSV file with columns: city, sqft, rooms, bathrooms, price."),
        ("What does R² mean?", "R² shows how well the model explains the data. 1.0 = perfect."),
        ("What is MAE?", "MAE = Mean Absolute Error. It shows average difference between predicted and actual price."),
        ("Why a license key?", "License unlocks Basic/Pro/Premium features.")
    ],
    "RU": [
        ("Как загрузить данные?", "Загрузите CSV файл со столбцами: city, sqft, rooms, bathrooms, price."),
        ("Что значит R²?", "Показывает, насколько хорошо модель объясняет данные. 1.0 = идеально."),
        ("Что такое MAE?", "MAE — средняя абсолютная ошибка прогноза."),
        ("Зачем ключ лицензии?", "Ключ даёт доступ к функциям Basic/Pro/Premium.")
    ]
}
for q,a in faqs[lang]:
    with st.expander(q):
        st.write(a)

st.markdown("---")
if lang == "RU":
    st.info("📧 Поддержка: viktormatrix37@gmail.com")
else:
    st.info("📧 Support: viktormatrix37@gmail.com")



















# app_premium.py
# 🏠 Real Estate AI Agency — Unified Streamlit app
# Combines old functionality + Premium (image valuation) + DEV_MODE + license checks
# Developed for Viktor Yevtushenko

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime, timedelta
from io import BytesIO

# Optional XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# --- Premium image model (TensorFlow ResNet50) ---
try:
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# --- DEV MODE (offline test mode) ---
# set to True to run without Google Sheets; "test" key or @example.com emails are accepted
DEV_MODE = st.secrets.get("DEV_MODE", False) if "DEV_MODE" in st.secrets else False

if not DEV_MODE:
    import gspread
    from google.oauth2.service_account import Credentials

# --- Google Sheets setup (only if not DEV_MODE) ---
def get_gcp_credentials_from_secrets():
    # expects st.secrets["gcp_service_account"] present in deployed env
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

    def ensure_headers():
        try:
            headers_licenses = ["key", "expiry", "email", "plan", "created_at", "status"]
            cur = licenses_sheet.get_all_values()
            if not cur or cur[0] != headers_licenses:
                licenses_sheet.clear()
                licenses_sheet.append_row(headers_licenses)
            headers_logs = ["key", "email", "plan", "role", "created_at"]
            cur2 = logs_sheet.get_all_values()
            if not cur2 or cur2[0] != headers_logs:
                logs_sheet.clear()
                logs_sheet.append_row(headers_logs)
        except Exception as e:
            st.warning(f"⚠️ Error ensuring sheet headers: {e}")

    ensure_headers()

# --- Localization texts ---
TEXTS = {
    "EN": {
        "title": "🏠 Real Estate AI",
        "auth_title": "🔑 Authorization",
        "auth_prompt": "Enter your license key",
        "email_prompt": "Enter your email",
        "csv_error": "❌ CSV must contain: city, sqft, rooms, bathrooms, price",
        "upload": "Upload CSV or property photo",
        "data_preview": "📊 Data Preview",
        "plot": "📈 Price vs. Sqft",
        "xlabel": "Square footage",
        "ylabel": "Price (€)",
        "download": "⬇️ Download Predictions (Excel)",
        "download_png": "⬇️ Download Plot (PNG)",
        "prediction_input": "Enter square footage for prediction",
        "prediction_result": "Predicted price: {price} €",
        "remember": "💾 Remember me",
        "continue": "Continue",
        "enter_credentials": "👉 Please enter your email and license key to continue.",
        "error_license": "❌ Invalid or expired license",
        "plan_info": "📌 Plan: {plan}",
        "expiry_info": "⏳ Valid until: {date}",
        "photo_upload": "📷 Upload property photo (Premium only)",
        "photo_result": "🏠 Estimated value: €{price} ±5%",
        "not_premium": "📷 Photo analysis is available only for Premium plan.",
    },
    "RU": {
        "title": "🏠 ИИ для недвижимости",
        "auth_title": "🔑 Авторизация",
        "auth_prompt": "Введите лицензионный ключ",
        "email_prompt": "Введите email",
        "csv_error": "❌ CSV должен содержать: city, sqft, rooms, bathrooms, price",
        "upload": "Загрузите CSV или фото недвижимости",
        "data_preview": "📊 Предпросмотр данных",
        "plot": "📈 Цена vs. Площадь",
        "xlabel": "Площадь (кв.м)",
        "ylabel": "Цена (€)",
        "download": "⬇️ Скачать прогнозы (Excel)",
        "download_png": "⬇️ Скачать график (PNG)",
        "prediction_input": "Введите площадь для прогноза",
        "prediction_result": "Прогнозируемая цена: {price} €",
        "remember": "💾 Запомнить меня",
        "continue": "Продолжить",
        "enter_credentials": "👉 Введите email и лицензионный ключ, чтобы продолжить.",
        "error_license": "❌ Лицензия недействительна или истекла",
        "plan_info": "📌 План: {plan}",
        "expiry_info": "⏳ Действует до: {date}",
        "photo_upload": "📷 Загрузите фото недвижимости (только Premium)",
        "photo_result": "🏠 Оценочная стоимость: €{price} ±5%",
        "not_premium": "📷 Анализ фото доступен только для тарифа Premium.",
    }
}

# --- License check & logging ---
def check_key_valid(key: str, email: str):
    """Return (valid:bool, role, plan, expiry_str, message)"""
    if DEV_MODE:
        # In dev mode, accept "test" key or emails at example.com as test users
        if key.strip().lower() == "test" or email.strip().endswith("@example.com"):
            return True, "user", "Pro", "2099-12-31", "✅ Test license active (DEV_MODE)"
        # fallback: allow basic offline
        return True, "user", "Basic", "2099-12-31", "✅ Offline license (DEV_MODE)"
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
        return False, None, None, None, f"⚠️ Error checking license: {e}"

def log_access(key: str, email: str, role: str, plan: str):
    if DEV_MODE:
        return
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logs_sheet.append_row([key, email, plan, role, now])
    except Exception:
        pass

# --- Caching utilities ---
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

@st.cache_data
def train_model(X, y, model_type="linear"):
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        if XGBOOST_AVAILABLE:
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    return model, preds

# --- ResNet loader (Premium) ---
@st.cache_resource
def load_resnet_model():
    if not TF_AVAILABLE:
        return None
    base = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return base

def predict_value_from_image_bytes(file_buffer):
    """Return estimated value (int) or None"""
    if not TF_AVAILABLE:
        return None
    model = load_resnet_model()
    try:
        img = load_img(file_buffer, target_size=(224,224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)  # (1, feat_dim)
        val = float(np.mean(features)) * 1000.0  # simple heuristic
        val = int(max(50000, min(val, 2_000_000)))
        return val
    except Exception as e:
        st.error(f"Image analysis error: {e}")
        return None

# --- Session defaults ---
if "email" not in st.session_state:
    st.session_state.email = ""
if "key" not in st.session_state:
    st.session_state.key = ""
if "remember" not in st.session_state:
    st.session_state.remember = True

# --- UI config ---
st.set_page_config(page_title="Real Estate AI", layout="wide")
lang = st.sidebar.selectbox("🌐 Language / Язык", ["EN","RU"])
TXT = TEXTS[lang]

st.sidebar.title(TXT["auth_title"])

# Query params auto-fill (for Activate button link)
try:
    params = st.query_params

    if "email" in params and params["email"]:
        st.session_state.email = params["email"][0]
    if "key" in params and params["key"]:
        st.session_state.key = params["key"][0]
except Exception:
    pass

email = st.sidebar.text_input(TXT["email_prompt"], value=st.session_state.email)
password = st.sidebar.text_input(TXT["auth_prompt"], value=st.session_state.key, type="password")
remember = st.sidebar.checkbox(TXT["remember"], value=st.session_state.remember)

if st.sidebar.button(TXT["continue"]):
    st.session_state.remember = remember
    if remember:
        st.session_state.email = email
        st.session_state.key = password

# Validate presence
if (not email) or (not password):
    st.info(TXT["enter_credentials"])
    st.stop()

# License check
valid, role, plan, expiry, message = check_key_valid(password.strip(), email.strip())
if not valid:
    st.error(message)
    st.stop()
else:
    st.success(message)
    # show plan / expiry in sidebar
    st.sidebar.markdown(
        f"""
        <div style='padding:12px;border-radius:8px;background:#0b1120;color:#e5e7eb'>
            <strong>{TXT['plan_info'].format(plan=plan)}</strong><br>
            <small>{TXT['expiry_info'].format(date=expiry)}</small>
        </div>
        """,
        unsafe_allow_html=True
    )
    log_access(password.strip(), email.strip(), role, plan)

# --- Main App UI: tabs for CSV analysis and Photo valuation (if Premium) ---
st.title(TXT["title"])
tabs = ["CSV Analysis"]
if str(plan).lower() == "premium":
    tabs.append("Photo Valuation")
elif str(plan).lower() == "pro":
    tabs.append("CSV Analysis (Pro Models)")  # same tab but with model choices

tab1, *rest = st.tabs(tabs)

with tab1:
    st.header(TXT["upload"])
    uploaded_file = st.file_uploader(TXT["upload"], type=["csv","jpg","jpeg","png"])
    if uploaded_file is None:
        st.info("Upload a CSV file for CSV analysis or an image for photo valuation (Premium).")
    else:
        # CSV branch
        if uploaded_file.type == "text/csv" or uploaded_file.name.lower().endswith(".csv"):
            try:
                df = load_csv(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                st.stop()

            st.subheader(TXT["data_preview"])
            st.dataframe(df.head())

            required = {"city","sqft","rooms","bathrooms","price"}
            if not required.issubset(set(df.columns)):
                st.error(TXT["csv_error"])
            else:
                X = df[["sqft","rooms","bathrooms"]].astype(float)
                y = df["price"].astype(float)

                # choose model depending on plan
                default_model = "linear"
                model_type = default_model
                if str(plan).lower() in ["pro","premium"]:
                    st.success("🚀 Pro / Premium — choose model.")
                    options = ["Linear Regression","Random Forest"]
                    if XGBOOST_AVAILABLE:
                        options.append("XGBoost")
                    model_choice = st.selectbox("Select model:", options)
                    if model_choice == "Linear Regression":
                        model_type = "linear"
                    elif model_choice == "Random Forest":
                        model_type = "rf"
                    else:
                        model_type = "xgb"
                else:
                    st.info("🔑 Basic plan — Linear Regression only.")

                model, preds = train_model(X,y, model_type=model_type)
                r2 = r2_score(y, preds)
                mae = mean_absolute_error(y, preds)
                mae_percent = (mae / y.mean())*100 if y.mean() else 0.0

                st.write(f"**R²:** {r2:.3f} | **MAE:** {mae:,.0f} € (~{mae_percent:.2f}%)")
                if mae_percent < 2:
                    st.success("📌 High accuracy (<2%).")
                elif mae_percent < 5:
                    st.info("📌 Reliable (<5%).")
                else:
                    st.warning("📌 Consider adding more data.")

                # Plot
                st.subheader(TXT["plot"])
                fig, ax = plt.subplots(figsize=(8,5))
                for city in df["city"].unique():
                    cd = df[df["city"]==city]
                    ax.scatter(cd["sqft"], cd["price"], label=city, alpha=0.7)
                sqft_vals = np.linspace(df["sqft"].min(), df["sqft"].max(), 200)
                sqft_df = pd.DataFrame({
                    "sqft": sqft_vals,
                    "rooms": np.full_like(sqft_vals, 3),
                    "bathrooms": np.full_like(sqft_vals, 2)
                })
                pred_line = model.predict(sqft_df)
                ax.plot(sqft_vals, pred_line, color="red", linewidth=2, label="Prediction")
                ax.set_xlabel(TXT["xlabel"])
                ax.set_ylabel(TXT["ylabel"])
                ax.legend()
                st.pyplot(fig)

                # Download plot
                png_buf = BytesIO()
                fig.savefig(png_buf, format="png", bbox_inches="tight")
                st.download_button(TXT["download_png"], data=png_buf.getvalue(), file_name="price_vs_sqft.png", mime="image/png")

                # Export Excel with preds
                df_export = df.copy()
                df_export["predicted_price"] = preds.astype(int)
                out = BytesIO()
                df_export.to_excel(out, index=False, engine="openpyxl")
                st.download_button(TXT["download"], data=out.getvalue(), file_name="predictions.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                # Single prediction
                st.subheader("🔮 Predict New Property")
                sqft_input = st.number_input(TXT["prediction_input"], min_value=1, max_value=10000, value=int(np.median(df["sqft"])), step=1)
                rooms_input = st.number_input("Rooms", min_value=1, max_value=10, value=3, step=1)
                baths_input = st.number_input("Bathrooms", min_value=1, max_value=5, value=2, step=1)
                if st.button("Predict Price"):
                    new_X = np.array([[sqft_input, rooms_input, baths_input]])
                    pred_price = model.predict(new_X)[0]
                    st.success(TXT["prediction_result"].format(price=int(pred_price)))
        # Image branch
        elif uploaded_file.type.startswith("image/"):
            if str(plan).lower() != "premium":
                st.warning(TXT["not_premium"])
            else:
                st.image(uploaded_file, caption="Uploaded photo", use_container_width=True)
                with st.spinner("Analyzing image..."):
                    val = predict_value_from_image_bytes(uploaded_file)
                if val:
                    # color logic for range
                    if val < 200_000:
                        bg = "#2ecc71"  # green
                        label = "Low"
                    elif val < 500_000:
                        bg = "#f39c12"  # yellow
                        label = "Medium"
                    else:
                        bg = "#e74c3c"  # red
                        label = "High"

                    st.markdown(
                        f"""
                        <div style="padding:18px;border-radius:12px;background:{bg};color:#0b1120;text-align:center;font-weight:700;font-size:20px;">
                            🏠 {TXT['photo_result'].format(price=val)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# --- FAQ (both langs) ---
st.markdown("---")
st.subheader("❓ FAQ")
faqs = {
    "EN": [
        ("How to upload data?", "Upload a CSV file with columns: city, sqft, rooms, bathrooms, price."),
        ("What does R² mean?", "R² shows how well the model explains the data. 1.0 = perfect."),
        ("What is MAE?", "MAE = Mean Absolute Error. It shows average difference between predicted and actual price."),
        ("Why a license key?", "License unlocks Basic/Pro/Premium features.")
    ],
    "RU": [
        ("Как загрузить данные?", "Загрузите CSV файл со столбцами: city, sqft, rooms, bathrooms, price."),
        ("Что значит R²?", "Показывает, насколько хорошо модель объясняет данные. 1.0 = идеально."),
        ("Что такое MAE?", "MAE — средняя абсолютная ошибка прогноза."),
        ("Зачем ключ лицензии?", "Ключ даёт доступ к функциям Basic/Pro/Premium.")
    ]
}
for q,a in faqs[lang]:
    with st.expander(q):
        st.write(a)

st.markdown("---")
if lang == "RU":
    st.info("📧 Поддержка: viktormatrix37@gmail.com")
else:
    st.info("📧 Support: viktormatrix37@gmail.com")




































































