# app.py — Real Estate AI with License Control (v2 — stable, localized, "Remember me")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime, timedelta
import gspread
from google.oauth2.service_account import Credentials
from io import BytesIO

# --- Try XGBoost ---
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# --- Google Sheets setup ---
def get_gcp_credentials():
    return Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )

creds = get_gcp_credentials()
client = gspread.authorize(creds)

SHEET_ID = st.secrets["SHEET_ID"]
licenses_sheet = client.open_by_key(SHEET_ID).worksheet("Licenses")
logs_sheet = client.open_by_key(SHEET_ID).worksheet("Logs")

# --- Ensure headers exist ---
def ensure_headers():
    try:
        headers_licenses = ["key", "expiry", "email", "plan", "created_at", "status"]
        if not licenses_sheet.get_all_values() or licenses_sheet.get_all_values()[0] != headers_licenses:
            licenses_sheet.clear()
            licenses_sheet.append_row(headers_licenses)

        headers_logs = ["key", "email", "plan", "role", "created_at"]
        if not logs_sheet.get_all_values() or logs_sheet.get_all_values()[0] != headers_logs:
            logs_sheet.clear()
            logs_sheet.append_row(headers_logs)
    except Exception as e:
        st.warning(f"⚠️ Ошибка при проверке заголовков: {e}")

ensure_headers()

# --- Language packs ---
TEXTS = {
    "EN": {
        "title": "🏠 Real Estate AI",
        "auth_title": "🔑 Authorization",
        "auth_prompt": "Enter your license key",
        "email_prompt": "Enter your email",
        "csv_error": "❌ CSV must contain: city, sqft, rooms, bathrooms, price",
        "upload": "Upload CSV with data",
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
    },
    "RU": {
        "title": "🏠 ИИ для недвижимости",
        "auth_title": "🔑 Авторизация",
        "auth_prompt": "Введите лицензионный ключ",
        "email_prompt": "Введите email",
        "csv_error": "❌ CSV должен содержать: city, sqft, rooms, bathrooms, price",
        "upload": "Загрузите CSV с данными",
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
    }
}

# --- License validation ---
def check_key_valid(key: str, email: str):
    try:
        records = licenses_sheet.get_all_records()
        for row in records:
            if row["key"] == key and row["email"].lower() == email.lower():
                expiry = datetime.strptime(row["expiry"], "%Y-%m-%d")
                if expiry < datetime.now():
                    return False, None, None, None, "❌ Срок действия лицензии истёк"
                return True, row.get("status", "user"), row.get("plan", "Basic"), row.get("expiry"), "✅ Лицензия активна"
        return False, None, None, None, "❌ Лицензия не найдена"
    except Exception as e:
        return False, None, None, None, f"⚠️ Ошибка проверки лицензии: {e}"

# --- Log access ---
def log_access(key: str, email: str, role: str, plan: str):
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logs_sheet.append_row([key, email, plan, role, now])
    except:
        pass

# --- Cache ---
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
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    return model, preds

# --- Session memory ---
if "email" not in st.session_state:
    st.session_state.email = ""
if "key" not in st.session_state:
    st.session_state.key = ""

# --- UI ---
lang = st.sidebar.selectbox("🌐 Language / Язык", ["EN", "RU"])
TXT = TEXTS[lang]

st.sidebar.title(TXT["auth_title"])

# Try load from URL params
try:
    params = st.query_params
    if "email" in params:
        st.session_state.email = params["email"][0]
    if "key" in params:
        st.session_state.key = params["key"][0]
except:
    pass

email = st.sidebar.text_input(TXT["email_prompt"], value=st.session_state.email)
password = st.sidebar.text_input(TXT["auth_prompt"], value=st.session_state.key, type="password")
remember = st.sidebar.checkbox(TXT["remember"], value=True)

if st.sidebar.button(TXT["continue"]):
    if remember:
        st.session_state.email = email
        st.session_state.key = password

if not email or not password:
    st.info("👉 Введите email и лицензионный ключ, чтобы продолжить.")
    st.stop()

valid, role, plan, expiry, message = check_key_valid(password, email)

if not valid:
    st.error(message)
    st.stop()
else:
    st.success(message)
    log_access(password, email, role, plan)
    st.sidebar.markdown(
        f"""
        <div style='padding:15px; border-radius:10px; background-color:#1E3A8A; color:white;'>
            <h4 style='margin:0;'>📌 План: {plan}</h4>
            <p style='margin:0;'>⏳ Действителен до: {expiry}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Main app ---
if role in ["user", "admin"]:
    st.title(TXT["title"])

    uploaded_file = st.file_uploader(TXT["upload"], type=["csv"])
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        st.subheader(TXT["data_preview"])
        st.dataframe(df.head())

        required = {"city", "sqft", "rooms", "bathrooms", "price"}
        if not required.issubset(df.columns):
            st.error(TXT["csv_error"])
            st.stop()

        X = df[["sqft", "rooms", "bathrooms"]].astype(float)
        y = df["price"].astype(float)

        model_type = "linear"
        if str(plan).lower() == "pro":
            st.success("🚀 Pro план — выбор модели.")
            options = ["Linear Regression", "Random Forest"]
            if XGBOOST_AVAILABLE:
                options.append("XGBoost")
            choice = st.selectbox("Выберите модель:", options)
            model_type = {"Linear Regression": "linear", "Random Forest": "rf", "XGBoost": "xgb"}[choice]
        else:
            st.info("🔑 Basic план — только Linear Regression.")

        model, preds = train_model(X, y, model_type=model_type)

        # --- Метрики ---
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        avg_price = y.mean()
        mae_percent = (mae / avg_price) * 100

        st.write(f"**R²:** {r2:.3f} | **MAE:** {mae:,.0f} € (~{mae_percent:.2f}%)")
        if mae_percent < 2:
            st.success("📌 Прогноз очень точный (<2%).")
        elif mae_percent < 5:
            st.info("📌 Прогноз надёжный (ошибка <5%).")
        else:
            st.warning("📌 Ошибка прогноза выше 5%. Добавьте больше данных.")

        # --- График ---
        st.subheader(TXT["plot"])
        fig, ax = plt.subplots(figsize=(8, 5))
        for city in df["city"].unique():
            subset = df[df["city"] == city]
            ax.scatter(subset["sqft"], subset["price"], label=city, alpha=0.7)

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

        # --- Скачать ---
        png_buf = BytesIO()
        fig.savefig(png_buf, format="png", bbox_inches="tight")
        png_buf.seek(0)
        st.download_button(TXT["download_png"], png_buf, file_name="price_vs_sqft.png", mime="image/png")

        df["predicted_price"] = preds.astype(int)
        excel_buf = BytesIO()
        df.to_excel(excel_buf, index=False, engine="openpyxl")
        st.download_button(TXT["download"], excel_buf.getvalue(),
                           file_name="predictions.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # --- Прогноз ---
        st.subheader("🔮 Прогноз для нового объекта")
        sqft_input = st.number_input(TXT["prediction_input"], 1, 10000, 50)
        rooms_input = st.number_input("Комнат", 1, 10, 3)
        baths_input = st.number_input("Ванных", 1, 5, 2)
        if st.button("Рассчитать цену"):
            pred_price = model.predict(np.array([[sqft_input, rooms_input, baths_input]]))[0]
            st.success(TXT["prediction_result"].format(price=int(pred_price)))

# --- FAQ ---
FAQS = {
    "RU": [
        ("Как загрузить данные?", "Загрузите CSV-файл со столбцами: city, sqft, rooms, bathrooms, price."),
        ("Что такое R²?", "Показывает, насколько хорошо модель объясняет данные."),
        ("Что такое MAE?", "Средняя ошибка прогноза в евро."),
        ("Зачем нужен лицензионный ключ?", "Открывает доступ к функциям Basic или Pro."),
    ],
    "EN": [
        ("How to upload data?", "Upload CSV with: city, sqft, rooms, bathrooms, price."),
        ("What is R²?", "Shows how well the model fits the data."),
        ("What is MAE?", "Average absolute error of predictions."),
        ("Why license key?", "Unlocks Basic or Pro features."),
    ],
}

st.subheader("❓ FAQ")
for q, a in FAQS[lang]:
    with st.expander(q):
        st.write(a)

st.markdown("---")
if lang == "RU":
    st.info("📧 Поддержка: viktormatrix37@gmail.com")
else:
    st.info("📧 Support: viktormatrix37@gmail.com")








