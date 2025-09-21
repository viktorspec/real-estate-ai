import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from io import BytesIO
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import requests

st.set_page_config(page_title="🏡 AI Real Estate Predictor", layout="centered")

# --- Language selector ---
lang = st.sidebar.selectbox("🌐 Language / Язык", ["English", "Русский"])

T = {
    "English": {
        "auth_title": "🔑 Authorization",
        "auth_prompt": "Enter your access key:",
        "auth_error": "⛔ Invalid key",
        "auth_expired": "⛔ Key expired",
        "auth_success": "✅ Access granted",
        "admin_success": "✅ Admin access granted",
        "title": "🏡 AI Real Estate Price Predictor",
        "upload": "Upload CSV (columns: city, sqft, rooms, bathrooms, price)",
        "data_preview": "### Data preview",
        "plot": "📈 Actual vs Predicted Prices",
        "download": "📥 Download predictions as Excel",
        "csv_error": "CSV must contain: city, sqft, rooms, bathrooms, price",
        "admin_title": "👑 Admin: Manage Users",
        "current_keys": "📋 Current Keys",
        "add_key": "➕ Add New Key",
        "delete_key": "🗑 Delete Key",
        "expiry_optional": "Expiry date (optional)",
        "delete_prompt": "Enter key to delete",
        "extend_key": "⏳ Extend Key",
        "extend_prompt": "Enter key to extend",
        "extend_date": "New expiry date",
        "email_prompt": "Enter your email:",
        "logs": "📜 Login Logs",
        "download_logs": "📥 Download Logs as Excel",
        "filter_email": "🔍 Filter logs by email"
    },
    "Русский": {
        "auth_title": "🔑 Авторизация",
        "auth_prompt": "Введите ключ доступа:",
        "auth_error": "⛔ Неверный ключ",
        "auth_expired": "⛔ Срок действия ключа истёк",
        "auth_success": "✅ Доступ разрешён",
        "admin_success": "✅ Доступ администратора",
        "title": "🏡 AI-Прогноз цен недвижимости",
        "upload": "Загрузите CSV (колонки: city, sqft, rooms, bathrooms, price)",
        "data_preview": "### Предпросмотр данных",
        "plot": "📈 Фактическая vs прогнозируемая цена",
        "download": "📥 Скачать прогнозы в Excel",
        "csv_error": "CSV должен содержать: city, sqft, rooms, bathrooms, price",
        "admin_title": "👑 Админ: Управление пользователями",
        "current_keys": "📋 Текущие ключи",
        "add_key": "➕ Добавить ключ",
        "delete_key": "🗑 Удалить ключ",
        "expiry_optional": "Дата окончания (необязательно)",
        "delete_prompt": "Введите ключ для удаления",
        "extend_key": "⏳ Продлить ключ",
        "extend_prompt": "Введите ключ для продления",
        "extend_date": "Новая дата окончания",
        "email_prompt": "Введите ваш email:",
        "logs": "📜 Логи входов",
        "download_logs": "📥 Скачать логи в Excel",
        "filter_email": "🔍 Фильтр логов по email"
    }
}

# --- Google Sheets API connection ---
creds_dict = dict(st.secrets["GCP_CREDENTIALS"])
creds = Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
)
client = gspread.authorize(creds)
SHEET_ID = st.secrets["SHEET_ID"]
sheet = client.open_by_key(SHEET_ID).sheet1

# --- Helper: get user IP ---
def get_user_ip():
    try:
        return requests.get("https://api.ipify.org").text
    except:
        return "unknown"

# --- Load keys ---
def load_keys():
    records = sheet.get_all_records()
    df = pd.DataFrame(records)
    if "expiry_date" in df.columns:
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    return df

# --- Add key ---
def add_key(new_key, expiry_date=""):
    sheet.append_row([new_key, expiry_date, ""])
    st.success(f"✅ Key {new_key} added!")

# --- Delete key ---
def delete_key(del_key):
    records = sheet.get_all_records()
    for idx, row in enumerate(records, start=2):
        if row["key"] == del_key:
            sheet.delete_rows(idx)
            st.success(f"✅ Key {del_key} deleted!")
            return
    st.error("⚠️ Key not found")

# --- Extend key ---
def extend_key(ext_key, new_expiry):
    records = sheet.get_all_records()
    for idx, row in enumerate(records, start=2):
        if row["key"] == ext_key:
            sheet.update_cell(idx, 2, str(new_expiry))
            st.success(f"✅ Key {ext_key} extended until {new_expiry}")
            return
    st.error("⚠️ Key not found")

# --- Logging ---
def log_access(user_key, email, role):
    try:
        log_sheet = client.open_by_key(SHEET_ID).worksheet("logs")
    except:
        sh = client.open_by_key(SHEET_ID)
        sh.add_worksheet(title="logs", rows="1000", cols="5")
        log_sheet = sh.worksheet("logs")
        log_sheet.append_row(["timestamp", "key", "email", "role", "ip"])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ip = get_user_ip()
    log_sheet.append_row([timestamp, user_key, email, role, ip])

# --- Key validation ---
def check_key_valid(user_key, email=""):
    if user_key == st.secrets["ADMIN_KEY"]:
        return True, "admin", T[lang]["admin_success"]

    df = load_keys()
    row = df[df["key"] == user_key]

    if row.empty:
        return False, "user", T[lang]["auth_error"]

    expiry = row["expiry_date"].values[0]
    if not pd.isna(expiry) and expiry < pd.Timestamp(datetime.now()):
        return False, "user", T[lang]["auth_expired"]

    return True, "user", T[lang]["auth_success"]

# --- Authorization ---
st.sidebar.title(T[lang]["auth_title"])
password = st.sidebar.text_input(T[lang]["auth_prompt"], type="password")
email = st.sidebar.text_input(T[lang]["email_prompt"])

valid, role, message = check_key_valid(password, email)

if not valid:
    st.error(message)
    st.stop()
else:
    st.success(message)
    log_access(password, email, role)

# --- Admin Panel ---
if role == "admin":
    st.title(T[lang]["admin_title"])
    st.dataframe(load_keys())

# --- Main App ---
if role in ["user", "admin"]:
    st.title(T[lang]["title"])

    uploaded_file = st.file_uploader(
        T[lang]["upload"], type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(T[lang]["data_preview"])
        st.dataframe(df.head())

        required_cols = {"city", "sqft", "rooms", "bathrooms", "price"}
        if not required_cols.issubset(df.columns):
            st.error(T[lang]["csv_error"])
        else:
            # --- Basic vs Pro ---
            if role == "admin":
                model_choice = st.selectbox("Choose ML Model", ["Linear Regression", "RandomForest", "XGBoost"])
            else:
                if "pro" in password.lower() or "pro" in email.lower():
                    model_choice = st.selectbox("Choose ML Model", ["Linear Regression", "RandomForest", "XGBoost"])
                else:
                    st.info("🔑 Your plan: **Basic** (Linear Regression only).")
                    model_choice = "Linear Regression"

            # --- Features ---
            X = df[["city", "sqft", "rooms", "bathrooms"]]
            y = df["price"]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("city", OneHotEncoder(handle_unknown="ignore"), ["city"]),
                    ("num", "passthrough", ["sqft", "rooms", "bathrooms"])
                ]
            )

            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "RandomForest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = XGBRegressor(n_estimators=100, random_state=42)

            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X, y)
            preds = pipeline.predict(X)

            # --- Metrics ---
            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            st.write(f"**R² Score:** {r2:.3f}")
            st.write(f"**MAE:** {mae:.2f} €")

            # --- Plot ---
            st.write(T[lang]["plot"])
            fig, ax = plt.subplots()
            ax.scatter(y, preds, alpha=0.7, label="Predictions")
            ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="Perfect Fit")
            ax.set_xlabel("Actual Price (€)")
            ax.set_ylabel("Predicted Price (€)")
            ax.legend()
            st.pyplot(fig)

            # --- New prediction ---
            st.subheader("🔮 Predict New Property")
            city_input = st.text_input("City", "Madrid")
            sqft_input = st.number_input("Square footage", min_value=20, max_value=500, value=70, step=5)
            rooms_input = st.number_input("Rooms", min_value=1, max_value=10, value=2, step=1)
            bathrooms_input = st.number_input("Bathrooms", min_value=1, max_value=5, value=1, step=1)

            if st.button("Predict Price"):
                new_data = pd.DataFrame([[city_input, sqft_input, rooms_input, bathrooms_input]],
                                        columns=["city", "sqft", "rooms", "bathrooms"])
                price_pred = pipeline.predict(new_data)[0]
                st.success(f"Predicted price: {int(price_pred):,} €")

            # --- Export ---
            df["predicted_price"] = preds.astype(int)
            output = BytesIO()
            df.to_excel(output, index=False, engine="openpyxl")
            st.download_button(
                label=T[lang]["download"],
                data=output.getvalue(),
                file_name="real_estate_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


