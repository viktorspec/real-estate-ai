# app.py — AI Real Estate SaaS с тарифами Basic / Pro и выбором моделей

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from io import BytesIO
from datetime import datetime, timedelta
import gspread
from google.oauth2.service_account import Credentials
import requests

st.set_page_config(page_title="🏡 AI Real Estate SaaS", layout="centered")

# --- Language selector ---
lang = st.sidebar.selectbox("🌐 Language / Язык", ["English", "Русский"])

T = {
    "English": {
        "auth_title": "🔑 Authorization",
        "auth_prompt": "Enter your access key:",
        "auth_error": "⛔ Invalid key",
        "auth_expired": "⛔ Key expired",
        "auth_success": "✅ Access granted (Plan: {plan})",
        "admin_success": "✅ Admin access granted",
        "title": "🏡 AI Real Estate Price Predictor",
        "upload": "Upload CSV (columns: city, sqft, rooms, bathrooms, price)",
        "data_preview": "### Data preview",
        "plot": "### Price vs. Square Footage",
        "xlabel": "Square Footage (sqft)",
        "ylabel": "Price (€)",
        "prediction_input": "Enter square footage:",
        "prediction_result": "Predicted price: {price:,} €",
        "download": "📥 Download predictions as Excel",
        "csv_error": "CSV must contain columns: city, sqft, rooms, bathrooms, price",
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
        "auth_success": "✅ Доступ разрешён (Тариф: {plan})",
        "admin_success": "✅ Доступ администратора",
        "title": "🏡 AI-Прогноз цен недвижимости",
        "upload": "Загрузите CSV (колонки: city, sqft, rooms, bathrooms, price)",
        "data_preview": "### Предпросмотр данных",
        "plot": "### Зависимость цены от площади",
        "xlabel": "Площадь (кв. футы)",
        "ylabel": "Цена (€)",
        "prediction_input": "Введите площадь:",
        "prediction_result": "Прогноз цены: {price:,} €",
        "download": "📥 Скачать прогнозы в Excel",
        "csv_error": "CSV должен содержать колонки: city, sqft, rooms, bathrooms, price",
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

# --- Get user IP ---
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
def add_key(new_key, expiry_date="", plan="Basic"):
    sheet.append_row([new_key, expiry_date, "", plan])
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

# --- Logging system ---
def log_access(user_key, email, role, plan="Basic"):
    try:
        try:
            log_sheet = client.open_by_key(SHEET_ID).worksheet("logs")
        except:
            sh = client.open_by_key(SHEET_ID)
            log_sheet = sh.add_worksheet(title="logs", rows="1000", cols="6")
            log_sheet.append_row(["timestamp", "key", "email", "role", "plan", "ip"])

        headers = log_sheet.row_values(1)
        expected = ["timestamp", "key", "email", "role", "plan", "ip"]
        if headers != expected:
            log_sheet.clear()
            log_sheet.append_row(expected)

        logs = log_sheet.get_all_records()
        cutoff = datetime.now() - timedelta(days=30)
        new_data = [expected]

        for row in logs:
            try:
                ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                if ts >= cutoff:
                    new_data.append([row["timestamp"], row["key"], row["email"], row["role"], row["plan"], row["ip"]])
            except:
                new_data.append([row["timestamp"], row["key"], row["email"], row["role"], row["plan"], row["ip"]])

        if len(new_data) != len(logs) + 1:
            log_sheet.clear()
            log_sheet.update(new_data)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ip = get_user_ip()
        log_sheet.append_row([timestamp, user_key, email, role, plan, ip])
    except Exception as e:
        st.warning(f"⚠️ Logging error: {e}")

# --- Check key validity ---
def check_key_valid(user_key, email=""):
    if user_key == st.secrets["ADMIN_KEY"]:
        return True, "admin", "Admin", T[lang]["admin_success"]

    df = load_keys()
    row = df[df["key"] == user_key]

    if row.empty:
        return False, "user", "Basic", T[lang]["auth_error"]

    expiry = row["expiry_date"].values[0]
    user_val = row["user"].values[0] if "user" in df.columns else ""
    plan = row["plan"].values[0] if "plan" in df.columns else "Basic"

    if not pd.isna(expiry) and expiry < pd.Timestamp(datetime.now()):
        return False, "user", plan, T[lang]["auth_expired"]

    if user_val:
        if email and email != user_val:
            return False, "user", plan, f"⚠️ This key is already used by {user_val}"
        else:
            return True, "user", plan, T[lang]["auth_success"].format(plan=plan)
    else:
        if email:
            records = sheet.get_all_records()
            for idx, r in enumerate(records, start=2):
                if r["key"] == user_key:
                    sheet.update_cell(idx, 3, email)
                    break
        return True, "user", plan, T[lang]["auth_success"].format(plan=plan)

# --- Authorization ---
st.sidebar.title(T[lang]["auth_title"])
password = st.sidebar.text_input(T[lang]["auth_prompt"], type="password")
email = st.sidebar.text_input(T[lang]["email_prompt"])

valid, role, plan, message = check_key_valid(password.strip(), email.strip())

if not valid:
    st.error(message)
    st.stop()
else:
    st.success(message)
    log_access(password.strip(), email.strip(), role, plan)

# --- Admin Panel ---
if role == "admin":
    st.title(T[lang]["admin_title"])
    keys_df = load_keys()
    st.dataframe(keys_df)

    new_key = st.text_input("Enter new key")
    expiry_date = st.date_input(T[lang]["expiry_optional"], value=None)
    plan_type = st.selectbox("Plan", ["Basic", "Pro"])
    if st.button("Add Key"):
        if new_key.strip() == "":
            st.error("⚠️ Key cannot be empty")
        else:
            add_key(new_key, str(expiry_date) if expiry_date else "", plan_type)

    del_key = st.text_input(T[lang]["delete_prompt"])
    if st.button("Delete Key"):
        delete_key(del_key)

    ext_key = st.text_input(T[lang]["extend_prompt"])
    new_expiry = st.date_input(T[lang]["extend_date"], value=datetime.now())
    if st.button("Extend Key"):
        extend_key(ext_key, new_expiry)

    try:
        logs = client.open_by_key(SHEET_ID).worksheet("logs").get_all_records()
        logs_df = pd.DataFrame(logs)
        email_filter = st.text_input(T[lang]["filter_email"])
        if email_filter:
            logs_df = logs_df[logs_df["email"].str.contains(email_filter, case=False, na=False)]
        st.dataframe(logs_df)

        output = BytesIO()
        logs_df.to_excel(output, index=False, engine="openpyxl")
        st.download_button(
            label=T[lang]["download_logs"],
            data=output.getvalue(),
            file_name="login_logs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except:
        st.info("ℹ️ No logs yet.")

# --- Main App ---
if role in ["user", "admin"]:
    st.title(T[lang]["title"])

    uploaded_file = st.file_uploader(T[lang]["upload"], type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(T[lang]["data_preview"])
        st.dataframe(df.head())

        if {"city", "sqft", "rooms", "bathrooms", "price"}.issubset(df.columns):
            X = df[["sqft", "rooms", "bathrooms"]]
            y = df["price"]

            # --- Basic Plan: only Linear Regression ---
            if plan == "Basic":
                st.info("🔑 Ваш тариф: Basic — только Linear Regression.")
                model = LinearRegression()
                model.fit(X, y)

            # --- Pro Plan: multiple models ---
            elif plan == "Pro":
                st.success("🚀 Ваш тариф: Pro — выбор модели.")
                model_choice = st.selectbox("Выберите модель:", ["Linear Regression", "Random Forest", "XGBoost"])

                if model_choice == "Linear Regression":
                    model = LinearRegression()
                elif model_choice == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_choice == "XGBoost":
                    model = xgb.XGBRegressor(n_estimators=100, random_state=42)

                model.fit(X, y)

            # --- Plot ---
            st.write(T[lang]["plot"])
            fig, ax = plt.subplots()
            for city in df["city"].unique():
                city_data = df[df["city"] == city]
                ax.scatter(city_data["sqft"], city_data["price"], label=city)

            ax.plot(df["sqft"], model.predict(X), color="red", linewidth=2, label="Prediction")
            ax.set_xlabel(T[lang]["xlabel"])
            ax.set_ylabel(T[lang]["ylabel"])
            ax.legend()
            st.pyplot(fig)

            # --- Prediction ---
            sqft_value = st.number_input(T[lang]["prediction_input"], min_value=200, max_value=5000, step=50)
            if sqft_value:
                price_pred = model.predict([[sqft_value, 3, 2]])[0]
                st.success(T[lang]["prediction_result"].format(price=int(price_pred)))

            # --- Export to Excel ---
            df["predicted_price"] = model.predict(X).astype(int)
            output = BytesIO()
            df.to_excel(output, index=False, engine="openpyxl")
            st.download_button(
                label=T[lang]["download"],
                data=output.getvalue(),
                file_name="real_estate_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error(T[lang]["csv_error"])
