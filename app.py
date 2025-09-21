# app.py — финальная версия с исправлением дублей
# Теперь сообщение об успешной авторизации выводится только один раз.
# Привязка ключа к email показывается отдельно голубым сообщением.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from io import BytesIO
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import requests  # для получения IP

st.set_page_config(page_title="🏡 AI Real Estate Predictor", layout="centered")

# --- Выбор языка интерфейса ---
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
        "auth_success": "✅ Доступ разрешён",
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

# --- Подключение к Google Sheets ---
creds_dict = dict(st.secrets["GCP_CREDENTIALS"])
creds = Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
)
client = gspread.authorize(creds)
SHEET_ID = st.secrets["SHEET_ID"]
sheet = client.open_by_key(SHEET_ID).sheet1

# --- Получение IP пользователя ---
def get_user_ip():
    try:
        return requests.get("https://api.ipify.org").text
    except:
        return "unknown"

# --- Загрузка ключей ---
def load_keys():
    records = sheet.get_all_records()
    df = pd.DataFrame(records)
    if "expiry_date" in df.columns:
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    return df

# --- Добавление ключа ---
def add_key(new_key, expiry_date=""):
    sheet.append_row([new_key, expiry_date, ""])
    st.success(f"✅ Key {new_key} added!")

# --- Удаление ключа ---
def delete_key(del_key):
    records = sheet.get_all_records()
    for idx, row in enumerate(records, start=2):
        if row["key"] == del_key:
            sheet.delete_rows(idx)
            st.success(f"✅ Key {del_key} deleted!")
            return
    st.error("⚠️ Key not found")

# --- Продление ключа ---
def extend_key(ext_key, new_expiry):
    records = sheet.get_all_records()
    for idx, row in enumerate(records, start=2):
        if row["key"] == ext_key:
            sheet.update_cell(idx, 2, str(new_expiry))
            st.success(f"✅ Key {ext_key} extended until {new_expiry}")
            return
    st.error("⚠️ Key not found")

# --- Логирование входов ---
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

# --- Проверка ключа ---
def check_key_valid(user_key, email=""):
    if user_key == st.secrets["ADMIN_KEY"]:
        return True, "admin", T[lang]["admin_success"]

    df = load_keys()
    row = df[df["key"] == user_key]

    if row.empty:
        return False, "user", T[lang]["auth_error"]

    expiry = row["expiry_date"].values[0]
    user_val = row["user"].values[0] if "user" in df.columns else ""

    if not pd.isna(expiry) and expiry < pd.Timestamp(datetime.now()):
        return False, "user", T[lang]["auth_expired"]

    if user_val:
        if email and email != user_val:
            return False, "user", f"⚠️ Этот ключ уже используется {user_val}"
        else:
            return True, "user", T[lang]["auth_success"]
    else:
        if email:
            records = sheet.get_all_records()
            for idx, r in enumerate(records, start=2):
                if r["key"] == user_key:
                    sheet.update_cell(idx, 3, email)
                    # 👇 Голубое уведомление вместо дубля
                    st.info(f"🔗 Ключ {user_key} привязан к {email}")
                    break
        return True, "user", T[lang]["auth_success"]

# --- Авторизация ---
st.sidebar.title(T[lang]["auth_title"])
password = st.sidebar.text_input(T[lang]["auth_prompt"], type="password")
email = st.sidebar.text_input(T[lang]["email_prompt"])

valid, role, message = check_key_valid(password.strip(), email.strip())

if not valid:
    st.error(message)
    st.stop()
else:
    st.success(message)  # ✅ Одно зелёное сообщение
    log_access(password.strip(), email.strip(), role)

# --- Админка ---
if role == "admin":
    st.title(T[lang]["admin_title"])
    st.subheader(T[lang]["current_keys"])
    keys_df = load_keys()
    st.dataframe(keys_df)

# --- Основное приложение ---
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

            model = LinearRegression()
            model.fit(X, y)

            st.write(T[lang]["plot"])
            fig, ax = plt.subplots()
            for city in df["city"].unique():
                city_data = df[df["city"] == city]
                ax.scatter(city_data["sqft"], city_data["price"], label=city)

            ax.set_xlabel(T[lang]["xlabel"])
            ax.set_ylabel(T[lang]["ylabel"])
            ax.legend()
            st.pyplot(fig)

            sqft_value = st.number_input(T[lang]["prediction_input"], min_value=200, max_value=5000, step=50)
            if sqft_value:
                price_pred = model.predict([[sqft_value, 3, 2]])[0]
                st.success(T[lang]["prediction_result"].format(price=int(price_pred)))

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
