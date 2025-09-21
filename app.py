# app.py — Полная версия: планы, админка, логи, модели, корректный график и экспорт
# Комментарии на русском языке — читай и учись :)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime, timedelta
import requests

# ML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# XGBoost опционально
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# Google Sheets (опционально — приложение будет работать локально без них)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except Exception:
    GSPREAD_AVAILABLE = False

st.set_page_config(page_title="🏡 AI Real Estate SaaS", layout="centered")

# ----------------- Тексты интерфейса -----------------
lang = st.sidebar.selectbox("🌐 Language / Язык", ["English", "Русский"])

T = {
    "English": {
        "auth_title": "🔑 Authorization",
        "auth_prompt": "Enter your access key:",
        "email_prompt": "Enter your email:",
        "auth_error": "⛔ Invalid key",
        "auth_expired": "⛔ Key expired",
        "auth_trial_expired": "⛔ Trial expired (7 days limit)",
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
        "download_png": "📥 Download plot as PNG",
        "csv_error": "CSV must contain columns: city, sqft, rooms, bathrooms, price",
        "admin_title": "👑 Admin: Manage Users / Keys",
        "current_keys": "📋 Current Keys (key | expiry_date | user | plan)",
        "add_key": "➕ Add New Key",
        "delete_key": "🗑 Delete Key",
        "expiry_optional": "Expiry date (optional)",
        "delete_prompt": "Enter key to delete",
        "extend_key": "⏳ Extend Key",
        "extend_prompt": "Enter key to extend",
        "extend_date": "New expiry date",
        "change_plan": "🔁 Change plan for key",
        "unbind_user": "🔓 Unbind user (clear email)",
        "logs": "📜 Login Logs",
        "download_logs": "📥 Download Logs as Excel",
        "filter_email": "🔍 Filter logs by email"
    },
    "Русский": {
        "auth_title": "🔑 Авторизация",
        "auth_prompt": "Введите ключ доступа:",
        "email_prompt": "Введите ваш email:",
        "auth_error": "⛔ Неверный ключ",
        "auth_expired": "⛔ Срок действия ключа истёк",
        "auth_trial_expired": "⛔ Trial истёк (ограничение 7 дней)",
        "auth_success": "✅ Доступ разрешён (Тариф: {plan})",
        "admin_success": "✅ Доступ администратора",
        "title": "🏡 AI-Прогноз цен недвижимости",
        "upload": "Загрузите CSV (колонки: city, sqft, rooms, bathrooms, price)",
        "data_preview": "### Предпросмотр данных",
        "plot": "### Зависимость цены от площади",
        "xlabel": "Площадь (sqft)",
        "ylabel": "Цена (€)",
        "prediction_input": "Введите площадь:",
        "prediction_result": "Прогноз цены: {price:,} €",
        "download": "📥 Скачать прогнозы в Excel",
        "download_png": "📥 Скачать график в PNG",
        "csv_error": "CSV должен содержать колонки: city, sqft, rooms, bathrooms, price",
        "admin_title": "👑 Админ: Управление пользователями / ключами",
        "current_keys": "📋 Текущие ключи (key | expiry_date | user | plan)",
        "add_key": "➕ Добавить ключ",
        "delete_key": "🗑 Удалить ключ",
        "expiry_optional": "Дата окончания (необязательно)",
        "delete_prompt": "Введите ключ для удаления",
        "extend_key": "⏳ Продлить ключ",
        "extend_prompt": "Введите ключ для продления",
        "extend_date": "Новая дата окончания",
        "change_plan": "🔁 Сменить тариф у ключа",
        "unbind_user": "🔓 Отвязать пользователя (очистить email)",
        "logs": "📜 Логи входов",
        "download_logs": "📥 Скачать логи в Excel",
        "filter_email": "🔍 Фильтр логов по email"
    }
}
TXT = T[lang]

# ----------------- Google Sheets: безопасное подключение -----------------
client = None
sheet = None
SHEET_ID = None
if GSPREAD_AVAILABLE:
    try:
        # Ожидается, что в Streamlit secrets есть GCP_CREDENTIALS и SHEET_ID
        creds_dict = dict(st.secrets["GCP_CREDENTIALS"])
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        )
        client = gspread.authorize(creds)
        SHEET_ID = st.secrets.get("SHEET_ID", None)
        if SHEET_ID:
            sheet = client.open_by_key(SHEET_ID).sheet1
    except Exception as e:
        # не ломаем приложение, просто выключаем интеграцию
        st.warning("⚠️ Google Sheets not configured or credentials invalid — admin/keys/logs disabled in this session.")
        client = None
        sheet = None

# ----------------- Вспомогательные функции для работы с Google Sheets -----------------
def load_keys():
    """Загрузить таблицу ключей (key, expiry_date, user, plan). Если нет sheet — вернуть пустой df."""
    if sheet is None:
        return pd.DataFrame(columns=["key", "expiry_date", "user", "plan"])
    records = sheet.get_all_records()
    df = pd.DataFrame(records)
    if "expiry_date" in df.columns:
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    else:
        df["expiry_date"] = pd.NaT
    if "user" not in df.columns:
        df["user"] = ""
    if "plan" not in df.columns:
        df["plan"] = "Basic"
    return df

def add_key(new_key, expiry_date="", plan="Basic"):
    """Добавить новый ключ в таблицу (key, expiry_date, user='', plan)."""
    if sheet is None:
        st.error("Google Sheets not configured.")
        return
    sheet.append_row([new_key, expiry_date, "", plan])
    st.success(f"✅ Key {new_key} ({plan}) added!")

def delete_key(del_key):
    if sheet is None:
        st.error("Google Sheets not configured.")
        return
    records = sheet.get_all_records()
    for idx, row in enumerate(records, start=2):
        if row.get("key") == del_key:
            sheet.delete_rows(idx)
            st.success(f"✅ Key {del_key} deleted!")
            return
    st.error("⚠️ Key not found")

def extend_key(ext_key, new_expiry):
    if sheet is None:
        st.error("Google Sheets not configured.")
        return
    records = sheet.get_all_records()
    for idx, row in enumerate(records, start=2):
        if row.get("key") == ext_key:
            sheet.update_cell(idx, 2, str(new_expiry))
            st.success(f"✅ Key {ext_key} extended until {new_expiry}")
            return
    st.error("⚠️ Key not found")

def update_plan(key_val, new_plan):
    """Обновить поле plan для ключа."""
    if sheet is None:
        st.error("Google Sheets not configured.")
        return
    records = sheet.get_all_records()
    for idx, row in enumerate(records, start=2):
        if row.get("key") == key_val:
            # столбец plan — 4-я колонка (key, expiry_date, user, plan)
            sheet.update_cell(idx, 4, new_plan)
            st.success(f"✅ Key {key_val} plan set to {new_plan}")
            return
    st.error("⚠️ Key not found")

def unbind_user(key_val):
    """Очистить колонку user для ключа."""
    if sheet is None:
        st.error("Google Sheets not configured.")
        return
    records = sheet.get_all_records()
    for idx, row in enumerate(records, start=2):
        if row.get("key") == key_val:
            sheet.update_cell(idx, 3, "")
            st.success(f"✅ User unbound from {key_val}")
            return
    st.error("⚠️ Key not found")

def bind_user_to_key(user_key, email):
    """Если ключ свободен — привязать email."""
    if sheet is None:
        return
    records = sheet.get_all_records()
    for idx, r in enumerate(records, start=2):
        if r.get("key") == user_key:
            current_user = r.get("user", "")
            if not current_user and email:
                sheet.update_cell(idx, 3, email)
                st.info(f"🔗 Key {user_key} linked to {email}")
            return

def ensure_logs_sheet():
    """Убедиться, что лист logs существует и имеет корректные заголовки."""
    if client is None or SHEET_ID is None:
        return None
    sh = client.open_by_key(SHEET_ID)
    try:
        log_sheet = sh.worksheet("logs")
    except gspread.exceptions.WorksheetNotFound:
        log_sheet = sh.add_worksheet(title="logs", rows="1000", cols="6")
        log_sheet.append_row(["timestamp", "key", "email", "role", "plan", "ip"])
    # если заголовки не те — исправим
    headers = log_sheet.row_values(1)
    expected = ["timestamp", "key", "email", "role", "plan", "ip"]
    if headers != expected:
        log_sheet.clear()
        log_sheet.append_row(expected)
    return log_sheet

def get_user_ip():
    """Попытка получить внешний IP (может вернуть IP сервера)."""
    try:
        return requests.get("https://api.ipify.org", timeout=3).text
    except Exception:
        return "unknown"

def log_access(user_key, email, role, plan="Basic"):
    """Логируем вход: timestamp, key, email, role, plan, ip + автоочистка старше 30 дней."""
    if client is None or SHEET_ID is None:
        # не критично — пропускаем логирование
        return
    try:
        log_sheet = ensure_logs_sheet()
        if log_sheet is None:
            return
        logs = log_sheet.get_all_records()
        cutoff = datetime.now() - timedelta(days=30)
        new_rows = [log_sheet.row_values(1)]  # header
        # фильтруем старые
        for r in logs:
            ts_str = r.get("timestamp", "")
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                if ts >= cutoff:
                    new_rows.append([r.get(c) for c in ["timestamp", "key", "email", "role", "plan", "ip"]])
            except Exception:
                # если парсить не удалось — сохраняем (без риска потерять)
                new_rows.append([r.get(c) for c in ["timestamp", "key", "email", "role", "plan", "ip"]])
        # если изменилось количество строк — обновим лист (убираем старые)
        if len(new_rows) != len(logs) + 1:
            log_sheet.clear()
            log_sheet.update(new_rows)
        # добавляем текущую запись
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ip = get_user_ip()
        log_sheet.append_row([timestamp, user_key, email, role, plan, ip])
    except Exception as e:
        st.warning(f"⚠️ Logging error: {e}")

# ----------------- Валидация ключа -----------------
def check_key_valid(user_key, email=""):
    """
    Возвращает: (valid:bool, role:str, plan:str, message:str)
    role: "admin" или "user"
    plan: Basic/Pro/Trial (если нет — Basic)
    """
    # ADMIN_KEY в st.secrets (опционально)
    ADMIN_KEY = st.secrets.get("ADMIN_KEY") if "ADMIN_KEY" in st.secrets else None
    if ADMIN_KEY and user_key == ADMIN_KEY:
        return True, "admin", "Admin", TXT["admin_success"]

    df = load_keys()
    if df.empty:
        return False, "user", "Basic", TXT["auth_error"]

    row = df[df["key"] == user_key]
    if row.empty:
        return False, "user", "Basic", TXT["auth_error"]

    expiry = row.iloc[0].get("expiry_date", pd.NaT)
    user_val = str(row.iloc[0].get("user", "")).strip()
    plan = row.iloc[0].get("plan", "Basic") or "Basic"

    # проверка expiry_date
    if pd.notna(expiry):
        try:
            if pd.Timestamp(expiry) < pd.Timestamp(datetime.now()):
                return False, "user", plan, TXT["auth_expired"]
        except Exception:
            # если не парсится — игнорируем
            pass

    # авто-блокировка Trial: ожидаем expiry_date (должна быть дата окончания триала)
    if str(plan).lower() == "trial":
        if pd.isna(expiry):
            return False, "user", plan, "⚠️ Trial must have expiry_date"
        if pd.Timestamp(datetime.now()) > pd.Timestamp(expiry):
            return False, "user", plan, TXT["auth_trial_expired"]

    # если ключ уже привязан к другому email
    if user_val:
        if email and email.strip().lower() != user_val.lower():
            return False, "user", plan, f"⚠️ This key is already used by {user_val}"
        else:
            return True, "user", plan, TXT["auth_success"].format(plan=plan)
    else:
        # если ключ свободен и пришёл email — привяжем
        if email:
            bind_user_to_key(user_key, email.strip())
        return True, "user", plan, TXT["auth_success"].format(plan=plan)

# ----------------- UI: Авторизация -----------------
st.sidebar.title(TXT["auth_title"])
password = st.sidebar.text_input(TXT["auth_prompt"], type="password")
email = st.sidebar.text_input(TXT["email_prompt"])

valid, role, plan, message = check_key_valid(password.strip(), email.strip())

if not valid:
    st.error(message)
    st.stop()
else:
    # Показываем единое подтверждение
    st.success(message)
    # Логируем (тихо, даже если Google Sheets не настроен — пропускаем)
    log_access(password.strip(), email.strip(), role, plan)

# ----------------- Admin panel -----------------
if role == "admin":
    st.title(TXT["admin_title"])

    # Показать таблицу ключей
    st.subheader(TXT["current_keys"])
    keys_df = load_keys()
    st.dataframe(keys_df)

    # Добавление ключа
    st.subheader(TXT["add_key"])
    new_key = st.text_input("Enter new key")
    expiry_date = st.date_input(TXT["expiry_optional"], value=None)
    plan_choice = st.selectbox("Select plan", ["Basic", "Pro", "Trial"])
    if st.button("Add Key"):
        add_key(new_key.strip(), str(expiry_date) if expiry_date else "", plan_choice)

    # Удаление ключа
    st.subheader(TXT["delete_key"])
    del_key = st.text_input(TXT["delete_prompt"])
    if st.button("Delete Key"):
        delete_key(del_key.strip())

    # Продление ключа
    st.subheader(TXT["extend_key"])
    ext_key = st.text_input(TXT["extend_prompt"])
    new_expiry = st.date_input(TXT["extend_date"], value=datetime.now())
    if st.button("Extend Key"):
        extend_key(ext_key.strip(), new_expiry)

    # Смена плана
    st.subheader(TXT["change_plan"])
    key_for_plan = st.text_input("Key to change plan")
    new_plan_val = st.selectbox("New plan", ["Basic", "Pro", "Trial"])
    if st.button("Update Plan"):
        update_plan(key_for_plan.strip(), new_plan_val)

    # Отвязать пользователя
    st.subheader(TXT["unbind_user"])
    key_unbind = st.text_input("Key to unbind user")
    if st.button("Unbind User"):
        unbind_user(key_unbind.strip())

    # Просмотр логов
    st.subheader(TXT["logs"])
    try:
        if client and SHEET_ID:
            log_sheet = ensure_logs_sheet()
            logs = log_sheet.get_all_records()
            logs_df = pd.DataFrame(logs)
            email_filter = st.text_input(TXT["filter_email"])
            if email_filter:
                logs_df = logs_df[logs_df["email"].str.contains(email_filter, case=False, na=False)]
            st.dataframe(logs_df)
            # Скачать логи
            out = BytesIO()
            logs_df.to_excel(out, index=False, engine="openpyxl")
            st.download_button(TXT["download_logs"], out.getvalue(), file_name="login_logs.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("ℹ️ Logs are disabled (Google Sheets not configured).")
    except Exception as e:
        st.warning(f"⚠️ Cannot load logs: {e}")

# ----------------- Main application (user / admin) -----------------
if role in ["user", "admin"]:
    st.title(TXT["title"])

    uploaded_file = st.file_uploader(TXT["upload"], type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader(TXT["data_preview"])
        st.dataframe(df.head())

        required = {"city", "sqft", "rooms", "bathrooms", "price"}
        if not required.issubset(df.columns):
            st.error(TXT["csv_error"])
        else:
            # Формируем X и y (модель учится на sqft, rooms, bathrooms)
            X = df[["sqft", "rooms", "bathrooms"]].astype(float)
            y = df["price"].astype(float)

            # Выбор модели по плану
            model = None
            if str(plan).lower() != "pro":
                st.info("🔑 Your plan: Basic/Trial — using Linear Regression.")
                model = LinearRegression()
                model.fit(X, y)
            else:
                st.success("🚀 Your plan: Pro — choose model.")
                options = ["Linear Regression", "Random Forest"]
                if XGBOOST_AVAILABLE:
                    options.append("XGBoost")
                model_choice = st.selectbox("Select model:", options)
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                elif model_choice == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_choice == "XGBoost":
                    if XGBOOST_AVAILABLE:
                        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                    else:
                        st.warning("XGBoost not installed — fallback to RandomForest.")
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                # Обучаем выбранную модель
                with st.spinner("🔧 Training model..."):
                    model.fit(X, y)

            # Предсказания на тренировочных данных
            preds = model.predict(X)
            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            st.write(f"**R²:** {r2:.3f}    **MAE:** {mae:,.0f} €")

            # График: scatter + линия предсказания по sqft (при фиксированных rooms=3, bathrooms=2)
            st.subheader(TXT["plot"])
            fig, ax = plt.subplots(figsize=(8, 5))
            for city in df["city"].unique():
                cd = df[df["city"] == city]
                ax.scatter(cd["sqft"], cd["price"], label=city, alpha=0.7)

            # Диапазон площадей для линии прогноза
            min_sqft = int(df["sqft"].min())
            max_sqft = int(df["sqft"].max())
            sqft_vals = np.linspace(min_sqft, max_sqft, 200)
            sqft_df = pd.DataFrame({
                "sqft": sqft_vals,
                "rooms": np.full_like(sqft_vals, 3),
                "bathrooms": np.full_like(sqft_vals, 2)
            })
            # predict принимает 2D array / DataFrame
            pred_line = model.predict(sqft_df)
            ax.plot(sqft_vals, pred_line, color="red", linewidth=2, label="Prediction")
            ax.set_xlabel(TXT["xlabel"])
            ax.set_ylabel(TXT["ylabel"])
            ax.legend()
            st.pyplot(fig)

            # Скачать график PNG
            png_buffer = BytesIO()
            fig.savefig(png_buffer, format="png", bbox_inches="tight")
            png_buffer.seek(0)
            st.download_button(TXT["download_png"], data=png_buffer.getvalue(),
                               file_name="price_vs_sqft.png", mime="image/png")

            # Прогноз по пользовательской площади
            st.subheader("🔮 Predict New Property")
            sqft_input = st.number_input(TXT["prediction_input"], min_value=1, max_value=10000, value=int(np.median(df["sqft"])), step=1)
            rooms_input = st.number_input("Rooms", min_value=1, max_value=10, value=3, step=1)
            baths_input = st.number_input("Bathrooms", min_value=1, max_value=5, value=2, step=1)
            if st.button("Predict Price"):
                new_X = np.array([[sqft_input, rooms_input, baths_input]])
                pred_price = model.predict(new_X)[0]
                st.success(TXT["prediction_result"].format(price=int(pred_price)))

            # Экспорт Excel с предсказаниями
            df_export = df.copy()
            df_export["predicted_price"] = preds.astype(int)
            out = BytesIO()
            df_export.to_excel(out, index=False, engine="openpyxl")
            st.download_button(TXT["download"], out.getvalue(), file_name="predictions.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

