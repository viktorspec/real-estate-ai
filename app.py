# app.py — финальная версия с комментариями на русском языке
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
import requests

# ML / preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False  # если xgboost не установлен — отключаем опцию
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# ----------------- Конфигурация страницы -----------------
st.set_page_config(page_title="🏡 AI Real Estate SaaS", layout="centered")

# ----------------- Тексты на двух языках -----------------
lang = st.sidebar.selectbox("🌐 Language / Язык", ["English", "Русский"])

T = {
    "English": {
        "auth_title": "🔑 Authorization",
        "auth_prompt": "Enter your access key:",
        "email_prompt": "Enter your email:",
        "auth_error": "⛔ Invalid key",
        "auth_expired": "⛔ Key expired",
        "auth_success": "✅ Access granted",
        "admin_success": "✅ Admin access granted",
        "title": "🏡 AI Real Estate Price Predictor",
        "upload": "Upload CSV (columns: city, sqft, rooms, bathrooms, price)",
        "data_preview": "### Data preview",
        "plot": "📈 Actual vs Predicted Prices",
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
        "logs": "📜 Login Logs",
        "download_logs": "📥 Download Logs as Excel",
        "filter_email": "🔍 Filter logs by email",
        "plan_basic_info": "🔑 Your plan: Basic — Linear Regression only.",
        "xgboost_missing": "XGBoost not installed — option disabled."
    },
    "Русский": {
        "auth_title": "🔑 Авторизация",
        "auth_prompt": "Введите ключ доступа:",
        "email_prompt": "Введите ваш email:",
        "auth_error": "⛔ Неверный ключ",
        "auth_expired": "⛔ Срок действия ключа истёк",
        "auth_success": "✅ Доступ разрешён",
        "admin_success": "✅ Доступ администратора",
        "title": "🏡 AI-Прогноз цен недвижимости",
        "upload": "Загрузите CSV (колонки: city, sqft, rooms, bathrooms, price)",
        "data_preview": "### Предпросмотр данных",
        "plot": "📈 Фактическая vs Прогнозируемая цена",
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
        "logs": "📜 Логи входов",
        "download_logs": "📥 Скачать логи в Excel",
        "filter_email": "🔍 Фильтр логов по email",
        "plan_basic_info": "🔑 Ваш тариф: Basic — только Linear Regression.",
        "xgboost_missing": "XGBoost не установлен — опция недоступна."
    }
}
# Удобный доступ к текущим текстам
TXT = T[lang]

# ----------------- Google Sheets: подключение через секреты Streamlit -----------------
# Ожидается, что в Streamlit Secrets заданы:
# - SHEET_ID (id таблицы)
# - ADMIN_KEY (строка)
# - GCP_CREDENTIALS (json сервисного аккаунта)
# НЕ ВКЛАДЫВАЙ СЮДА СЕКРЕТЫ В ЯВНОМ ВИДЕ.
try:
    creds_dict = dict(st.secrets["GCP_CREDENTIALS"])
    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    )
    client = gspread.authorize(creds)
    SHEET_ID = st.secrets["SHEET_ID"]
    sheet = client.open_by_key(SHEET_ID).sheet1  # лист access_keys (ожидается)
except Exception as e:
    # Если нет секретов (локальная разработка) — отключаем интеграцию с Google Sheets,
    # но приложение всё ещё будет работать локально (без авторизации/админки).
    sheet = None
    client = None
    st.warning("⚠️ Google Sheets not configured (check Streamlit secrets). Admin / keys will be disabled in this session.")
    # можно логировать e в отладочных целях:
    # st.write(e)

# ----------------- Вспомогательные функции -----------------

def get_user_ip():
    """
    Пытаемся получить внешний IP клиента через внешний сервис.
    Внимание: в некоторых окружениях (Streamlit Cloud) запрос возвращает IP сервера,
    но чаще всего это достаточно для грубой геолокации/мониторинга.
    """
    try:
        return requests.get("https://api.ipify.org").text
    except:
        return "unknown"

def load_keys():
    """
    Загружает таблицу access_keys из первого листа Google Sheets.
    Ожидается структура:
    key | expiry_date | user
    """
    if sheet is None:
        # возвращаем пустую таблицу локально
        return pd.DataFrame(columns=["key", "expiry_date", "user"])
    records = sheet.get_all_records()
    df = pd.DataFrame(records)
    if "expiry_date" in df.columns:
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    else:
        df["expiry_date"] = pd.NaT
    if "user" not in df.columns:
        df["user"] = ""
    return df

def add_key(new_key, expiry_date=""):
    """Добавляет новый ключ в конец таблицы (key, expiry_date, user='')."""
    if sheet is None:
        st.error("Google Sheets not configured.")
        return
    sheet.append_row([new_key, expiry_date, ""])
    st.success(f"✅ Key {new_key} added!")

def delete_key(del_key):
    """Удаляет ключ по значению (ищет в колонке key)."""
    if sheet is None:
        st.error("Google Sheets not configured.")
        return
    records = sheet.get_all_records()
    for idx, row in enumerate(records, start=2):  # строка 1 — заголовки
        if row.get("key") == del_key:
            sheet.delete_rows(idx)
            st.success(f"✅ Key {del_key} deleted!")
            return
    st.error("⚠️ Key not found")

def extend_key(ext_key, new_expiry):
    """Обновляет expiry_date для заданного ключа."""
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

def bind_user_to_key(user_key, email):
    """Если ключ ещё не занят, привязывает email пользователя в колонку user."""
    if sheet is None:
        return
    records = sheet.get_all_records()
    for idx, row in enumerate(records, start=2):
        if row.get("key") == user_key:
            current_user = row.get("user", "")
            if not current_user:
                sheet.update_cell(idx, 3, email)
            return

def log_access(user_key, email, role, plan="Basic"):
    """
    Записывает лог в лист "logs": timestamp, key, email, role, plan, ip
    """
    if client is None:
        return
    sh = client.open_by_key(SHEET_ID)
    try:
        log_sheet = sh.worksheet("logs")
    except gspread.exceptions.WorksheetNotFound:
        sh.add_worksheet(title="logs", rows="1000", cols="6")
        log_sheet = sh.worksheet("logs")
        log_sheet.append_row(["timestamp", "key", "email", "role", "plan", "ip"])

    # добавляем запись
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ip = get_user_ip()
    log_sheet.append_row([timestamp, user_key, email, role, plan, ip])


    # авто-очистка старых записей (30 дней)
    try:
        logs = log_sheet.get_all_records()
        cutoff = pd.Timestamp(datetime.now()) - pd.Timedelta(days=30)
        rows_to_keep = [0]  # индекс заголовка
        all_values = log_sheet.get_all_values()
        for idx, row in enumerate(logs, start=2):
            try:
                ts = pd.to_datetime(row.get("timestamp"))
                if ts >= cutoff:
                    rows_to_keep.append(idx)
            except:
                rows_to_keep.append(idx)
        # если есть что удалить — обновляем лист
        if len(rows_to_keep) - 1 < len(logs):
            header = all_values[0]
            new_data = [header] + [all_values[i-1] for i in rows_to_keep[1:]]
            log_sheet.clear()
            log_sheet.update(new_data)
    except Exception:
        # если что-то пошло не так — не критично, продолжаем
        pass

    # добавляем текущую запись
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ip = get_user_ip()
    log_sheet.append_row([timestamp, user_key, email, role, ip])

def check_key_valid(user_key, email=""):
    """
    Проверяет валидность ключа:
    - если key == ADMIN_KEY из secrets => role admin
    - иначе ищем в таблице access_keys
      * если нет — неверный
      * если есть expiry_date и он в прошлом — истёк
      * если есть user и он не пуст — проверяем совпадение с email (чтобы ключ не "угнали")
      * иначе — валиден; при первом использовании привязываем email к ключу
    Возвращает: (valid: bool, role: "admin"|"user", message: str)
    """
    ADMIN_KEY = st.secrets.get("ADMIN_KEY") if "ADMIN_KEY" in st.secrets else None
    if ADMIN_KEY and user_key == ADMIN_KEY:
        return True, "admin", TXT["admin_success"]

    df = load_keys()
    if df.empty:
        return False, "user", TXT["auth_error"]

    row = df[df["key"] == user_key]
    if row.empty:
        return False, "user", TXT["auth_error"]

    expiry = row["expiry_date"].values[0]
    user_val = row["user"].values[0] if "user" in row.columns else ""

    # проверяем срок действия
    if pd.notna(expiry) and expiry < pd.Timestamp(datetime.now()):
        return False, "user", TXT["auth_expired"]

    # если ключ уже привязан к другому email — блокируем
    if user_val and email and (email.lower() != str(user_val).lower()):
        return False, "user", f"⚠️ This key is already used by {user_val}"

    # если ключ свободен и мы получили email — привязываем
    if (not user_val) and email:
        bind_user_to_key(user_key, email)

    return True, "user", TXT["auth_success"]

# ----------------- Интерфейс: авторизация -----------------
st.sidebar.title(TXT["auth_title"])
password = st.sidebar.text_input(TXT["auth_prompt"], type="password")
email = st.sidebar.text_input(TXT["email_prompt"])

# Проверяем ключ/аккаунт
valid, role, message = check_key_valid(password.strip(), email.strip())

if not valid:
    st.error(message)
    st.stop()
else:
    st.success(message)
    # логируем успешный вход (если настроен Google Sheets)
    valid, role, message = check_key_valid(password, email)

if not valid:
    st.error(message)
    st.stop()
else:
    st.success(message)

    # Определяем тариф
    plan = "Basic"
    if role == "admin":
        plan = "Admin"
    else:
        if email and ("pro" in email.lower() or "pro" in password.lower()):
            plan = "Pro"

    # Записываем лог
    log_access(password.strip(), email.strip(), role, plan)


# ----------------- Admin Panel (видно только админу) -----------------
if role == "admin":
    st.title(TXT["admin_title"])

    # Показать текущие ключи
    st.subheader(TXT["current_keys"])
    keys_df = load_keys()
    st.dataframe(keys_df)

    # Форма добавления ключа
    st.subheader(TXT["add_key"])
    new_key = st.text_input("Enter new key")
    expiry_date = st.date_input(TXT["expiry_optional"], value=None)
    if st.button("Add Key"):
        if new_key.strip() == "":
            st.error("⚠️ Key cannot be empty")
        else:
            add_key(new_key.strip(), str(expiry_date) if expiry_date else "")

    # Удаление ключа
    st.subheader(TXT["delete_key"])
    del_key = st.text_input(TXT["delete_prompt"])
    if st.button("Delete Key"):
        if del_key.strip() == "":
            st.error("⚠️ Please enter a key")
        else:
            delete_key(del_key.strip())

    # Продление ключа
    st.subheader(TXT["extend_key"])
    ext_key = st.text_input(TXT["extend_prompt"])
    new_expiry = st.date_input(TXT["extend_date"], value=datetime.now())
    if st.button("Extend Key"):
        if ext_key.strip() == "":
            st.error("⚠️ Please enter a key")
        else:
            extend_key(ext_key.strip(), new_expiry)

    # Просмотр логов + фильтр + скачивание
    st.subheader(TXT["logs"])
    try:
        logs = client.open_by_key(SHEET_ID).worksheet("logs").get_all_records()
        logs_df = pd.DataFrame(logs)
        email_filter = st.text_input(TXT["filter_email"])
        if email_filter:
            filtered_logs = logs_df[logs_df["email"].str.contains(email_filter, case=False, na=False)]
            st.dataframe(filtered_logs)
        else:
            st.dataframe(logs_df)

        # кнопка экспорта логов
        output = BytesIO()
        logs_df.to_excel(output, index=False, engine="openpyxl")
        st.download_button(
            label=TXT["download_logs"],
            data=output.getvalue(),
            file_name="login_logs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception:
        st.info("ℹ️ No logs yet or Google Sheets not configured.")

# ----------------- Основное приложение: загрузка данных и ML -----------------
if role in ["user", "admin"]:
    st.title(TXT["title"])
    uploaded_file = st.file_uploader(TXT["upload"], type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader(TXT["data_preview"])
        st.dataframe(df.head())

        required_cols = {"city", "sqft", "rooms", "bathrooms", "price"}
        if not required_cols.issubset(df.columns):
            st.error(TXT["csv_error"])
        else:
            # определяем права: Basic vs Pro
            # Правило: если в ключе или в email есть 'pro' (простая тестовая логика),
            # считаем пользователя Pro; иначе — Basic.
            is_pro = False
            if role == "admin":
                is_pro = True
            else:
                # безопасная проверка (если email пуст — Basic)
                if email and ("pro" in email.lower() or "pro" in password.lower()):
                    is_pro = True

            # если xgboost не установлен — скрываем опцию
            model_options = ["Linear Regression", "RandomForest"]
            if XGBOOST_AVAILABLE:
                model_options.append("XGBoost")
            else:
                # сообщаем админу/пользователю, что xgboost отсутствует
                st.info(TXT["xgboost_missing"])

            # если Basic — ограничиваем выбор только линейной моделью
            if not is_pro:
                st.info(TXT["plan_basic_info"])
                model_choice = "Linear Regression"
            else:
                model_choice = st.selectbox("Choose ML Model", model_options)

            # Подготовка признаков
            X = df[["city", "sqft", "rooms", "bathrooms"]]
            y = df["price"]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("city", OneHotEncoder(handle_unknown="ignore"), ["city"]),
                    ("num", "passthrough", ["sqft", "rooms", "bathrooms"])
                ]
            )

            # Выбор модели
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "RandomForest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = XGBRegressor(n_estimators=100, random_state=42)

            # Сборка пайплайна и обучение
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            with st.spinner("🔧 Training model..."):
                pipeline.fit(X, y)
            preds = pipeline.predict(X)

            # Метрики качества модели
            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            st.write(f"**R² Score:** {r2:.3f}")
            st.write(f"**MAE:** {mae:.2f} €")

            # График: фактические vs предсказанные
            st.subheader(TXT["plot"])
            fig, ax = plt.subplots()
            ax.scatter(y, preds, alpha=0.7, label="Predictions")
            ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="Perfect Fit")
            ax.set_xlabel("Actual Price (€)")
            ax.set_ylabel("Predicted Price (€)")
            ax.legend()
            st.pyplot(fig)

            # Форма для предсказания по новой квартире
            st.subheader("🔮 Predict New Property")
            city_input = st.text_input("City", "Madrid")
            sqft_input = st.number_input(TXT["prediction_input"], min_value=20, max_value=2000, value=70, step=5)
            rooms_input = st.number_input("Rooms", min_value=1, max_value=10, value=2, step=1)
            bathrooms_input = st.number_input("Bathrooms", min_value=1, max_value=5, value=1, step=1)

            if st.button("Predict Price"):
                new_data = pd.DataFrame([[city_input, sqft_input, rooms_input, bathrooms_input]],
                                        columns=["city", "sqft", "rooms", "bathrooms"])
                price_pred = pipeline.predict(new_data)[0]
                st.success(TXT["prediction_result"].format(price=int(price_pred)))

            # Экспорт предсказаний для загруженного CSV
            st.subheader(TXT["download"])
            df["predicted_price"] = preds.astype(int)
            output = BytesIO()
            df.to_excel(output, index=False, engine="openpyxl")
            st.download_button(
                label=TXT["download"],
                data=output.getvalue(),
                file_name="real_estate_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

