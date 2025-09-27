# app.py — Real Estate AI with License Control (optimized with cache + session_state)
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

# --- Ensure headers ---
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

# --- Language dictionaries ---
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
    }
}

# --- License check ---
def check_key_valid(key: str, email: str):
    try:
        records = licenses_sheet.get_all_records()
        for row in records:
            if row["key"] == key and row["email"].lower() == email.lower():
                expiry = datetime.strptime(row["expiry"], "%Y-%m-%d")
                if expiry < datetime.now():
                    return False, None, None, None, "❌ License expired"
                return True, row.get("status", "user"), row.get("plan", "Basic"), row.get("expiry"), "✅ License valid"
        return False, None, None, None, ""
    except Exception as e:
        return False, None, None, None, f"⚠️ Error checking key: {e}"

# --- Logging ---
def log_access(key: str, email: str, role: str, plan: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        logs_sheet.append_row([key, email, plan, role, now])
    except:
        pass

# --- Auto-clean logs ---
def cleanup_logs():
    try:
        records = logs_sheet.get_all_records()
        headers = ["key", "email", "plan", "role", "created_at"]
        new_rows = [headers]
        for row in records:
            created_at = row.get("created_at")
            if created_at:
                try:
                    dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
                    if dt >= datetime.now() - timedelta(days=30):
                        new_rows.append(list(row.values()))
                except:
                    new_rows.append(list(row.values()))
        logs_sheet.clear()
        for row in new_rows:
            logs_sheet.append_row(row)
    except:
        pass

cleanup_logs()

# --- Cache загрузки CSV ---
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

# --- Cache обучения модели ---
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

# --- Session state ---
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "preds" not in st.session_state:
    st.session_state.preds = None

# --- UI ---
lang = st.sidebar.selectbox("🌐 Language / Язык", ["EN", "RU"])
TXT = TEXTS[lang]

st.sidebar.title(TXT["auth_title"])
password = st.sidebar.text_input(TXT["auth_prompt"], type="password")
email = st.sidebar.text_input(TXT["email_prompt"])

valid, role, plan, expiry, message = check_key_valid(password.strip(), email.strip())

if password and email:
    if not valid:
        st.error(message)
        st.stop()
    else:
        st.success(message)
        log_access(password.strip(), email.strip(), role, plan)
        st.sidebar.markdown(
            f"""
            <div style='padding:15px; border-radius:10px; background-color:#1E3A8A; color:white;'>
                <h4 style='margin:0;'>📌 Plan: {plan}</h4>
                <p style='margin:0;'>⏳ Valid until: {expiry}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.info("👉 Please enter license key and email to continue")
    st.stop()

# --- Main App ---
if role in ["user", "admin"]:
    st.title(TXT["title"])

    uploaded_file = st.file_uploader(TXT["upload"], type=["csv"])
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        st.session_state.df = df
        st.subheader(TXT["data_preview"])
        st.dataframe(df.head())

        required = {"city", "sqft", "rooms", "bathrooms", "price"}
        if not required.issubset(df.columns):
            st.error(TXT["csv_error"])
        else:
            X = df[["sqft", "rooms", "bathrooms"]].astype(float)
            y = df["price"].astype(float)

            model_type = "linear"
            if str(plan).lower() == "pro":
                st.success("🚀 Pro plan — choose model.")
                options = ["Linear Regression", "Random Forest"]
                if XGBOOST_AVAILABLE:
                    options.append("XGBoost")
                model_choice = st.selectbox("Select model:", options)
                if model_choice == "Linear Regression":
                    model_type = "linear"
                elif model_choice == "Random Forest":
                    model_type = "rf"
                elif model_choice == "XGBoost":
                    model_type = "xgb"
            else:
                st.info("🔑 Basic plan — Linear Regression only.")

            st.session_state.model, st.session_state.preds = train_model(X, y, model_type=model_type)
            preds = st.session_state.preds

            # --- Метрики ---
            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            avg_price = y.mean()
            mae_percent = (mae / avg_price) * 100

            st.write(f"**R²:** {r2:.3f}    **MAE:** {mae:,.0f} € (~{mae_percent:.2f}% от средней цены)")
            st.caption("ℹ️ R² показывает, насколько хорошо модель объясняет данные (1.0 = идеально). "
                       "MAE показывает, насколько в среднем прогноз отличается от реальной цены.")
            avg_rent = 500
            rent_months = mae / avg_rent
            st.caption(f"📊 Это примерно {rent_months:.1f} месяцев аренды при средней ставке {avg_rent} €/мес.")

            if mae_percent < 2:
                st.success("📌 Прогноз очень точный: средняя ошибка меньше 2% от рыночной стоимости.")
            elif mae_percent < 5:
                st.info("📌 Прогноз надёжный: ошибка в пределах 5% от рыночной стоимости.")
            else:
                st.warning("📌 Ошибка прогноза выше 5%. Рекомендуем добавить больше данных для повышения точности.")

            # --- Plot ---
            st.subheader(TXT["plot"])
            fig, ax = plt.subplots(figsize=(8, 5))
            for city in df["city"].unique():
                cd = df[df["city"] == city]
                ax.scatter(cd["sqft"], cd["price"], label=city, alpha=0.7)

            sqft_vals = np.linspace(df["sqft"].min(), df["sqft"].max(), 200)
            sqft_df = pd.DataFrame({
                "sqft": sqft_vals,
                "rooms": np.full_like(sqft_vals, 3),
                "bathrooms": np.full_like(sqft_vals, 2)
            })
            pred_line = st.session_state.model.predict(sqft_df)
            ax.plot(sqft_vals, pred_line, color="red", linewidth=2, label="Prediction")
            ax.set_xlabel(TXT["xlabel"])
            ax.set_ylabel(TXT["ylabel"])
            ax.legend()
            st.pyplot(fig)

            # --- Download Plot ---
            png_buffer = BytesIO()
            fig.savefig(png_buffer, format="png", bbox_inches="tight")
            png_buffer.seek(0)
            st.download_button(TXT["download_png"], data=png_buffer.getvalue(),
                               file_name="price_vs_sqft.png", mime="image/png")

            # --- Predict new ---
            st.subheader("🔮 Predict New Property")
            sqft_input = st.number_input(TXT["prediction_input"], min_value=1, max_value=10000,
                                         value=int(np.median(df["sqft"])), step=1)
            rooms_input = st.number_input("Rooms", min_value=1, max_value=10, value=3, step=1)
            baths_input = st.number_input("Bathrooms", min_value=1, max_value=5, value=2, step=1)
            if st.button("Predict Price"):
                new_X = np.array([[sqft_input, rooms_input, baths_input]])
                pred_price = st.session_state.model.predict(new_X)[0]
                st.success(TXT["prediction_result"].format(price=int(pred_price)))

            # --- Export Excel ---
            df_export = df.copy()
            df_export["predicted_price"] = preds.astype(int)
            out = BytesIO()
            df_export.to_excel(out, index=False, engine="openpyxl")
            st.download_button(TXT["download"], out.getvalue(),
                               file_name="predictions.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- FAQ --- 
FAQS = {
    "EN": [
        ("How to upload data?", "Upload a CSV file with columns: city, sqft, rooms, bathrooms, price."),
        ("What does R² mean?", "R² shows how well the model explains the data. 1.0 = perfect."),
        ("What is MAE?", "MAE = Mean Absolute Error. It shows the average difference between prediction and real price."),
        ("Why do I need a license?", "License gives you access to Basic or Pro features."),
    ],
    "RU": [
        ("Как загрузить данные?", "Загрузите CSV файл со столбцами: city, sqft, rooms, bathrooms, price."),
        ("Что значит R²?", "R² показывает, насколько хорошо модель объясняет данные. 1.0 = идеально."),
        ("Что такое MAE?", "MAE — это средняя абсолютная ошибка. Показывает, насколько в среднем прогноз отличается от реальной цены."),
        ("Зачем нужен ключ лицензии?", "Ключ открывает доступ к возможностям Basic или Pro."),
    ]
}

st.subheader("❓ FAQ")
for question, answer in FAQS[lang]:
    with st.expander(question):
        st.write(answer)
        
st.markdown("---")
if lang == "EN":
    st.info("📧 Need help? Contact support: viktormatrix37@gmail.com")
else:
    st.info("📧 Нужна помощь? Свяжитесь с поддержкой: viktormatrix37@gmail.com")



