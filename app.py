# app.py — Real Estate AI with License Control
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
sheet = client.open_by_key(SHEET_ID).sheet1


# --- Ensure headers exist ---
def ensure_headers():
    try:
        values = sheet.get_all_values()
        headers = ["key", "expiry", "email", "plan", "created_at", "status"]
        if not values or values[0] != headers:
            sheet.clear()
            sheet.append_row(headers)
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
        records = sheet.get_all_records()
        for row in records:
            if row["key"] == key and row["email"].lower() == email.lower():
                expiry = datetime.strptime(row["expiry"], "%Y-%m-%d")
                if expiry < datetime.now():
                    return False, None, None, "❌ License expired"
                return True, row.get("status", "user"), row.get("plan", "Basic"), "✅ License valid"
        return False, None, None, "❌ Invalid key or email"
    except Exception as e:
        return False, None, None, f"⚠️ Error checking key: {e}"


# --- Logging ---
def log_access(key: str, email: str, role: str, plan: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        sheet.append_row([key, "", email, plan, now, role])
    except:
        pass


# --- Auto-clean logs (keep only last 30 days) ---
def cleanup_logs():
    try:
        records = sheet.get_all_records()
        headers = ["key", "expiry", "email", "plan", "created_at", "status"]
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
        sheet.clear()
        for row in new_rows:
            sheet.append_row(row)
    except:
        pass


cleanup_logs()


# --- UI ---
lang = st.sidebar.selectbox("🌐 Language / Язык", ["EN", "RU"])
TXT = TEXTS[lang]

st.sidebar.title(TXT["auth_title"])
password = st.sidebar.text_input(TXT["auth_prompt"], type="password")
email = st.sidebar.text_input(TXT["email_prompt"])

valid, role, plan, message = check_key_valid(password.strip(), email.strip())

if not valid:
    st.error(message)
    st.stop()
else:
    st.success(message)
    log_access(password.strip(), email.strip(), role, plan)


# --- Main App ---
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
            X = df[["sqft", "rooms", "bathrooms"]].astype(float)
            y = df["price"].astype(float)

            model = None
            if str(plan).lower() != "pro":
                st.info("🔑 Basic plan — Linear Regression only.")
                model = LinearRegression()
                model.fit(X, y)
            else:
                st.success("🚀 Pro plan — choose model.")
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
                        st.warning("XGBoost not installed — fallback to RF.")
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                with st.spinner("🔧 Training model..."):
                    model.fit(X, y)

            preds = model.predict(X)
            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            st.write(f"**R²:** {r2:.3f}    **MAE:** {mae:,.0f} €")

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
            pred_line = model.predict(sqft_df)
            ax.plot(sqft_vals, pred_line, color="red", linewidth=2, label="Prediction")
            ax.set_xlabel(TXT["xlabel"])
            ax.set_ylabel(TXT["ylabel"])
            ax.legend()
            st.pyplot(fig)

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
                pred_price = model.predict(new_X)[0]
                st.success(TXT["prediction_result"].format(price=int(pred_price)))

            # --- Export Excel ---
            df_export = df.copy()
            df_export["predicted_price"] = preds.astype(int)
            out = BytesIO()
            df_export.to_excel(out, index=False, engine="openpyxl")
            st.download_button(TXT["download"], out.getvalue(),
                               file_name="predictions.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
