import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from io import BytesIO
from datetime import datetime

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
        "upload": "Upload CSV (columns: city, sqft, price)",
        "data_preview": "### Data preview",
        "plot": "### Price vs. Square Footage",
        "xlabel": "Square Footage (sqft)",
        "ylabel": "Price (€)",
        "prediction_input": "Enter square footage:",
        "prediction_result": "Predicted price: {price:,} €",
        "download": "📥 Download predictions as Excel",
        "csv_error": "CSV must contain columns: city, sqft, price"
    },
    "Русский": {
        "auth_title": "🔑 Авторизация",
        "auth_prompt": "Введите ключ доступа:",
        "auth_error": "⛔ Неверный ключ",
        "auth_expired": "⛔ Срок действия ключа истёк",
        "auth_success": "✅ Доступ разрешён",
        "admin_success": "✅ Доступ администратора",
        "title": "🏡 AI-Прогноз цен недвижимости",
        "upload": "Загрузите CSV (колонки: city, sqft, price)",
        "data_preview": "### Предпросмотр данных",
        "plot": "### Зависимость цены от площади",
        "xlabel": "Площадь (кв. футы)",
        "ylabel": "Цена (€)",
        "prediction_input": "Введите площадь:",
        "prediction_result": "Прогноз цены: {price:,} €",
        "download": "📥 Скачать прогнозы в Excel",
        "csv_error": "CSV должен содержать колонки: city, sqft, price"
    }
}

# --- Load keys from Google Sheets ---
SHEET_URL = st.secrets["SHEET_URL"]

try:
    keys_df = pd.read_csv(SHEET_URL)
    keys_df["expiry_date"] = pd.to_datetime(keys_df["expiry_date"], errors="coerce")
except Exception as e:
    st.error("❌ Cannot load keys from Google Sheets.")
    st.stop()

# --- Check key validity ---
def check_key_valid(user_key):
    if user_key == st.secrets["ADMIN_KEY"]:
        return True, "admin", T[lang]["admin_success"]

    row = keys_df[keys_df["key"] == user_key]
    if row.empty:
        return False, "user", T[lang]["auth_error"]

    expiry = row["expiry_date"].values[0]
    if pd.isna(expiry) or expiry >= pd.Timestamp(datetime.now()):
        return True, "user", T[lang]["auth_success"]
    else:
        return False, "user", T[lang]["auth_expired"]

# --- Authorization ---
st.sidebar.title(T[lang]["auth_title"])
password = st.sidebar.text_input(T[lang]["auth_prompt"], type="password")

valid, role, message = check_key_valid(password)

if not valid:
    st.error(message)
    st.stop()
else:
    st.success(message)

# --- Admin Panel ---
if role == "admin":
    st.sidebar.markdown("### 🛠 Admin Panel")
    st.sidebar.info("Future: view logs, manage users, etc.")

# --- Main App ---
st.title(T[lang]["title"])

uploaded_file = st.file_uploader(T[lang]["upload"], type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write(T[lang]["data_preview"])
    st.dataframe(df.head())

    if {"city", "sqft", "price"}.issubset(df.columns):
        X = df[["sqft"]]
        y = df["price"]

        model = LinearRegression()
        model.fit(X, y)

        # --- Plot ---
        st.write(T[lang]["plot"])
        fig, ax = plt.subplots()
        for city in df['city'].unique():
            city_data = df[df['city'] == city]
            ax.scatter(city_data["sqft"], city_data["price"], label=city)

        ax.plot(X, model.predict(X), color="red", linewidth=2, label="Prediction")
        ax.set_xlabel(T[lang]["xlabel"])
        ax.set_ylabel(T[lang]["ylabel"])
        ax.legend()
        st.pyplot(fig)

        # --- Prediction ---
        sqft_value = st.number_input(T[lang]["prediction_input"], min_value=200, max_value=5000, step=50)
        if sqft_value:
            price_pred = model.predict([[sqft_value]])[0]
            st.success(T[lang]["prediction_result"].format(price=int(price_pred)))

        # --- Export to Excel ---
        df["predicted_price"] = model.predict(df[["sqft"]]).astype(int)

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




