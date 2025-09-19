import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from io import BytesIO

st.set_page_config(page_title="🏡 AI Real Estate Predictor", layout="centered")

# --- Language selector ---
lang = st.sidebar.selectbox("🌐 Language / Язык", ["English", "Русский"])

# --- Dictionary for translations ---
T = {
    "English": {
        "auth_title": "🔑 Authorization",
        "auth_prompt": "Enter your access key:",
        "auth_error": "⛔ Access denied. Please enter a valid key.",
        "auth_success": "✅ Access granted! Welcome.",
        "title": "🏡 AI-Powered Real Estate Price Predictor",
        "upload": "Upload CSV (columns required: city, sqft, price)",
        "data_preview": "### Data preview",
        "plot": "### Price vs. Square Footage",
        "xlabel": "Square Footage (sqft)",
        "ylabel": "Price (€)",
        "prediction_input": "Enter square footage:",
        "prediction_result": "Predicted price: {price:,} €",
        "download": "📥 Download predictions as Excel",
        "csv_error": "CSV must contain the following columns: city, sqft, price"
    },
    "Русский": {
        "auth_title": "🔑 Авторизация",
        "auth_prompt": "Введите ключ доступа:",
        "auth_error": "⛔ Доступ запрещён. Введите правильный ключ.",
        "auth_success": "✅ Доступ разрешён! Добро пожаловать.",
        "title": "🏡 AI-Прогноз цен недвижимости",
        "upload": "Загрузите CSV (колонки: city, sqft, price)",
        "data_preview": "### Данные (первые строки)",
        "plot": "### Зависимость цены от площади",
        "xlabel": "Площадь (кв. футы)",
        "ylabel": "Цена (€)",
        "prediction_input": "Введите площадь:",
        "prediction_result": "Прогноз цены: {price:,} €",
        "download": "📥 Скачать прогнозы в Excel",
        "csv_error": "CSV должен содержать колонки: city, sqft, price"
    }
}

# --- Google Sheets Access Keys ---
SHEET_URL = st.secrets["SHEET_URL"]

# ⚠️ Replace ID with your Google Sheets ID

try:
    keys_df = pd.read_csv(SHEET_URL)
    VALID_KEYS = set(keys_df["key"].astype(str).tolist())
except Exception as e:
    st.error("Error: failed to load keys from Google Sheets")
    st.stop()

# --- Authorization ---
st.sidebar.title(T[lang]["auth_title"])
password = st.sidebar.text_input(T[lang]["auth_prompt"], type="password")

if password not in VALID_KEYS:
    st.error(T[lang]["auth_error"])
    st.stop()

# --- After login ---
st.success(T[lang]["auth_success"])
st.title(T[lang]["title"])

# --- File upload ---
uploaded_file = st.file_uploader(T[lang]["upload"], type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write(T[lang]["data_preview"])
    st.dataframe(df.head())

    if {"city", "sqft", "price"}.issubset(df.columns):
        # Train model
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


