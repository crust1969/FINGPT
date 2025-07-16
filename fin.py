import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import openai
import os
from dotenv import load_dotenv  # 🆕 für .env
from datetime import datetime, timedelta

# 🔐 .env-Datei laden
load_dotenv()

# 🔐 OpenAI API Key aus .env-Datei holen
openai.api_key = os.getenv("OPENAI_API_KEY")

# 📌 App-Titel
st.title("📈 Aktienanalyse für Stillhalter-Strategien")


# 📌 App-Titel
st.title("📈 Aktienanalyse für Stillhalter-Strategien")

# 🧾 Eingabe
symbol = st.text_input("📎 Aktiensymbol eingeben (z. B. AAPL, MSFT):", "AAPL")

# 📅 Zeitraum festlegen
end_date = datetime.today()
start_date = end_date - timedelta(days=180)

if st.button("🔍 Analyse starten"):

    # 📉 Kursdaten laden
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)

    if df.empty:
        st.error("Keine Daten gefunden. Prüfe das Symbol.")
    else:
        # 🎯 Technische Indikatoren berechnen
        df["SMA20"] = df["Close"].rolling(window=20).mean()
        df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(window=14).mean()))

        # 📊 Plot
        st.subheader("📈 Kursverlauf & SMA")
        fig, ax = plt.subplots()
        ax.plot(df["Close"], label="Kurs")
        ax.plot(df["SMA20"], label="SMA 20", linestyle="--")
        ax.set_title(f"{symbol} – Kurs & SMA")
        ax.legend()
        st.pyplot(fig)

        # 🧠 GPT-Prognose generieren
        st.subheader("🤖 Kursprognose (LLM-basiert)")

        latest_price = df["Close"][-1]
        rsi = df["RSI"].dropna().iloc[-1]
        trend = "seitwärts"
        if df["SMA20"].iloc[-1] < df["Close"].iloc[-1]:
            trend = "aufwärts"
        elif df["SMA20"].iloc[-1] > df["Close"].iloc[-1]:
            trend = "abwärts"

        gpt_prompt = (
            f"Eine Aktie ({symbol}) hat derzeit einen Kurs von {latest_price:.2f} USD, "
            f"einen RSI von {rsi:.1f} und zeigt einen {trend}-Trend basierend auf dem SMA. "
            "Welche Kursentwicklung ist in den nächsten 10–30 Tagen wahrscheinlich? "
            "Und welche Stillhalterstrategie (Covered Call, CSP) ist sinnvoll?"
        )

        try:
            with st.spinner("GPT analysiert..."):
              from openai import OpenAI

client = OpenAI()  # API-Key wird automatisch aus Umgebungsvariable gelesen

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": gpt_prompt}],
    temperature=0.3,
)

prediction = response.choices[0].message.content


                st.success("📬 Prognose erhalten:")
                st.markdown(prediction)
        except Exception as e:
            st.error(f"Fehler bei OpenAI-Anfrage: {e}")
