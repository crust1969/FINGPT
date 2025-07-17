import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from openai import OpenAI

# 🌱 .env laden
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY nicht gefunden!")

client = OpenAI(api_key=api_key)

# 🎯 App-Titel
st.title("📈 Aktienanalyse für Stillhalter-Strategien")

# 📎 Eingabe: Aktiensymbol
symbol = st.text_input("Gib das Aktiensymbol ein (z. B. AAPL, MSFT, ALV.DE):", "AAPL")

# 📅 Zeitraum für Kursanalyse
end_date = datetime.today()
start_date = end_date - timedelta(days=180)

# 🟢 Wenn Button geklickt → Analyse starten
if st.button("🔍 Analyse starten"):
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)

    if df.empty:
        st.error("⚠️ Keine Kursdaten gefunden. Bitte Symbol prüfen.")
    else:
        # Technische Indikatoren
        df["SMA20"] = df["Close"].rolling(window=20).mean()
        df["SMA50"] = df["Close"].rolling(window=50).mean()
        df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(window=14).mean()))

        # Aktueller Kurs aus dem letzten Close
        latest_price = df["Close"].iloc[-1]
        latest_rsi = df["RSI"].dropna().iloc[-1]
        sma20 = df["SMA20"].iloc[-1]

        trend = "seitwärts"
        if latest_price > sma20:
            trend = "aufwärts"
        elif latest_price < sma20:
            trend = "abwärts"

        # Chart
        st.subheader(f"Kursverlauf & SMA für {symbol}")
        fig, ax = plt.subplots()
        ax.plot(df["Close"], label="Kurs", linewidth=2)
        ax.plot(df["SMA20"], label="SMA 20", linestyle="--", color="orange")
        ax.plot(df["SMA50"], label="SMA 50", linestyle=":", color="green")
        ax.set_title(f"{symbol} – Kursverlauf")
        ax.legend()
        st.pyplot(fig)

        st.markdown(f"**Aktueller Kurs von {symbol}:** {latest_price:.2f} USD (Yahoo Finance)")

        # GPT Prompt
        gpt_prompt = (
            f"Die Aktie {symbol} notiert aktuell bei {latest_price:.2f} USD. "
            f"Der RSI liegt bei {latest_rsi:.1f}, der Trend laut SMA ist {trend}. "
            f"Welche Kursentwicklung ist in den nächsten 10–30 Tagen wahrscheinlich? "
            f"Welche Stillhalterstrategie (z. B. Covered Call, Cash Secured Put) wäre dafür geeignet? "
            f"Nenne auch Strike-Überlegungen und Laufzeiten für eine konservative Prämieneinnahme. "
            f"Nenne drei verschiedene Strike-Preise und Laufzeiten (1 Woche, 2 Wochen, 3 Wochen) mit möglichen Prämieneinnahmen und Deltawerten."
        )

        try:
            with st.spinner("GPT analysiert die Daten..."):
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": gpt_prompt}],
                    temperature=0.3,
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
        except Exception as e:
            st.error(f"Fehler bei der GPT-Anfrage: {e}")
