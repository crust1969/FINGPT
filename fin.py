import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# 🌱 .env laden

load_dotenv()  # .env wird automatisch geladen

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY nicht gefunden!")

#load_dotenv()
#client = OpenAI()  # API-Key wird automatisch aus .env gelesen

# 🎯 App-Titel
st.title("📈 Aktienanalyse für Stillhalter-Strategien")

# 📎 Eingabe: Aktiensymbol
symbol = st.text_input("Gib das Aktiensymbol ein (z. B. AAPL, MSFT):", "AAPL")

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
        # 📊 Technische Indikatoren berechnen
        df["SMA20"] = df["Close"].rolling(window=20).mean()
        df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(window=14).mean()))

        # 📈 Kurs & SMA visualisieren
        st.subheader(f"📊 Kursverlauf & SMA20 für {symbol}")
        fig, ax = plt.subplots()
        ax.plot(df["Close"], label="Kurs", linewidth=2)
        ax.plot(df["SMA20"], label="SMA 20", linestyle="--", color="orange")
        ax.set_title(f"{symbol} – Kursverlauf")
        ax.legend()
        st.pyplot(fig)

        # 📬 GPT-Prognose vorbereiten
        st.subheader("🤖 Kursprognose & Strategie")

        latest_price = df["Close"].iloc[-1]
        latest_rsi = df["RSI"].dropna().iloc[-1]
        sma = df["SMA20"].iloc[-1]
        trend = "seitwärts"
        if latest_price > sma:
            trend = "aufwärts"
        elif latest_price < sma:
            trend = "abwärts"

        gpt_prompt = (
            f"Die Aktie {symbol} notiert aktuell bei {latest_price:.2f} USD. "
            f"Der RSI liegt bei {latest_rsi:.1f}, der Trend laut SMA ist {trend}. "
            f"Welche Kursentwicklung ist in den nächsten 10–30 Tagen wahrscheinlich? "
            f"Welche Stillhalterstrategie (z. B. Covered Call, Cash Secured Put) wäre dafür geeignet? "
            f"Nenne den aktuellen Kurs der Aktie {symbol}."
            f"Nenne auch Strike-Überlegungen und Laufzeiten für eine konservative Prämieneinnahme."
            f"Nenne auch drei verschiedene Strike-Preise und Laufzeiten für mögliche Prämieneinnahmen, füge den jeweiligen Deltawert hinzu."
            f"Nenne auch für die drei verschiedenen Strike-Preise unterschiedliche Laufzeiten von 1 Woche, 2 Wochen und drei wochen sowie die jeweiligen mögliche Prämieneinnahmen, füge den jeweiligen Deltawert hinzu."
            
        )

        # 🧠 GPT abfragen
        try:
            with st.spinner("GPT analysiert die Daten..."):
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": gpt_prompt}],
                    temperature=0.3,
                )
                answer = response.choices[0].message.content
                st.success("📬 GPT-Antwort:")
                st.markdown(answer)
        except Exception as e:
            st.error(f"Fehler bei der GPT-Anfrage: {e}")
