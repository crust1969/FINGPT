import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

# 🌱 .env laden
load_dotenv()

# 🔐 API-Keys
finnhub_api_key = os.getenv("FINNHUB_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not finnhub_api_key or not openai_api_key:
    st.error("API-Keys nicht gefunden. Bitte .env prüfen.")

client = OpenAI(api_key=openai_api_key)

# 🎯 App-Titel
st.title("Aktienanalyse für Stillhalter-Strategien")

# 📎 Eingabe: Aktiensymbol
symbol = st.text_input("Gib das Aktiensymbol ein (z. B. AAPL, MSFT, ALV.DE):", "AAPL")

# 📅 Zeitraum für Kursanalyse
end_date = datetime.today()
start_date = end_date - timedelta(days=180)

# 🟢 Wenn Button geklickt → Analyse starten
if st.button("Analyse starten"):
    from_timestamp = int(start_date.timestamp())
    to_timestamp = int(end_date.timestamp())

    candle_url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol": symbol,
        "resolution": "D",
        "from": from_timestamp,
        "to": to_timestamp,
        "token": finnhub_api_key
    }
    response = requests.get(candle_url, params=params)
    data = response.json()

    if data.get("s") != "ok":
        st.error("⚠️ Keine Kursdaten gefunden. Bitte Symbol prüfen.")
    else:
        df = pd.DataFrame({
            "Date": pd.to_datetime(data["t"], unit="s"),
            "Open": data["o"],
            "High": data["h"],
            "Low": data["l"],
            "Close": data["c"],
            "Volume": data["v"]
        })
        df.set_index("Date", inplace=True)

        # 📊 Technische Indikatoren berechnen
        df["SMA20"] = df["Close"].rolling(window=20).mean()
        df["SMA50"] = df["Close"].rolling(window=50).mean()
        df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(window=14).mean()))

        # 📈 Kurs & SMAs visualisieren
        st.subheader(f"Kursverlauf für {symbol}")
        fig, ax = plt.subplots()
        ax.plot(df["Close"], label="Kurs", linewidth=2)
        ax.plot(df["SMA20"], label="SMA 20", linestyle="--", color="orange")
        ax.plot(df["SMA50"], label="SMA 50", linestyle=":", color="green")
        ax.set_title(f"{symbol} – Kursverlauf")
        ax.legend()
        st.pyplot(fig)

        # 📬 GPT-Prognose vorbereiten
        st.subheader("Kursprognose & Strategie")

        latest_price = df["Close"].iloc[-1]
        latest_rsi = df["RSI"].dropna().iloc[-1]
        sma = df["SMA20"].iloc[-1]
        trend = "seitwärts"
        if latest_price > sma:
            trend = "aufwärts"
        elif latest_price < sma:
            trend = "abwärts"

        st.markdown(f"**Aktueller Kurs von {symbol}:** {latest_price:.2f} (Datenquelle: Finnhub)")

        gpt_prompt = (
            f"Die Aktie {symbol} notiert aktuell bei {latest_price:.2f} Währungseinheiten. "
            f"Der RSI liegt bei {latest_rsi:.1f}, der Trend laut SMA ist {trend}. "
            f"Welche Kursentwicklung ist in den nächsten 10–30 Tagen wahrscheinlich? "
            f"Welche Stillhalterstrategie (z. B. Covered Call, Cash Secured Put) wäre dafür geeignet? "
            f"Nenne den aktuellen Kurs der Aktie {symbol}. "
            f"Nenne auch Strike-Überlegungen und Laufzeiten für eine konservative Prämieneinnahme. "
            f"Nenne auch drei verschiedene Strike-Preise und Laufzeiten für mögliche Prämieneinnahmen, füge den jeweiligen Deltawert hinzu. "
            f"Nenne auch für die drei verschiedenen Strike-Preise unterschiedliche Laufzeiten von 1 Woche, 2 Wochen und drei Wochen sowie die jeweiligen möglichen Prämieneinnahmen, füge den jeweiligen Deltawert hinzu."
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
