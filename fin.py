import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import requests
from openai import OpenAI

# .env laden
load_dotenv()

# API-Keys laden
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY nicht gefunden!")
if not FINNHUB_API_KEY:
    st.error("FINNHUB_API_KEY nicht gefunden!")

client = OpenAI(api_key=OPENAI_API_KEY)

def get_finnhub_price(symbol, api_key):
    url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('c')  # aktueller Preis ("current")
    return None

# App-Titel
st.title("\ud83d\udcc8 Aktienanalyse f\u00fcr Stillhalter-Strategien")

# Eingabe: Aktiensymbol
symbol = st.text_input("Gib das Aktiensymbol ein (z.\u202fB. AAPL, MSFT, ALV):", "AAPL")

# Symbol automatisch f\u00fcr deutsche B\u00f6rse erg\u00e4nzen
if "." not in symbol:
    symbol += ".DE"

# Zeitraum f\u00fcr Kursanalyse
end_date = datetime.today()
start_date = end_date - timedelta(days=180)

if st.button("\ud83d\udd0d Analyse starten"):

    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)

    if df.empty:
        st.error("\u26a0\ufe0f Keine Kursdaten gefunden. Bitte Symbol pr\u00fcfen.")
    else:
        # Technische Indikatoren berechnen
        df["SMA20"] = df["Close"].rolling(window=20).mean()
        df["SMA50"] = df["Close"].rolling(window=50).mean()
        df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(window=14).mean()))

        # Aktuellen Kurs via Finnhub holen
        aktueller_kurs = get_finnhub_price(symbol, FINNHUB_API_KEY)
        if not aktueller_kurs:
            aktueller_kurs = df["Close"].iloc[-1]
            kursquelle = " (letzter Schlusskurs von yfinance)"
        else:
            kursquelle = " (live von Finnhub)"

        # W\u00e4hrung bestimmen
        waehrung = "EUR" if symbol.endswith(".DE") else "USD"

        st.write(f"**Aktueller Kurs von {symbol}: {aktueller_kurs:.2f} {waehrung}{kursquelle}**")

        # Kurs & SMAs visualisieren
        st.subheader(f"\ud83d\udcca Kursverlauf, SMA20 & SMA50 f\u00fcr {symbol}")
        fig, ax = plt.subplots()
        ax.plot(df["Close"], label="Kurs", linewidth=2)
        ax.plot(df["SMA20"], label="SMA 20", linestyle="--", color="orange")
        ax.plot(df["SMA50"], label="SMA 50", linestyle=":", color="green")
        ax.set_title(f"{symbol} – Kursverlauf")
        ax.legend()
        st.pyplot(fig)

        # GPT-Prognose vorbereiten
        st.subheader("\ud83e\uddd0 Kursprognose & Strategie")

        latest_rsi = df["RSI"].dropna().iloc[-1]
        sma20 = df["SMA20"].iloc[-1]
        trend = "seitw\u00e4rts"
        if aktueller_kurs > sma20:
            trend = "aufw\u00e4rts"
        elif aktueller_kurs < sma20:
            trend = "abw\u00e4rts"

        gpt_prompt = (
            f"Die Aktie {symbol} notiert aktuell bei {aktueller_kurs:.2f} {waehrung}. "
            f"Der RSI liegt bei {latest_rsi:.1f}, der Trend laut SMA ist {trend}. "
            f"Welche Kursentwicklung ist in den n\u00e4chsten 10–30 Tagen wahrscheinlich? "
            f"Welche Stillhalterstrategie (z.\u202fB. Covered Call, Cash Secured Put) w\u00e4re daf\u00fcr geeignet? "
            f"Nenne den aktuellen Kurs der Aktie {symbol}. "
            f"Nenne auch Strike-\u00dcberlegungen und Laufzeiten f\u00fcr eine konservative Pr\u00e4mieneinnahme. "
            f"Nenne auch drei verschiedene Strike-Preise und Laufzeiten f\u00fcr m\u00f6gliche Pr\u00e4mieneinnahmen, f\u00fcge den jeweiligen Deltawert hinzu. "
            f"Nenne auch f\u00fcr die drei verschiedenen Strike-Preise unterschiedliche Laufzeiten von 1 Woche, 2 Wochen und drei Wochen sowie die jeweiligen m\u00f6glichen Pr\u00e4mieneinnahmen, f\u00fcge den jeweiligen Deltawert hinzu."
        )

        try:
            with st.spinner("GPT analysiert die Daten..."):
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": gpt_prompt}],
                    temperature=0.3,
                )
                answer = response.choices[0].message.content
                st.success("\ud83d\udcec GPT-Antwort:")
                st.markdown(answer)
        except Exception as e:
            st.error(f"Fehler bei der GPT-Anfrage: {e}")
