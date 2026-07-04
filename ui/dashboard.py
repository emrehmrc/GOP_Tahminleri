"""
dashboard.py — ADM + GDZ İzleme Paneli (Streamlit giriş noktası)
===================================================================
Çalıştırma:
    cd "adm live"
    streamlit run ui/dashboard.py
"""

import streamlit as st

import tab_veri_durumu
import tab_veri_yukleme
import tab_tahmin_uret

st.set_page_config(page_title="Talep Tahmini İzleme Paneli", layout="wide")

with st.sidebar:
    st.header("Talep Tahmini")
    st.button("🔄 Yenile")

st.title("📈 Talep Tahmini İzleme Paneli")

tab1, tab2, tab3 = st.tabs(["📊 Veri Durumu", "📥 Veri Yükleme", "🔮 Tahmin Üret"])

with tab1:
    tab_veri_durumu.render()
with tab2:
    tab_veri_yukleme.render()
with tab3:
    tab_tahmin_uret.render()
