from config.config import Config
import streamlit as st


def load_css():
    with open(Config.CSS_FILE) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
