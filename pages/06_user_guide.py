import streamlit as st
import os

st.title("📝 Руководство пользователя")

readme_path = "README.md"
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    st.markdown(content)
else:
    st.error("Файл `README.md` не найден.")
