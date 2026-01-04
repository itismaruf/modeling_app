import streamlit as st
from Utils.chat import continue_chat, render_message, reset_chat_history

st.title("💬 ИИ-Интерпретатор")
st.markdown("Задайте вопросы о полученных моделях, данных или методах анализа. ИИ имеет доступ к статистике ваших данных.")

if st.button("🗑 Очистить историю диалога"):
    reset_chat_history()
    st.rerun()

st.session_state.setdefault("chat_history", [])

# Контейнер для чата
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.info("👋 Привет! Я готов помочь вам проанализировать результаты. Спросите, например: 'Какие признаки самые важные?' или 'Почему модель ошибается?'")
    
    for msg in st.session_state.chat_history:
        render_message(msg["text"], msg["sender"])

# Ввод
if prompt := st.chat_input("Ваш вопрос..."):
    # User message
    st.session_state.chat_history.append({"text": prompt, "sender": "user"})
    render_message(prompt, "user") # Render immediately? No, re-render loop will handle usually, but force render here to look fast
    
    # AI response
    with st.spinner("ИИ думает..."):
        try:
            answer = continue_chat(prompt)
            st.session_state.chat_history.append({"text": answer, "sender": "ai"})
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка обращения к ИИ: {e}")
