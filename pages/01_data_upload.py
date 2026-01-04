import streamlit as st
import pandas as pd
from Utils.upload_utils import load_data, get_base_info, show_data_head, show_descriptive_stats, display_base_info
from Utils.AI_helper import update_context, notify_ai_dataset_and_goal, get_chatgpt_response

st.title("📥 Загрузка данных")
st.caption("Первый шаг в любом исследовании — загрузка и первичный анализ данных.")

# --- Загрузка данных ---
if "df" not in st.session_state:
    uploaded_file = st.file_uploader("Выберите файл (CSV или Excel)", type=["csv", "xlsx", "xls"])
    if not uploaded_file:
        st.info("⬆ Загрузите файл для начала работы.", icon="📁")
    else:
        try:
            df = load_data(uploaded_file)
            st.session_state["df"] = df
            st.success("Данные успешно загружены!", icon="✅")
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка при обработке данных: {e}", icon="🚫")
else:
    df = st.session_state["df"]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"✅ Файл загружен: {df.shape[0]} строк, {df.shape[1]} столбцов")
    with col2:
        if st.button("🗑 Сбросить файл"):
            del st.session_state["df"]
            if "modeling" in st.session_state:
                del st.session_state["modeling"]
            if "catboost_state" in st.session_state:
                del st.session_state["catboost_state"]
            st.rerun()

# --- Если данные загружены ---
if "df" in st.session_state:
    st.markdown("---")

    # Превью данных в экспандере
    with st.expander("🔍 Просмотр данных (первые 5 строк)", expanded=True):
        show_data_head(df)

    # Описательная статистика
    with st.expander("📊 Описательная статистика", expanded=False):
        st.caption("Основные статистические показатели для числовых и категориальных переменных.")
        show_descriptive_stats(df)

    # Метрики
    st.markdown("### Общая сводка")
    base_info = get_base_info(df)
    display_base_info(base_info)

    # — Контекст для ИИ —
    data_sig = (tuple(df.columns), df.shape)
    if st.session_state.get("_data_sig") != data_sig:
        summary = f"{df.shape[0]} строк, {df.shape[1]} столбцов; признаки: {', '.join(map(str, df.columns))}"
        st.session_state["_data_sig"] = data_sig
        st.session_state["data_summary"] = summary
        try:
            update_context("data_summary", summary)
        except Exception:
            pass
    else:
        summary = st.session_state.get(
            "data_summary",
            f"{df.shape[0]} строк, {df.shape[1]} столбцов; признаки: {', '.join(map(str, df.columns))}"
        )

    st.markdown("---")
    # Блок подключения ИИ
    st.subheader("💡 Подключить ИИ-ассистента")
    with st.expander("Настройка контекста анализа", expanded=False):
        st.caption("Опишите вашу задачу, чтобы ИИ лучше понимал контекст и давал более точные интерпретации.")
        
        user_desc = st.text_area(
            label="Цель исследования",
            placeholder="Например: Хочу выявить факторы риска для развития диабета...",
            value=st.session_state.get("analysis_goal", ""),
            height=100
        )
        
        if st.button("Отправить контекст ИИ"):
            st.session_state["analysis_goal"] = user_desc
            with st.spinner("Анализирую контекст..."):
                msg = notify_ai_dataset_and_goal(df, user_desc, get_chatgpt_response)
            st.success(msg)
