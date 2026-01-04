import streamlit as st
import os
from Utils.styles import apply_custom_styles, show_splash_screen
from Utils.AI_helper import reset_ai_conversation

# --- Config (Must be first) ---
st.set_page_config(
    page_title="Medical ML Modeling",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Init ---
apply_custom_styles()
show_splash_screen()

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if "_ai_session_inited" not in st.session_state:
    reset_ai_conversation()
    st.session_state["_ai_session_inited"] = True

# --- Home Page Content ---
def show_home():
    st.title("🧬 Medical ML Modeling Platform")
    st.caption("Платформа для медицинского машинного обучения и анализа данных")
    st.markdown("---")

    # --- Карточки разделов ---
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container():
            st.success("""
            📥 **Загрузка данных**
            
            Начните работу с загрузки вашего датасета (CSV, Excel). 
            Просмотрите базовую статистику и подключите ИИ для первичного анализа.
            """)
            
        with st.container():
            st.info("""
            📊 **Логистическая регрессия**
            
            Интерпретируемая модель для понимания влияния каждого признака.
            Идеально для медицинских исследований.
            """)

    with col2:
        with st.container():
            st.warning("""
            🐈‍⬛ **CatBoost моделирование**
            
            Мощный алгоритм градиентного бустинга для высокой точности.
            Отлично работает с категориальными данными.
            """)
            
        with st.container():
            st.error("""
            💬 **ИИ Интерпретация**
            
            Обсудите результаты с ИИ-ассистентом.
            Получите объяснения метрик и рекомендации.
            """)

    with col3:
        with st.container():
            st.info("""
            📝 **Руководство**
            
            Подробная документация по использованию платформы.
            Читайте README для быстрого старта.
            """)
            
        with st.container():
            st.success("""
            ⚡ **Быстрый старт**
            
            1. Загрузите данные
            2. Обучите модель
            3. Получите интерпретацию от ИИ
            """)

    st.markdown("---")
# --- Navigation Setup ---
pages = {
    "Главное": [
        st.Page(show_home, title="Главная", icon="🏠", default=True),
    ],
    "Данные": [
        st.Page("pages/01_data_upload.py", title="Загрузка данных", icon="📥"),
        st.Page("pages/02_data_insights.py", title="Анализ данных", icon="🔍"),
    ],
    "Моделирование": [
        st.Page("pages/03_logistic_regression.py", title="Логистическая регрессия", icon="📊"),
        st.Page("pages/04_catboost_modeling.py", title="CatBoost", icon="🐈‍⬛"),
    ],
    "Искусственный интеллект": [
        st.Page("pages/05_ai_interpretation.py", title="ИИ Интерпретация", icon="💬"),
    ],
    "Справка": [
        st.Page("pages/06_user_guide.py", title="Руководство", icon="📝"),
    ]
}

pg = st.navigation(pages)
pg.run()
