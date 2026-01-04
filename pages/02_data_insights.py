import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Utils.AI_helper import get_chatgpt_response, update_context

st.title("🔍 Анализ и Понимание Данных")
st.caption("Получите глубокое понимание вашего датасета с помощью ИИ и визуализаций")

if "df" not in st.session_state:
    st.warning("⚠️ Сначала загрузите данные на странице «Загрузка данных».")
    st.stop()

df = st.session_state["df"]

st.markdown("---")

# === 1. ИИ Анализ Данных ===
st.subheader("🤖 ИИ Анализ Датасета")

with st.expander("📊 Получить аналитику от ИИ", expanded=True):
    st.caption("ИИ-ассистент проанализирует ваши данные и даст рекомендации")
    
    analysis_type = st.radio(
        "Что вас интересует?",
        ["Общий анализ данных", "Потенциальные проблемы", "Рекомендации по моделированию"],
        horizontal=True
    )
    
    if st.button("🚀 Анализировать", type="primary"):
        with st.spinner("ИИ анализирует ваши данные..."):
            # Собираем краткую статистику
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            missing = df.isnull().sum()
            missing_cols = missing[missing > 0].to_dict()
            
            if analysis_type == "Общий анализ данных":
                prompt = f"""
                Датасет содержит {df.shape[0]} строк и {df.shape[1]} признаков.
                Числовые признаки ({len(num_cols)}): {', '.join(num_cols[:10])}
                Категориальные признаки ({len(cat_cols)}): {', '.join(cat_cols[:10])}
                
                Дай краткий обзор: что можно сказать о структуре данных и какие признаки выглядят важными?
                """
            elif analysis_type == "Потенциальные проблемы":
                prompt = f"""
                В датасете обнаружено:
                - Пропущенные значения: {len(missing_cols)} признаков с пропусками
                - Числовых признаков: {len(num_cols)}
                - Категориальных признаков: {len(cat_cols)}
                
                Какие потенциальные проблемы могут возникнуть при моделировании? 
                Что нужно исправить в первую очередь?
                """
            else:  # Рекомендации
                prompt = f"""
                У нас есть медицинский датасет с {df.shape[0]} записями и {df.shape[1]} признаками.
                Числовые: {len(num_cols)}, Категориальные: {len(cat_cols)}
                
                Какой алгоритм машинного обучения лучше использовать? 
                На что обратить внимание при обучении модели для медицинских данных?
                """
            
            response = get_chatgpt_response(prompt)
            
            # Красивое отображение ответа
            st.markdown("### 📝 Ответ ИИ-аналитика")
            st.info(response, icon="🤖")

st.markdown("---")

# === 2. Автоматический Анализ Типов ===
st.subheader("📋 Типы Переменных")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Числовые признаки")
    num_features = df.select_dtypes(include=['number']).columns.tolist()
    if num_features:
        for feat in num_features[:10]:  # Показываем первые 10
            min_val = df[feat].min()
            max_val = df[feat].max()
            st.success(f"**{feat}**: {min_val:.2f} ↔ {max_val:.2f}", icon="📊")
    else:
        st.warning("Числовых признаков не найдено")

with col2:
    st.markdown("#### Категориальные признаки")
    cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_features:
        for feat in cat_features[:10]:
            unique_count = df[feat].nunique()
            st.info(f"**{feat}**: {unique_count} уникальных значений", icon="🏷️")
    else:
        st.warning("Категориальных признаков не найдено")

st.markdown("---")

# === 3. Проблемы с Данными ===
st.subheader("⚠️ Проверка Качества Данных")

missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0]

if len(missing_data) > 0:
    st.error(f"Обнаружено {len(missing_data)} признаков с пропущенными значениями", icon="🚨")
    
    # Визуализация пропусков
    fig = px.bar(
        x=missing_data.index,
        y=missing_data.values,
        labels={'x': 'Признак', 'y': 'Количество пропусков'},
        title='Пропущенные значения по признакам'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Детальная таблица
    with st.expander("📊 Детальная информация о пропусках"):
        missing_df = pd.DataFrame({
            'Признак': missing_data.index,
            'Пропусков': missing_data.values,
            'Процент': (missing_data.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True)
else:
    st.success("✅ Пропущенных значений не обнаружено!", icon="✨")

st.markdown("---")

# === 4. Корреляционный Анализ ===
st.subheader("🔗 Корреляции Между Признаками")

num_df = df.select_dtypes(include=['number'])
if num_df.shape[1] >= 2:
    with st.expander("📈 Показать корреляционную матрицу", expanded=False):
        corr_matrix = num_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title='Корреляционная матрица числовых признаков'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Топ корреляций
        st.markdown("#### 🔝 Сильные корреляции")
        corr_pairs = corr_matrix.unstack()
        corr_pairs = corr_pairs[corr_pairs < 1].abs().sort_values(ascending=False)[:10]
        
        for (var1, var2), corr_val in corr_pairs.items():
            if var1 != var2:
                st.caption(f"**{var1}** ↔ **{var2}**: {corr_val:.3f}")
else:
    st.warning("Недостаточно числовых признаков для корреляционного анализа")

st.markdown("---")

# === 5. Распределения ===
st.subheader("📊 Распределение Признаков")

with st.expander("🎲 Визуализировать распределение", expanded=False):
    selected_feature = st.selectbox("Выберите признак для анализа:", df.columns)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if pd.api.types.is_numeric_dtype(df[selected_feature]):
            fig = px.histogram(
                df,
                x=selected_feature,
                marginal="box",
                title=f'Распределение: {selected_feature}'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            value_counts = df[selected_feature].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                labels={'x': selected_feature, 'y': 'Частота'},
                title=f'Распределение: {selected_feature}'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**📈 Статистика**")
        if pd.api.types.is_numeric_dtype(df[selected_feature]):
            stats = df[selected_feature].describe()
            st.dataframe(stats, use_container_width=True)
        else:
            unique = df[selected_feature].nunique()
            most_common = df[selected_feature].mode()[0]
            st.metric("Уникальных значений", unique)
            st.metric("Самое частое", most_common)

st.markdown("---")
st.success("💡 **Совет**: Используйте эту информацию для выбора признаков и настройки модели!", icon="💡")
