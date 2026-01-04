import streamlit as st
import time
import pandas as pd
from sklearn.model_selection import train_test_split

from Utils.modeling_utils import (
    ensure_modeling_state, sticky_selectbox, show_model_settings, 
    prepare_features_and_target, train_logistic_regression, evaluate_model, 
    compute_feature_importance, interpret_feature_importance, mark_model_trained, 
    show_results_and_analysis, show_single_prediction, show_export_buttons
)
from Utils.AI_helper import chat_with_context

st.title("📊 Логистическая Регрессия")
st.caption("В медицинских задачах логистическая регрессия ценится за интерпретируемость - мы можем оценить шансы (Odds Ratio) и понять вклад каждого симптома в вероятность диагноза.")

if "df" not in st.session_state:
    st.warning("⚠️ Сначала загрузите данные на странице «Загрузка данных».")
    st.stop()

df = st.session_state["df"]
ms = ensure_modeling_state(df)

st.markdown("---")

# === 1. Выбор целевой переменной (FULL WIDTH) ===
st.subheader("1️⃣ Выбор целевой переменной")
options = list(df.columns)
target_col, _ = sticky_selectbox("modeling_state", "target", "🎯 Целевая переменная (бинарный исход)", options, ui_key="modeling_target_ui")

if target_col:
    unique_vals = df[target_col].dropna().unique()
    if len(unique_vals) > 2:
        st.error(f"❌ Целевая переменная '{target_col}' содержит {len(unique_vals)} классов. Логистическая регрессия требует 2 класса (0 и 1).")
        st.stop()
    
    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        st.error("❌ Нет доступных признаков для обучения.")
        st.stop()
    
    st.success(f"✅ Выбрано признаков: {len(feature_cols)}")

st.markdown("---")

# === 2. Гиперпараметры (В ОСНОВНОЙ ЧАСТИ) ===
with st.expander("⚙️ 2️⃣ Настройка гиперпараметров", expanded=False):
    st.caption("Настройте параметры модели для оптимальной работы")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        C_value = st.number_input("Параметр регуляризации C", 0.01, 100.0, 1.0, 0.01, 
                                   help="Обратная сила регуляризации. Меньше = сильнее регуляризация")
        penalty = st.selectbox("Тип регуляризации", ["l1", "l2"], index=1,
                               help="L1: разреженная модель, L2: гладкая модель")
        
    with col2:
        max_iter = st.number_input("Макс. итераций", 100, 5000, 1000, 100)
        threshold = st.slider("Порог классификации", 0.05, 0.95, 0.5, 0.05,
                              help="Вероятность отсечения для класса 1. Снизьте для повышения Recall")
        
    with col3:
        test_size = st.slider("Размер тестовой выборки (%)", 10, 50, 20, 5) / 100
        use_class_weight = st.checkbox("Сбалансировать веса классов", value=False,
                                        help="Автоматически балансирует классы при несбалансированных данных")

st.markdown("---")

# === 3. Обучение ===
st.subheader("3️⃣ Обучение модели")

if st.button("🚀 Обучить модель", use_container_width=True, type="primary"):
    try:
        with st.spinner("⏳ Тренировка модели... Идет поиск зависимостей..."):
            # Подготовка
            X, y_encoded, le, num_cols, cat_cols = prepare_features_and_target(df, target_col)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )

            # Обучение
            class_weight = "balanced" if use_class_weight else None
            model, meta = train_logistic_regression(
                X_train, y_train,
                C=C_value, penalty=penalty,
                class_weight=class_weight, max_iter=max_iter,
                label_encoder=le
            )

            # Оценка
            metrics, roc_data, pr_data = evaluate_model(model, X_test, y_test, meta, threshold)
            importance_df = compute_feature_importance(model, meta)
            short_text = interpret_feature_importance(importance_df, top_n=5)

            # Сохранение
            st.session_state["modeling"] = {
                "model": model, "meta": meta,
                "threshold": threshold, "metrics": metrics,
                "roc": roc_data, "pr": pr_data,
                "importance_df": importance_df, "short_text": short_text,
                "target_col": target_col, "feature_cols": feature_cols,
                "params": {
                    "C": C_value, "penalty": penalty,
                    "class_weight": class_weight, "max_iter": max_iter,
                    "test_size": test_size
                }
            }
            mark_model_trained()
        st.success("✅ Модель успешно обучена!")

    except Exception as e:
        st.error(f"Ошибка во время обучения: {e}")

# === 4. Результаты ===
if "modeling" in st.session_state:
    data = st.session_state["modeling"]
    
    st.markdown("---")
    st.subheader("4️⃣ Результаты и Интерпретация")
    
    show_results_and_analysis(data)

    if st.button("🤖 Объяснить метрики (ИИ)"):
        with st.spinner("Спрашиваю у ИИ..."):
            prompt = f"У меня получились такие метрики модели: {data['metrics']}. Объясни, хорошие ли это результаты для медицинской задачи? На что обратить внимание?"
            chat_with_context(prompt)
        st.info("✅ Ответ готов! Перейдите в раздел **ИИ Интерпретация**, чтобы прочитать.")
    
    st.markdown("---")
    col_pred, col_export = st.columns(2)
    with col_pred:
         show_single_prediction(data, df)
    with col_export:
         show_export_buttons(data)
