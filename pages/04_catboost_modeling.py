import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

from Utils.catboost_modeling import (
    detect_task,
    prepare_features_and_target_catboost,
    train_catboost_universal,
    evaluate_catboost_universal,
    compute_catboost_feature_importance,
    plot_feature_importance_signed,
    build_confusion_matrix,
    valid_eval_metrics_for_task,
    predict_new_object
)
from Utils.AI_helper import chat_with_context

st.title("🐈‍⬛ CatBoost Моделирование")
st.caption("CatBoost отлично справляется с категориальными данными и часто показывает более высокую точность. В медицине важно следить за Recall (Чувствительность), чтобы не пропускать больных пациентов.")

df = st.session_state.get("df")
if df is None or df.empty:
    st.warning("⚠️ Сначала загрузите данные.")
    st.stop()

if "catboost_state" not in st.session_state:
    st.session_state["catboost_state"] = {}

st.markdown("---")

# === 1. Выбор целевой переменной (FULL WIDTH) ===
st.subheader("1️⃣ Выбор целевой переменной")
options = list(df.columns)
target_col = st.selectbox("🎯 Целевая переменная", options)
if not target_col:
    st.stop()

task = detect_task(df, target_col)
st.info(f"Тип задачи: **{task.upper()}**")

st.markdown("---")

# === 2. Параметры модели (В ОСНОВНОЙ ЧАСТИ) ===
with st.expander("⚙️ 2️⃣ Параметры модели", expanded=False):
    st.caption("Настройте гиперпараметры CatBoost для оптимизации модели")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        iterations = st.slider("Итерации (Iterations)", 100, 3000, 800, step=50, 
                               help="Сколько деревьев строить. Больше = точнее, но дольше")
        depth = st.slider("Глубина (Depth)", 2, 10, 6, 
                          help="Глубина деревьев. Меньше = быстрее и меньше переобучение")
        learning_rate = st.slider("Learning rate", 0.005, 0.2, 0.05,
                                   help="Скорость обучения. Меньше = стабильнее, но медленнее")
    
    with col2:
        test_size = st.slider("Размер тестовой выборки", 0.1, 0.5, 0.2)
        
        if task == "binary":
            use_manual_class_weights = st.checkbox("Баланс классов (Auto Class Weights)", value=True, 
                                                    help="Автоматически увеличить вес редкого класса")
            threshold = st.slider("Порог вероятности (Threshold)", 0.05, 0.95, 0.5, 
                                  help="Порог для отнесения к классу 1. Снизьте для повышения Recall")
        else:
            use_manual_class_weights = False
            threshold = 0.5
    
    with col3:
        if task == "binary":
            eval_metric = st.selectbox("Метрика оптимизации", valid_eval_metrics_for_task(task), index=0)
        elif task == "regression":
            eval_metric = "RMSE"
        else:
            eval_metric = "MultiClass"
        
        st.info(f"Метрика: **{eval_metric}**")

# Расчет весов классов
class_weights = None
if use_manual_class_weights and task == "binary":
    y_tmp = df[target_col]
    if not pd.api.types.is_numeric_dtype(y_tmp):
        y_tmp = pd.factorize(y_tmp)[0]
    pos_rate = float((y_tmp == 1).mean())
    if pos_rate > 0:
        auto_w = round(1.0 / pos_rate, 2)
        st.caption(f"⚖️ Вес положительного класса будет ≈ {auto_w} (так как их всего {pos_rate:.1%})")
        class_weights = [1.0, auto_w]

st.markdown("---")

# === 3. Обучение ===
st.subheader("3️⃣ Обучение модели")

if st.button("🚀 Запустить обучение", use_container_width=True, type="primary"):
    with st.spinner("⏳ Cat Boost обучается..."):
        try:
            X, y, cat_features = prepare_features_and_target_catboost(df, target_col)
            stratify = y if task in ("binary", "multiclass") else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify
            )

            model = train_catboost_universal(
                X_train, y_train, X_test, y_test, cat_features,
                task=task,
                iterations=iterations, depth=depth, lr=learning_rate,
                class_weights=class_weights, eval_metric=eval_metric,
            )

            metrics, y_pred, y_proba, viz = evaluate_catboost_universal(
                model, X_test, y_test, task=task, threshold=threshold
            )

            imp_df = compute_catboost_feature_importance(model, X.columns.tolist(), signed=True)
            imp_figs = plot_feature_importance_signed(imp_df, top_n=15)
            
            cm_fig = None
            if task in ("binary", "multiclass") and y_pred is not None:
                cm_fig = build_confusion_matrix(y_test, y_pred, labels=np.unique(y_test))

            st.session_state["catboost_state"] = {
                "model": model, "metrics": metrics, "viz": viz,
                "importance_df": imp_df, "importance_figs": imp_figs,
                "confusion_matrix": cm_fig, "target_col": target_col,
                "feature_cols": X.columns.tolist(),
                "threshold": threshold, "task": task,
                "params": {"iterations": iterations, "depth": depth}
            }
            
            st.success("✅ Модель успешно обучена!")
        except Exception as e:
            st.error(f"Ошибка: {e}")

# === 4. Результаты ===
state = st.session_state.get("catboost_state")
if state:
    st.markdown("---")
    st.subheader("4️⃣ Результаты и Анализ")
    
    # Метрики
    st.markdown("#### Метрики качества")
    m_cols = st.columns(len(state["metrics"]))
    for i, (k, v) in enumerate(state["metrics"].items()):
        with m_cols[i % len(m_cols)]:
            st.metric(k, f"{v:.3f}")

    if st.button("🤖 Объяснить эти метрики (ИИ)"):
        with st.spinner("Анализирую..."):
            p = f"Модель CatBoost показала: {state['metrics']}. Задача: {state['task']}. Дай краткий анализ качества."
            chat_with_context(p)
        st.info("✅ ИИ ответил. Проверьте вкладку **ИИ Интерпретация**!")

    # Графики в табах
    tab1, tab2, tab3 = st.tabs(["📊 Важность признаков", "📈 Визуализация", "🔮 Прогноз"])
    
    with tab1:
        st.caption("Какие факторы сильнее всего влияют на предсказание?")
        colp, coln = st.columns(2)
        with colp:
            if state["importance_figs"].get("pos"):
                st.plotly_chart(state["importance_figs"]["pos"], use_container_width=True)
        with coln:
            if state["importance_figs"].get("neg"):
                st.plotly_chart(state["importance_figs"]["neg"], use_container_width=True)
    
    with tab2:
        if state["task"] == "binary":
            st.plotly_chart(state["viz"]["roc_fig"], use_container_width=True)
            st.plotly_chart(state["viz"]["pr_fig"], use_container_width=True)
        if state["confusion_matrix"]:
            st.plotly_chart(state["confusion_matrix"], use_container_width=True)

    with tab3:
        st.markdown("### 🔮 Прогноз для Нового Пациента")
        st.caption("Введите данные пациента для получения прогноза и его интерпретации")
        
        # Инициализируем session_state для прогноза
        if "last_prediction" not in st.session_state:
            st.session_state["last_prediction"] = None
        
        # Форма ввода данных
        with st.form("prediction_form"):
            st.markdown("#### Введите значения признаков:")
            
            feature_inputs = {}
            # Разбиваем на колонки для компактности
            num_cols_per_row = 3
            total_features = len(state["feature_cols"])
            
            for row_start in range(0, total_features, num_cols_per_row):
                cols = st.columns(num_cols_per_row)
                for i, col_idx in enumerate(range(row_start, min(row_start + num_cols_per_row, total_features))):
                    col_name = state["feature_cols"][col_idx]
                    with cols[i]:
                        if pd.api.types.is_numeric_dtype(df[col_name]):
                            median_val = float(df[col_name].median())
                            feature_inputs[col_name] = st.number_input(
                                f"{col_name}",
                                value=median_val,
                                key=f"input_{col_name}"
                            )
                        else:
                            unique_vals = df[col_name].unique().tolist()
                            feature_inputs[col_name] = st.selectbox(
                                f"{col_name}",
                                unique_vals,
                                key=f"select_{col_name}"
                            )
            
            submitted = st.form_submit_button("🚀 Сделать прогноз", use_container_width=True, type="primary")
        
        # Обработка формы и сохранение в session_state
        if submitted:
            res = predict_new_object(state["model"], feature_inputs, task=state["task"], threshold=state["threshold"])
            
            # Сохраняем в session_state
            st.session_state["last_prediction"] = {
                "result": res,
                "inputs": feature_inputs.copy(),
                "task": state["task"],
                "threshold": state["threshold"],
                "importance_df": state.get("importance_df")
            }
        
        # Отображение результатов из session_state
        if st.session_state["last_prediction"] is not None:
            pred_data = st.session_state["last_prediction"]
            res = pred_data["result"]
            feature_inputs = pred_data["inputs"]
            
            # === Отображение результата ===
            st.markdown("---")
            st.markdown("### 📊 Результат Прогноза")
            
            if pred_data["task"] == "binary":
                prob = res["probability"]
                pred_class = res["prediction"]
                
                # Красивая визуализация результата
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Основной результат
                    if pred_class == 1:
                        st.error(f"**Прогноз: ПОЛОЖИТЕЛЬНЫЙ (класс 1)**", icon="🔴")
                        st.metric("Вероятность", f"{prob:.1%}", delta=f"+{(prob-0.5)*100:.1f}%")
                    else:
                        st.success(f"**Прогноз: ОТРИЦАТЕЛЬНЫЙ (класс 0)**", icon="🟢")
                        st.metric("Вероятность негативного исхода", f"{(1-prob):.1%}")
                    
                    st.caption(f"Порог отсечения: {pred_data['threshold']:.2f}")
                
                with col2:
                    # Gauge chart для вероятности
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Вероятность положительного класса (%)"},
                        delta = {'reference': pred_data['threshold'] * 100},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred" if prob > pred_data['threshold'] else "darkgreen"},
                            'steps': [
                                {'range': [0, pred_data['threshold']*100], 'color': "lightgreen"},
                                {'range': [pred_data['threshold']*100, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': pred_data['threshold'] * 100
                            }
                        }
                    ))
                    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(res)
            
            st.markdown("---")
            
            # === Feature Importance для этого предсказания ===
            st.markdown("### 📈 Что повлияло на прогноз?")
            st.caption("Вклад каждого признака в итоговое предсказание для данного пациента")
            
            # Используем feature importance модели
            if pred_data["importance_df"] is not None:
                imp_df = pred_data["importance_df"].copy()
                
                # Берем топ-10 самых важных признаков
                top_features = imp_df.nlargest(10, 'importance')
                
                # Создаем данные для графика с реальными значениями пациента
                chart_data = []
                for _, row in top_features.iterrows():
                    feat_name = row['feature']
                    importance = row['importance']
                    patient_value = feature_inputs.get(feat_name, "N/A")
                    
                    chart_data.append({
                        'Признак': feat_name,
                        'Важность': importance,
                        'Значение пациента': str(patient_value)
                    })
                
                chart_df = pd.DataFrame(chart_data)
                
                # Визуализация важности с помощью Plotly
                fig = px.bar(
                    chart_df,
                    y='Признак',
                    x='Важность',
                    orientation='h',
                    title='Важность признаков для данного прогноза',
                    hover_data=['Значение пациента'],
                    color='Важность',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Детальная таблица
                with st.expander("📋 Детальная таблица важности признаков"):
                    display_df = chart_df.copy()
                    display_df['Важность'] = display_df['Важность'].apply(lambda x: f"{x:.4f}")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # === AI Интерпретация ===
            st.markdown("### 🤖 ИИ Объяснение Прогноза")
            
            if st.button("💬 Получить объяснение от ИИ", type="secondary"):
                with st.spinner("ИИ анализирует прогноз..."):
                    # Формируем контекст для ИИ
                    top_feat_str = ", ".join([f"{row['Признак']}={feature_inputs.get(row['Признак'])}" 
                                              for _, row in chart_df.iterrows()])
                    
                    if pred_data["task"] == "binary":
                        prob = res["probability"]
                        pred_class = res["prediction"]
                        ai_prompt = f"""
                        Модель CatBoost сделала прогноз для пациента:
                        - Вероятность положительного исхода: {prob:.1%}
                        - Итоговый класс: {pred_class}
                        - Порог отсечения: {pred_data['threshold']}
                        
                        Самые важные признаки этого пациента:
                        {top_feat_str}
                        
                        Объясни простым языком, почему модель сделала такой прогноз. 
                        Какие признаки сыграли ключевую роль? Что это значит для пациента?
                        """
                    else:
                        ai_prompt = f"""
                        Модель CatBoost сделала прогноз: {res['prediction']}
                        
                        Ключевые признаки пациента:
                        {top_feat_str}
                        
                        Объясни, почему получился такой результат и что влияет на прогноз.
                        """
                    
                    explanation = chat_with_context(ai_prompt)
                    
                    # Красивое отображение объяснения
                    st.info(explanation, icon="🤖")
                    st.caption("💡 Для продолжения диалога перейдите в раздел **ИИ Интерпретация**")


