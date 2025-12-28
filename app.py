# ============ Модули
import streamlit as st
import pandas as pd
import os
from catboost import Pool
import time
from sklearn.model_selection import train_test_split
import numpy as np

from Utils.upload_utils import load_data, get_base_info, show_data_head, show_descriptive_stats, display_base_info


from Utils.modeling_utils import ensure_modeling_state, sticky_selectbox, show_model_settings, \
                                 prepare_features_and_target, train_logistic_regression, evaluate_model, \
                                 compute_feature_importance, interpret_feature_importance, mark_model_trained, \
                                 show_results_and_analysis, show_single_prediction, show_export_buttons

from Utils.chat import continue_chat, render_message, reset_chat_history

from AI_helper import update_context, reset_ai_conversation, get_chatgpt_response, notify_ai_dataset_and_goal, chat_with_context


# Конфигурация страницы
st.set_page_config(layout="wide")

import time
import streamlit as st

if "app_loaded" not in st.session_state:
    st.markdown("""
        <style>
            :root {
                --bg1: #0b0f19;
                --bg2: #1a2238;
                --bg3: #243b55;
                --accent: #8ab6ff;
                --accent2: #b6d6ff;
            }

            .splash-root {
                position: fixed;
                inset: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(120deg, var(--bg1), var(--bg2), var(--bg3));
                background-size: 400% 400%;
                animation: gradientMove 12s ease infinite;
                z-index: 99999;
            }

            @keyframes gradientMove {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            .splash-card {
                position: relative;
                padding: 40px;
                border-radius: 20px;
                background: rgba(255,255,255,0.06);
                backdrop-filter: blur(20px);
                text-align: center;
                color: white;
                box-shadow: 0 0 40px rgba(138,182,255,0.3);
                animation: fadeInUp 1.2s ease forwards;
            }

            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(40px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .splash-icon {
                font-size: 4em;
                margin-bottom: 20px;
                animation: pulse 2.5s infinite, rotate 8s linear infinite;
            }

            @keyframes pulse {
                0%,100% { transform: scale(1); }
                50% { transform: scale(1.1); }
            }
            @keyframes rotate {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }

            .splash-title {
                font-size: 2.2em;
                font-weight: 700;
                margin-bottom: 10px;
                border-right: 2px solid var(--accent);
                white-space: nowrap;
                overflow: hidden;
                animation: typing 3s steps(30, end), blink 0.8s infinite;
            }

            @keyframes typing {
                from { width: 0; }
                to { width: 100%; }
            }
            @keyframes blink {
                50% { border-color: transparent; }
            }

            .splash-sub {
                font-size: 1.1em;
                color: #d2dbff;
                margin-bottom: 20px;
                opacity: 0;
                animation: fadeIn 2s ease forwards;
                animation-delay: 1.5s;
            }

            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            .splash-footer {
                margin-top: 20px;
                font-size: 0.9em;
                color: #98a2c6;
            }

            .fade-out {
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.8s ease;
            }
        </style>

        <div class="splash-root" id="splash">
            <div class="splash-card">
                <div class="splash-icon">🧬</div>
                <div class="splash-title">ML‑модели для медицинских целей</div>
                <div class="splash-sub">Логистическая регрессия • CatBoost • Интерпретация и оценка</div>
                <div class="splash-footer">© Разработано Rahimov M.A.</div>
            </div>
        </div>

        <script>
            const splash = document.getElementById("splash");
            setTimeout(() => {
                splash.classList.add("fade-out");
                setTimeout(() => splash.remove(), 900);
            }, 4000);
        </script>
    """, unsafe_allow_html=True)

    time.sleep(7)
    st.session_state.app_loaded = True
    st.rerun()


    
# --- Установка API-ключа из секретов, если есть ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if "_ai_session_inited" not in st.session_state:
    reset_ai_conversation()                 # сброс глобальной истории для этой сессии
    st.session_state["_ai_session_inited"] = True

# --- Инициализация первой страницы при запуске ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'Загрузка данных'

st.markdown("""
    <style>
        /* Когда сайдбар открыт (aria-expanded="true"), основной контент смещается вправо */
        [data-testid="stSidebar"][aria-expanded="true"] ~ .main .block-container {
            margin-left: 300px;
            transition: margin-left 0.3s ease;
        }
        /* Когда сайдбар свернут (aria-expanded="false"), основной контент возвращается в исходное положение */
        [data-testid="stSidebar"][aria-expanded="false"] ~ .main .block-container {
            margin-left: 1rem;
            transition: margin-left 0.3s ease;
        }
    </style>
""", unsafe_allow_html=True)

# --- Функция переключения страниц ---
def set_page(page_name):
    st.session_state['page'] = page_name

# --- Сайдбар с навигацией и стилем кнопок ---
st.sidebar.header("🔧 Навигация")
pages = {
    "Загрузка данных": "📥",
    "Логистическая регрессия": "⑀",
    "CatBoost моделирование": "🐈‍⬛",
    "Разъяснение результатов (с ИИ)": "💬",
    "Руководство пользователя": "📝"
}

# Настройка CSS для кнопок (цвета при наведении)
st.markdown("""
    <style>
        div.stButton > button {
            background-color: #f0f2f6;
            color: black;
            border: 1px solid #ccc;
            border-radius: 6px;
        }
        div.stButton > button:hover {
            background-color: #e0f0ff;
            color: #007BFF;
            border: 1px solid #007BFF;
        }
    </style>
""", unsafe_allow_html=True)

# Навигационные кнопки
for name, icon in pages.items():
    st.sidebar.button(f"{icon} {name}", on_click=set_page, args=(name,))

# Кнопка для очистки всех данных
if st.sidebar.button("🔄 Очистить всё"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ===================== СТРАНИЦЫ =======================
# === Загрузка данных ===
if st.session_state['page'] == "Загрузка данных":
    st.caption('💡Если вы не пользовались ClaryData, сначала перейдите в раздел "Руководство пользователя"!')
    st.title("📥 Загрузка данных")

    # --- Загрузка данных ---
    if "df" not in st.session_state:
        uploaded_file = st.file_uploader(" ", type=["csv", "xlsx", "xls"])
        if not uploaded_file:
            st.info("⬆ Загрузите файл для анализа.", icon="📁")
        else:
            try:
                df = load_data(uploaded_file)
                st.session_state["df"] = df
                st.success("Данные успешно загружены", icon="✅")
            except Exception as e:
                st.error(f"Ошибка при обработке данных: {e}", icon="🚫")
    else:
        df = st.session_state["df"]
        st.success("Данные уже загружены ✅")

    # --- Если данные загружены ---
    if "df" in st.session_state:
        st.markdown("---")

        # Превью данных в экспандере
        with st.expander("Пример данных (первые строки)", expanded=False):
            show_data_head(df)

        # Описательная статистика в отдельном экспандере
        with st.expander("📑 Описательная статистика", expanded=False):
            show_descriptive_stats(df)

        # Метрики
        base_info = get_base_info(df)
        display_base_info(base_info)

        # — Инициализация/обновление краткого summary —
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
        # Блок подключения ИИ в экспандере
        with st.expander("🤖 Подключение ИИ", expanded=False):
            st.caption("При желании укажите цель анализа — ИИ адаптирует помощь под неё.")

            user_desc = st.text_area(
                label="Цель анализа",
                placeholder="Например: Хочу проанализировать, как меняются цены на жильё по регионам",
                value=st.session_state.get("analysis_goal", ""),
                height=100,
                label_visibility="collapsed",
                key="analysis_goal_input" 
            )

            if st.button("Подключить ИИ"):
                msg = notify_ai_dataset_and_goal(df, user_desc, get_chatgpt_response)
                st.success(msg)

        if st.button("🤖 Подключить ИИ"):
            msg = notify_ai_dataset_and_goal(df, user_desc, get_chatgpt_response)
            st.success(msg)


# === Моделирование и предсказание ===
if st.session_state.get("page") == "Логистическая регрессия":
    st.title("Логистическая регрессия")
    st.caption("ℹ Фокус: понять, как и почему признаки влияют на целевую переменную")

    if "df" not in st.session_state:
        st.warning("📥 Сначала загрузите данные.")
        st.stop()

    df = st.session_state["df"]
    ms = ensure_modeling_state(df)

    options = list(df.columns)
    target_col, _ = sticky_selectbox("modeling_state", "target", "🎯 Целевая переменная (binary target)", options, ui_key="modeling_target_ui")

    if len(pd.Series(df[target_col].dropna().unique())) > 2:
        st.error("Целевая переменная должна быть бинарной")
        st.stop()

    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        st.error("Нет признаков для обучения")
        st.stop()

    C_value, penalty, max_iter, threshold, test_size, use_class_weight = show_model_settings()


    if st.button("🚀 Обучить / переобучить модель", use_container_width=True):
        try:
            with st.spinner("⏳ Обучение модели..."):
                time.sleep(5)

                # Подготовка данных
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
                short_text = interpret_feature_importance(importance_df, top_n=3)

                # Сохраняем в сессию
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

            st.success("✅ Модель обучена и сохранена")

        except Exception as e:
            st.error(f"Не удалось обучить модель: {e}")

    # Если модель уже обучена — показываем результаты
    if "modeling" in st.session_state:
        data = st.session_state["modeling"]

        show_results_and_analysis(data)
        show_single_prediction(data, df)
        show_export_buttons(data)


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

if st.session_state.get("page") == "CatBoost моделирование":
    st.title("CatBoost моделирование")
    st.caption("ℹ Медицинский фокус: высокая полнота (Recall) при разумной точности")

    # --- Страховка от отсутствия df ---
    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning("📥 Сначала загрузите данные.")
        st.stop()

    # --- Инициализация состояния ---
    if "catboost_state" not in st.session_state:
        st.session_state["catboost_state"] = {}

    # --- Выбор целевой переменной ---
    options = list(df.columns)
    target_col = st.selectbox("🎯 Целевая переменная", options)
    if not target_col:
        st.error("❌ Не выбрана целевая переменная")
        st.stop()

    # --- Определение задачи ---
    task = detect_task(df, target_col)
    st.info(f"Задача: {task}")

    # --- Настройки модели ---
    with st.expander("⚙️ Настройки модели", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            iterations = st.slider("Iterations", 100, 3000, 800, step=50)
            depth = st.slider("Depth", 2, 10, 6)
            learning_rate = st.slider("Learning rate", 0.005, 0.2, 0.05)
            l2_leaf_reg = st.slider("L2 leaf reg", 1.0, 10.0, 3.0)
        with col2:
            subsample = st.slider("Subsample", 0.3, 1.0, 0.8)
            colsample_bylevel = st.slider("Colsample by level", 0.3, 1.0, 0.8)
            min_data_in_leaf = st.slider("Min data in leaf", 1, 100, 20)
            test_size = st.slider("Test size", 0.1, 0.5, 0.2)

        threshold = st.slider("Threshold (binary only)", 0.05, 0.95, 0.5) if task == "binary" else 0.5

        available_metrics = valid_eval_metrics_for_task(task)
        eval_metric = st.selectbox("Eval metric", available_metrics, index=0)

        use_recall_monitor = st.checkbox("Мониторить Recall (custom_metric)", value=(task == "binary"))

        use_class_weight = st.checkbox("Баланс классов (binary)", value=(task == "binary"))
        class_weights = None
        if use_class_weight and task == "binary":
            y_tmp = df[target_col]
            if not pd.api.types.is_numeric_dtype(y_tmp):
                y_tmp = pd.factorize(y_tmp)[0]
            if len(np.unique(pd.Series(y_tmp).dropna())) == 2:
                pos_rate = float((y_tmp == 1).mean())
                auto_w = round(1.0 / max(pos_rate, 1e-6), 2)
                st.caption(f"Автовес положительного класса ≈ {auto_w}")
                class_weights = [1.0, auto_w]

    # --- Кнопка обучения ---
    if st.button("🚀 Обучить модель CatBoost", use_container_width=True):
        try:
            with st.spinner("⏳ Обучение модели..."):
                from sklearn.model_selection import train_test_split

                X, y, cat_features = prepare_features_and_target_catboost(df, target_col)
                stratify = y if task in ("binary", "multiclass") else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=stratify
                )

                custom_metric = ["Recall"] if (task == "binary" and use_recall_monitor) else None
                model = train_catboost_universal(
                    X_train, y_train, X_test, y_test, cat_features,
                    task=task,
                    iterations=iterations,
                    depth=depth,
                    lr=learning_rate,
                    l2_leaf_reg=l2_leaf_reg,
                    subsample=subsample,
                    colsample_bylevel=colsample_bylevel,
                    min_data_in_leaf=min_data_in_leaf,
                    class_weights=class_weights,
                    eval_metric=eval_metric,
                    custom_metric=custom_metric,
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
                    "model": model,
                    "metrics": metrics,
                    "viz": viz,
                    "importance_df": imp_df,
                    "importance_figs": imp_figs,
                    "confusion_matrix": cm_fig,
                    "target_col": target_col,
                    "feature_cols": X.columns.tolist(),
                    "threshold": threshold,
                    "task": task,
                    # сохраняем индексы категориальных признаков
                    "cat_features_idx": st.session_state.get("cat_features_idx", []),
                    "params": {
                        "iterations": iterations,
                        "depth": depth,
                        "learning_rate": learning_rate,
                        "l2_leaf_reg": l2_leaf_reg,
                        "subsample": subsample,
                        "colsample_bylevel": colsample_bylevel,
                        "min_data_in_leaf": min_data_in_leaf,
                        "test_size": test_size,
                        "class_weights": class_weights,
                        "eval_metric": eval_metric,
                        "use_recall_monitor": use_recall_monitor,
                    }
                }


            st.success("✅ CatBoost модель обучена и сохранена")

        except Exception as e:
            st.error(f"❌ Не удалось обучить модель: {e}")

    # --- Если модель уже обучена ---
    state = st.session_state.get("catboost_state")
    if state:
        tabs = st.tabs(["📊 Результаты модели", "🔮 Прогноз нового объекта"])

        # --- Вкладка 1: результаты ---
        with tabs[0]:
            with st.expander("📊 Метрики (таблица)", expanded=False):
                # Форматируем значения метрик как строки с 3 знаками после запятой
                metrics_df = pd.DataFrame(
                    [(k, f"{v:.3f}") for k, v in state["metrics"].items()],
                    columns=["Метрика", "Значение"]
                )
                st.table(metrics_df)

            if state["task"] == "binary" and "roc_fig" in state["viz"] and "pr_fig" in state["viz"]:
                with st.expander("📈 Визуализация метрик (ROC и PR)", expanded=True):
                    st.plotly_chart(state["viz"]["roc_fig"], use_container_width=True)
                    st.plotly_chart(state["viz"]["pr_fig"], use_container_width=True)

            if state.get("confusion_matrix") is not None:
                with st.expander("🧩 Confusion Matrix", expanded=True):
                    st.plotly_chart(state["confusion_matrix"], use_container_width=True)

            with st.expander("🔥 Важность признаков (топ-15)", expanded=True):
                colp, coln = st.columns(2)
                with colp:
                    fig_pos = state["importance_figs"].get("pos")
                    if fig_pos:
                        st.plotly_chart(fig_pos, use_container_width=True)
                with coln:
                    fig_neg = state["importance_figs"].get("neg")
                    if fig_neg:
                        st.plotly_chart(fig_neg, use_container_width=True)

        # --- Вкладка 2: прогноз нового объекта ---
        with tabs[1]:
            st.subheader("🔮 Прогнозирование нового объекта")

            feature_inputs = {}
            cols = st.columns(2)

            for i, col in enumerate(state["feature_cols"]):
                if pd.api.types.is_numeric_dtype(df[col]):
                    with cols[i % 2]:
                        feature_inputs[col] = st.number_input(f"{col}", value=float(df[col].median()))
                else:
                    with cols[i % 2]:
                        options = df[col].dropna().unique().tolist()
                        feature_inputs[col] = st.selectbox(f"{col}", options)

            import time
            import pandas as pd
            import plotly.express as px

            output_area = st.empty()

            if st.button("📌 Сделать прогноз", use_container_width=True):
                with st.spinner("⏳ Модель делает прогноз..."):
                    time.sleep(1.5)

                    try:
                        result = predict_new_object(
                            state["model"], feature_inputs,
                            task=state["task"], threshold=state["threshold"]
                        )
                        st.session_state["last_prediction"] = result

                        with output_area.container():
                            st.success("✅ Прогноз готов")

                            if state["task"] == "binary":
                                st.write(f"**Предсказанный класс:** {result['prediction']}")
                                st.write(f"**Вероятность положительного класса:** {result['probability']:.3f}")

                                # Визуализация вероятности
                                fig = px.bar(
                                    x=["Negative", "Positive"],
                                    y=[1 - result["probability"], result["probability"]],
                                    labels={"x": "Класс", "y": "Вероятность"},
                                    title="Вероятности классов",
                                    color=["Negative", "Positive"],
                                    color_discrete_map={"Negative": "steelblue", "Positive": "crimson"}
                                )
                                st.plotly_chart(fig, use_container_width=True, key="binary_probs")

                            elif state["task"] == "multiclass":
                                st.write(f"**Предсказанный класс:** {result['prediction']}")
                                st.write("**Вероятности по классам:**")

                                # Красивый бар-чарт для вероятностей
                                fig = px.bar(
                                    x=list(range(len(result["probabilities"]))),
                                    y=result["probabilities"],
                                    labels={"x": "Класс", "y": "Вероятность"},
                                    title="Вероятности по классам",
                                    color=result["probabilities"],
                                    color_continuous_scale="Viridis"
                                )
                                st.plotly_chart(fig, use_container_width=True, key="multiclass_probs")

                            else:  # regression
                                st.write(f"**Предсказанное значение:** {result['prediction']:.3f}")

                            # --- Важность признаков (из обучения) ---
                            st.markdown("### Признаки которые влияли на прогноз")
                            fig_pos = state["importance_figs"].get("pos")
                            fig_neg = state["importance_figs"].get("neg")

                            if fig_pos:
                                st.plotly_chart(fig_pos, use_container_width=True, key="feat_imp_pos")
                            if fig_neg:
                                st.plotly_chart(fig_neg, use_container_width=True, key="feat_imp_neg")

                    except Exception as e:
                        st.error(f"❌ Ошибка при прогнозировании: {e}")








# === Разъяснение результатов (с ИИ) ===
if st.session_state.get("page") == "Разъяснение результатов (с ИИ)":
    st.title("💬 Поговорим о ваших данных?")
    st.markdown("---")

    if st.button("🗑 Очистить чат"):
        reset_chat_history()
        st.success("Чат очищен.")
        st.stop()

    st.session_state.setdefault("chat_history", [])

    # Ввод нового сообщения
    question = st.chat_input("Напишите свой вопрос…")

    if question:
        # Добавляем вопрос в историю
        st.session_state.chat_history.append({"text": question, "sender": "user"})

        # Сначала рендерим всю историю (включая новый вопрос)
        for msg in st.session_state.chat_history:
            render_message(msg["text"], msg["sender"])

        # Временный индикатор "ИИ печатает..."
        placeholder = st.empty()
        placeholder.markdown(
            """
            <style>
            @keyframes blink {
                0%   { opacity: 0.2; }
                20%  { opacity: 1; }
                100% { opacity: 0.2; }
            }
            .dot {
                display: inline-block;
                margin-left: 2px;
                animation: blink 1.4s infinite both;
            }
            .dot:nth-child(2) { animation-delay: 0.2s; }
            .dot:nth-child(3) { animation-delay: 0.4s; }
            </style>

            <div style='
                background: var(--background-color);
                color: var(--text-color);
                padding: 10px 14px;
                border-radius: 12px;
                text-align: left;
                margin: 6px 0;
                font-style: italic;
                opacity: 0.85;
                box-shadow: 0 1px 3px rgba(0,0,0,0.15);
            '>
                🤖 ИИ печатает<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


        # Получаем ответ ИИ (это занимает время)
        answer = continue_chat(question)

        # Заменяем индикатор на настоящий ответ
        placeholder.empty()
        st.session_state.chat_history.append({"text": answer, "sender": "ai"})
        render_message(answer, "ai")

    else:
        # Если нового вопроса нет — просто рендерим историю
        for msg in st.session_state.chat_history:
            render_message(msg["text"], msg["sender"])



# === Руководство пользователя ===
elif st.session_state['page'] == "Руководство пользователя":
    
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(content)
    except FileNotFoundError:
        st.warning("Файл README.md не найден — проверь путь или название файла.")


# === Футер внизу страницы (автор) ===
# Постоянная надпись внизу лево, вне зависимости от содержимого
# st.markdown("""
#     <style>
#         .bottom-right {
#             position: fixed;
#             right: 15px;
#             bottom: 10px;
#             font-size: 0.75em;
#             color: #333333;
#             z-index: 9999;
#         }
#     </style>
#     <div class="bottom-right">© Created by Rahimov M.A. TTU 2025</div>
# """, unsafe_allow_html=True)
