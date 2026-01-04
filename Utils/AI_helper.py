import os
import requests
import streamlit as st
from dotenv import load_dotenv

# === Загрузка переменных окружения ===
load_dotenv()
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ API-ключ не найден. Убедитесь, что он задан в .env или secrets.")

# === История диалога и контекст проекта ===
INITIAL_SYSTEM_PROMPT = (
    "Ты профессиональный ассистент по анализу медицинских данных и машинному обучению. "
    "Твоя задача - помогать пользователям понимать их данные и результаты моделей.\n\n"
    "ВАЖНЫЕ ПРАВИЛА:\n"
    "1. Отвечай на русском языке, четко и по существу\n"
    "2. НЕ ПРЕДЛАГАЙ КОД - пользователь работает через интерфейс приложения\n"
    "3. Длина ответа: 3-7 предложений (не слишком кратко, не слишком подробно)\n"
    "4. Фокусируйся на интерпретации и практических советах\n"
    "5. В медицинском контексте обращай внимание на:\n"
    "   - Чувствительность (Recall) - важно не пропускать больных\n"
    "   - Специфичность - важно не пугать здоровых\n"
    "   - Интерпретируемость результатов для врачей\n"
    "6. Используй простой язык, избегай лишних технических терминов\n"
)

chat_history = [
    {
        "role": "system",
        "content": INITIAL_SYSTEM_PROMPT
    }
]
context = {}

def update_context(key, value):
    """Добавление или обновление глобального контекста проекта."""
    context[key] = value

def reset_ai_conversation():
    """
    Полный сброс памяти ИИ (истории и контекста).
    Вызывай при очистке чата или при старте новой сессии.
    """
    global chat_history
    chat_history = [
        {
            "role": "system",
            "content": INITIAL_SYSTEM_PROMPT
        }
    ]
    context.clear()

# === Универсальная функция с учётом контекста ===
def get_chatgpt_response(prompt, model="mistralai/mistral-7b-instruct:free"):
    """Запрос в ИИ с подстановкой глобального контекста."""
    if not prompt or not isinstance(prompt, str):
        return "❌ Пустой или некорректный запрос."

    context_info = "\n".join([f"{k}: {v}" for k, v in context.items()])
    full_prompt = f"Контекст:\n{context_info}\n\n{prompt}, не когда не давай код, просто отвечай на то что просять конкретно, коротко если надо!"

    chat_history.append({"role": "user", "content": full_prompt})

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": chat_history},
            timeout=20
        )
        data = resp.json()
        if "choices" in data and data["choices"]:
            reply = data["choices"][0]["message"]["content"]
            chat_history.append({"role": "assistant", "content": reply})
            return reply
        return f"❌ Пустой ответ от API: {data}"
    except Exception as e:
        return f"❌ Ошибка при запросе: {e}"

def chat_with_context(message, model="mistralai/mistral-7b-instruct:free"):
    """Общение с ИИ с учётом сохранённого контекста (после подключения)."""
    if not message or not isinstance(message, str):
        return "❌ Пустой или некорректный запрос."

    chat_history.append({"role": "user", "content": message})

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": chat_history},
            timeout=20
        )
        data = resp.json()
        if "choices" in data and data["choices"]:
            reply = data["choices"][0]["message"]["content"]
            chat_history.append({"role": "assistant", "content": reply})
            return reply
        return f"❌ Пустой ответ от API: {data}"
    except Exception as e:
        return f"❌ Ошибка при запросе: {e}"

    

def notify_ai_dataset_and_goal(df, user_desc, get_fn=get_chatgpt_response):
    """
    Отправляет в ИИ расширенную информацию о датасете и, при наличии, цель анализа.
    """
    try:
        # === Базовая информация ===
        info = [f"Размер: {df.shape[0]} строк, {df.shape[1]} столбцов"]

        # === Подробности по колонкам ===
        col_details = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = round(missing / len(df) * 100, 2)

            if df[col].dtype in ["int64", "float64"]:
                desc = df[col].describe()
                detail = (
                    f"{col} ({dtype}) → min={desc['min']}, max={desc['max']}, "
                    f"mean={round(desc['mean'],2)}, std={round(desc['std'],2)}, "
                    f"пропуски={missing} ({missing_pct}%)"
                )
            elif df[col].dtype == "object" or df[col].dtype.name == "category":
                uniques = df[col].nunique()
                examples = df[col].dropna().unique()[:3]
                detail = (
                    f"{col} ({dtype}) → {uniques} уникальных значений "
                    f"(примеры: {', '.join(map(str, examples))}), "
                    f"пропуски={missing} ({missing_pct}%)"
                )
            elif "datetime" in dtype:
                min_date, max_date = df[col].min(), df[col].max()
                detail = (
                    f"{col} ({dtype}) → диапазон дат: {min_date} — {max_date}, "
                    f"пропуски={missing} ({missing_pct}%)"
                )
            else:
                detail = f"{col} ({dtype}) → пропуски={missing} ({missing_pct}%)"

            col_details.append(detail)

        info.append("Колонки:\n- " + "\n- ".join(col_details))

        # === Примеры строк (первые 2) ===
        sample_rows = df.head(2).to_dict(orient="records")
        info.append(f"Примеры строк: {sample_rows}")

        # === Формируем сообщение для ИИ ===
        dataset_info = "\n".join(info)
        instruction = (
            "Ты получил информацию о медицинском датасете. "
            "Запомни эту структуру - она будет использоваться для обучения моделей. "
            "В следующих вопросах пользователь может спросить о признаках, метриках или интерпретации. "
            "НЕ предлагай код - только практические советы по анализу и моделированию."
        )

        if user_desc.strip():
            prompt = (
                f"[DATASET STRUCTURE]\n{dataset_info}\n\n"
                f"[ANALYSIS GOAL]\n{user_desc}\n\n"
                f"[INSTRUCTION]\n{instruction}"
            )
            update_context("user_goal", user_desc)
        else:
            prompt = (
                f"[DATASET STRUCTURE]\n{dataset_info}\n\n"
                f"[INSTRUCTION]\n{instruction}"
            )

        update_context("dataset_structure", dataset_info)


        # === Отправляем в ИИ ===
        with st.spinner("📡 Отправляем данные в ИИ..."):
            get_fn(prompt)

        # === Сообщение об успехе ===
        if user_desc.strip():
            return "✅ Учитывая вашу цель, ИИ подключён"
        else:
            return "✅ ИИ подключён"

    except Exception as e:
        return f"❌ Ошибка при отправке данных в ИИ: {e}"


# === Отправка корреляций ===
def send_correlation_to_ai(df):
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return "📉 Недостаточно числовых переменных для корреляции."

    corr = numeric_df.corr().abs().unstack().sort_values(ascending=False)
    corr = corr[corr < 1].drop_duplicates()
    top_corr = corr.head(10)

    formatted_corr = "\n".join([f"{a} и {b}: корреляция {v:.2f}" for (a, b), v in top_corr.items()])
    prompt = f"Топ-10 корреляций между переменными:\n{formatted_corr}"
    return chat_with_context(prompt)

# === Отправка сводной таблицы ===
def send_pivot_to_ai(pivot_df, index_col, value_col, agg_func):
    try:
        if pivot_df is None:
            return "❌ Невозможно отправить пустую сводную таблицу."

        top_rows = pivot_df.head(10).to_dict(orient="records")
        formatted = "\n".join(map(str, top_rows))
        prompt = f"Сводная таблица по {index_col}, агрегируя {value_col} методом {agg_func}:\n{formatted}"
        return chat_with_context(prompt)
    except Exception as e:
        return f"❌ Ошибка при отправке сводной таблицы: {e}"