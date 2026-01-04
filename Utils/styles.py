import streamlit as st
import time

def apply_custom_styles():
    """Applies global custom styles for the application."""
    st.markdown("""
        <style>
            :root {
                --bg1: #0b0f19;
                --bg2: #1a2238;
                --bg3: #243b55;
                --accent: #8ab6ff;
                --accent2: #b6d6ff;
            }
            /* Global Streamlit Cleanups */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
            }
            div.stButton > button {
                background-color: #f0f2f6;
                color: black;
                border: 1px solid #ccc;
                border-radius: 6px;
                transition: all 0.2s ease;
            }
            div.stButton > button:hover {
                background-color: #e0f0ff;
                color: #007BFF;
                border: 1px solid #007BFF;
                transform: scale(1.02);
            }
        </style>
    """, unsafe_allow_html=True)

def show_splash_screen():
    """Shows the animated splash screen if the app hasn't loaded yet."""
    if "app_loaded" not in st.session_state:
        # Hide sidebar during splash
        st.markdown("""
            <style>
                [data-testid="stSidebar"] {
                    display: none;
                }
                .splash-root {
                    position: fixed;
                    inset: 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: linear-gradient(120deg, #0b0f19, #1a2238, #243b55);
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
                    animation: pulse 2.5s infinite;
                }
                @keyframes pulse {
                    0%,100% { transform: scale(1); }
                    50% { transform: scale(1.1); }
                }
                .splash-title {
                    font-size: 2.2em;
                    font-weight: 700;
                    margin-bottom: 10px;
                    border-right: 2px solid #8ab6ff;
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
                .fade-out {
                    opacity: 0;
                    pointer-events: none;
                    transition: opacity 0.8s ease;
                }
            </style>
            
            <div class="splash-root" id="splash">
                <div class="splash-card">
                    <div class="splash-icon">🧬</div>
                    <div class="splash-title">Medical ML Modeling</div>
                    <div style="margin-top: 10px; color: #d2dbff;">Логистическая регрессия • CatBoost • Интерпретация</div>
                </div>
            </div>
            
            <script>
                const splash = document.getElementById("splash");
                setTimeout(() => {
                    splash.classList.add("fade-out");
                    setTimeout(() => splash.remove(), 900);
                }, 3500);
            </script>
        """, unsafe_allow_html=True)
        time.sleep(4)
        st.session_state.app_loaded = True
        st.rerun()
