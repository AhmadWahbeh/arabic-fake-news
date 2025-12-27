import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import plotly.graph_objects as go
from datetime import datetime

# ==============================================
# CONFIGURATION
# ==============================================
CONFIDENCE_THRESHOLD = 0.8
MODEL_PATH = "AhmadDS04/arabert-fake-news"

st.set_page_config(
    page_title="Arabic Fake News Detection System",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==============================================
# ENHANCED PROFESSIONAL CSS
# ==============================================
st.markdown("""
<style>
    /* ===== Global App Foundation ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f2e 50%, #0f1419 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        max-width: 900px !important;
    }
    
    /* ===== Header & Branding ===== */
    .app-header {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(168, 85, 247, 0.05) 100%);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 50%, #c7d2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        padding: 0;
        letter-spacing: -1px;
        line-height: 1.2;
    }
    
    .subtitle {
        color: #9ca3af;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 0.75rem;
        letter-spacing: 0.3px;
        line-height: 1.6;
    }
    
    .subtitle-ar {
        font-family: 'Tahoma', 'Arial', sans-serif;
        direction: rtl;
        font-size: 1.05rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }
    
    /* ===== Input Section ===== */
    .input-section {
        background: linear-gradient(145deg, #1a1f2e 0%, #252a3d 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .input-label {
        display: block;
        font-size: 1rem;
        font-weight: 700;
        color: #f3f4f6;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .input-label-ar {
        font-family: 'Tahoma', 'Arial', sans-serif;
        direction: rtl;
        font-weight: 600;
        font-size: 1.05rem;
    }
    
    /* Enhanced Text Area */
    .stTextArea > label {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #f9fafb !important;
        margin-bottom: 1rem !important;
        letter-spacing: 0.3px !important;
    }
    
    .stTextArea textarea {
        font-size: 1.15rem !important;
        line-height: 2 !important;
        font-family: 'Tahoma', 'Arial', 'Segoe UI', sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
        background: #1f2937 !important;
        color: #f9fafb !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        border: 2px solid #374151 !important;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        resize: vertical !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15), inset 0 2px 8px rgba(0, 0, 0, 0.2) !important;
        outline: none !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #6b7280 !important;
        opacity: 0.7 !important;
        font-style: italic !important;
    }
    
    /* ===== Button Styling ===== */
    .stButton {
        margin-top: 1.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 1rem 2.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        width: 100% !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        text-transform: uppercase !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(139, 92, 246, 0.45) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3) !important;
    }
    
    /* ===== Results Section ===== */
    .results-container {
        margin-top: 3rem;
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Decision Cards */
    .decision-card {
        background: linear-gradient(145deg, #1f2937 0%, #111827 100%);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin: 2rem 0;
        border: 2px solid;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .decision-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, transparent, currentColor, transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .decision-card-real {
        border-color: #10b981;
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.05) 0%, rgba(5, 150, 105, 0.05) 100%);
    }
    
    .decision-card-real::before {
        color: #10b981;
    }
    
    .decision-card-fake {
        border-color: #ef4444;
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.05) 0%, rgba(220, 38, 38, 0.05) 100%);
    }
    
    .decision-card-fake::before {
        color: #ef4444;
    }
    
    .decision-card-uncertain {
        border-color: #f59e0b;
        background: linear-gradient(145deg, rgba(245, 158, 11, 0.05) 0%, rgba(217, 119, 6, 0.05) 100%);
    }
    
    .decision-card-uncertain::before {
        color: #f59e0b;
    }
    
    .decision-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
        display: block;
        text-align: center;
        filter: drop-shadow(0 4px 8px currentColor);
    }
    
    .decision-title {
        font-size: 2.25rem;
        font-weight: 900;
        text-align: center;
        margin: 0.5rem 0;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    
    .decision-title-real {
        color: #10b981;
        text-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
    }
    
    .decision-title-fake {
        color: #ef4444;
        text-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
    }
    
    .decision-title-uncertain {
        color: #f59e0b;
        text-shadow: 0 0 30px rgba(245, 158, 11, 0.3);
    }
    
    .decision-subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #9ca3af;
        margin-top: 0.5rem;
        font-family: 'Tahoma', 'Arial', sans-serif;
        direction: rtl;
        font-weight: 500;
    }
    
    /* ===== Score Cards ===== */
    .score-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .score-card {
        background: linear-gradient(145deg, #1f2937 0%, #111827 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .score-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    .score-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6b7280;
        margin-bottom: 0.75rem;
    }
    
    .score-value {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .score-value-confidence {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .score-value-real {
        color: #10b981;
    }
    
    .score-value-fake {
        color: #ef4444;
    }
    
    /* ===== Section Headers ===== */
    .section-header {
        font-size: 1.25rem;
        font-weight: 800;
        color: #f3f4f6;
        margin: 2.5rem 0 1.5rem 0;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        padding-bottom: 1rem;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, transparent, #6366f1, transparent);
        border-radius: 2px;
    }
    
    /* ===== Warnings & Alerts ===== */
    .stAlert {
        background: linear-gradient(145deg, #1f2937 0%, #111827 100%) !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 12px !important;
        padding: 1.25rem !important;
        color: #fbbf24 !important;
        margin: 1.5rem 0 !important;
    }
    
    /* ===== Footer ===== */
    .app-footer {
        margin-top: 4rem;
        padding: 2rem 1rem;
        text-align: center;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        color: #6b7280;
        font-size: 0.875rem;
    }
    
    .app-footer p {
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    .footer-highlight {
        color: #8b5cf6;
        font-weight: 600;
    }
    
    /* ===== Loading Spinner ===== */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
        border-right-color: #8b5cf6 !important;
    }
    
    /* ===== Plotly Chart Styling ===== */
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
    }
    
    /* ===== Divider ===== */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        margin: 2rem 0;
    }
    
    /* ===== Responsive Design ===== */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        
        .score-grid {
            grid-template-columns: 1fr;
        }
        
        .decision-title {
            font-size: 1.75rem;
        }
    }
    
    /* ===== Accessibility ===== */
    *:focus {
        outline: 2px solid #6366f1;
        outline-offset: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================
# MODEL LOADING (CACHED)
# ==============================================
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the pre-trained AraBERT model and create a classification pipeline.
    Cached to prevent reloading on every interaction.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

        # Configure label mapping
        model.config.id2label = {0: "Real", 1: "Fake"}
        model.config.label2id = {"Real": 0, "Fake": 1}

        # Create classification pipeline
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True
        )
        
        return classifier
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

# ==============================================
# CONFIDENCE GAUGE VISUALIZATION
# ==============================================
def create_confidence_gauge(confidence_score, decision_type):
    """
    Create an elegant, professional confidence gauge using Plotly.
    
    Args:
        confidence_score: Float between 0 and 1
        decision_type: 'real', 'fake', or 'uncertain'
    
    Returns:
        Plotly figure object
    """
    # Color mapping based on decision type
    color_map = {
        'real': '#10b981',
        'fake': '#ef4444',
        'uncertain': '#f59e0b'
    }
    
    gauge_color = color_map.get(decision_type, '#6366f1')
    confidence_percent = confidence_score * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={
            'suffix': "%",
            'font': {
                'size': 56,
                'color': gauge_color,
                'family': 'Inter, sans-serif',
                'weight': 900
            },
            'valueformat': '.1f'
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': '#374151',
                'tickfont': {'size': 13, 'color': '#9ca3af', 'family': 'Inter'}
            },
            'bar': {
                'color': gauge_color,
                'thickness': 0.75,
                'line': {'width': 0}
            },
            'bgcolor': '#1f2937',
            'borderwidth': 3,
            'bordercolor': '#374151',
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},
                {'range': [50, 80], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
            ],
            'threshold': {
                'line': {'color': '#f9fafb', 'width': 4},
                'thickness': 0.8,
                'value': CONFIDENCE_THRESHOLD * 100
            }
        },
        title={
            'text': "Confidence Level",
            'font': {'size': 18, 'color': '#e5e7eb', 'family': 'Inter', 'weight': 700}
        }
    ))
    
    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=80, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#f9fafb", 'family': "Inter, sans-serif"},
        annotations=[
            dict(
                text=f"Decision Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%",
                x=0.5,
                y=-0.15,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=13, color="#6b7280", family="Inter")
            )
        ]
    )
    
    return fig

# ==============================================
# MAIN APPLICATION
# ==============================================
def main():
    """Main application logic"""
    
    # Load the model
    with st.spinner("🔄 Loading AI model..."):
        classifier = load_model()
    
    if classifier is None:
        st.error("❌ Unable to initialize the application. Please check the model files.")
        return
    
    # ==============================================
    # HEADER SECTION
    # ==============================================
    st.markdown("""
    <div class='app-header'>
        <h1 class='main-title'>🔍 Arabic Fake News Detection</h1>
        <p class='subtitle'>AI-Powered Misinformation Detection System</p>
        <p class='subtitle-ar'>نظام كشف الأخبار المزيفة بالذكاء الاصطناعي</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ==============================================
    # INPUT SECTION
    # ==============================================
    st.markdown("<div class='section-header'>📝 Input Analysis</div>", unsafe_allow_html=True)
    
    # Text input area with clear instructions
    news_text = st.text_area(
        "Enter Arabic news text for analysis | أدخل النص الإخباري العربي للتحليل",
        height=280,
        placeholder="مثال: أعلنت وزارة الصحة اليوم عن اكتشاف علاج جديد يقضي على فيروس كورونا بنسبة 100٪ خلال 24 ساعة فقط...\n\n"
                    ,
        help="Paste or type Arabic news text here. Minimum 20 characters recommended for accurate analysis.",
        key="news_text_input"
    )
    
    # Character count indicator
    char_count = len(news_text.strip())
    if char_count > 0:
        st.caption(f"📊 Character count: {char_count} | عدد الأحرف: {char_count}")
    
    # Analysis button
    analyze_clicked = st.button("🔍 Analyze Text | تحليل النص")
    
    # ==============================================
    # ANALYSIS LOGIC
    # ==============================================
    if analyze_clicked:
        # Input validation
        if char_count < 10:
            st.warning("⚠️ **Input too short** | النص قصير جداً\n\nPlease provide at least 10 characters for meaningful analysis. | الرجاء إدخال 10 أحرف على الأقل للحصول على تحليل دقيق.")
        else:
            # Show processing state
            with st.spinner("🔄 Analyzing text... Please wait | جاري التحليل... يرجى الانتظار"):
                try:
                    # Get model predictions
                    outputs = classifier(news_text)[0]
                    
                    # Extract probabilities
                    prob_real = outputs[0]["score"]
                    prob_fake = outputs[1]["score"]
                    confidence = max(prob_real, prob_fake)
                    
                    # Determine decision based on confidence threshold
                    if confidence < CONFIDENCE_THRESHOLD:
                        decision = "uncertain"
                        decision_en = "Uncertain — Requires Review"
                        decision_ar = "غير مؤكد — يحتاج إلى مراجعة"
                        emoji = "⚠️"
                        card_class = "decision-card-uncertain"
                        title_class = "decision-title-uncertain"
                    elif prob_fake > prob_real:
                        decision = "fake"
                        decision_en = "Fake News Detected"
                        decision_ar = "تم اكتشاف خبر مزيف"
                        emoji = "🚨"
                        card_class = "decision-card-fake"
                        title_class = "decision-title-fake"
                    else:
                        decision = "real"
                        decision_en = "Real News Detected"
                        decision_ar = "تم اكتشاف خبر حقيقي"
                        emoji = "✅"
                        card_class = "decision-card-real"
                        title_class = "decision-title-real"
                    
                    # ==============================================
                    # RESULTS DISPLAY
                    # ==============================================
                    st.markdown("<div class='results-container'>", unsafe_allow_html=True)
                    
                    # Custom divider
                    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                    
                    # Decision card
                    st.markdown(f"""
                    <div class='decision-card {card_class}'>
                        <span class='decision-icon'>{emoji}</span>
                        <h2 class='decision-title {title_class}'>{decision_en}</h2>
                        <p class='decision-subtitle'>{decision_ar}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence gauge section
                    st.markdown("<div class='section-header'>📊 Confidence Analysis</div>", unsafe_allow_html=True)
                    
                    # Display the gauge
                    gauge_fig = create_confidence_gauge(confidence, decision)
                    st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})
                    
                    # Score breakdown
                    st.markdown("<div class='section-header'>🔢 Detailed Scores</div>", unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class='score-card'>
                            <div class='score-label'>Confidence</div>
                            <div class='score-value score-value-confidence'>{confidence:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class='score-card'>
                            <div class='score-label'>Real Score</div>
                            <div class='score-value score-value-real'>{prob_real:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class='score-card'>
                            <div class='score-label'>Fake Score</div>
                            <div class='score-value score-value-fake'>{prob_fake:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Interpretation guidance
                    st.markdown("<div class='section-header'>💡 Interpretation Guide</div>", unsafe_allow_html=True)
                    
                    if decision == "uncertain":
                        st.info("""
                        **⚠️ Low Confidence Alert**
                        
                        The model's confidence is below the threshold (80%). This could indicate:
                        - Ambiguous or mixed content patterns
                        - Text characteristics falling between fake and real news patterns
                        - Unusual writing style or structure
                        
                        **Recommendation:** Exercise caution and verify through additional trusted sources.
                        """)
                    elif decision == "fake":
                        st.warning("""
                        **🚨 Fake News Indicators Detected**
                        
                        The analysis suggests patterns commonly associated with misinformation:
                        - Sensationalized language or claims
                        - Lack of credible source attribution
                        - Emotional manipulation tactics
                        
                        **Recommendation:** Verify claims through official sources and fact-checking organizations.
                        """)
                    else:
                        st.success("""
                        **✅ Authentic News Indicators Detected**
                        
                        The analysis suggests patterns commonly associated with legitimate news:
                        - Balanced and objective language
                        - Credible source references
                        - Factual presentation style
                        
                        **Note:** Always maintain critical thinking and cross-reference important information.
                        """)
                    
                    # Timestamp
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.caption(f"🕒 Analysis completed at: {current_time}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ **Analysis Error**\n\nAn error occurred during processing: {str(e)}\n\nPlease try again or contact support if the issue persists.")
    
    # ==============================================
    # FOOTER SECTION
    # ==============================================
    st.markdown("""
    <div class='app-footer'>
        <p>🤖 <span class='footer-highlight'>Powered by AraBERT Transformer Model</span></p>
        <p>Confidence Threshold: <strong>80%</strong> | عتبة الثقة: <strong>80%</strong></p>
        <p style='margin-top: 1rem; font-size: 0.8rem; color: #4b5563;'>
            This is an assistive AI tool for educational and research purposes. 
            Results should be verified through multiple reliable sources.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================
# APPLICATION ENTRY POINT
# ==============================================
if __name__ == "__main__":

    main()

