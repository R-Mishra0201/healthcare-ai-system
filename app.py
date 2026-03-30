import streamlit as st
import logging
import re
import time

# IMPORT THE NEW SDK
from google import genai
from google.genai import types

# ==========================================
# 1. SYSTEM CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Enterprise Healthcare AI",
    page_icon="🩺",
    layout="centered"
)

# ==========================================
# 2. CORE FUNCTIONS
# ==========================================
@st.cache_resource
def initialize_ai():
    """Securely initializes the modern Gemini AI Client."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        # The new architecture relies on a Client object
        client = genai.Client(api_key=api_key)
        return client
    except KeyError:
        st.error("🚨 Configuration Error: GEMINI_API_KEY not found in secrets.toml.")
        st.stop()
    except Exception as e:
        logger.error(f"AI Initialization Failure: {e}")
        st.error("🚨 Critical System Failure: Could not initialize AI client.")
        st.stop()

def sanitize_input(text: str) -> str:
    """Basic sanitization to strip dangerous characters from prompts."""
    if not text:
        return ""
    return re.sub(r'[^a-zA-Z0-9\s,\.\-]', '', text).strip()

def generate_medical_analysis(client, name, age, gender, symptoms, max_retries=3):
    """Executes the AI prompt using the new client routing with fault tolerance."""
    system_instruction = (
        "You are a medical triage assistant for educational purposes only. "
        "You must explicitly state that you are an AI and not a doctor. "
        "Do not provide a definitive diagnosis."
    )
    
    user_payload = f"""
    Analyze the following patient data:
    Name: {name}
    Age: {age}
    Gender: {gender}
    Symptoms: {symptoms}
    
    Provide the response strictly in this format:
    1. Potential Conditions (List 3 possibilities)
    2. Over-the-Counter (OTC) Recommendations for symptom relief
    3. Immediate Precautions
    4. When to seek emergency care
    """
    
    backoff_factor = 2 # Start with a 2-second delay
    
    for attempt in range(max_retries):
        try:
            # The new SDK uses client.models.generate_content
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_payload,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2 # Lower temperature for medical data stability
                )
            )
            return response.text
            
        except Exception as e:
            error_str = str(e)
            # Catch transient network errors (503 Service Unavailable or 429 Rate Limit)
            if "503" in error_str or "429" in error_str:
                logger.warning(f"Upstream bottleneck (Attempt {attempt + 1}/{max_retries}). Retrying in {backoff_factor} seconds...")
                if attempt < max_retries - 1:
                    time.sleep(backoff_factor)
                    backoff_factor *= 2 # Double the wait time
                else:
                    logger.error("Max retries exhausted. Upstream provider is completely down.")
                    return None
            else:
                logger.error(f"Fatal Generation Error: {e}")
                return None

# ==========================================
# 3. USER INTERFACE
# ==========================================
def main():
    st.title("🩺 AI Health Analysis Portal")
    st.caption("Secure, Educational Triage System v1.1")
    
    model = initialize_ai()
    
    # Form Container
    with st.form("patient_form"):
        st.subheader("Patient Demographics")
        col1, col2 = st.columns(2)
        
        with col1:
            raw_name = st.text_input("Full Name")
            raw_age = st.number_input("Age", min_value=0, max_value=120, step=1)
        
        with col2:
            raw_gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
            
        st.subheader("Clinical Information")
        raw_symptoms = st.text_area("Describe symptoms (e.g., fever, headache, nausea)")
        
        submitted = st.form_submit_button("Analyze Symptoms", type="primary")

    # ==========================================
    # 4. EXECUTION PIPELINE
    # ==========================================
    if submitted:
        # Validation
        if not raw_name or raw_gender == "Select" or not raw_symptoms:
            st.warning("⚠️ Please complete all required fields before submission.")
            return
            
        # Sanitization
        name = sanitize_input(raw_name)
        age = str(raw_age)
        gender = sanitize_input(raw_gender)
        symptoms = sanitize_input(raw_symptoms)
        
        # Processing
        with st.spinner("Executing AI Analysis Protocol... (This may take a moment if servers are busy)"):
            analysis = generate_medical_analysis(model, name, age, gender, symptoms)
            
        if analysis:
            st.success("Analysis Complete.")
            st.markdown("### 📋 Triage Report")
            st.write(analysis)
            
            # Downloadable Artifact
            st.download_button(
                label="📄 Download Secure Report",
                data=f"CONFIDENTIAL REPORT FOR: {name}\n\n{analysis}",
                file_name=f"triage_report_{name.replace(' ', '_')}.txt",
                mime="text/plain"
            )
        else:
            st.error("❌ The AI service is currently unavailable due to high demand. Please try again later.")

    st.divider()
    st.info("DISCLAIMER: This tool generates educational information using artificial intelligence. It is not a substitute for professional medical advice, diagnosis, or treatment.")

if __name__ == "__main__":
    main()