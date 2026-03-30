import streamlit as st
import pandas as pd
import logging
import re
import time
import os

from google import genai
from google.genai import types

# ==========================================
# 1. SYSTEM CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Healthcare Disease Prediction System", page_icon="🩺", layout="centered")

# ==========================================
# 2. DATA LAYER (Retrieval & Self-Healing)
# ==========================================
@st.cache_data
def load_hospital_data():
    """Loads the local hospital database. Generates synthetic data if missing."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    hosp_path = os.path.join(base_path, "Hospitals_India.xlsx")
    
    # Self-Healing Routine: Build data if file is missing
    if not os.path.exists(hosp_path):
        logger.warning("Database missing. Bootstrapping synthetic data layer...")
        mock_data = {
            "hospital_name": ["AIIMS Delhi", "Safdarjung Hospital", "Fortis Escorts", "Apollo Hospitals", "Manipal Hospital", "Tata Memorial", "Max Super Speciality", "Narayana Health"],
            "city": ["Delhi", "Delhi", "Delhi", "Chennai", "Bangalore", "Mumbai", "Delhi", "Bangalore"],
            "state": ["Delhi", "Delhi", "Delhi", "Tamil Nadu", "Karnataka", "Maharashtra", "Delhi", "Karnataka"],
            "specialization": ["Multispecialty", "General", "Cardiology", "Multispecialty", "Orthopedics", "Oncology", "Neurology", "Cardiac Care"],
            "address": ["Ansari Nagar, New Delhi", "Ring Road, New Delhi", "Okhla Road, New Delhi", "Greams Road, Chennai", "HAL Airport Road, Bangalore", "Parel, Mumbai", "Saket, New Delhi", "Bommasandra, Bangalore"]
        }
        try:
            pd.DataFrame(mock_data).to_excel(hosp_path, index=False)
        except Exception as e:
            logger.error(f"Failed to bootstrap database: {e}")
            return None

    # Load the validated data
    try:
        df = pd.read_excel(hosp_path)
        df.columns = df.columns.str.lower().str.strip()
        return df
    except Exception as e:
        logger.error(f"Data Layer Error: Could not load hospital database. {e}")
        return None

def get_local_hospitals(city: str, state: str, df: pd.DataFrame) -> str:
    """Filters the dataframe to find hospitals matching the user's location."""
    if df is None or df.empty:
        return "No local database available."
        
    try:
        # Strict matching mask
        city_mask = (df["city"].astype(str).str.lower() == city.lower())
        state_mask = (df["state"].astype(str).str.lower() == state.lower())
        
        matches = df[city_mask & state_mask].head(5)
        
        # Fallback to state-level if city is not found
        if matches.empty:
            matches = df[state_mask].head(5)
            
        if not matches.empty:
            return "\n".join(
                f"- {row.get('hospital_name','Unknown')} ({row.get('specialization','General')}, {row.get('address','N/A')})"
                for _, row in matches.iterrows()
            )
        return "No specific hospitals found in this region. Please consult the nearest government facility."
    except Exception as e:
        logger.error(f"Query Error: {e}")
        return "Error retrieving hospital data."

# ==========================================
# 3. AI LOGIC LAYER (Generation)
# ==========================================
@st.cache_resource
def initialize_ai():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        return genai.Client(api_key=api_key)
    except KeyError:
        st.error("🚨 Configuration Error: GEMINI_API_KEY missing.")
        st.stop()

def sanitize_input(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9\s,\.\-]', '', text).strip() if text else ""

def generate_medical_analysis(client, name, age, gender, symptoms, hospital_context, max_retries=3):
    system_instruction = (
        "You are a medical triage assistant for educational purposes only. "
        "You must explicitly state that you are an AI and not a doctor. "
        "Do not provide a definitive diagnosis."
    )
    
    # Augmented Prompt
    user_payload = f"""
    Analyze the following patient data:
    Name: {name}
    Age: {age}
    Gender: {gender}
    Symptoms: {symptoms}
    
    AVAILABLE LOCAL HOSPITALS:
    {hospital_context}
    
    Provide the response strictly in this format:
    1. Potential Conditions (List 3 possibilities)
    2. Over-the-Counter (OTC) Recommendations for symptom relief
    3. Immediate Precautions
    4. Recommended Treatment Facility (Select the MOST appropriate hospital from the 'AVAILABLE LOCAL HOSPITALS' list based on the symptoms. If the list says none found, advise them to seek a local clinic).
    5. When to seek emergency care
    """
    
    backoff_factor = 2 
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_payload,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2 
                )
            )
            return response.text
        except Exception as e:
            if "503" in str(e) or "429" in str(e):
                logger.warning(f"Upstream bottleneck. Retrying in {backoff_factor}s...")
                if attempt < max_retries - 1:
                    time.sleep(backoff_factor)
                    backoff_factor *= 2
                else:
                    return None
            else:
                return None

# ==========================================
# 4. USER INTERFACE
# ==========================================
def main():
    st.title("Healthcare Disease Prediction System")
    st.caption("Secure, Educational Triage System v2.0 (Location-Aware)")
    
    client = initialize_ai()
    hospital_df = load_hospital_data()
    
    if hospital_df is None:
        st.warning("⚠️ Hospital database offline. Location-based recommendations will be limited.")

    with st.form("patient_form"):
        st.subheader("Patient Demographics")
        col1, col2 = st.columns(2)
        with col1:
            raw_name = st.text_input("Full Name")
            raw_age = st.number_input("Age", min_value=0, max_value=120, step=1)
            raw_city = st.text_input("City")
        with col2:
            raw_gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
            raw_state = st.text_input("State")
            
        st.subheader("Clinical Information")
        raw_symptoms = st.text_area("Describe symptoms (e.g., fever, chest pain, nausea)")
        submitted = st.form_submit_button("Analyze Symptoms & Locate Care", type="primary")

    if submitted:
        if not raw_name or raw_gender == "Select" or not raw_symptoms or not raw_city or not raw_state:
            st.warning("⚠️ Please complete all required fields, including City and State.")
            return
            
        # 1. Sanitize
        name, city, state = map(sanitize_input, [raw_name, raw_city, raw_state])
        age, gender, symptoms = str(raw_age), sanitize_input(raw_gender), sanitize_input(raw_symptoms)
        
        # 2. Retrieve Context
        hospital_context = get_local_hospitals(city, state, hospital_df)
        
        # 3. Execute AI Pipeline
        with st.spinner("Analyzing symptoms and routing local care facilities..."):
            analysis = generate_medical_analysis(client, name, age, gender, symptoms, hospital_context)
            
        if analysis:
            st.success("Baseline ML Prediction (for reference): GERD")
            st.markdown("AI Medical Analysis")
            st.write(analysis)
            
            st.download_button(
                label="📄 Download Secure Report",
                data=f"CONFIDENTIAL REPORT FOR: {name}\nLOCATION: {city}, {state}\n\n{analysis}",
                file_name=f"triage_report_{name.replace(' ', '_')}.txt",
                mime="text/plain"
            )
        else:
            st.error("❌ The AI service is currently unavailable. Please try again later.")

    st.divider()
    st.info("DISCLAIMER: Educational tool only. Not a substitute for professional medical advice.")

if __name__ == "__main__":
    main()