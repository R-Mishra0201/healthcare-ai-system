import pandas as pd
import os

def build_synthetic_database():
    """Generates a mock hospital database for local testing."""
    print("Initializing synthetic data generation...")
    
    mock_data = {
        "hospital_name": [
            "AIIMS Delhi", "Safdarjung Hospital", "Fortis Escorts", 
            "Apollo Hospitals", "Manipal Hospital", "Tata Memorial", 
            "Max Super Speciality", "Narayana Health"
        ],
        "city": [
            "Delhi", "Delhi", "Delhi", 
            "Chennai", "Bangalore", "Mumbai", 
            "Delhi", "Bangalore"
        ],
        "state": [
            "Delhi", "Delhi", "Delhi", 
            "Tamil Nadu", "Karnataka", "Maharashtra", 
            "Delhi", "Karnataka"
        ],
        "specialization": [
            "Multispecialty", "General", "Cardiology", 
            "Multispecialty", "Orthopedics", "Oncology", 
            "Neurology", "Cardiac Care"
        ],
        "address": [
            "Ansari Nagar, New Delhi", "Ring Road, New Delhi", "Okhla Road, New Delhi", 
            "Greams Road, Chennai", "HAL Airport Road, Bangalore", "Parel, Mumbai", 
            "Saket, New Delhi", "Bommasandra, Bangalore"
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(mock_data)
    
    # Resolve absolute path to ensure it drops next to app.py
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "Hospitals_India.xlsx")
    
    try:
        # Requires openpyxl installed in your venv
        df.to_excel(file_path, index=False)
        print(f"✅ SUCCESS: Synthetic database deployed to {file_path}")
    except ModuleNotFoundError:
        print("❌ ERROR: 'openpyxl' is missing. Run: pip install openpyxl")
    except Exception as e:
        print(f"❌ ERROR: Failed to write file. {e}")

if __name__ == "__main__":
    build_synthetic_database()
