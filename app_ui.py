import streamlit as st
import requests
import pandas as pd

# ------------------ Streamlit Page Setup ------------------
st.set_page_config(page_title="AI Medical Coding Assistant", page_icon="🩺", layout="wide")
st.title("🩺 AI Medical Coding Assistant")

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ Tabs ------------------
tab1, tab2 = st.tabs(["New Prediction", "History"])

# ------------------ New Prediction Tab ------------------
with tab1:
    doctor_notes = st.text_area("Doctor Notes", height=150)
    diagnosis = st.text_input("Diagnosis (Optional)")
    lab_report = st.text_area("Lab Report (Optional)", height=100)

    if st.button("Get ICD-10 Codes"):
        if not doctor_notes.strip():
            st.warning("Please enter doctor notes before requesting ICD-10 codes.")
        else:
            payload = {
                "doctor_notes": doctor_notes,
                "diagnosis": diagnosis,
                "lab_report": lab_report
            }
            try:
                response = requests.post("http://127.0.0.1:8000/predict", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    codes = result.get("ICD10_Codes", [])

                    if codes:
                        st.success("ICD-10 Codes Retrieved!")

                        # Ensure results are always a DataFrame
                        df = pd.DataFrame(codes) if isinstance(codes, list) else pd.DataFrame([codes])
                        st.dataframe(df, use_container_width=True)

                        # Save to session history
                        st.session_state.history.append({
                            "Doctor Notes": doctor_notes,
                            "Diagnosis": diagnosis,
                            "Lab Report": lab_report,
                            "Results": df.copy()
                        })

                        # Download current result as CSV
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download This Result (CSV)",
                            data=csv,
                            file_name="icd10_codes.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No ICD-10 codes found for the input.")
                else:
                    st.error(f"Error from API: {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

# ------------------ History Tab ------------------
with tab2:
    st.subheader("Previous Predictions")
    
    if st.session_state.history:
        for i, record in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"### 🔹 Case {i}")
            st.markdown(f"**Doctor Notes:** {record['Doctor Notes']}")
            if record['Diagnosis']:
                st.markdown(f"**Diagnosis:** {record['Diagnosis']}")
            if record['Lab Report']:
                st.markdown(f"**Lab Report:** {record['Lab Report']}")
            st.dataframe(record["Results"], use_container_width=True)
            st.markdown("---")

        # Combine all results for full history download
        all_results = []
        for record in st.session_state.history:
            df_temp = record["Results"].copy()
            df_temp["Doctor Notes"] = record["Doctor Notes"]
            df_temp["Diagnosis"] = record["Diagnosis"]
            df_temp["Lab Report"] = record["Lab Report"]
            all_results.append(df_temp)

        if all_results:
            df_all = pd.concat(all_results, ignore_index=True)
            csv_all = df_all.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full History (CSV)",
                data=csv_all,
                file_name="icd10_history.csv",
                mime="text/csv"
            )
    else:
        st.info("No history yet.")
