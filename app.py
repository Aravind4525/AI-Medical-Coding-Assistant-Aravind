from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from code_medical import get_medical_info

class MedicalQuery(BaseModel):
    doctor_notes: str
    diagnosis: Optional[str] = ""
    lab_report: Optional[str] = ""

app = FastAPI(title="AI Medical Coding API", version="1.0")

@app.get("/")
def read_root():
    return {"message": "AI ICD-10 Coding API is running!"}

@app.post("/predict")
def predict_icd10(query: MedicalQuery):
    result = get_medical_info(
        doctor_notes=query.doctor_notes,
        diagnosis=query.diagnosis,
        lab_report=query.lab_report
    )
    return {"ICD10_Codes": result}
