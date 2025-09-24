from google import genai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import json 
import re

# Shared config
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# Gemini client
client = genai.Client(api_key="AIzaSyAmZJ--m8mWPmS_EbqDil4OKm62wqh6SAs")

# Load FAISS index
vectorstore = FAISS.load_local("icd10_faiss_index", embeddings, allow_dangerous_deserialization=True)

# AI Assisted Medical Coding function
def get_medical_info(doctor_notes: str, diagnosis: str = "", lab_report: str = ""):
    # Combine all inputs into a single query
    query_parts = [doctor_notes]
    if diagnosis:
        query_parts.append(f"Diagnosis: {diagnosis}")
    if lab_report:
        query_parts.append(f"Lab Report: {lab_report}")
    query = "\n".join(query_parts)

    # Retrieve relevant knowledge from FAISS (Document + score)
    retriever = vectorstore.similarity_search_with_score(query, k=3)
    context = "\n".join([doc.page_content for doc, _ in retriever])

    # Ask Gemini for ICD-10 code in JSON format
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=f"""
        You are an Expert medical coder.

        Your task: Assign the most accurate and specific ICD-10 code(s) based on the doctor's note and your knowledge of ICD-10.
        Use the provided context from ICD-10 documents to support your reasoning.
        If the context does not contain the exact match, rely on your ICD-10 coding knowledge to provide the most specific code possible.
        If the context is not relevant, respond with:
        {{
            "ICD-10 Code": "",
            "Description": "I'm sorry, I don't have that information."
        }}

        STRICT OUTPUT RULES:
        - Return ONLY a valid JSON object.
        - No backticks, no extra text.
        - Keys must be: "ICD-10 Code", "Description".
        - If multiple diagnoses apply (primary, secondary, complications, comorbidities), include them all in the array.
        - If nothing relevant is found, return an empty array [].

        Output Example:
        [
          {{
            "ICD-10 Code": "A00.1",
            "Description": "Cholera due to Vibrio cholerae 01, biovar eltor"
          }},
          {{
            "ICD-10 Code": "E86.0",
            "Description": "Dehydration"
          }}
        ]

        Context:
        {context}

        Doctor Note (Diagnosis Query):
        {query}
        """
        )

    # Extract model answer safely
    answer_text = response.candidates[0].content.parts[0].text.strip()
    if answer_text.startswith("```"):
        answer_text = re.sub(r"^```[a-zA-Z]*\n?", "", answer_text)
        answer_text = re.sub(r"```$", "", answer_text)
        answer_text = answer_text.strip()

    try:
        return json.loads(answer_text)
    except json.JSONDecodeError:
        return {
            "ICD-10 Code": "",
            "Description": f"Parsing failed, raw output: {answer_text}"
        }
    

# Example usage
# if __name__ == "__main__":
#     # doctor_query = "A 35-year-old man presents with profuse watery diarrhea. Stool culture confirms Vibrio cholerae 01, biovar eltor."
#     doctor_query = "Confirmed Plasmodium falciparum malaria in traveler from Africa presenting with high fever and chills — started on antimalarial therapy and supportive care."
#     lab_notes = "A traveler returning from Africa develops high fever and chills. Lab confirms Plasmodium falciparum malaria."
#     result = get_medical_info(doctor_notes=doctor_query, lab_report=lab_notes)
#     print(result)
