import os
import pytest
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "local" in data["supported_providers"]

def test_cohort_endpoint():
    response = client.get("/api/cohort")
    assert response.status_code == 200
    cohort = response.json()
    assert len(cohort) >= 6
    ids = [p["patient_id"] for p in cohort]
    assert "PT_001" in ids
    assert "PT_002" in ids

def test_guidelines_endpoint():
    response = client.get("/api/guidelines")
    assert response.status_code == 200
    guidelines = response.json()
    assert len(guidelines) >= 5

def test_ontology_endpoint():
    response = client.get("/api/ontology")
    assert response.status_code == 200
    onto = response.json()
    assert onto["axioms_count"] >= 2

def test_upload_pdf():
    pdf_path = os.path.join("data", "sample_reports", "PT_001_Clinical_Lab_Report.pdf")
    assert os.path.exists(pdf_path), "Sample PDF should exist"
    
    with open(pdf_path, "rb") as f:
        response = client.post(
            "/api/upload-pdf",
            files={"file": ("PT_001_Clinical_Lab_Report.pdf", f, "application/pdf")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    extracted = data["data"]
    assert extracted["patient_id"] == "PT_001"
    assert extracted["egfr"] == 22.0
    assert extracted["hba1c"] == 9.2

def test_evaluate_patient_pt001():
    payload = {
        "patient_id": "PT_001",
        "patient_name": "Robert Miller",
        "age": 68,
        "gender": "Male",
        "conditions": ["CKD_Stage_4_5", "Type_2_Diabetes"],
        "lab_values": {"egfr": 22.0, "hba1c": 9.2, "sbp": 138, "dbp": 84},
        "clinical_narrative": "68-year-old male with CKD Stage 4 and Type 2 Diabetes.",
        "provider": "local"
    }
    response = client.post("/api/evaluate", json=payload)
    assert response.status_code == 200
    res = response.json()["result"]
    assert res["status"] == "APPROVED"
    assert "Insulin" in res["recommended_treatment"]
    assert res["verification_details"]["status"] == "PASS"

def test_evaluate_patient_pt002_self_correction():
    payload = {
        "patient_id": "PT_002",
        "patient_name": "Eleanor Vance",
        "age": 72,
        "gender": "Female",
        "conditions": ["CKD_Stage_4_5", "Hypertension"],
        "lab_values": {"egfr": 14.0, "sbp": 165, "dbp": 95},
        "clinical_narrative": "72yo female with CKD Stage 5 and Stage 2 HTN.",
        "provider": "local"
    }
    response = client.post("/api/evaluate", json=payload)
    assert response.status_code == 200
    res = response.json()["result"]
    assert res["status"] == "APPROVED"
    assert "Amlodipine" in res["recommended_treatment"]
    assert res["total_attempts"] == 2
    assert len(res["intercepted_violations"]) > 0

def test_benchmark_endpoint():
    response = client.post("/api/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    bm = data["benchmark"]
    assert bm["metrics"]["overall_safe_rate"] == 100.0
    assert bm["metrics"]["formal_violation_escape_rate"] == 0.0
    assert bm["metrics"]["self_correction_rate"] == 100.0

def test_patient_chat_endpoint():
    payload = {
        "message": "I have Type 2 Diabetes and my eGFR is 22 mL/min. What medicine can I take?",
        "provider": "local"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "reply" in data
    assert "Insulin" in data["reply"] or "Metformin avoided" in data["reply"]
    assert "CKD_Stage_4" in data["detected_conditions"] or "CKD_Stage_4_5" in data["detected_conditions"]

def test_evaluate_pediatric_patient_pt007():
    payload = {
        "patient_id": "PT_007",
        "patient_name": "Leo Martinez",
        "age": 14,
        "gender": "Male",
        "conditions": ["PediatricNephropathy", "Type_1_Diabetes"],
        "lab_values": {
            "egfr": 42.0,
            "sbp": 128,
            "dbp": 82,
            "hba1c": 9.4,
            "creatinine": 1.4
        },
        "clinical_narrative": "14-year-old adolescent with early diabetic nephropathy and Type 1 Diabetes.",
        "provider": "local"
    }
    response = client.post("/api/evaluate", json=payload)
    assert response.status_code == 200
    res = response.json()["result"]
    assert res["status"] == "APPROVED"
    assert "Insulin" in res["recommended_treatment"]
    assert res["icd10_code"] == "E10.21"
    assert res["verification_details"]["status"] == "PASS"

def test_evaluate_patient_pt008_sarah_johnson():
    payload = {
        "patient_id": "PT_008",
        "patient_name": "Sarah Johnson",
        "age": 62,
        "gender": "Female",
        "conditions": ["CKD_Stage_3b", "Type_2_Diabetes", "Hypertension"],
        "lab_values": {
            "egfr": 38.0,
            "uacr": 420.0,
            "hba1c": 8.1,
            "sbp": 146,
            "dbp": 88,
            "creatinine": 1.6,
            "potassium": 4.4
        },
        "clinical_narrative": "62yo female with CKD Stage 3b, Type 2 Diabetes, and Hypertension.",
        "provider": "local"
    }
    response = client.post("/api/evaluate", json=payload)
    assert response.status_code == 200
    res = response.json()["result"]
    assert res["status"] == "APPROVED"
    # Deterministic KDIGO Stage 3b (G3b) and ICD-10 N18.32
    assert res["icd10_code"] == "N18.32"
    assert "Stage 3b" in res["diagnosed_condition"] or "G3b" in res["diagnosed_condition"]
    # Metformin is NOT contraindicated at eGFR 38; dose reduction mandated
    assert "Metformin" in res["recommended_treatment"]
    assert "1000mg" in res["recommended_treatment"] or "dose-reduced" in res["recommended_treatment"].lower()
    # Insulin must not be knee-jerk recommended
    assert "Insulin" not in res["recommended_treatment"]
    # SGLT2i renoprotection mentioned in rationale
    assert "SGLT2" in res["clinical_rationale"] or "SGLT2" in " ".join(res["warnings"])

def test_evaluate_patient_pt009_david_anderson():
    payload = {
        "patient_id": "PT_009",
        "patient_name": "David Anderson",
        "age": 69,
        "gender": "Male",
        "conditions": ["CKD_Stage_4", "Type_2_Diabetes", "Hypertension", "Heart_Failure", "Hyperkalemia"],
        "lab_values": {
            "egfr": 24.0,
            "uacr": 780.0,
            "hba1c": 7.4,
            "sbp": 154,
            "dbp": 92,
            "creatinine": 2.7,
            "potassium": 5.3
        },
        "symptoms": ["Bilateral pedal edema", "Breathlessness on exertion", "Reduced exercise tolerance"],
        "clinical_narrative": "69yo male with Heart Failure, CKD Stage 4, pedal edema, dyspnea, and K+ 5.3 mmol/L.",
        "provider": "local"
    }
    response = client.post("/api/evaluate", json=payload)
    assert response.status_code == 200
    res = response.json()["result"]
    assert res["status"] == "APPROVED"
    # Deterministic Stage 4 (G4) and ICD-10 N18.4
    assert res["icd10_code"] == "N18.4"
    # Critical Hyperkalemia Alert raised
    assert any("Hyperkalemia" in a or "5.3" in a for a in res.get("clinical_alerts", [])) or any("5.3" in w for w in res.get("warnings", []))
    # SGLT2 inhibitor recommended for HF and cardiorenal protection
    assert "SGLT2" in res["recommended_treatment"] or "Empagliflozin" in res["recommended_treatment"]
    # Metformin strictly contraindicated at eGFR 24
    assert any("Metformin" in w for w in res["warnings"])



def test_regression_case_1_pediatric_t1d_single_uacr():
    """
    Regression Test Case 1 (Requirement 8):
    14-year-old with Type 1 Diabetes, HbA1c 9.1%, eGFR 82, UACR 65.
    Verifies:
    1. All original values are preserved (Age 14, eGFR 82, HbA1c 9.1%, UACR 65).
    2. Essential insulin therapy in T1D is never suggested to be replaced by oral antidiabetic agents.
    3. Single elevated UACR (65 mg/g) is NOT automatically classified as confirmed diabetic nephropathy;
       it requires 2 of 3 repeat morning spot tests over 3-6 months.
    4. Evaluates pediatric kidney function (Bedside Schwartz equation) and age-appropriate guidelines.
    5. No blanket claim that all oral antidiabetics are unsafe in children (acknowledges youth >= 10 T2D approvals).
    6. Individual assessment of inquired medications if asked.
    """
    payload = {
        "message": (
            "My 14-year-old child has Type 1 Diabetes, HbA1c 9.1%, eGFR 82, and UACR 65 mg/g. "
            "Can we switch from insulin to Metformin or oral diabetes pills?"
        ),
        "conditions": ["Type_1_Diabetes"],
        "lab_values": {
            "age": 14,
            "hba1c": 9.1,
            "egfr": 82.0,
            "uacr": 65.0
        },
        "provider": "local"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    reply = data["reply"]
    eval_res = data["evaluation"]

    # 1. Preserved original values
    labs = data["detected_labs"]
    assert labs["age"] == 14
    assert labs["hba1c"] == 9.1
    assert labs["egfr"] == 82.0
    assert labs["uacr"] == 65.0

    # 2. Never replace essential insulin in T1D
    assert "Insulin" in reply or "insulin" in reply
    assert "never" in reply.lower() or "life-essential" in reply.lower() or "cannot replace" in reply.lower() or "mandatory" in reply.lower()

    # 3. Single elevated UACR is suspected, not confirmed diabetic nephropathy
    assert "repeat" in reply.lower() or "suspected" in reply.lower() or "3 to 6 months" in reply.lower() or "2 out of 3" in reply.lower()

    # 4. Inquired medication (Metformin) answered individually
    assert "Metformin" in data["inquired_medications"]
    assert "metformin" in reply.lower()

    # 5. No blanket claim; nuances T1D vs T2D
    assert "type 1 diabetes" in reply.lower()


def test_regression_case_2_adult_t2d_ckd4_hf_hyperkalemia_meds():
    """
    Regression Test Case 2 (Requirement 8):
    67-year-old with Type 2 Diabetes, CKD G4, eGFR 22, potassium 5.6, heart failure, Metformin and Lisinopril.
    Verifies:
    1. All original values are preserved (Age 67, eGFR 22, K+ 5.6, HF).
    2. Potassium 5.6 mmol/L flagged as Moderate Hyperkalemia with prompt assessment recommendation (24-72 hours, ECG).
    3. Both Metformin and Lisinopril receive individual assessments:
       - Metformin: Strictly contraindicated at eGFR 22 (< 30) due to lactic acidosis risk.
       - Lisinopril: Identified as contributing to hyperkalemia; managed with dietary review/binders,
         NOT automatically discontinued solely because CKD is advanced.
    4. SGLT2 inhibitor (Empagliflozin 10mg) indicated for cardiorenal and HF protection down to eGFR >= 20.
    5. Loop diuretic indicated for active volume overload.
    """
    payload = {
        "message": (
            "I am a 67-year-old with Type 2 Diabetes, CKD Stage 4, eGFR 22, potassium 5.6, and heart failure with swollen legs. "
            "Are Metformin and Lisinopril safe for me to take?"
        ),
        "conditions": ["Type_2_Diabetes", "CKD_Stage_4", "Heart_Failure"],
        "lab_values": {
            "age": 67,
            "egfr": 22.0,
            "potassium": 5.6,
            "hba1c": 7.4
        },
        "provider": "local"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    reply = data["reply"]
    eval_res = data["evaluation"]

    # 1. Preserved exact lab values
    labs = data["detected_labs"]
    assert labs["age"] == 67
    assert labs["egfr"] == 22.0
    assert labs["potassium"] == 5.6

    # 2. Both Metformin and Lisinopril individually addressed
    assert "Metformin" in data["inquired_medications"]
    assert "Lisinopril" in data["inquired_medications"]
    assert "metformin" in reply.lower()
    assert "lisinopril" in reply.lower()

    # 3. Metformin is contraindicated
    assert "contraindicated" in reply.lower() or "lactic acidosis" in reply.lower() or "unsafe" in reply.lower()

    # 4. Potassium 5.6 hyperkalemia alert & prompt assessment (not delayed)
    assert "5.6" in reply
    assert "hyperkalemia" in reply.lower()
    assert ("24" in reply or "72" in reply or "prompt" in reply.lower() or "ecg" in reply.lower())

    # 5. Lisinopril not permanently stopped solely due to CKD 4; managed carefully
    assert "potassium" in reply.lower()
    assert "kdigo" in reply.lower() or "guideline" in reply.lower()

    # 6. SGLT2 inhibitor / Loop diuretic cardiorenal protection for HF
    assert "sglt2" in reply.lower() or "empagliflozin" in reply.lower() or "diuretic" in reply.lower()


def test_regression_case_3_pt_chat_003_adult_t2d_ckd3b_a3_losartan_metformin():
    """
    Regression Test Case 3 (PT_CHAT_003):
    56-year-old female with Type 2 Diabetes, eGFR 38, Cr 1.7, HbA1c 8.4, BP 142/86, UACR 560, K+ 4.7.
    Current Medications: Metformin 1000 mg twice daily, Losartan 50 mg daily.
    Verifies:
    1. CKD Stage: G3b (Moderate-to-Severe GFR Reduction), ICD-10 N18.32.
    2. Albuminuria Stage: A3 (Severely increased albuminuria), NOT microalbuminuria or A2.
    3. Metformin Assessment: Current dose (1000 mg twice daily = 2000 mg/day) exceeds recommended maximum for CKD G3b.
       Reduce dose to 500 mg twice daily (maximum 1000 mg/day). Safe to continue at reduced dose; not contraindicated.
    4. Losartan Assessment: Appropriate for continuation as first-line renal protection in adult T2D with CKD Stage 3b and A3.
       Monitor serum creatinine and potassium. Do not apply pediatric restrictions.
    5. SGLT2 Inhibitor Recommendation: Strongly recommended (Empagliflozin 10 mg daily) for kidney and cardiovascular protection.
    6. Potassium Status: 4.7 mmol/L (Normal / Normokalemia). No hyperkalemia alert, no emergency protocols.
    7. Zero pediatric rules, zero references to youth diabetes, zero pediatric ACEi/ARB restrictions.
    8. Context Isolation: No contamination from previous tests (Leo Martinez, David Anderson, etc.).
    """
    payload = {
        "message": (
            "I am a 56-year-old female with Type 2 Diabetes, eGFR 38 mL/min/1.73m2, serum creatinine 1.7 mg/dL, "
            "HbA1c 8.4%, blood pressure 142/86 mmHg, UACR 560 mg/g, and serum potassium 4.7 mmol/L. "
            "My current medications are Metformin 1000 mg twice daily and Losartan 50 mg daily. "
            "Are my current medications safe with these lab results, and do I need to change anything?"
        ),
        "conditions": ["Type_2_Diabetes", "CKD_Stage_3b", "Hypertension"],
        "lab_values": {
            "age": 56,
            "gender": "Female",
            "egfr": 38.0,
            "creatinine": 1.7,
            "hba1c": 8.4,
            "sbp": 142,
            "dbp": 86,
            "uacr": 560.0,
            "potassium": 4.7,
            "current_medications": ["Metformin 1000 mg twice daily", "Losartan 50 mg daily"]
        },
        "provider": "local"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    reply = data["reply"]
    eval_res = data["evaluation"]

    # 1. Active patient values preserved canonical
    labs = data["detected_labs"]
    assert labs["age"] == 56
    assert labs["egfr"] == 38.0
    assert labs["potassium"] == 4.7
    assert labs["uacr"] == 560.0
    assert labs["creatinine"] == 1.7
    assert labs["hba1c"] == 8.4

    # 2. CKD Stage G3b and ICD-10 N18.32
    assert eval_res["icd10_code"] == "N18.32"
    assert "Stage 3b" in eval_res["diagnosed_condition"] or "G3b" in eval_res["diagnosed_condition"]

    # 3. Albuminuria Stage A3 (Severely increased), NEVER microalbuminuria or A2
    assert eval_res.get("albuminuria_stage") == "A3"
    assert "microalbuminuria" not in reply.lower()
    assert "a2" not in eval_res.get("albuminuria_stage", "").lower()

    # 4. Potassium 4.7 mmol/L is Normal / Normokalemia (NO false hyperkalemia, NO emergency protocol)
    assert "10.0" not in reply
    assert "severe hyperkalemia" not in reply.lower()
    assert "emergency department" not in reply.lower()
    assert "telemetry" not in reply.lower()
    assert eval_res.get("potassium_status") == "NORMOKALEMIA"

    # 5. Metformin: safe at reduced dose, NOT contraindicated, dose review to max 1000 mg/d
    assert "metformin" in reply.lower()
    assert "1000" in reply
    assert ("not contraindicated" in reply.lower() or "safe" in reply.lower() or "not automatically contraindicated" in reply.lower())

    # 6. Losartan: appropriate continuation with monitoring
    assert "losartan" in reply.lower()
    assert ("continue" in reply.lower() or "continuation" in reply.lower() or "appropriate" in reply.lower())

    # 7. SGLT2 inhibitor recommended
    assert "sglt2" in reply.lower() or "empagliflozin" in reply.lower()

    # 8. ABSOLUTELY NO pediatric rule leakage or stale patient leakage
    forbidden = [
        "pediatric", "teenager", "child", "ispad", "schwartz",
        "youth", "adolescent", "leo martinez", "david anderson"
    ]
    for term in forbidden:
        assert term not in reply.lower(), f"Forbidden term '{term}' leaked into adult patient output!"
