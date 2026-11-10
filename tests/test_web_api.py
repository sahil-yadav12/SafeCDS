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


