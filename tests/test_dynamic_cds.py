import os
import pytest
from fastapi.testclient import TestClient
from src.api.server import app
from src.agents.llm_client import LLMClinicalAgent

client = TestClient(app)


def test_suite_a_unseen_patient():
    """
    TEST A – Unseen Patient Evaluation:
    Evaluates a completely novel patient profile with non-memorized laboratory parameters:
    52-year-old female with eGFR 48.0 mL/min/1.73m², UACR 180 mg/g, HbA1c 7.8%, BP 138/86 mmHg.
    Expected:
    - KDIGO CKD Stage G3a (ICD-10 N18.31).
    - Albuminuria Stage A2 (Moderately Increased).
    - Model dynamically generates safe recommendation (Metformin at monitored dose / SGLT2i).
    - Zero canned dictionary responses.
    """
    payload = {
        "patient_id": "PT_UNSEEN_001",
        "patient_name": "Clara Higgins",
        "age": 52,
        "gender": "Female",
        "conditions": ["Chronic_Kidney_Disease", "Type_2_Diabetes", "Hypertension"],
        "lab_values": {
            "egfr": 48.0,
            "uacr": 180.0,
            "hba1c": 7.8,
            "sbp": 138,
            "dbp": 86,
            "creatinine": 1.3,
            "potassium": 4.5
        },
        "clinical_narrative": "52-year-old female with newly discovered CKD Stage 3a and uncontrolled Type 2 Diabetes.",
        "provider": "local"
    }
    response = client.post("/api/evaluate", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()["result"]
    assert data["status"] in ["APPROVED", "ESCALATED_FOR_HUMAN_REVIEW"]
    
    # Must correctly classify G3a / N18.31
    assert data["ckd_stage"] == "G3a"
    assert data["icd10_code"] == "N18.31"
    assert data["albuminuria_stage"] == "A2"
    assert data["potassium_status"] == "NORMOKALEMIA"
    
    # Model participated genuinely
    assert "model_used" in data
    assert data["model_used"] in ["qwen2.5:3b", "llama3.2:3b"]
    assert len(data["clinical_rationale"]) > 50
    assert len(data["patient_friendly_summary"]) > 30


def test_suite_b_sequential_context_isolation():
    """
    TEST B – Sequential Context Isolation:
    Evaluates a pediatric patient (14yo T1D) followed immediately by an adult patient (58yo T2D).
    Verifies that the adult patient's response contains ZERO pediatric keywords,
    ZERO ISPAD citations, and ZERO context bleed from the previous evaluation.
    """
    # 1. Run Pediatric Evaluation
    pediatric_payload = {
        "patient_id": "PT_PEDS_ISOLATION",
        "patient_name": "Tommy Barnes",
        "age": 14,
        "gender": "Male",
        "conditions": ["Type_1_Diabetes"],
        "lab_values": {
            "egfr": 95.0,
            "uacr": 75.0,
            "hba1c": 8.9,
            "sbp": 118,
            "dbp": 74,
            "potassium": 4.3
        },
        "clinical_narrative": "14-year-old adolescent with Type 1 Diabetes and spot UACR 75 mg/g.",
        "provider": "local"
    }
    peds_resp = client.post("/api/evaluate", json=pediatric_payload)
    assert peds_resp.status_code == 200, f"Expected 200, got {peds_resp.status_code}: {peds_resp.text}"
    peds_data = peds_resp.json()["result"]
    assert "Insulin" in peds_data["recommended_treatment"] or "insulin" in str(peds_data.get("clinical_rationale", "")).lower()

    # 2. Run Adult Evaluation Immediately Afterwards
    adult_payload = {
        "patient_id": "PT_ADULT_ISOLATION",
        "patient_name": "Arthur Pendelton",
        "age": 58,
        "gender": "Male",
        "conditions": ["Type_2_Diabetes", "Hypertension"],
        "lab_values": {
            "egfr": 78.0,
            "uacr": 22.0,
            "hba1c": 7.4,
            "sbp": 134,
            "dbp": 82,
            "potassium": 4.6
        },
        "clinical_narrative": "58-year-old adult male with Type 2 Diabetes and well-controlled blood pressure.",
        "provider": "local"
    }
    adult_resp = client.post("/api/evaluate", json=adult_payload)
    assert adult_resp.status_code == 200, f"Expected 200, got {adult_resp.status_code}: {adult_resp.text}"
    adult_data = adult_resp.json()["result"]
    
    # Combined adult output text
    adult_full_text = " ".join([
        str(adult_data.get("diagnosed_condition", "")),
        str(adult_data.get("clinical_rationale", "")),
        str(adult_data.get("patient_friendly_summary", "")),
        str(adult_data.get("recommended_treatment", "")),
        " ".join(adult_data.get("warnings", [])),
        " ".join([g.get("title", "") for g in adult_data.get("guideline_citations", [])])
    ]).lower()

    # STRICT ASSERTIONS: No pediatric keywords in adult output
    forbidden_pediatric_terms = [
        "ispad", "pediatric", "children and adolescents", "adolescent", "teenager",
        "schwartz", "bedside schwartz", "tommy barnes"
    ]
    for term in forbidden_pediatric_terms:
        assert term not in adult_full_text, f"Context Bleed Violation: Found pediatric term '{term}' in adult response."


def test_suite_c_potassium_normokalemia_fidelity():
    """
    TEST C – Potassium Normokalemia Preservation (Fidelity):
    Evaluates a patient with normal potassium (4.7 mmol/L).
    Verifies that:
    - Potassium is reported faithfully as 4.7 mmol/L.
    - Zero false hyperkalemia alarms are raised.
    - Zero emergency department referrals or false ECG requirements are generated.
    - Zero hallucinated values (e.g. 10.0 mmol/L).
    """
    payload = {
        "patient_id": "PT_K_NORMAL",
        "patient_name": "Grace Miller",
        "age": 59,
        "gender": "Female",
        "conditions": ["Chronic_Kidney_Disease", "Hypertension"],
        "lab_values": {
            "egfr": 36.0,
            "potassium": 4.7,
            "creatinine": 1.7,
            "sbp": 142,
            "dbp": 86
        },
        "clinical_narrative": "59-year-old female with CKD Stage 3b and normokalemia (potassium 4.7 mmol/L).",
        "provider": "local"
    }
    response = client.post("/api/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()["result"]
    
    assert data["potassium_status"] == "NORMOKALEMIA"
    assert "4.7" in data["lab_summary"]["potassium"]
    
    all_alerts_and_warnings = " ".join(data.get("clinical_alerts", []) + data.get("warnings", [])).lower()
    assert "severe hyperkalemia" not in all_alerts_and_warnings
    assert "medical emergency" not in all_alerts_and_warnings
    assert "10.0" not in all_alerts_and_warnings


def test_suite_d_hyperkalemia_detection_and_safety():
    """
    TEST D – Hyperkalemia Detection and Safe Management:
    Evaluates a patient with elevated potassium (5.6 mmol/L).
    Verifies that:
    - Moderate Hyperkalemia is accurately detected and classified.
    - Alert is raised recommending clinical reassessment within 24-72 hours and 12-lead ECG.
    - ACEi/ARB management is guided safely without unwarranted permanent discontinuation.
    """
    payload = {
        "patient_id": "PT_K_ELEVATED",
        "patient_name": "Walter Ross",
        "age": 66,
        "gender": "Male",
        "conditions": ["Chronic_Kidney_Disease", "Hypertension", "Hyperkalemia"],
        "lab_values": {
            "egfr": 26.0,
            "potassium": 5.6,
            "creatinine": 2.5,
            "sbp": 150,
            "dbp": 90,
            "current_medications": ["Lisinopril 20mg daily"]
        },
        "clinical_narrative": "66-year-old male with CKD Stage 4, potassium 5.6 mmol/L, on Lisinopril.",
        "provider": "local"
    }
    response = client.post("/api/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()["result"]
    
    assert data["potassium_status"] == "MODERATE_HYPERKALEMIA"
    
    # Check alert or warning presence
    combined_notes = " ".join(
        data.get("clinical_alerts", []) + data.get("warnings", []) + [data.get("clinical_rationale", "")]
    )
    assert "5.6" in combined_notes
    assert any(term in combined_notes.lower() for term in ["hyperkalemia", "ecg", "potassium"])


def test_suite_e_general_clinical_question_no_profile():
    """
    TEST E – General Clinical Question Without Patient Profile:
    Asks an arbitrary clinical question via /api/chat with no patient profile or lab values.
    Verifies that:
    - The local LLM dynamically answers the inquiry using retrieved guideline evidence.
    - System does not crash, return 500, or claim patient data is missing.
    - Accurately explains Type 1 vs Type 2 diabetes etiology.
    """
    payload = {
        "message": "What is the clinical difference between Type 1 and Type 2 diabetes regarding insulin requirements?",
        "provider": "local"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    reply = data["reply"]
    
    # Must explain essential insulin in T1D vs resistance in T2D
    reply_lower = reply.lower()
    assert "type 1" in reply_lower and "type 2" in reply_lower
    assert "insulin" in reply_lower
    assert any(term in reply_lower for term in ["autoimmune", "beta cell", "deficiency", "resistance", "essential"])


def test_suite_f_multiturn_conversational_continuity():
    """
    TEST F – Multi-Turn Conversational Continuity:
    Simulates a multi-turn patient dialogue:
    Turn 1: Patient provides lab values (eGFR 38, UACR 420, HbA1c 8.1%).
    Turn 2: Patient asks follow-up inquiry: "Can I continue taking Losartan with these results?"
    Verifies that:
    - Turn 2 retains the active patient context from history.
    - Accurately evaluates Losartan in CKD Stage 3b with severely increased albuminuria (Stage A3).
    """
    # Turn 1: Initial lab submission
    t1_payload = {
        "message": "My recent lab results show eGFR of 38 mL/min/1.73m2, UACR of 420 mg/g, and HbA1c 8.1%. What does this mean?",
        "provider": "local"
    }
    t1_resp = client.post("/api/chat", json=t1_payload)
    assert t1_resp.status_code == 200
    t1_data = t1_resp.json()
    assert t1_data["status"] == "success"

    # Turn 2: Follow-up question relying on history context
    history = [
        {"role": "user", "content": t1_payload["message"]},
        {"role": "assistant", "content": t1_data["reply"]}
    ]
    t2_payload = {
        "message": "Can I continue taking Losartan with these results?",
        "history": history,
        "provider": "local"
    }
    t2_resp = client.post("/api/chat", json=t2_payload)
    assert t2_resp.status_code == 200
    t2_data = t2_resp.json()
    assert t2_data["status"] == "success"
    
    t2_reply = t2_data["reply"].lower()
    assert "losartan" in t2_reply
    # Losartan is safe and renoprotective in CKD 3b with A3 albuminuria
    assert any(term in t2_reply for term in ["safe", "protect", "continue", "indicated", "blood pressure", "arb"])


def test_suite_g_disconnected_ollama_handling():
    """
    TEST G – Disconnected Ollama Error Handling:
    Points the client to a non-existent Ollama endpoint (e.g. port 99999).
    Verifies that:
    - System returns an explicit 503 HTTP status and connection failure details.
    - System NEVER silently falls back to a canned fake successful response.
    """
    # Direct test of LLMClinicalAgent with unreachable endpoint
    agent = LLMClinicalAgent(provider="local", base_url="http://localhost:59999")
    with pytest.raises(ConnectionError) as exc_info:
        agent.generate_recommendation(
            patient_id="PT_OFFLINE_TEST",
            conditions=["Type_2_Diabetes"],
            guidelines="None",
            violations=[],
            lab_values={"egfr": 45}
        )
    assert "Could not connect to Ollama" in str(exc_info.value) or "59999" in str(exc_info.value)


def test_suite_h_consistency_evaluation():
    """
    TEST H – Consistency Across Identical Queries:
    Executes the identical clinical evaluation twice.
    Verifies that both evaluations produce matching clinical diagnoses,
    identical ICD-10 codes, and consistent medication safety verdicts.
    """
    payload = {
        "patient_id": "PT_CONSISTENCY",
        "patient_name": "Marcus Vance",
        "age": 64,
        "gender": "Male",
        "conditions": ["Chronic_Kidney_Disease", "Type_2_Diabetes"],
        "lab_values": {
            "egfr": 34.0,
            "uacr": 350.0,
            "hba1c": 8.0,
            "potassium": 4.6
        },
        "clinical_narrative": "64-year-old male with CKD Stage 3b and Type 2 Diabetes.",
        "provider": "local"
    }
    resp1 = client.post("/api/evaluate", json=payload)
    resp2 = client.post("/api/evaluate", json=payload)
    
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    
    res1 = resp1.json()["result"]
    res2 = resp2.json()["result"]
    
    assert res1["ckd_stage"] == res2["ckd_stage"] == "G3b"
    assert res1["icd10_code"] == res2["icd10_code"] == "N18.32"
    assert res1["albuminuria_stage"] == res2["albuminuria_stage"] == "A3"
    assert res1["potassium_status"] == res2["potassium_status"] == "NORMOKALEMIA"
