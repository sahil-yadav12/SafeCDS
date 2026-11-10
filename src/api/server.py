import os
import json
import time
import shutil
import uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.reasoning.verifier import ClinicalVerifier
from src.retrieval.hybrid_retriever import HybridClinicalRetriever
from src.retrieval.pdf_parser import ClinicalPDFParser
from src.agents.llm_client import LLMClinicalAgent
from src.agents.workflow import build_safecds_graph
from src.reasoning.clinical_rules import (
    extract_inquired_medications,
    evaluate_individual_medication,
    evaluate_inquired_medications
)
from src.models.patient_context import PatientContext, format_lab_value
from src.evaluation.run_benchmark import run_cohort_benchmark

ONTOLOGY_PATH = os.path.join("ontologies", "cardiometabolic_core.owl")
GUIDELINES_PATH = os.path.join("data", "guidelines.json")
COHORT_PATH = os.path.join("data", "cohort.json")

# Initialize app
app = FastAPI(
    title="SafeCDS: Neuro-Symbolic Agentic RAG Clinical Decision Support",
    description="Locally grounded, formally verified clinical decision support system for cardiometabolic multi-morbidity.",
    version="2.2.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons
verifier = ClinicalVerifier(ontology_path=ONTOLOGY_PATH)
retriever = HybridClinicalRetriever(guidelines_path=GUIDELINES_PATH)
pdf_parser = ClinicalPDFParser()
llm_agent = LLMClinicalAgent(provider="local")
safecds_app = build_safecds_graph(verifier, llm_agent=llm_agent, retriever=retriever)

# Request Models
class EvaluateRequest(BaseModel):
    patient_id: str = Field(default="PT_CUSTOM")
    patient_name: Optional[str] = Field(default="Anonymous Patient")
    evaluation_id: Optional[str] = Field(default=None)
    age: Optional[int] = Field(default=None)
    gender: Optional[str] = Field(default=None)
    conditions: List[str] = Field(default_factory=list)
    symptoms: List[str] = Field(default_factory=list)
    lab_values: Optional[Dict[str, Any]] = Field(default_factory=dict)
    clinical_narrative: Optional[str] = Field(default="")
    provider: Optional[str] = Field(default="local")
    model_name: Optional[str] = Field(default="")
    api_key: Optional[str] = Field(default="")
    max_retries: Optional[int] = Field(default=3)


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    patient_id: Optional[str] = Field(default="PT_PATIENT")
    evaluation_id: Optional[str] = Field(default=None)
    conditions: Optional[List[str]] = Field(default_factory=list)
    lab_values: Optional[Dict[str, Any]] = Field(default_factory=dict)
    provider: Optional[str] = Field(default="local")
    model_name: Optional[str] = Field(default="")
    api_key: Optional[str] = Field(default="")


@app.get("/api/health")
def get_health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "verifier": "online (axiomatic fast-path & HermiT DL ready)",
        "retriever": f"online ({len(retriever.corpus)} guideline passages indexed)",
        "pdf_parser": "ready (pypdfium2)",
        "active_provider": llm_agent.provider,
        "supported_providers": ["local", "gemini", "openai", "antigravity"]
    }


@app.get("/api/cohort")
def get_cohort():
    """Returns benchmark patients from data/cohort.json."""
    if os.path.exists(COHORT_PATH):
        with open(COHORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.get("/api/guidelines")
def get_guidelines():
    """Returns all guideline documents from data/guidelines.json."""
    if os.path.exists(GUIDELINES_PATH):
        with open(GUIDELINES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.get("/api/ontology")
def get_ontology():
    """Returns parsed ontology axioms and metadata."""
    return {
        "ontology_file": ONTOLOGY_PATH,
        "axioms_count": len(verifier.axioms),
        "axioms": verifier.axioms,
        "canonical_conditions": verifier.CANONICAL_CONDITIONS,
        "canonical_medications": verifier.CANONICAL_MEDICATIONS,
        "cache_stats": verifier.cache_stats
    }


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Extracts text and clinical variables from an uploaded medical report PDF."""
    try:
        temp_dir = os.path.join("data", "uploads")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        text = pdf_parser.extract_text_from_pdf(file_path)
        extracted = pdf_parser.parse_clinical_entities(text)
        extracted["filename"] = file.filename
        
        return JSONResponse(content={"status": "success", "data": extracted})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")


@app.post("/api/evaluate")
def evaluate_patient(req: EvaluateRequest):
    """
    Executes the SafeCDS Neuro-Symbolic Agentic RAG state machine:
    1. Ingestion -> 2. Hybrid RAG (FAISS+BM25 RRF) -> 3. Neural Proposer -> 4. HermiT DL Verifier -> 5. Self-Correction -> 6. Approved/Escalated Output
    """
    start_t = time.perf_counter()
    
    # Configure custom provider if specified
    active_agent = llm_agent
    if req.provider and (req.provider != llm_agent.provider or req.api_key or req.model_name):
        active_agent = LLMClinicalAgent(
            provider=req.provider,
            model_name=req.model_name or None,
            api_key=req.api_key or None
        )
        
    app_instance = build_safecds_graph(verifier, llm_agent=active_agent, retriever=retriever)
    
    eval_id = req.evaluation_id or f"eval_{uuid.uuid4().hex[:12]}"
    state_input = {
        "patient_id": req.patient_id,
        "patient_name": req.patient_name,
        "evaluation_id": eval_id,
        "age": req.age,
        "gender": req.gender,
        "conditions": req.conditions,
        "symptoms": req.symptoms,
        "lab_values": req.lab_values,
        "clinical_narrative": req.clinical_narrative,
        "retrieved_context": None,
        "retrieved_passages": None,
        "proposed_medication": None,
        "diagnosed_condition": None,
        "icd10_code": None,
        "differential_diagnoses": None,
        "patient_friendly_summary": None,
        "warnings": None,
        "reasoning": None,
        "retries": 0,
        "max_retries": req.max_retries,
        "violations": [],
        "status": "PENDING",
        "verification_details": None,
        "chronological_trace": [],
        "provider": req.provider,
        "model_name": req.model_name,
        "api_key": req.api_key,
        "final_output": {}
    }
    
    try:
        result = app_instance.invoke(state_input)
    except ConnectionError as ce:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error_type": "OLLAMA_DISCONNECTED",
                "message": f"Local LLM (Ollama) connection failure: {str(ce)}. Ensure Ollama is running ('ollama serve') with model '{active_agent.model_name}'."
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Evaluation pipeline failed: {str(e)}"
            }
        )
    elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
    
    final_output = result.get("final_output", {})
    final_output["total_execution_time_ms"] = elapsed_ms
    
    return JSONResponse(content={
        "status": "success",
        "result": final_output
    })


@app.post("/api/chat")
def patient_chat(req: ChatRequest):
    """
    Conversational Patient Assistant endpoint.
    Extracts clinical signals, inquired medications, and lab values from the patient query,
    preserves original laboratory values without default substitution,
    executes SafeCDS LangGraph state machine with 8 clinical validation checks,
    and returns a structured, plain-language response with individual medication safety evaluations.
    """
    # 1. Parse natural text entities
    parsed = pdf_parser.parse_clinical_entities(req.message)
    
    # Merge existing conditions with newly detected conditions
    detected_conds = parsed.get("conditions", []) if isinstance(parsed, dict) else getattr(parsed, "conditions", [])
    conditions = list(dict.fromkeys((req.conditions or []) + detected_conds))
    
    # Preserve exact lab values without loss
    lab_vals = dict(req.lab_values or {})
    for k in ["egfr", "sbp", "dbp", "hba1c", "creatinine", "potassium", "uacr", "age", "gender"]:
        if parsed.get(k) is not None and lab_vals.get(k) is None:
            lab_vals[k] = parsed[k]

    # Context continuity: scan prior conversation history if labs/conditions are missing
    if req.history:
        for past_turn in reversed(req.history):
            past_msg = past_turn.get("content", "")
            past_parsed = pdf_parser.parse_clinical_entities(past_msg)
            for k in ["egfr", "sbp", "dbp", "hba1c", "creatinine", "potassium", "uacr", "age", "gender"]:
                if past_parsed.get(k) is not None and lab_vals.get(k) is None:
                    lab_vals[k] = past_parsed[k]
            for past_c in past_parsed.get("conditions", []):
                if past_c not in conditions:
                    conditions.append(past_c)

    # Detect inquired medications from message and current medications
    curr_meds = lab_vals.get("current_medications", [])
    curr_meds_text = " ".join([str(m) for m in curr_meds]) if isinstance(curr_meds, list) else str(curr_meds)
    inquired_meds = extract_inquired_medications(req.message + " " + curr_meds_text)

    # Check if this is a general clinical inquiry or simple greeting
    is_greeting = req.message.strip().lower() in ["hi", "hello", "hey", "help", "good morning", "good evening"]
    has_patient_labs = bool(req.lab_values) or (any(k in req.message.lower() for k in ["egfr", "uacr", "hba1c", "potassium", "k+", "creatinine", "blood pressure", "sbp", "dbp"]) and any(c.isdigit() for c in req.message))
    is_general_question = any(q_starter in req.message.lower() for q_starter in ["what is", "what are", "difference between", "how do", "how does", "can you explain", "tell me about", "why is", "why are", "could you explain"])

    if (not has_patient_labs and is_general_question) or (not conditions and not lab_vals and not inquired_meds):
        if is_greeting:
            return JSONResponse(content={
                "status": "success",
                "reply": (
                    "Hello! I am your SafeCDS health assistant. I can help evaluate your cardiometabolic symptoms, "
                    "such as Type 1 or Type 2 Diabetes, High Blood Pressure (Hypertension), Kidney Disease (CKD), or Heart Failure.\n\n"
                    "Please describe your symptoms, share your lab results (such as eGFR, potassium, blood pressure, or HbA1c), "
                    "or ask about specific medications (such as Metformin, Lisinopril, SGLT2 inhibitors, or Insulin) to review their safety."
                ),
                "verification": None,
                "detected_conditions": [],
                "detected_labs": {}
            })
        else:
            # Dynamic general clinical question answering using local LLM with RAG
            active_chat_agent = llm_agent
            if req.provider and (req.provider != llm_agent.provider or req.model_name or req.api_key):
                active_chat_agent = LLMClinicalAgent(
                    provider=req.provider,
                    model_name=req.model_name or None,
                    api_key=req.api_key or None
                )
            passages = retriever.retrieve_passages_with_scores(req.message, top_k=3)
            chat_resp = active_chat_agent.generate_chat_response(
                message=req.message,
                history=req.history,
                retrieved_passages=passages,
                model_name=req.model_name
            )
            return JSONResponse(content={
                "status": chat_resp.get("status", "success"),
                "reply": chat_resp.get("reply", ""),
                "model_used": chat_resp.get("model_used", active_chat_agent.model_name),
                "detected_conditions": [],
                "detected_labs": {},
                "guideline_passages": passages
            })

    # Age extraction for patient context
    age_val = lab_vals.get("age")
    eval_id = req.evaluation_id or f"chat_{uuid.uuid4().hex[:12]}"
    pid = req.patient_id if req.patient_id and req.patient_id not in ["PT_PATIENT", "PT_CUSTOM"] else f"Patient_{eval_id[:10]}"

    eval_req = EvaluateRequest(
        patient_id=pid,
        patient_name="Patient",
        evaluation_id=eval_id,
        age=age_val,
        conditions=conditions,
        lab_values=lab_vals,
        clinical_narrative=req.message,
        provider=req.provider,
        model_name=req.model_name,
        api_key=req.api_key
    )
    eval_res = evaluate_patient(eval_req)
    if eval_res.status_code != 200:
        return eval_res
    data = json.loads(eval_res.body.decode("utf-8"))
    if data.get("status") == "error":
        return eval_res
    res = data.get("result", {})

    # Extract structured fields
    diag = res.get("diagnosed_condition", "Cardiometabolic Health Profile")
    icd = res.get("icd10_code", "")
    med = res.get("recommended_treatment", "Targeted Regimen")
    summary = res.get("patient_friendly_summary", res.get("clinical_rationale", ""))
    confirmed_dx = res.get("confirmed_diagnoses", [])
    suspected_conds = res.get("suspected_conditions", [])
    missing_ev = res.get("missing_evidence", [])
    med_assessments = res.get("medication_assessments", {})
    warnings = res.get("warnings", [])
    violations = res.get("intercepted_violations", [])
    proof = res.get("verification_details", {})
    guidelines = res.get("guideline_citations", [])

    reply_parts = [
        f"Based on your clinical history and lab results, your profile is consistent with **{diag}** (ICD-10: `{icd}`).",
        "",
        f"### Safe Treatment Consideration (For Physician Review)\n**{med}**",
        f"\n{summary}",
    ]

    # Individual assessment for every medication inquired by the user
    if inquired_meds:
        reply_parts.append("\n### Individual Medication Safety Analysis")
        for drug in inquired_meds:
            # Find matching evaluation
            eval_data = None
            for k, v in med_assessments.items():
                if k.lower() == drug.lower() or drug.lower() in k.lower():
                    eval_data = v
                    break
            if not eval_data:
                eval_data = evaluate_individual_medication(drug, {
                    "age": age_val,
                    "is_pediatric": (age_val is not None and age_val < 18),
                    "is_type_1_diabetes": any("type_1" in c.lower() or "t1d" in c.lower() for c in conditions),
                    "is_type_2_diabetes": any("type_2" in c.lower() or "t2d" in c.lower() for c in conditions),
                    "conditions": conditions,
                    "lab_values": lab_vals,
                    "cardiorenal_status": res.get("cardiorenal_status", {}),
                    "current_medications": lab_vals.get("current_medications", [])
                })
            
            status_badge = "❌ Contraindicated / Unsafe" if eval_data.get("contraindicated") else "✅ Considered Safe / Indicated with Monitoring"
            reply_parts.append(
                f"- **{eval_data.get('medication', drug)}** [{status_badge}]: "
                f"{eval_data.get('safety_verdict', '')}. {eval_data.get('rationale', '')} "
                f"*(Guideline: {eval_data.get('guideline', 'Clinical Consensus')})*"
            )

    # Cardiorenal Organ Protection Guideline Recommendation (KDIGO 2023 Level 1A)
    uacr_val = lab_vals.get("uacr")
    is_t1 = any("type_1" in c.lower() or "t1d" in c.lower() for c in conditions)
    if uacr_val is not None and uacr_val > 300 and not is_t1:
        reply_parts.append(
            "\n### Cardiorenal Organ Protection Recommendation (KDIGO 2023)\n"
            "- **SGLT2 Inhibitor (Empagliflozin 10 mg daily)**: Strongly recommended (KDIGO 2023 Level 1A) for patients with "
            "Type 2 Diabetes and CKD with severely increased albuminuria (Stage A3) to provide kidney and cardiovascular protection."
        )

    # Diagnostic Staging: Confirmed vs Suspected
    if confirmed_dx or suspected_conds:
        reply_parts.append("\n### Diagnostic Findings & Monitoring Status")
        if confirmed_dx:
            reply_parts.append("**Confirmed Diagnoses:**")
            for c in confirmed_dx:
                reply_parts.append(f"- {c}")
        if suspected_conds:
            reply_parts.append("**Suspected Conditions & Monitoring Alerts:**")
            for s in suspected_conds:
                reply_parts.append(f"- {s}")

    # Missing Evidence / Uncertainty
    if missing_ev:
        reply_parts.append("\n### Missing Information & Recommended Workup")
        for m in missing_ev:
            reply_parts.append(f"- {m}")

    all_warnings = warnings + violations
    if all_warnings:
        reply_parts.append("\n### Clinical Safety Precautions & Intercepted Risks")
        for w in all_warnings:
            reply_parts.append(f"- {w}")

    # Guideline citations (strictly exclude pediatric citations for adult patients)
    if guidelines:
        if age_val is not None and age_val >= 18:
            guidelines = [
                g for g in guidelines
                if "ispad" not in g.get("id", "").lower()
                and "child" not in g.get("title", "").lower()
                and "pediatric" not in g.get("title", "").lower()
            ]
        reply_parts.append("\n### Authoritative Guideline Citations")
        for g in guidelines[:3]:
            reply_parts.append(f"- **{g.get('organization', 'Clinical Guideline')} ({g.get('year', '')})**: {g.get('title', '')}")

    # Scientifically rigorous disclaimer (NO claim of 0% escape rate!)
    reply_parts.append(
        "\n*Formal Description Logic verification confirms logical axiomatic consistency against defined OWL safety rules. "
        "This platform provides evidence-based decision support; final diagnosis, laboratory validation, and prescription decisions "
        "must be performed by a licensed physician.*"
    )

    return JSONResponse(content={
        "status": "success",
        "reply": "\n".join(reply_parts),
        "evaluation": res,
        "verification": proof,
        "detected_conditions": conditions,
        "detected_labs": lab_vals,
        "inquired_medications": inquired_meds,
        "medication_assessments": med_assessments
    })


@app.post("/api/benchmark")
def run_benchmark():
    """Executes the batch cohort benchmark over all 6 cardiometabolic cases."""
    try:
        run_cohort_benchmark()
        results_file = os.path.join("results", "cohort_benchmark_results.json")
        if os.path.exists(results_file):
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return JSONResponse(content={"status": "success", "benchmark": data})
        return JSONResponse(content={"status": "error", "message": "Benchmark results file not found"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Static files mount
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Sample reports mount
sample_reports_dir = os.path.join("data", "sample_reports")
if os.path.exists(sample_reports_dir):
    app.mount("/sample_reports", StaticFiles(directory=sample_reports_dir), name="sample_reports")


@app.get("/")
def serve_home():
    """Serves the main single page web application."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "SafeCDS Server running. Web UI loading..."}
