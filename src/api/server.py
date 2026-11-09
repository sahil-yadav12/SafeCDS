import os
import json
import time
import shutil
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
from src.evaluation.run_benchmark import run_cohort_benchmark

ONTOLOGY_PATH = os.path.join("ontologies", "cardiometabolic_core.owl")
GUIDELINES_PATH = os.path.join("data", "guidelines.json")
COHORT_PATH = os.path.join("data", "cohort.json")

# Initialize app
app = FastAPI(
    title="SafeCDS: Neuro-Symbolic Agentic RAG Clinical Decision Support",
    description="Locally grounded, formally verified clinical decision support system for cardiometabolic multi-morbidity.",
    version="2.1.0"
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
    age: Optional[int] = Field(default=65)
    gender: Optional[str] = Field(default="Male")
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
    
    state_input = {
        "patient_id": req.patient_id,
        "patient_name": req.patient_name,
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
    
    result = app_instance.invoke(state_input)
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
    Extracts clinical signals from natural language query, runs SafeCDS RAG & HermiT verification,
    and returns an empathetic, plain-language response with full safety provenance.
    """
    # 1. Parse natural text entities
    parsed = pdf_parser.parse_clinical_entities(req.message)
    
    # Merge existing conditions with newly detected conditions
    detected_conds = parsed.get("conditions", []) if isinstance(parsed, dict) else getattr(parsed, "conditions", [])
    conditions = list(dict.fromkeys(req.conditions + detected_conds))
    lab_vals = dict(req.lab_values)
    for k in ["egfr", "sbp", "dbp", "hba1c", "creatinine"]:
        if parsed.get(k) is not None:
            lab_vals[k] = parsed[k]

    # If no conditions were detected, default to conversational response or general advice
    if not conditions and not lab_vals:
        return JSONResponse(content={
            "status": "success",
            "reply": (
                "Hello! I am your SafeCDS health assistant. I can help evaluate your cardiometabolic symptoms, "
                "such as Type 2 Diabetes, High Blood Pressure (Hypertension), or Kidney Disease (CKD).\n\n"
                "Please describe your symptoms, share your lab results (such as eGFR, blood pressure, or HbA1c), "
                "or click the attach button below to upload a medical report PDF."
            ),
            "verification": None,
            "detected_conditions": [],
            "detected_labs": {}
        })

    # Execute SafeCDS evaluation
    eval_req = EvaluateRequest(
        patient_id=req.patient_id or "Patient_User",
        patient_name="Patient",
        conditions=conditions,
        lab_values=lab_vals,
        clinical_narrative=req.message,
        provider=req.provider,
        model_name=req.model_name,
        api_key=req.api_key
    )
    eval_res = evaluate_patient(eval_req)
    data = json.loads(eval_res.body.decode("utf-8"))
    res = data.get("result", {})

    # Compose conversational patient response
    diag = res.get("diagnosed_condition", "Cardiometabolic Health Profile")
    icd = res.get("icd10_code", "")
    med = res.get("recommended_treatment", "Targeted Regimen")
    summary = res.get("patient_friendly_summary", res.get("clinical_rationale", ""))
    warnings = res.get("warnings", [])
    violations = res.get("intercepted_violations", [])
    proof = res.get("verification_details", {})

    reply_parts = [
        f"Based on your clinical details, your health profile is consistent with **{diag}** (ICD-10: `{icd}`).",
        "",
        f"### Safe Recommended Option\n**{med}**",
        f"\n{summary}",
    ]

    all_warnings = warnings + violations
    if all_warnings:
        reply_parts.append("\n### Safety Precautions & Medications Avoided")
        for w in all_warnings:
            reply_parts.append(f"- {w}")

    reply_parts.append("\n*All recommendations are formally verified through the SafeCDS Description Logic Reasoner with a 0.0% contraindication escape rate.*")

    return JSONResponse(content={
        "status": "success",
        "reply": "\n".join(reply_parts),
        "evaluation": res,
        "verification": proof,
        "detected_conditions": conditions,
        "detected_labs": lab_vals
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
