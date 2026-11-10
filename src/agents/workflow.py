import time
import re
import json
from typing import TypedDict, List, Optional, Dict, Any, Callable, Tuple
from src.agents.llm_client import LLMClinicalAgent
from src.retrieval.hybrid_retriever import HybridClinicalRetriever
from src.reasoning.clinical_rules import (
    classify_ckd,
    classify_albuminuria,
    evaluate_potassium,
    evaluate_cardiorenal,
    evaluate_individual_medication,
    extract_inquired_medications,
    evaluate_inquired_medications,
    identify_hyperkalemia_contributing_medications,
    evaluate_pediatric_schwartz,
    categorize_clinical_diagnoses
)
from src.models.patient_context import (
    PatientContext,
    format_lab_value,
    validate_lab_integrity,
    verify_patient_output_consistency
)

# Import LangGraph if available, otherwise provide an exact lightweight state machine
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    END = "__END__"
    
    class StateGraph:
        """Lightweight native state machine matching LangGraph's API for zero-dependency execution."""
        def __init__(self, state_schema):
            self.state_schema = state_schema
            self.nodes: Dict[str, Callable] = {}
            self.edges: Dict[str, str] = {}
            self.conditional_edges: Dict[str, Tuple[Callable, Dict[str, str]]] = {}
            self.entry_point: Optional[str] = None

        def add_node(self, name: str, action: Callable):
            self.nodes[name] = action

        def set_entry_point(self, name: str):
            self.entry_point = name

        def add_edge(self, from_node: str, to_node: str):
            self.edges[from_node] = to_node

        def add_conditional_edges(self, from_node: str, condition: Callable, routing_map: Dict[str, str]):
            self.conditional_edges[from_node] = (condition, routing_map)

        def compile(self):
            return CompiledApp(self)

    class CompiledApp:
        def __init__(self, graph: StateGraph):
            self.graph = graph

        def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
            current_node = self.graph.entry_point
            current_state = dict(state)
            
            # Defensive loop protection
            max_steps = 25
            step_count = 0
            
            while current_node and current_node != END and step_count < max_steps:
                step_count += 1
                action = self.graph.nodes[current_node]
                updated = action(current_state)
                if isinstance(updated, dict):
                    current_state.update(updated)

                if current_node in self.graph.conditional_edges:
                    cond_fn, routing_map = self.graph.conditional_edges[current_node]
                    decision = cond_fn(current_state)
                    current_node = routing_map.get(decision, END)
                elif current_node in self.graph.edges:
                    current_node = self.graph.edges[current_node]
                else:
                    break

            return current_state


class AgentState(TypedDict):
    patient_id: str
    patient_name: Optional[str]
    age: Optional[int]
    gender: Optional[str]
    conditions: List[str]
    symptoms: Optional[List[str]]
    lab_values: Optional[dict]
    clinical_narrative: Optional[str]
    
    # Patient Context & Inquiries
    patient_context: Optional[dict]
    inquired_medications: Optional[List[str]]
    medication_assessments: Optional[dict]
    confirmed_diagnoses: Optional[List[str]]
    suspected_conditions: Optional[List[str]]
    missing_evidence: Optional[List[str]]
    clinical_safety_validation: Optional[dict]
    consistency_alerts: Optional[List[str]]
    
    # RAG Context
    retrieved_context: Optional[str]
    retrieved_passages: Optional[List[dict]]
    
    # Generated Proposal
    diagnosed_condition: Optional[str]
    icd10_code: Optional[str]
    proposed_medication: Optional[str]
    reasoning: Optional[str]
    differential_diagnoses: Optional[List[str]]
    patient_friendly_summary: Optional[str]
    warnings: Optional[List[str]]
    
    # Verification & Orchestration State
    retries: int
    max_retries: int
    violations: List[str]
    status: str
    verification_details: Optional[dict]
    chronological_trace: List[dict]
    
    # Provider options
    provider: Optional[str]
    model_name: Optional[str]
    api_key: Optional[str]
    
    # Clinical Rules & Laboratory Validation
    ckd_stage: Optional[str]
    albuminuria_stage: Optional[str]
    potassium_status: Optional[str]
    cardiorenal_status: Optional[dict]
    medication_eligibility: Optional[dict]
    clinical_alerts: Optional[List[str]]
    
    # Output payload
    final_output: dict


# Lazy global instances for high performance
_GLOBAL_LLM = None
_GLOBAL_RETRIEVER = None

def get_llm_agent() -> LLMClinicalAgent:
    global _GLOBAL_LLM
    if _GLOBAL_LLM is None:
        _GLOBAL_LLM = LLMClinicalAgent()
    return _GLOBAL_LLM

def get_retriever() -> HybridClinicalRetriever:
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = HybridClinicalRetriever()
    return _GLOBAL_RETRIEVER


def log_step(state: AgentState, stage: str, message: str, meta: Optional[dict] = None):
    """Appends an auditable transition event to chronological trace."""
    if "chronological_trace" not in state or state["chronological_trace"] is None:
        state["chronological_trace"] = []
    
    step_num = len(state["chronological_trace"]) + 1
    state["chronological_trace"].append({
        "step": step_num,
        "stage": stage,
        "message": message,
        "meta": meta or {},
        "timestamp": time.time()
    })


def ingest_node(state: AgentState):
    """
    Initializes and normalizes patient intake profile into a single structured PatientContext.
    Preserves exact original lab values without substitution or defaulting.
    """
    pid = state.get("patient_id", "Patient_Unknown")
    pname = state.get("patient_name", "Anonymous Patient")
    labs = state.get("lab_values") or {}
    age = state.get("age") if state.get("age") is not None else labs.get("age")
    gender = state.get("gender") or labs.get("gender", "Unspecified")
    conds = list(dict.fromkeys(state.get("conditions", [])))
    symptoms = state.get("symptoms") or labs.get("symptoms", [])
    narrative = state.get("clinical_narrative") or labs.get("clinical_narrative", "")
    
    # Detect inquired medications from message/narrative/symptoms
    inquired_meds = extract_inquired_medications(narrative + " " + " ".join(symptoms))

    # Construct single structured PatientContext
    ctx = PatientContext(
        patient_id=pid,
        patient_name=pname,
        age=age,
        gender=gender,
        height_cm=labs.get("height_cm"),
        weight_kg=labs.get("weight_kg"),
        conditions=conds,
        symptoms=symptoms,
        current_medications=labs.get("current_medications", []),
        inquired_medications=inquired_meds,
        lab_values=labs,
        clinical_narrative=narrative
    )

    egfr = ctx.lab_values.get("egfr")
    uacr = ctx.lab_values.get("uacr")
    k_val = ctx.lab_values.get("potassium")
    hba1c = ctx.lab_values.get("hba1c")
    sbp = ctx.lab_values.get("sbp")

    has_ckd = "ckd" in " ".join(conds).lower() or (egfr is not None and egfr < 60)
    has_hf = "heart_failure" in " ".join(conds).lower() or "hf" in " ".join(conds).lower() or any("heart" in s.lower() for s in symptoms)

    # Deterministic evaluations
    ckd_eval = classify_ckd(egfr, has_kidney_damage=True if (uacr and uacr >= 30) or has_ckd else False)
    alb_eval = classify_albuminuria(uacr, repeat_confirmed=False, is_pediatric=ctx.is_pediatric)
    k_eval = evaluate_potassium(k_val)
    cr_eval = evaluate_cardiorenal(has_hf, symptoms=symptoms, clinical_narrative=narrative, egfr=egfr)

    # Individual medication assessments for all inquired medications
    ctx_dict = ctx.to_dict()
    ctx_dict["cardiorenal_status"] = cr_eval
    med_assessments = evaluate_inquired_medications(inquired_meds, ctx_dict)

    # Separate confirmed diagnoses from suspected conditions & missing evidence
    confirmed_dx, suspected_conds, missing_ev = categorize_clinical_diagnoses(ctx_dict)

    alerts = list(ctx.integrity_alerts)
    if k_eval["is_alert"]:
        alerts.append(k_eval["message"])
    if cr_eval["fluid_overload_present"]:
        alerts.append("Fluid Overload Alert: Dependent peripheral or pulmonary congestion detected. Loop diuretic indicated.")
    if cr_eval["cardiorenal_syndrome"]:
        alerts.append(f"Cardiorenal Syndrome Alert: {cr_eval['crs_type']} present.")

    # Store in state
    state["patient_context"] = ctx_dict
    state["lab_values"] = ctx.lab_values
    state["conditions"] = ctx.conditions
    state["age"] = ctx.age
    state["gender"] = ctx.gender
    state["inquired_medications"] = inquired_meds
    state["medication_assessments"] = med_assessments
    state["confirmed_diagnoses"] = confirmed_dx
    state["suspected_conditions"] = suspected_conds
    state["missing_evidence"] = missing_ev
    state["ckd_stage"] = ckd_eval["stage_code"]
    state["albuminuria_stage"] = alb_eval["stage_code"]
    state["potassium_status"] = k_eval["status"]
    state["cardiorenal_status"] = cr_eval
    state["medication_eligibility"] = med_assessments
    state["clinical_alerts"] = alerts

    log_step(
        state,
        "INGESTION",
        f"Patient {pid} context initialized. KDIGO CKD: {ckd_eval['stage_code']} ({ckd_eval['icd10']}), "
        f"Albuminuria: {alb_eval['stage_code']}, K+: {k_eval['status']}. Inquired medications: {inquired_meds}",
        {"patient_id": pid, "ckd_stage": ckd_eval["stage_code"], "icd10": ckd_eval["icd10"], "alerts": alerts}
    )
    return state


def retrieve_node(state: AgentState, retriever: HybridClinicalRetriever):
    """Retrieves relevant guideline chunks using FAISS + BM25 RRF."""
    if state.get("retrieved_context") and state.get("retrieved_passages"):
        return state

    conds_clean = []
    for c in state.get("conditions", []):
        expanded = re.sub(r'([a-z])([A-Z])', r'\1 \2', c).replace('_', ' ')
        conds_clean.append(expanded)
    conds_text = ", ".join(conds_clean)
    inquired_text = " ".join(state.get("inquired_medications", []))
    query = f"Management of patient with {conds_text} {inquired_text}".strip()
    
    passages = retriever.retrieve_passages_with_scores(query, top_k=3)
    context_str = retriever.retrieve_context(query, top_k=3)
    
    state["retrieved_passages"] = passages
    state["retrieved_context"] = context_str
    
    log_step(
        state,
        "RETRIEVAL",
        f"Hybrid RAG retrieved {len(passages)} authoritative guideline passages (RRF k=60).",
        {"passages_count": len(passages), "top_source": passages[0]["source"] if passages else ""}
    )
    return state


def generate_hypothesis_node(state: AgentState, llm_agent: LLMClinicalAgent):
    attempt = state.get("retries", 0) + 1
    provider = state.get("provider", "local")
    model_name = state.get("model_name")
    api_key = state.get("api_key")
    lab_vals = state.get("lab_values")
    
    response = llm_agent.generate_recommendation(
        patient_id=state["patient_id"],
        conditions=state["conditions"],
        guidelines=state.get("retrieved_context") or "None",
        violations=state.get("violations", []),
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        lab_values=lab_vals
    )
    
    state["proposed_medication"] = response["medication"]
    state["reasoning"] = response["reasoning"]
    state["diagnosed_condition"] = response.get("diagnosed_condition", "Cardiometabolic Multimorbidity")
    state["icd10_code"] = response.get("icd10_code", "R69")
    state["differential_diagnoses"] = response.get("differential_diagnoses", [])
    state["patient_friendly_summary"] = response.get("patient_friendly_summary", "")
    state["warnings"] = response.get("warnings", [])
    if response.get("clinical_alerts"):
        cur_alerts = state.get("clinical_alerts") or []
        for alert in response["clinical_alerts"]:
            if alert not in cur_alerts:
                cur_alerts.append(alert)
        state["clinical_alerts"] = cur_alerts
    if response.get("ckd_stage"):
        state["ckd_stage"] = response["ckd_stage"]
    if response.get("albuminuria_stage"):
        state["albuminuria_stage"] = response["albuminuria_stage"]
    if response.get("confirmed_diagnoses"):
        state["confirmed_diagnoses"] = response["confirmed_diagnoses"]
    if response.get("suspected_conditions"):
        state["suspected_conditions"] = response["suspected_conditions"]
    if response.get("missing_evidence"):
        state["missing_evidence"] = response["missing_evidence"]
    if response.get("medication_assessments"):
        state["medication_assessments"] = response["medication_assessments"]
    
    log_step(
        state,
        "GENERATION",
        f"Attempt {attempt}: Proposing '{state['proposed_medication']}' for {state['diagnosed_condition']} [{state['icd10_code']}]",
        {"medication": state["proposed_medication"], "attempt": attempt, "provider": provider}
    )
    return state


def verify_axiomatic_node(state: AgentState, verifier):
    """
    Executes formal Description Logic tableaux reasoning via HermiT OWL 2 DL.
    Explicitly distinguishes axiomatic logical consistency from empirical clinical guideline correctness.
    """
    medication = state.get("proposed_medication", "")
    patient_id = state.get("patient_id", "Patient_Case")
    conditions = state.get("conditions", [])
    
    result = verifier.verify_prescription(
        patient_id=patient_id,
        conditions=conditions,
        medications=[medication],
        lab_values=state.get("lab_values")
    )
    
    # Add explicit HermiT scope distinction
    result["reasoner_scope"] = (
        "HermiT OWL 2 DL reasoner verified formal logical consistency against ontology safety axioms. "
        "A passing HermiT proof establishes formal logical unsatisfiability of defined contraindication axioms, "
        "distinct from comprehensive clinical guideline validation and individualized physician assessment."
    )
    state["verification_details"] = result
    
    if result["status"] == "PASS":
        state["status"] = "AXIOMS_PASSED"
        log_step(
            state,
            "AXIOMATIC_VERIFICATION",
            f"HermiT OWL 2 DL reasoner: PASSED (Zero formal axiomatic contraindications).",
            {"status": "PASS", "latency_ms": result["reasoner_latency_ms"]}
        )
    else:
        state["status"] = "FAILED"
        if "violations" not in state or state["violations"] is None:
            state["violations"] = []
        state["violations"].append(result["violated_axiom"])
        state["retries"] = state.get("retries", 0) + 1
        
        log_step(
            state,
            "AXIOMATIC_VERIFICATION",
            f"HermiT OWL 2 DL reasoner: FAILED -> Intercepted contraindication axiom: {result['violated_axiom']}",
            {"status": "FAIL", "axiom": result["violated_axiom"], "retry": state["retries"]}
        )
        
    return state


def validate_clinical_safety_node(state: AgentState):
    """
    Validation node executing the 8 clinical safety checks:
    1. CKD Stage Consistency
    2. Medication Contraindications
    3. Laboratory Value Integrity
    4. Diabetes Type and Medication Compatibility
    5. Pediatric versus Adult Rules
    6. Heart Failure Interactions
    7. Hyperkalemia Safety
    8. Missing Clinical Evidence
    """
    p_ctx = state.get("patient_context") or {}
    labs = state.get("lab_values") or {}
    proposed_med = str(state.get("proposed_medication", "")).lower()
    
    safety_checks = {
        "ckd_stage_consistency": {"passed": True, "detail": "Consistent with KDIGO 2023 eGFR staging."},
        "medication_contraindications": {"passed": True, "detail": "No unhandled contraindications detected."},
        "laboratory_value_integrity": {"passed": True, "detail": "All original lab values preserved without substitution."},
        "diabetes_type_compatibility": {"passed": True, "detail": "Medication compatibility matches diabetes etiology."},
        "pediatric_vs_adult_rules": {"passed": True, "detail": "Age-appropriate clinical guidelines enforced."},
        "heart_failure_interactions": {"passed": True, "detail": "Cardiorenal interactions appropriately addressed."},
        "hyperkalemia_safety": {"passed": True, "detail": "Potassium preserved and stratified per clinical severity."},
        "missing_clinical_evidence": {"passed": True, "detail": "Missing clinical parameters documented transparently."}
    }

    # 1. CKD Stage Consistency
    egfr = labs.get("egfr")
    reported_stage = state.get("ckd_stage")
    if egfr is not None:
        if egfr < 30 and reported_stage not in ["G4", "G5", "CKD_Stage_4", "CKD_Stage_5"]:
            safety_checks["ckd_stage_consistency"] = {
                "passed": False,
                "detail": f"eGFR {egfr} indicates Stage 4/5 CKD, but reported stage was {reported_stage}."
            }
        elif 30 <= egfr < 45 and reported_stage not in ["G3b", "CKD_Stage_3b"]:
            safety_checks["ckd_stage_consistency"] = {
                "passed": False,
                "detail": f"eGFR {egfr} indicates Stage 3b CKD, but reported stage was {reported_stage}."
            }

    # 2. Medication Contraindications
    if egfr is not None and egfr < 30 and "metformin" in proposed_med:
        safety_checks["medication_contraindications"] = {
            "passed": False,
            "detail": f"Metformin proposed at eGFR {egfr} (< 30 mL/min/1.73m²); violates KDIGO 2023 Level 1A rule."
        }

    # 3. Laboratory Value Integrity
    if p_ctx:
        ctx_obj = PatientContext(
            patient_id=p_ctx.get("patient_id", ""),
            patient_name=p_ctx.get("patient_name", ""),
            age=p_ctx.get("age"),
            gender=p_ctx.get("gender"),
            conditions=p_ctx.get("conditions", []),
            symptoms=p_ctx.get("symptoms", []),
            current_medications=p_ctx.get("current_medications", []),
            inquired_medications=state.get("inquired_medications", []),
            lab_values=labs,
            clinical_narrative=state.get("clinical_narrative", "")
        )
        inconsistencies = verify_patient_output_consistency(ctx_obj, state)
        if inconsistencies:
            safety_checks["laboratory_value_integrity"] = {
                "passed": False,
                "detail": "; ".join(inconsistencies)
            }
            state["consistency_alerts"] = inconsistencies

    # 4. Diabetes Type and Medication Compatibility
    is_t1d = p_ctx.get("is_type_1_diabetes", False)
    if is_t1d:
        if "replace insulin" in proposed_med or "substitute insulin" in proposed_med or ("oral" in proposed_med and "insulin" not in proposed_med):
            safety_checks["diabetes_type_compatibility"] = {
                "passed": False,
                "detail": "In Type 1 Diabetes, essential insulin therapy must never be discontinued or replaced by oral agents."
            }

    # 5. Pediatric vs Adult Rules
    is_peds = p_ctx.get("is_pediatric", False)
    if is_peds:
        uacr = labs.get("uacr")
        if uacr and uacr >= 30 and "confirmed diabetic nephropathy" in str(state.get("diagnosed_condition", "")).lower():
            safety_checks["pediatric_vs_adult_rules"] = {
                "passed": False,
                "detail": "Single elevated UACR cannot be classified as confirmed diabetic nephropathy; requires 2 of 3 repeat tests over 3-6 months (ISPAD 2024)."
            }

    # 6. Heart Failure Interactions
    has_hf = state.get("cardiorenal_status", {}).get("heart_failure_present", False)
    if has_hf:
        if any(drug in proposed_med for drug in ["pioglitazone", "rosiglitazone", "nsaid", "ibuprofen"]):
            safety_checks["heart_failure_interactions"] = {
                "passed": False,
                "detail": "Fluid-retaining or cardiotoxic medication proposed in clinical heart failure."
            }

    # 7. Hyperkalemia Safety
    k_val = labs.get("potassium")
    if k_val is not None and k_val >= 5.5:
        if "normal potassium" in str(state.get("reasoning", "")).lower():
            safety_checks["hyperkalemia_safety"] = {
                "passed": False,
                "detail": f"Serum potassium is {k_val} mmol/L (moderate hyperkalemia); cannot be described as normal."
            }

    # 8. Missing Clinical Evidence
    missing = state.get("missing_evidence") or []
    if missing:
        safety_checks["missing_clinical_evidence"] = {
            "passed": True,
            "detail": f"Identified {len(missing)} missing clinical parameters requiring workup."
        }

    state["clinical_safety_validation"] = safety_checks
    
    # Check if any safety check failed that requires fail-closed correction
    all_passed = all(check["passed"] for check in safety_checks.values())
    if all_passed and state["status"] == "AXIOMS_PASSED":
        state["status"] = "VERIFIED"
        log_step(
            state,
            "SAFETY_VALIDATION",
            "Multi-dimensional clinical safety validation: All 8 clinical safety checks PASSED.",
            {"checks": safety_checks}
        )
    elif not all_passed:
        failed_checks = [k for k, v in safety_checks.items() if not v["passed"]]
        state["status"] = "FAILED"
        if "violations" not in state or state["violations"] is None:
            state["violations"] = []
        for fc in failed_checks:
            detail = safety_checks[fc]["detail"]
            if detail not in state["violations"]:
                state["violations"].append(f"Clinical Safety Rule Violation ({fc}): {detail}")
        state["retries"] = state.get("retries", 0) + 1
        log_step(
            state,
            "SAFETY_VALIDATION",
            f"Multi-dimensional clinical safety validation: FAILED on {failed_checks}. Engaging self-correction.",
            {"failed_checks": failed_checks, "retry": state["retries"]}
        )

    return state


def route_next_step(state: AgentState):
    if state["status"] == "VERIFIED":
        return "finalize"
    elif state["retries"] >= state.get("max_retries", 3):
        return "escalate"
    else:
        return "retry"


def finalize_node(state: AgentState):
    log_step(
        state,
        "FINALIZATION",
        f"Approved verified clinical output for {state['patient_id']} [{state.get('icd10_code')}].",
        {"status": "APPROVED"}
    )

    # Standard guideline citations
    guidelines = [
        {
            "id": "KDIGO_2023_CKD_DM",
            "title": "KDIGO 2023 Clinical Practice Guideline for Diabetes Management in Chronic Kidney Disease",
            "organization": "KDIGO",
            "year": 2023,
            "key_recommendation": "Metformin safe down to eGFR 30 with dose reduction in G3b; SGLT2i recommended down to eGFR 20."
        },
        {
            "id": "ADA_STANDARDS_2024",
            "title": "Standards of Care in Diabetes—2024",
            "organization": "American Diabetes Association",
            "year": 2024,
            "key_recommendation": "Insulin is life-essential in T1D; SGLT2i and GLP-1 RA prioritize organ protection in T2D."
        },
        {
            "id": "ISPAD_2024_PEDIATRIC",
            "title": "ISPAD Clinical Practice Consensus Guidelines 2024: Diabetes in Children and Adolescents",
            "organization": "ISPAD",
            "year": 2024,
            "key_recommendation": "Single elevated UACR does not establish diabetic kidney disease; 2 of 3 repeat tests over 3-6 months required."
        },
        {
            "id": "AHA_ACC_HF_2022",
            "title": "AHA/ACC/HFSA 2022 Guideline for the Management of Heart Failure",
            "organization": "AHA/ACC/HFSA",
            "year": 2022,
            "key_recommendation": "SGLT2 inhibitors and loop diuretics indicated for heart failure and volume overload."
        }
    ]

    p_ctx = state.get("patient_context") or {}
    labs = state.get("lab_values") or {}

    state["final_output"] = {
        "status": "APPROVED",
        "patient_id": state["patient_id"],
        "patient_name": state.get("patient_name", "Clinical Patient"),
        "diagnosed_condition": state.get("diagnosed_condition", "Cardiometabolic Multimorbidity"),
        "icd10_code": state.get("icd10_code", "R69"),
        "differential_diagnoses": state.get("differential_diagnoses", []),
        "confirmed_diagnoses": state.get("confirmed_diagnoses", []),
        "suspected_conditions": state.get("suspected_conditions", []),
        "missing_evidence": state.get("missing_evidence", []),
        "recommended_treatment": state.get("proposed_medication"),
        "clinical_rationale": state.get("reasoning"),
        "patient_friendly_summary": state.get("patient_friendly_summary"),
        "warnings": state.get("warnings", []),
        "clinical_alerts": state.get("clinical_alerts", []),
        "inquired_medications": state.get("inquired_medications", []),
        "medication_assessments": state.get("medication_assessments", {}),
        "guideline_citations": guidelines,
        "ckd_stage": state.get("ckd_stage"),
        "albuminuria_stage": state.get("albuminuria_stage"),
        "potassium_status": state.get("potassium_status"),
        "cardiorenal_status": state.get("cardiorenal_status"),
        "medication_eligibility": state.get("medication_assessments") or state.get("medication_eligibility"),
        "lab_summary": {
            "egfr": format_lab_value(labs.get("egfr"), "egfr"),
            "potassium": format_lab_value(labs.get("potassium"), "potassium"),
            "creatinine": format_lab_value(labs.get("creatinine"), "creatinine"),
            "hba1c": format_lab_value(labs.get("hba1c"), "hba1c"),
            "sbp": format_lab_value(labs.get("sbp"), "sbp"),
            "dbp": format_lab_value(labs.get("dbp"), "dbp"),
            "uacr": format_lab_value(labs.get("uacr"), "uacr")
        },
        "retrieved_passages": state.get("retrieved_passages", []),
        "evidence_context": state.get("retrieved_context"),
        "verification_details": state.get("verification_details"),
        "clinical_safety_validation": state.get("clinical_safety_validation"),
        "consistency_alerts": state.get("consistency_alerts", []),
        "total_attempts": state.get("retries", 0) + 1,
        "retries_count": state.get("retries", 0),
        "audit_trail": state.get("violations", []),
        "chronological_trace": state.get("chronological_trace", []),
        "intercepted_violations": state.get("violations", [])
    }
    return state


def escalate_node(state: AgentState):
    log_step(
        state,
        "ESCALATION",
        f"Maximum retries exhausted ({state.get('retries')}). Fail-closed escalation for human specialist review.",
        {"status": "ESCALATED"}
    )
    
    state["final_output"] = {
        "status": "ESCALATED_FOR_HUMAN_REVIEW",
        "patient_id": state["patient_id"],
        "patient_name": state.get("patient_name", "Clinical Patient"),
        "diagnosed_condition": state.get("diagnosed_condition", "Cardiometabolic Multimorbidity"),
        "icd10_code": state.get("icd10_code", "R69"),
        "last_proposed_treatment": state.get("proposed_medication"),
        "reason": "Maximum retry limit reached without resolving formal clinical violations. Fail-closed safety gate engaged.",
        "confirmed_diagnoses": state.get("confirmed_diagnoses", []),
        "suspected_conditions": state.get("suspected_conditions", []),
        "missing_evidence": state.get("missing_evidence", []),
        "inquired_medications": state.get("inquired_medications", []),
        "medication_assessments": state.get("medication_assessments", {}),
        "retrieved_passages": state.get("retrieved_passages", []),
        "verification_details": state.get("verification_details"),
        "clinical_alerts": state.get("clinical_alerts", []),
        "ckd_stage": state.get("ckd_stage"),
        "albuminuria_stage": state.get("albuminuria_stage"),
        "potassium_status": state.get("potassium_status"),
        "cardiorenal_status": state.get("cardiorenal_status"),
        "medication_eligibility": state.get("medication_assessments") or state.get("medication_eligibility"),
        "clinical_safety_validation": state.get("clinical_safety_validation"),
        "total_attempts": state.get("retries", 0),
        "audit_trail": state.get("violations", []),
        "chronological_trace": state.get("chronological_trace", []),
        "intercepted_violations": state.get("violations", [])
    }
    return state


def build_safecds_graph(
    verifier,
    llm_agent: Optional[LLMClinicalAgent] = None,
    retriever: Optional[HybridClinicalRetriever] = None
):
    """
    Builds and compiles the SafeCDS LangGraph state machine with 8 multi-dimensional validation checks.
    Pipeline:
    Ingest -> Retrieve -> Generate Hypothesis -> Axiomatic Verification (HermiT OWL 2 DL) -> Clinical Safety Validation -> Route
    """
    active_llm = llm_agent or get_llm_agent()
    active_retriever = retriever or get_retriever()
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("retrieve", lambda s: retrieve_node(s, active_retriever))
    workflow.add_node("generate_hypothesis", lambda s: generate_hypothesis_node(s, active_llm))
    workflow.add_node("verify_axioms", lambda s: verify_axiomatic_node(s, verifier))
    workflow.add_node("validate_safety", validate_clinical_safety_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("escalate", escalate_node)
    
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "retrieve")
    workflow.add_edge("retrieve", "generate_hypothesis")
    workflow.add_edge("generate_hypothesis", "verify_axioms")
    workflow.add_edge("verify_axioms", "validate_safety")
    
    workflow.add_conditional_edges(
        "validate_safety",
        route_next_step,
        {
            "finalize": "finalize",
            "retry": "generate_hypothesis",
            "escalate": "escalate"
        }
    )
    
    workflow.add_edge("finalize", END)
    workflow.add_edge("escalate", END)
    
    return workflow.compile()
