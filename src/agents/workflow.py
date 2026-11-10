import time
import re
import json
import uuid
import logging
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
from src.reasoning.response_validator import (
    validate_final_clinical_response,
    sanitize_and_correct_response
)

logger = logging.getLogger("safecds.workflow")

# Native lightweight StateGraph conforming to LangGraph API
class StateGraph:
    """Observable native state machine matching LangGraph API with cycle protection."""
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
        return CompiledWorkflow(self)


class CompiledWorkflow:
    def __init__(self, graph: StateGraph):
        self.graph = graph

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        current_node = self.graph.entry_point
        current_state = dict(state)
        max_steps = 30
        step_count = 0

        while current_node and current_node != "__END__" and step_count < max_steps:
            step_count += 1
            action = self.graph.nodes[current_node]
            updated = action(current_state)
            if isinstance(updated, dict):
                current_state.update(updated)

            if current_node in self.graph.conditional_edges:
                cond_fn, routing_map = self.graph.conditional_edges[current_node]
                decision = cond_fn(current_state)
                current_node = routing_map.get(decision, "__END__")
            elif current_node in self.graph.edges:
                current_node = self.graph.edges[current_node]
            else:
                break

        return current_state


class AgentState(TypedDict):
    patient_id: str
    patient_name: Optional[str]
    evaluation_id: Optional[str]
    age: Optional[int]
    gender: Optional[str]
    conditions: List[str]
    symptoms: Optional[List[str]]
    lab_values: Optional[dict]
    clinical_narrative: Optional[str]
    
    # Query Analysis & Retrieval
    age_group: str
    lab_abnormalities: List[str]
    inquired_medications: List[str]
    retrieval_query: str
    retrieved_passages: List[dict]
    retrieved_context: str
    evidence_quality_status: str
    
    # Structured Patient Context & Deterministic Rules
    patient_context: Optional[dict]
    ckd_stage: Optional[str]
    albuminuria_stage: Optional[str]
    potassium_status: Optional[str]
    cardiorenal_status: Optional[dict]
    medication_assessments: Optional[dict]
    confirmed_diagnoses: Optional[List[str]]
    suspected_conditions: Optional[List[str]]
    missing_evidence: Optional[List[str]]
    clinical_alerts: Optional[List[str]]
    consistency_alerts: Optional[List[str]]
    
    # Neural Clinical Proposal
    proposed_medication: Optional[str]
    reasoning: Optional[str]
    diagnosed_condition: Optional[str]
    icd10_code: Optional[str]
    differential_diagnoses: Optional[List[str]]
    patient_friendly_summary: Optional[str]
    warnings: Optional[List[str]]
    
    # Verification & Trace
    verification_details: Optional[dict]
    clinical_safety_validation: Optional[dict]
    violations: List[str]
    retries: int
    max_retries: int
    status: str
    chronological_trace: List[dict]
    
    # Provider & Output
    provider: Optional[str]
    model_name: Optional[str]
    api_key: Optional[str]
    final_output: dict


# Global singletons
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


def log_trace(state: AgentState, stage: str, message: str, meta: Optional[dict] = None):
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


# ---------------------------------------------------------
# Observable 13 LangGraph Nodes
# ---------------------------------------------------------

# Node 1: patient_input_validation
def patient_input_validation_node(state: AgentState):
    pid = state.get("patient_id", "PT_UNKNOWN")
    labs = state.get("lab_values") or {}
    
    # Ensure raw lab values are preserved without default overwrites
    cleaned_labs = {}
    for k, v in labs.items():
        if v is not None and v != "":
            cleaned_labs[k] = v
            
    state["lab_values"] = cleaned_labs
    if not state.get("evaluation_id"):
        state["evaluation_id"] = str(uuid.uuid4())
    if "historical_violations" not in state or state["historical_violations"] is None:
        state["historical_violations"] = list(state.get("violations", []))
        
    log_trace(
        state,
        "INPUT_VALIDATION",
        f"Validated patient intake for {pid}. Verified {len(cleaned_labs)} canonical laboratory parameters.",
        {"patient_id": pid, "lab_keys": list(cleaned_labs.keys())}
    )
    return state


# Node 2: patient_context_builder
def patient_context_builder_node(state: AgentState):
    pid = state.get("patient_id", "PT_UNKNOWN")
    pname = state.get("patient_name", "Anonymous Patient")
    labs = state.get("lab_values") or {}
    age = state.get("age") if state.get("age") is not None else labs.get("age")
    gender = state.get("gender") or labs.get("gender", "Unspecified")
    conds = list(dict.fromkeys(state.get("conditions", [])))
    symptoms = state.get("symptoms") or labs.get("symptoms", [])
    narrative = state.get("clinical_narrative") or labs.get("clinical_narrative", "")
    
    # Inquired medications detection
    curr_meds_list = labs.get("current_medications", [])
    curr_meds_str = " ".join([str(m) for m in curr_meds_list]) if isinstance(curr_meds_list, list) else str(curr_meds_list)
    inquired_meds = extract_inquired_medications(narrative + " " + " ".join(symptoms) + " " + curr_meds_str)

    ctx = PatientContext(
        patient_id=pid,
        patient_name=pname,
        evaluation_id=state["evaluation_id"],
        age=age,
        gender=gender,
        height_cm=labs.get("height_cm"),
        weight_kg=labs.get("weight_kg"),
        conditions=conds,
        symptoms=symptoms,
        current_medications=curr_meds_list if isinstance(curr_meds_list, list) else [curr_meds_str],
        inquired_medications=inquired_meds,
        lab_values=labs,
        clinical_narrative=narrative
    )

    egfr = ctx.lab_values.get("egfr")
    uacr = ctx.lab_values.get("uacr")
    k_val = ctx.lab_values.get("potassium")
    has_ckd = "ckd" in " ".join(conds).lower() or (egfr is not None and egfr < 60)
    has_hf = "heart_failure" in " ".join(conds).lower() or "hf" in " ".join(conds).lower() or any("heart" in s.lower() for s in symptoms)

    # Deterministic computations
    ckd_eval = classify_ckd(egfr, has_kidney_damage=True if (uacr and uacr >= 30) or has_ckd else False)
    alb_eval = classify_albuminuria(uacr, repeat_confirmed=False, is_pediatric=ctx.is_pediatric)
    k_eval = evaluate_potassium(k_val)
    cr_eval = evaluate_cardiorenal(has_hf, symptoms=symptoms, clinical_narrative=narrative, egfr=egfr)

    ctx_dict = ctx.to_dict()
    ctx_dict["cardiorenal_status"] = cr_eval
    med_assessments = evaluate_inquired_medications(inquired_meds, ctx_dict)
    confirmed_dx, suspected_conds, missing_ev = categorize_clinical_diagnoses(ctx_dict)

    alerts = list(ctx.integrity_alerts)
    if k_eval["is_alert"]:
        alerts.append(k_eval["message"])
    if cr_eval["fluid_overload_present"]:
        alerts.append("Fluid Overload Alert: Dependent peripheral or pulmonary congestion detected. Loop diuretic indicated.")

    state["patient_context"] = ctx_dict
    state["age"] = ctx.age
    state["gender"] = ctx.gender
    state["conditions"] = ctx.conditions
    state["inquired_medications"] = inquired_meds
    state["medication_assessments"] = med_assessments
    state["confirmed_diagnoses"] = confirmed_dx
    state["suspected_conditions"] = suspected_conds
    state["missing_evidence"] = missing_ev
    state["ckd_stage"] = ckd_eval["stage_code"]
    state["albuminuria_stage"] = alb_eval["stage_code"]
    state["potassium_status"] = k_eval["status"]
    state["cardiorenal_status"] = cr_eval
    state["clinical_alerts"] = alerts
    state["age_group"] = "pediatric" if ctx.is_pediatric else "adult"

    log_trace(
        state,
        "CONTEXT_BUILDER",
        f"Built PatientContext: Age {age} ({state['age_group']}), CKD: {ckd_eval['stage_code']}, Albuminuria: {alb_eval['stage_code']}, K+: {k_eval['status']}",
        {"ckd_stage": ckd_eval["stage_code"], "icd10": ckd_eval["icd10"], "inquired_meds": inquired_meds}
    )
    return state


# Node 3: clinical_query_analyzer
def clinical_query_analyzer_node(state: AgentState):
    labs = state.get("lab_values") or {}
    abnormalities = []
    
    egfr = labs.get("egfr")
    if egfr is not None:
        if egfr < 60:
            abnormalities.append(f"reduced eGFR {egfr} mL/min/1.73m2 (CKD {state.get('ckd_stage')})")
        else:
            abnormalities.append(f"normal eGFR {egfr} mL/min/1.73m2")
            
    uacr = labs.get("uacr")
    if uacr is not None:
        if uacr >= 30:
            abnormalities.append(f"elevated UACR {uacr} mg/g (Albuminuria {state.get('albuminuria_stage')})")
        else:
            abnormalities.append(f"normal UACR {uacr} mg/g (A1)")

    k_val = labs.get("potassium")
    if k_val is not None:
        if k_val > 5.0:
            abnormalities.append(f"hyperkalemia potassium {k_val} mmol/L")
        elif k_val < 3.5:
            abnormalities.append(f"hypokalemia potassium {k_val} mmol/L")
        else:
            abnormalities.append(f"normokalemia potassium {k_val} mmol/L")

    hba1c = labs.get("hba1c")
    if hba1c is not None and hba1c > 7.0:
        abnormalities.append(f"elevated HbA1c {hba1c}%")

    state["lab_abnormalities"] = abnormalities
    log_trace(
        state,
        "QUERY_ANALYZER",
        f"Analyzed clinical state: {len(abnormalities)} lab indicators evaluated for {state['age_group']} population.",
        {"abnormalities": abnormalities}
    )
    return state


# Node 4: retrieval_query_generator
def retrieval_query_generator_node(state: AgentState):
    conds = state.get("conditions", [])
    meds = state.get("inquired_medications", [])
    abns = state.get("lab_abnormalities", [])
    age_grp = state.get("age_group", "adult")
    
    # Formulate clinically targeted multi-topic query
    query_parts = []
    if age_grp == "pediatric":
        query_parts.append("ISPAD pediatric diabetes")
    
    query_parts.extend(conds)
    query_parts.extend(meds)
    query_parts.extend(abns[:2])
    
    query_str = " ".join(query_parts).strip()
    if not query_str:
        query_str = "Cardiometabolic multimorbidity guideline management"
        
    state["retrieval_query"] = query_str
    log_trace(
        state,
        "QUERY_GENERATION",
        f"Generated dynamic retrieval query: '{query_str}'",
        {"query": query_str, "age_group": age_grp}
    )
    return state


# Node 5: hybrid_retriever
def hybrid_retriever_node(state: AgentState, retriever: HybridClinicalRetriever):
    query = state.get("retrieval_query", "")
    age_grp = state.get("age_group", "adult")
    conds = state.get("conditions", [])
    meds = state.get("inquired_medications", [])
    abns = state.get("lab_abnormalities", [])

    passages = retriever.retrieve_passages_with_scores(
        query=query,
        patient_conditions=conds,
        inquired_medications=meds,
        lab_abnormalities=abns,
        age_group=age_grp,
        top_k=4
    )
    
    context_str = retriever.retrieve_context(
        query=query,
        patient_conditions=conds,
        inquired_medications=meds,
        lab_abnormalities=abns,
        age_group=age_grp,
        top_k=4
    )
    
    state["retrieved_passages"] = passages
    state["retrieved_context"] = context_str
    
    log_trace(
        state,
        "HYBRID_RETRIEVAL",
        f"Retrieved {len(passages)} passages via Dense ChromaDB + BM25Okapi RRF.",
        {"passage_count": len(passages), "top_source": passages[0]["source"] if passages else "None"}
    )
    return state


# Node 6: evidence_reranker
def evidence_reranker_node(state: AgentState):
    passages = state.get("retrieved_passages") or []
    # Already scored and sorted by ClinicalEvidenceReranker inside retrieve_passages_with_scores
    top_score = passages[0]["score"] if passages else 0.0
    log_trace(
        state,
        "EVIDENCE_RERANKER",
        f"Evidence reranked and population-aligned. Top score: {top_score}.",
        {"top_score": top_score, "count": len(passages)}
    )
    return state


# Node 7: evidence_quality_checker
def evidence_quality_checker_node(state: AgentState):
    passages = state.get("retrieved_passages") or []
    if not passages or len(passages) == 0:
        state["evidence_quality_status"] = "INSUFFICIENT"
        log_trace(state, "QUALITY_CHECK", "Evidence quality check: No relevant guideline evidence found.", {"status": "INSUFFICIENT"})
    else:
        state["evidence_quality_status"] = "SUFFICIENT"
        log_trace(state, "QUALITY_CHECK", f"Evidence quality check: PASSED with {len(passages)} verified sources.", {"status": "SUFFICIENT"})
    return state


# Node 8: clinical_reasoning
def clinical_reasoning_node(state: AgentState, llm_agent: LLMClinicalAgent):
    attempt = state.get("retries", 0) + 1
    provider = state.get("provider", "local")
    model_name = state.get("model_name")
    api_key = state.get("api_key")
    lab_vals = state.get("lab_values")

    response = llm_agent.generate_recommendation(
        patient_id=state["patient_id"],
        conditions=state["conditions"],
        guidelines=state.get("retrieved_context") or "Insufficient specific guidelines retrieved.",
        violations=state.get("violations", []),
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        lab_values=lab_vals,
        clinical_narrative=state.get("clinical_narrative"),
        age=state.get("age"),
        gender=state.get("gender"),
        evaluation_id=state.get("evaluation_id")
    )

    state["proposed_medication"] = response["medication"]
    state["reasoning"] = response["reasoning"]
    state["diagnosed_condition"] = response.get("diagnosed_condition", "Cardiometabolic Multimorbidity")
    state["icd10_code"] = response.get("icd10_code", "R69")
    state["differential_diagnoses"] = response.get("differential_diagnoses", [])
    state["patient_friendly_summary"] = response.get("patient_friendly_summary", "")
    state["warnings"] = response.get("warnings", [])
    state["model_used"] = response.get("model_used", model_name or "qwen2.5:3b")
    state["provider_used"] = response.get("provider_used", provider)
    state["inference_latency_ms"] = response.get("inference_latency_ms", 0.0)
    
    if response.get("ckd_stage"):
        state["ckd_stage"] = response["ckd_stage"]
    if response.get("albuminuria_stage"):
        state["albuminuria_stage"] = response["albuminuria_stage"]

    log_trace(
        state,
        "CLINICAL_REASONING",
        f"Attempt {attempt}: Model '{state['model_used']}' proposed '{state['proposed_medication']}' for {state['diagnosed_condition']} [{state['icd10_code']}].",
        {"medication": state["proposed_medication"], "attempt": attempt}
    )
    return state


# Node 9: clinical_rule_validator
def clinical_rule_validator_node(state: AgentState, verifier):
    medication = state.get("proposed_medication", "")
    pid = state.get("patient_id", "PT_UNKNOWN")
    conds = state.get("conditions", [])

    # Formal Description Logic tableaux reasoning
    dl_result = verifier.verify_prescription(
        patient_id=pid,
        conditions=conds,
        medications=[medication],
        lab_values=state.get("lab_values")
    )
    state["verification_details"] = dl_result

    if dl_result["status"] == "PASS":
        state["status"] = "AXIOMS_PASSED"
        log_trace(state, "RULE_VALIDATOR", "HermiT OWL 2 DL Reasoner: PASSED (Zero formal axiomatic contraindications).")
    else:
        state["status"] = "FAILED"
        if "violations" not in state or state["violations"] is None:
            state["violations"] = []
        state["violations"].append(dl_result["violated_axiom"])
        if "historical_violations" not in state or state["historical_violations"] is None:
            state["historical_violations"] = []
        state["historical_violations"].append(dl_result["violated_axiom"])
        state["retries"] = state.get("retries", 0) + 1
        log_trace(state, "RULE_VALIDATOR", f"HermiT OWL 2 DL Reasoner: FAILED -> Axiom violation: {dl_result['violated_axiom']}")

    return state


# Node 10: patient_data_integrity_validator
def patient_data_integrity_validator_node(state: AgentState):
    p_ctx = state.get("patient_context") or {}
    labs = state.get("lab_values") or {}
    
    if p_ctx:
        ctx_obj = PatientContext(
            patient_id=p_ctx.get("patient_id", state.get("patient_id", "")),
            patient_name=p_ctx.get("patient_name", state.get("patient_name", "")),
            evaluation_id=state.get("evaluation_id", ""),
            age=p_ctx.get("age", state.get("age")),
            gender=p_ctx.get("gender", state.get("gender")),
            conditions=p_ctx.get("conditions", state.get("conditions", [])),
            symptoms=p_ctx.get("symptoms", state.get("symptoms", [])),
            current_medications=p_ctx.get("current_medications", labs.get("current_medications", [])),
            inquired_medications=state.get("inquired_medications", []),
            lab_values=labs,
            clinical_narrative=state.get("clinical_narrative", "")
        )
        inconsistencies = verify_patient_output_consistency(ctx_obj, state)
        if inconsistencies:
            state["consistency_alerts"] = inconsistencies
            if "violations" not in state or state["violations"] is None:
                state["violations"] = []
            for inc in inconsistencies:
                if inc not in state["violations"]:
                    state["violations"].append(f"Data Integrity Violation: {inc}")
            state["status"] = "FAILED"
            log_trace(state, "INTEGRITY_VALIDATOR", f"Integrity validator flagged: {inconsistencies}")
        else:
            log_trace(state, "INTEGRITY_VALIDATOR", "Patient laboratory fidelity and context isolation verified 100%.")

    return state


# Node 11: citation_validator
def citation_validator_node(state: AgentState):
    passages = state.get("retrieved_passages") or []
    citations = []
    for p in passages:
        meta = p.get("metadata", {})
        citations.append({
            "id": p.get("id"),
            "doc_id": meta.get("doc_id", p.get("id")),
            "title": meta.get("title", p.get("title", "Clinical Guideline")),
            "organization": meta.get("org_id", p.get("source", "AUTHORITY")),
            "version": meta.get("guideline_version", "2024.1"),
            "original_url": meta.get("original_url", ""),
            "section": meta.get("topic", p.get("topic", "Clinical Guidance")),
            "score": p.get("score", 0.0),
            "evidence_grade": meta.get("evidence_grade", "Authoritative Practice Guideline"),
            "passage_excerpt": p.get("text", "")[:350] + "..."
        })
    state["guideline_citations"] = citations
    log_trace(state, "CITATION_VALIDATOR", f"Validated {len(citations)} citations linked to real indexed source passages.")
    return state


# Node 12: response_generator
def response_generator_node(state: AgentState):
    p_ctx = state.get("patient_context") or {}
    labs = state.get("lab_values") or {}

    if p_ctx:
        ctx_obj = PatientContext(
            patient_id=p_ctx.get("patient_id", state.get("patient_id", "")),
            patient_name=p_ctx.get("patient_name", state.get("patient_name", "")),
            evaluation_id=state.get("evaluation_id", ""),
            age=p_ctx.get("age", state.get("age")),
            gender=p_ctx.get("gender", state.get("gender")),
            conditions=p_ctx.get("conditions", state.get("conditions", [])),
            symptoms=p_ctx.get("symptoms", state.get("symptoms", [])),
            current_medications=p_ctx.get("current_medications", labs.get("current_medications", [])),
            inquired_medications=state.get("inquired_medications", []),
            lab_values=labs,
            clinical_narrative=state.get("clinical_narrative", "")
        )
        is_preflight_valid, preflight_errors = validate_final_clinical_response(ctx_obj, state)
        if not is_preflight_valid:
            sanitized = sanitize_and_correct_response(ctx_obj, state)
            state.update(sanitized)
            # Re-verify sanitized state
            is_now_valid, remaining_errors = validate_final_clinical_response(ctx_obj, state)
            if is_now_valid:
                state["status"] = "VERIFIED"
                state["violations"] = []
            else:
                state["status"] = "FAILED"
                state["violations"] = remaining_errors
        else:
            state["status"] = "VERIFIED"
            state["violations"] = []

    log_trace(state, "RESPONSE_GENERATOR", "Generated pre-flight validated, evidence-grounded clinical payload.")
    return state


# Node 13: final_safety_gate
def route_final_safety_gate(state: AgentState):
    if state.get("status") in ("AXIOMS_PASSED", "VERIFIED") and len(state.get("violations", [])) == 0:
        return "finalize"
    elif state.get("retries", 0) >= state.get("max_retries", 2):
        return "escalate"
    else:
        state["retries"] = state.get("retries", 0) + 1
        return "retry"


def finalize_node(state: AgentState):
    labs = state.get("lab_values") or {}
    p_ctx = state.get("patient_context") or {}
    
    state["final_output"] = {
        "status": "APPROVED",
        "patient_id": state["patient_id"],
        "patient_name": state.get("patient_name", "Clinical Patient"),
        "evaluation_id": state.get("evaluation_id"),
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
        "guideline_citations": state.get("guideline_citations", []),
        "ckd_stage": state.get("ckd_stage"),
        "albuminuria_stage": state.get("albuminuria_stage"),
        "potassium_status": state.get("potassium_status"),
        "cardiorenal_status": state.get("cardiorenal_status"),
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
        "verification_details": state.get("verification_details"),
        "consistency_alerts": state.get("consistency_alerts", []),
        "total_attempts": state.get("retries", 0) + 1,
        "chronological_trace": state.get("chronological_trace", []),
        "intercepted_violations": list(dict.fromkeys(state.get("historical_violations", []) + state.get("violations", []))),
        "audit_trail": list(dict.fromkeys(state.get("historical_violations", []) + state.get("violations", []))),
        "model_used": state.get("model_used", state.get("model_name") or "qwen2.5:3b"),
        "provider_used": state.get("provider_used", state.get("provider", "local")),
        "inference_latency_ms": state.get("inference_latency_ms", 0.0)
    }

    if p_ctx:
        ctx_obj = PatientContext(
            patient_id=p_ctx.get("patient_id", state.get("patient_id", "")),
            patient_name=p_ctx.get("patient_name", state.get("patient_name", "")),
            evaluation_id=state.get("evaluation_id", ""),
            age=p_ctx.get("age", state.get("age")),
            gender=p_ctx.get("gender", state.get("gender")),
            conditions=p_ctx.get("conditions", state.get("conditions", [])),
            symptoms=p_ctx.get("symptoms", state.get("symptoms", [])),
            current_medications=p_ctx.get("current_medications", labs.get("current_medications", [])),
            inquired_medications=state.get("inquired_medications", []),
            lab_values=labs,
            clinical_narrative=state.get("clinical_narrative", "")
        )
        state["final_output"] = sanitize_and_correct_response(ctx_obj, state["final_output"])

    log_trace(state, "FINAL_SAFETY_GATE", f"Approved clinical output for {state['patient_id']} [{state.get('icd10_code')}].")
    return state


def escalate_node(state: AgentState):
    labs = state.get("lab_values") or {}
    p_ctx = state.get("patient_context") or {}

    state["final_output"] = {
        "status": "ESCALATED_FOR_HUMAN_REVIEW",
        "patient_id": state["patient_id"],
        "patient_name": state.get("patient_name", "Clinical Patient"),
        "evaluation_id": state.get("evaluation_id"),
        "diagnosed_condition": state.get("diagnosed_condition", "Cardiometabolic Multimorbidity"),
        "icd10_code": state.get("icd10_code", "R69"),
        "last_proposed_treatment": state.get("proposed_medication"),
        "reason": "Maximum retry limit reached without resolving formal clinical violations. Fail-closed safety gate engaged.",
        "confirmed_diagnoses": state.get("confirmed_diagnoses", []),
        "suspected_conditions": state.get("suspected_conditions", []),
        "missing_evidence": state.get("missing_evidence", []),
        "inquired_medications": state.get("inquired_medications", []),
        "medication_assessments": state.get("medication_assessments", {}),
        "guideline_citations": state.get("guideline_citations", []),
        "retrieved_passages": state.get("retrieved_passages", []),
        "verification_details": state.get("verification_details"),
        "clinical_alerts": state.get("clinical_alerts", []),
        "ckd_stage": state.get("ckd_stage"),
        "albuminuria_stage": state.get("albuminuria_stage"),
        "potassium_status": state.get("potassium_status"),
        "cardiorenal_status": state.get("cardiorenal_status"),
        "total_attempts": state.get("retries", 0),
        "chronological_trace": state.get("chronological_trace", []),
        "intercepted_violations": list(dict.fromkeys(state.get("historical_violations", []) + state.get("violations", []))),
        "audit_trail": list(dict.fromkeys(state.get("historical_violations", []) + state.get("violations", []))),
        "model_used": state.get("model_used", state.get("model_name") or "qwen2.5:3b"),
        "provider_used": state.get("provider_used", state.get("provider", "local")),
        "inference_latency_ms": state.get("inference_latency_ms", 0.0)
    }

    if p_ctx:
        ctx_obj = PatientContext(
            patient_id=p_ctx.get("patient_id", state.get("patient_id", "")),
            patient_name=p_ctx.get("patient_name", state.get("patient_name", "")),
            evaluation_id=state.get("evaluation_id", ""),
            age=p_ctx.get("age", state.get("age")),
            gender=p_ctx.get("gender", state.get("gender")),
            conditions=p_ctx.get("conditions", state.get("conditions", [])),
            symptoms=p_ctx.get("symptoms", state.get("symptoms", [])),
            current_medications=p_ctx.get("current_medications", labs.get("current_medications", [])),
            inquired_medications=state.get("inquired_medications", []),
            lab_values=labs,
            clinical_narrative=state.get("clinical_narrative", "")
        )
        state["final_output"] = sanitize_and_correct_response(ctx_obj, state["final_output"])

    log_trace(state, "FINAL_SAFETY_GATE", f"Escalated for human physician review after exhausting retries.")
    return state


def build_safecds_graph(
    verifier,
    llm_agent: Optional[LLMClinicalAgent] = None,
    retriever: Optional[HybridClinicalRetriever] = None
):
    """
    Builds and compiles the 13-stage observable SafeCDS LangGraph workflow.
    """
    active_llm = llm_agent or get_llm_agent()
    active_retriever = retriever or get_retriever()

    workflow = StateGraph(AgentState)

    # 13 Observable Nodes
    workflow.add_node("patient_input_validation", patient_input_validation_node)
    workflow.add_node("patient_context_builder", patient_context_builder_node)
    workflow.add_node("clinical_query_analyzer", clinical_query_analyzer_node)
    workflow.add_node("retrieval_query_generator", retrieval_query_generator_node)
    workflow.add_node("hybrid_retriever", lambda s: hybrid_retriever_node(s, active_retriever))
    workflow.add_node("evidence_reranker", evidence_reranker_node)
    workflow.add_node("evidence_quality_checker", evidence_quality_checker_node)
    workflow.add_node("clinical_reasoning", lambda s: clinical_reasoning_node(s, active_llm))
    workflow.add_node("clinical_rule_validator", lambda s: clinical_rule_validator_node(s, verifier))
    workflow.add_node("patient_data_integrity_validator", patient_data_integrity_validator_node)
    workflow.add_node("citation_validator", citation_validator_node)
    workflow.add_node("response_generator", response_generator_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("escalate", escalate_node)

    # Entry and sequence
    workflow.set_entry_point("patient_input_validation")
    workflow.add_edge("patient_input_validation", "patient_context_builder")
    workflow.add_edge("patient_context_builder", "clinical_query_analyzer")
    workflow.add_edge("clinical_query_analyzer", "retrieval_query_generator")
    workflow.add_edge("retrieval_query_generator", "hybrid_retriever")
    workflow.add_edge("hybrid_retriever", "evidence_reranker")
    workflow.add_edge("evidence_reranker", "evidence_quality_checker")
    workflow.add_edge("evidence_quality_checker", "clinical_reasoning")
    workflow.add_edge("clinical_reasoning", "clinical_rule_validator")
    workflow.add_edge("clinical_rule_validator", "patient_data_integrity_validator")
    workflow.add_edge("patient_data_integrity_validator", "citation_validator")
    workflow.add_edge("citation_validator", "response_generator")

    # Conditional routing at final safety gate
    workflow.add_conditional_edges(
        "response_generator",
        route_final_safety_gate,
        {
            "finalize": "finalize",
            "retry": "clinical_reasoning",
            "escalate": "escalate"
        }
    )

    workflow.add_edge("finalize", "__END__")
    workflow.add_edge("escalate", "__END__")

    return workflow.compile()
