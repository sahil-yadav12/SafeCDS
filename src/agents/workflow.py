import time
import re
import json
from typing import TypedDict, List, Optional, Dict, Any, Callable, Tuple
from src.agents.llm_client import LLMClinicalAgent
from src.retrieval.hybrid_retriever import HybridClinicalRetriever

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
    """Initializes and normalizes patient intake profile."""
    pid = state.get("patient_id", "Patient_Unknown")
    conds = state.get("conditions", [])
    log_step(
        state,
        "INGESTION",
        f"Patient {pid} presentation ingested with documented conditions: {', '.join(conds) if conds else 'None'}",
        {"patient_id": pid, "conditions": conds}
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
    query = f"Management of patient with {conds_text}"
    
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
    
    log_step(
        state,
        "GENERATION",
        f"Attempt {attempt}: Proposing '{state['proposed_medication']}' for {state['diagnosed_condition']} [{state['icd10_code']}]",
        {"medication": state["proposed_medication"], "attempt": attempt, "provider": provider}
    )
    return state


def verify_node(state: AgentState, verifier):
    medication = state.get("proposed_medication", "")
    patient_id = state.get("patient_id", "Patient_Case")
    conditions = state.get("conditions", [])
    
    result = verifier.verify_prescription(
        patient_id=patient_id,
        conditions=conditions,
        medications=[medication]
    )
    
    state["verification_details"] = result
    
    if result["status"] == "PASS":
        state["status"] = "VERIFIED"
        log_step(
            state,
            "VERIFICATION",
            f"HermiT OWL 2 DL reasoner: PASSED (Zero formal contraindications detected).",
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
            "VERIFICATION",
            f"HermiT OWL 2 DL reasoner: FAILED -> Intercepted contraindication: {result['violated_axiom']}",
            {"status": "FAIL", "axiom": result["violated_axiom"], "retry": state["retries"]}
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
    
    state["final_output"] = {
        "status": "APPROVED",
        "patient_id": state["patient_id"],
        "patient_name": state.get("patient_name", "Clinical Patient"),
        "diagnosed_condition": state.get("diagnosed_condition", "Cardiometabolic Multimorbidity"),
        "icd10_code": state.get("icd10_code", "R69"),
        "differential_diagnoses": state.get("differential_diagnoses", []),
        "recommended_treatment": state.get("proposed_medication"),
        "clinical_rationale": state.get("reasoning"),
        "patient_friendly_summary": state.get("patient_friendly_summary"),
        "warnings": state.get("warnings", []),
        "retrieved_passages": state.get("retrieved_passages", []),
        "evidence_context": state.get("retrieved_context"),
        "verification_details": state.get("verification_details"),
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
        "retrieved_passages": state.get("retrieved_passages", []),
        "verification_details": state.get("verification_details"),
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
    Builds and compiles the SafeCDS LangGraph state machine.
    Enforces the fail-closed neuro-symbolic verification and self-correction loop.
    """
    active_llm = llm_agent or get_llm_agent()
    active_retriever = retriever or get_retriever()
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("retrieve", lambda s: retrieve_node(s, active_retriever))
    workflow.add_node("generate_hypothesis", lambda s: generate_hypothesis_node(s, active_llm))
    workflow.add_node("verify", lambda s: verify_node(s, verifier))
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("escalate", escalate_node)
    
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "retrieve")
    workflow.add_edge("retrieve", "generate_hypothesis")
    workflow.add_edge("generate_hypothesis", "verify")
    
    workflow.add_conditional_edges(
        "verify",
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
