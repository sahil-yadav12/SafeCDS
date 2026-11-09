import os
import sys
import json
import time
from src.reasoning.verifier import ClinicalVerifier
from src.agents.workflow import build_safecds_graph
from src.retrieval.hybrid_retriever import HybridClinicalRetriever

ONTOLOGY_PATH = os.path.join("ontologies", "cardiometabolic_core.owl")
GUIDELINES_PATH = os.path.join("data", "guidelines.json")


def main():
    if "--web" in sys.argv:
        import uvicorn
        print("Starting SafeCDS Web Server at http://127.0.0.1:8000 ...")
        uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=False)
        return

    print("Initializing SafeCDS Reasoning Engine...")
    verifier = ClinicalVerifier(ontology_path=ONTOLOGY_PATH)
    retriever = HybridClinicalRetriever(guidelines_path=GUIDELINES_PATH)

    print("Building LangGraph Orchestration State Machine...")
    safecds_app = build_safecds_graph(verifier, retriever=retriever)

    # Test Patient: Patient presenting with CKD Stage 4/5 (where standard Type-2 diabetes first-line Metformin is unsafe)
    patient_input = {
        "patient_id": "Patient_Cardio_102",
        "patient_name": "Eleanor Vance",
        "age": 72,
        "gender": "Female",
        "conditions": ["CKD_Stage_4_5", "Type_2_Diabetes"],
        "retrieved_context": None,
        "proposed_medication": None,
        "reasoning": None,
        "retries": 0,
        "max_retries": 3,
        "violations": [],
        "status": "PENDING",
        "final_output": {}
    }

    print("\nExecuting SafeCDS Pipeline with Dynamic Reasoning & Formal HermiT Verification...")
    start_t = time.perf_counter()
    result = safecds_app.invoke(patient_input)
    elapsed_ms = (time.perf_counter() - start_t) * 1000

    print("\n" + "=" * 55)
    print("             FINAL DECISION AUDIT            ")
    print("=" * 55)
    print(json.dumps(result["final_output"], indent=4))
    print(f"\nExecution Latency: {elapsed_ms:.2f} ms")
    print(f"Verifier Cache Stats: {verifier.cache_stats}")
    print("\nTip: To start the interactive web application, run: python main.py --web")


if __name__ == "__main__":
    main()
