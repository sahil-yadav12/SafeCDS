import os
import time
import logging
from typing import List, Dict, Any, Optional

from src.knowledge.source_registry import SourceRegistry
from src.knowledge.vector_store import PersistentMedicalVectorStore
from src.knowledge.ingest_pipeline import ClinicalDocumentIngestionPipeline
from src.knowledge.source_discovery import MedicalSourceDiscovery

logger = logging.getLogger("safecds.acquisition_agent")


class AutonomousKnowledgeAcquisitionAgent:
    """
    Autonomous Medical Knowledge Acquisition Agent.
    Coordinates discovery, validation, version comparison, and incremental ingestion:
    Discovered -> Retrieved -> Extracted -> Validated -> Indexed -> Active.
    """

    def __init__(
        self,
        registry: Optional[SourceRegistry] = None,
        vector_store: Optional[PersistentMedicalVectorStore] = None,
        discovery: Optional[MedicalSourceDiscovery] = None,
        ingestion_pipeline: Optional[ClinicalDocumentIngestionPipeline] = None
    ):
        self.registry = registry or SourceRegistry()
        self.vector_store = vector_store or PersistentMedicalVectorStore()
        self.discovery = discovery or MedicalSourceDiscovery()
        self.pipeline = ingestion_pipeline or ClinicalDocumentIngestionPipeline(
            registry=self.registry,
            vector_store=self.vector_store
        )

    def sync_initial_knowledge_base(self, guidelines_path: str = "data/guidelines.json") -> Dict[str, Any]:
        """
        One-time or startup initialization:
        1. Migrates and indexes the canonical guidelines (KDIGO 2023, ADA 2024, AHA/ACC 2022, ISPAD 2024).
        2. Discovers and indexes openFDA prescribing labels for core medications.
        """
        logger.info("Starting initial medical knowledge base synchronization...")
        corpus_res = self.pipeline.ingest_existing_corpus(guidelines_path)
        
        # Discover openFDA labels for core cardiometabolic medications
        fda_results = []
        for drug in ["metformin", "empagliflozin", "lisinopril"]:
            label = self.discovery.discover_openfda_drug_label(drug)
            if label:
                ing_res = self.pipeline.ingest_document(
                    doc_metadata=label,
                    raw_text=label["content"],
                    target_collection="drug_information"
                )
                fda_results.append(ing_res)

        return {
            "status": "INITIALIZED",
            "canonical_guidelines": corpus_res,
            "fda_labels_ingested": len(fda_results),
            "total_chunks": self.vector_store.get_stats()
        }

    def run_discovery_cycle(
        self,
        inquire_drugs: Optional[List[str]] = None,
        literature_queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes an autonomous discovery and ingestion cycle.
        1. Checks FDA drug labels for inquired drugs.
        2. Queries Europe PMC for systematic reviews on missing topics.
        3. Enforces 6-stage lifecycle validation.
        """
        drugs = inquire_drugs or ["dapagliflozin", "semaglutide", "finerenone"]
        discovered_records = []
        ingestion_results = []

        # 1. FDA Drug Labels
        for d in drugs:
            rec = self.discovery.discover_openfda_drug_label(d)
            if rec:
                discovered_records.append(rec)

        # 2. Peer-Reviewed Biomedical Evidence
        queries = literature_queries or [
            "KDIGO SGLT2 inhibitor chronic kidney disease guideline",
            "ISPAD pediatric diabetic kidney disease albuminuria"
        ]
        for q in queries:
            epmc_records = self.discovery.discover_europe_pmc_guideline(q, max_results=1)
            discovered_records.extend(epmc_records)

        # 3. Ingestion & Validation
        for doc in discovered_records:
            res = self.pipeline.ingest_document(
                doc_metadata=doc,
                raw_text=doc.get("content", ""),
                target_collection=doc.get("collection", "clinical_research")
            )
            ingestion_results.append(res)

        return {
            "status": "DISCOVERY_CYCLE_COMPLETE",
            "records_discovered": len(discovered_records),
            "records_ingested": len(ingestion_results),
            "results": ingestion_results,
            "vector_store_stats": self.vector_store.get_stats(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def deactivate_superseded_source(self, doc_id: str, reason: str) -> Dict[str, Any]:
        """Deactivates superseded document in registry and vector store."""
        self.registry.supersede_document(doc_id, new_doc_id="SUPERSEDED", reason=reason)
        self.vector_store.deactivate_document(doc_id)
        return {
            "status": "DEACTIVATED",
            "doc_id": doc_id,
            "reason": reason
        }
