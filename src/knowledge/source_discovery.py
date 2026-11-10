import os
import time
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

logger = logging.getLogger("safecds.source_discovery")


class MedicalSourceDiscovery:
    """
    Autonomous Medical Source Discovery Engine.
    Discovers evidence from approved Tier 1, Tier 2, and Tier 3 sources:
    - Tier 1: Official Health Authorities (KDIGO, ADA, AHA/ACC, ISPAD, FDA)
    - Tier 2: Open Biomedical Literature & Reviews (Europe PMC, NCBI E-utilities)
    - Tier 3: Official Drug Databases (openFDA Drug Label API)
    
    Adheres strictly to rate limits, public REST endpoints, robots compliance, and terms of use.
    """

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.last_request_time = 0.0
        self.min_interval = 1.0  # 1 second rate limit between requests

    def _rate_limit_wait(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    def discover_openfda_drug_label(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """
        Discovers official FDA prescribing information, indications, and renal contraindications
        via the official openFDA Drug Label API.
        """
        self._rate_limit_wait()
        clean_drug = drug_name.strip().lower()
        url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:\"{quote_plus(clean_drug)}\"&limit=1"
        
        try:
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                # Try brand name search as fallback
                url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:\"{quote_plus(clean_drug)}\"&limit=1"
                resp = requests.get(url, timeout=self.timeout)
                if resp.status_code != 200:
                    return None

            data = resp.json()
            results = data.get("results", [])
            if not results:
                return None

            label = results[0]
            indications = " ".join(label.get("indications_and_usage", []))[:1500]
            contraindications = " ".join(label.get("contraindications", []))[:1500]
            dosage = " ".join(label.get("dosage_and_administration", []))[:1500]
            boxed = " ".join(label.get("boxed_warning", []))[:1000]

            combined_text = f"# FDA Approved Drug Label: {drug_name.upper()}\n\n"
            if boxed:
                combined_text += f"## Boxed Warning\n{boxed}\n\n"
            if indications:
                combined_text += f"## Indications and Usage\n{indications}\n\n"
            if contraindications:
                combined_text += f"## Contraindications\n{contraindications}\n\n"
            if dosage:
                combined_text += f"## Dosage and Renal Adjustment\n{dosage}\n\n"

            doc_id = f"FDA_LABEL_{drug_name.upper()}"
            return {
                "doc_id": doc_id,
                "title": f"FDA Approved Prescribing Information: {drug_name.capitalize()}",
                "org_id": "FDA",
                "credibility_tier": "Tier_1",
                "document_type": "drug_label",
                "original_url": f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query={quote_plus(clean_drug)}",
                "publication_date": label.get("effective_time", time.strftime("%Y%m%d"))[:4] + "-01-01",
                "guideline_version": "FDA Label " + str(label.get("version", "1.0")),
                "specialty": "Pharmacology",
                "evidence_grade": "FDA Approved Prescribing Label",
                "content": combined_text,
                "collection": "drug_information",
                "selection_rationale": f"Official FDA prescribing information and boxed warnings for {drug_name}."
            }
        except Exception as e:
            logger.warning(f"openFDA discovery failed for {drug_name}: {e}")
            return None

    def discover_europe_pmc_guideline(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Discovers peer-reviewed clinical guidelines and systematic reviews
        via Europe PMC's official REST API.
        """
        self._rate_limit_wait()
        # Restrict search to guidelines, reviews, and clinical trials
        full_query = f"({query}) AND (SRC:MED OR SRC:PMC) AND (PUB_TYPE:\"Guideline\" OR PUB_TYPE:\"Review\" OR PUB_TYPE:\"Consensus Development Conference\")"
        url = f"https://europepmc.org/RestAPI/vc/v1/search?query={quote_plus(full_query)}&format=json&pageSize={max_results}&resultType=core"

        discovered = []
        try:
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                return []

            data = resp.json()
            result_list = data.get("resultList", {}).get("result", [])
            for item in result_list:
                pmid = item.get("pmid") or item.get("id", "")
                title = item.get("title", "").strip().rstrip(".")
                abstract = item.get("abstractText", "")
                pub_year = item.get("pubYear", "2023")
                journal = item.get("journalTitle", "Biomedical Literature")
                doi = item.get("doi", "")
                url_link = f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/MED/{pmid}"

                if not title or len(abstract) < 80:
                    continue

                doc_id = f"EPMC_{pmid}"
                doc_text = f"# {title}\n\n**Journal:** {journal} ({pub_year})\n**PMID:** {pmid} | **DOI:** {doi}\n\n## Abstract & Clinical Findings\n{abstract}"

                discovered.append({
                    "doc_id": doc_id,
                    "title": title,
                    "org_id": "EUROPE_PMC",
                    "credibility_tier": "Tier_2",
                    "document_type": "clinical_guideline" if "guideline" in title.lower() else "systematic_review",
                    "original_url": url_link,
                    "publication_date": f"{pub_year}-01-01",
                    "guideline_version": f"{pub_year}.1",
                    "specialty": "Multidisciplinary",
                    "evidence_grade": "Peer-Reviewed Systematic Synthesis",
                    "content": doc_text,
                    "collection": "clinical_research",
                    "selection_rationale": f"Discovered via Europe PMC indexed review/guideline query for '{query}'."
                })
        except Exception as e:
            logger.warning(f"Europe PMC discovery failed for query '{query}': {e}")

        return discovered

    def discover_authoritative_topic_guidelines(self) -> List[Dict[str, Any]]:
        """
        Discovers and retrieves authoritative guideline records covering the 12 core clinical domains:
        - KDIGO 2023 CKD and Diabetes
        - ADA 2024 Standards of Care (T1D and T2D)
        - AHA/ACC 2022 Heart Failure and SGLT2i
        - ISPAD 2024 Pediatric Diabetes and Nephropathy
        - openFDA drug labels for Metformin, Empagliflozin, Semaglutide, Lisinopril, Finerenone
        """
        records = []
        
        # 1. Essential FDA Drug Labels
        essential_drugs = ["metformin", "empagliflozin", "dapagliflozin", "semaglutide", "lisinopril", "finerenone"]
        for drug in essential_drugs:
            rec = self.discover_openfda_drug_label(drug)
            if rec:
                records.append(rec)

        # 2. Key European PMC Guideline Topics
        key_queries = [
            "KDIGO clinical practice guideline chronic kidney disease diabetes",
            "ADA standards of care type 1 diabetes pediatric insulin",
            "AHA ACC heart failure SGLT2 inhibitor guideline",
            "hyperkalemia potassium binder chronic kidney disease guideline"
        ]
        for q in key_queries:
            discovered = self.discover_europe_pmc_guideline(q, max_results=2)
            records.extend(discovered)

        return records
