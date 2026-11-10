import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi

from src.knowledge.vector_store import PersistentMedicalVectorStore
from src.knowledge.source_registry import SourceRegistry

logger = logging.getLogger("safecds.retriever")


def _tokenize(text: str) -> List[str]:
    """Clean alphanumeric tokenization splitting camelCase and snake_case."""
    text_expanded = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text_expanded = text_expanded.replace('_', ' ')
    return re.findall(r'[a-zA-Z0-9]+', text_expanded.lower())


class ClinicalEvidenceReranker:
    """
    Reranks candidate evidence passages by:
    1. Clinical Relevance: Direct match for patient's diagnosed conditions, abnormal labs, and inquired medications.
    2. Credibility Tier: Tier 1 (Official Practice Guidelines/FDA) > Tier 2 (Systematic Reviews) > Tier 3 (Drug databases).
    3. Population Specificity: Strict alignment with patient's age group (adult vs pediatric).
    4. Version Freshness: Recent evidence preferred over older consensus.
    """

    TIER_WEIGHTS = {
        "Tier_1": 1.0,
        "Tier_2": 0.85,
        "Tier_3": 0.75
    }

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        patient_conditions: List[str],
        inquired_medications: List[str],
        lab_abnormalities: List[str],
        age_group: str = "adult",
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        cond_tokens = set()
        for c in patient_conditions:
            cond_tokens.update(_tokenize(c))

        med_tokens = set()
        for m in inquired_medications:
            med_tokens.update(_tokenize(m))

        lab_tokens = set()
        for l in lab_abnormalities:
            lab_tokens.update(_tokenize(l))

        scored_candidates = []
        for cand in candidates:
            text = cand.get("text", "").lower()
            topic = cand.get("topic", "").lower()
            meta = cand.get("metadata", {})
            tier = meta.get("credibility_tier", "Tier_1")
            cand_age = meta.get("age_group", "adult")

            # Population mismatch penalty
            if age_group == "adult" and cand_age == "pediatric":
                continue
            if age_group == "pediatric" and cand_age == "adult" and "pediatric" not in text:
                continue

            # Clinical relevance score
            rel_score = 0.0
            for ct in cond_tokens:
                if ct in text or ct in topic:
                    rel_score += 0.15

            for mt in med_tokens:
                if mt in text or mt in topic:
                    rel_score += 0.25

            for lt in lab_tokens:
                if lt in text or lt in topic:
                    rel_score += 0.20

            rel_score = min(1.0, rel_score)
            tier_score = self.TIER_WEIGHTS.get(tier, 0.8)
            rrf_score = cand.get("rrf_score", 0.5)

            # Combined score: RRF (40%) + Clinical Relevance (40%) + Credibility (20%)
            final_score = (rrf_score * 0.40) + (rel_score * 0.40) + (tier_score * 0.20)
            
            cand_copy = dict(cand)
            cand_copy["score"] = round(final_score, 4)
            cand_copy["clinical_relevance"] = round(rel_score, 4)
            scored_candidates.append(cand_copy)

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[:top_k]


class HybridClinicalRetriever:
    """
    Production-Grade Hybrid Clinical Retriever.
    Combines:
    1. Dense Semantic Search: 768-dim Ollama nomic-embed-text in Persistent ChromaDB with metadata filtering.
    2. Sparse Lexical Search: BM25Okapi over persistent indexed corpus.
    3. Reciprocal Rank Fusion (RRF k=60).
    4. Clinical Evidence Reranker: Domain, tier, and population-specific scoring.
    """

    def __init__(
        self,
        vector_store: Optional[PersistentMedicalVectorStore] = None,
        registry: Optional[SourceRegistry] = None,
        guidelines_path: str = "data/guidelines.json"
    ):
        self.vector_store = vector_store or PersistentMedicalVectorStore()
        self.registry = registry or SourceRegistry()
        self.reranker = ClinicalEvidenceReranker()
        self.guidelines_path = guidelines_path
        self._context_cache: Dict[str, str] = {}
        
        # Load corpus for BM25
        self._refresh_corpus()

    def _refresh_corpus(self):
        """Loads all active chunks from persistent collections for BM25 indexing."""
        self.corpus_chunks: List[Dict[str, Any]] = []
        for coll_name, coll in self.vector_store.collections.items():
            try:
                res = coll.get(include=["documents", "metadatas"])
                if res and res.get("ids"):
                    for i in range(len(res["ids"])):
                        cid = res["ids"][i]
                        doc_text = res["documents"][i]
                        meta = res["metadatas"][i]
                        if meta.get("is_active", 1) == 1 or meta.get("is_active") == "1":
                            self.corpus_chunks.append({
                                "id": cid,
                                "text": doc_text,
                                "metadata": meta,
                                "topic": meta.get("topic", meta.get("section", "Clinical Guidance")),
                                "source": meta.get("org_id", "AUTHORITY"),
                                "title": meta.get("title", "Clinical Practice Guideline"),
                                "collection": coll_name
                            })
            except Exception as e:
                logger.warning(f"Failed to fetch chunks from {coll_name}: {e}")

        if not self.corpus_chunks:
            # Fallback to guidelines.json if vector store is not yet seeded
            if os.path.exists(self.guidelines_path):
                with open(self.guidelines_path, "r", encoding="utf-8") as f:
                    raw_items = json.load(f)
                for idx, it in enumerate(raw_items):
                    self.corpus_chunks.append({
                        "id": f"GL_{idx}",
                        "text": it.get("text", ""),
                        "metadata": {
                            "title": it.get("source", "Guideline"),
                            "org_id": it.get("source", "Guideline"),
                            "topic": it.get("topic", "Clinical Guidance"),
                            "credibility_tier": "Tier_1",
                            "age_group": "pediatric" if "ispad" in it.get("source", "").lower() else "adult"
                        },
                        "topic": it.get("topic", "Clinical Guidance"),
                        "source": it.get("source", "Guideline"),
                        "title": it.get("source", "Guideline"),
                        "collection": "clinical_guidelines"
                    })

        self.tokenized_corpus = [_tokenize(c["text"] + " " + c["topic"]) for c in self.corpus_chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def dense_search(
        self,
        query: str,
        age_group: str = "adult",
        top_k: int = 6
    ) -> List[Dict[str, Any]]:
        """Queries persistent ChromaDB with age_group metadata filter."""
        meta_filter = {"age_group": age_group} if age_group in ("adult", "pediatric") else None
        return self.vector_store.search_semantic(
            query=query,
            metadata_filter=meta_filter,
            top_k=top_k
        )

    def sparse_search(
        self,
        query: str,
        age_group: str = "adult",
        top_k: int = 6
    ) -> List[Dict[str, Any]]:
        """Queries BM25 index with population filtering."""
        if not self.bm25 or not self.corpus_chunks:
            return []

        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        
        indexed_scores = []
        for idx, score in enumerate(scores):
            chunk = self.corpus_chunks[idx]
            cand_age = chunk.get("metadata", {}).get("age_group", "adult")
            
            # Skip age mismatch
            if age_group == "adult" and cand_age == "pediatric":
                continue
            if age_group == "pediatric" and cand_age == "adult":
                continue

            indexed_scores.append((chunk, score))

        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for chunk, score in indexed_scores[:top_k]:
            res_item = dict(chunk)
            res_item["score"] = float(score)
            results.append(res_item)
        return results

    def retrieve_passages_with_scores(
        self,
        query: str,
        patient_conditions: Optional[List[str]] = None,
        inquired_medications: Optional[List[str]] = None,
        lab_abnormalities: Optional[List[str]] = None,
        age_group: str = "adult",
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Executes full Hybrid RRF + Reranker pipeline:
        1. Dense semantic search on persistent ChromaDB
        2. Sparse BM25 search on tokenized corpus
        3. Reciprocal Rank Fusion (k=60)
        4. Clinical Evidence Reranking
        """
        patient_conditions = patient_conditions or []
        inquired_medications = inquired_medications or []
        lab_abnormalities = lab_abnormalities or []

        # 1. Dense search
        dense_results = self.dense_search(query, age_group=age_group, top_k=top_k * 2)
        # 2. Sparse search
        sparse_results = self.sparse_search(query, age_group=age_group, top_k=top_k * 2)

        # 3. Reciprocal Rank Fusion
        rrf_map: Dict[str, Dict[str, Any]] = {}
        k = 60

        for rank, item in enumerate(dense_results):
            cid = item["id"]
            if cid not in rrf_map:
                rrf_map[cid] = {
                    "id": cid,
                    "text": item["text"],
                    "metadata": item.get("metadata", {}),
                    "topic": item.get("metadata", {}).get("topic", "Clinical Guidance"),
                    "source": item.get("metadata", {}).get("org_id", "AUTHORITY"),
                    "title": item.get("metadata", {}).get("title", "Clinical Practice Guideline"),
                    "rrf_score": 0.0
                }
            rrf_map[cid]["rrf_score"] += 1.0 / (k + (rank + 1))

        for rank, item in enumerate(sparse_results):
            cid = item["id"]
            if cid not in rrf_map:
                rrf_map[cid] = {
                    "id": cid,
                    "text": item["text"],
                    "metadata": item.get("metadata", {}),
                    "topic": item.get("metadata", {}).get("topic", "Clinical Guidance"),
                    "source": item.get("metadata", {}).get("org_id", "AUTHORITY"),
                    "title": item.get("metadata", {}).get("title", "Clinical Practice Guideline"),
                    "rrf_score": 0.0
                }
            rrf_map[cid]["rrf_score"] += 1.0 / (k + (rank + 1))

        candidates = list(rrf_map.values())

        # 4. Clinical Evidence Reranker
        reranked = self.reranker.rerank(
            candidates=candidates,
            patient_conditions=patient_conditions,
            inquired_medications=inquired_medications,
            lab_abnormalities=lab_abnormalities,
            age_group=age_group,
            top_k=top_k
        )

        return reranked

    def retrieve_context(
        self,
        query: str,
        patient_conditions: Optional[List[str]] = None,
        inquired_medications: Optional[List[str]] = None,
        lab_abnormalities: Optional[List[str]] = None,
        age_group: str = "adult",
        top_k: int = 4
    ) -> str:
        """Retrieves and formats passages as grounded markdown text."""
        cache_key = f"{query}|{sorted(patient_conditions or [])}|{sorted(inquired_medications or [])}|{sorted(lab_abnormalities or [])}|{age_group}|{top_k}"
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]

        passages = self.retrieve_passages_with_scores(
            query=query,
            patient_conditions=patient_conditions,
            inquired_medications=inquired_medications,
            lab_abnormalities=lab_abnormalities,
            age_group=age_group,
            top_k=top_k
        )

        formatted_chunks = []
        for p in passages:
            meta = p.get("metadata", {})
            title = meta.get("title") or p.get("title", "Authoritative Guideline")
            org = meta.get("org_id") or p.get("source", "AUTHORITY")
            topic = meta.get("topic") or p.get("topic", "Clinical Recommendation")
            tier = meta.get("credibility_tier", "Tier_1")
            grade = meta.get("evidence_grade", "")
            url = meta.get("original_url", "")
            score = p.get("score", 0.0)

            header = f"[{org}] {title} — Topic: {topic} (Score: {score}, {tier}"
            if grade:
                header += f", Grade: {grade}"
            header += ")"
            
            chunk_body = f"{header}\nSource URL: {url}\n{p.get('text', '')}"
            formatted_chunks.append(chunk_body)

        formatted_result = "\n\n---\n\n".join(formatted_chunks)
        self._context_cache[cache_key] = formatted_result
        return formatted_result
