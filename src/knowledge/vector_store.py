import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger("safecds.vector_store")


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Local Ollama Embedding Function utilizing nomic-embed-text (768 dimensions).
    Conforms directly to Chroma's EmbeddingFunction protocol.
    """
    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def name(self) -> str:
        return "ollama_nomic_embed_text"

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: Embeddings = []
        url = f"{self.base_url}/api/embeddings"
        for text in input:
            try:
                res = requests.post(url, json={"model": self.model_name, "prompt": text}, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    embeddings.append(data.get("embedding", []))
                else:
                    embeddings.append([])
            except Exception as e:
                logger.warning(f"Ollama embedding failed for text excerpt: {e}")
                embeddings.append([])
        return embeddings


class PersistentMedicalVectorStore:
    """
    Persistent ChromaDB Vector Store for Medical Knowledge.
    Maintains domain-partitioned collections:
    1. clinical_guidelines
    2. drug_information
    3. clinical_research
    4. diagnostic_criteria
    5. clinical_terminology
    6. safety_and_contraindications
    
    Uses Ollama's nomic-embed-text for consistent 768-dimensional embeddings in cosine metric space.
    Supports metadata filtering (age_group, specialty, is_active, evidence_grade),
    document version tracking, and deactivation of superseded chunks.
    """

    COLLECTIONS = [
        "clinical_guidelines",
        "drug_information",
        "clinical_research",
        "diagnostic_criteria",
        "clinical_terminology",
        "safety_and_contraindications"
    ]

    def __init__(self, persist_dir: str = "data/chroma_db", ollama_base_url: str = "http://localhost:11434"):
        self.persist_dir = os.path.abspath(persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)
        self.ollama_base_url = ollama_base_url
        self.embedding_fn = OllamaEmbeddingFunction(model_name="nomic-embed-text", base_url=ollama_base_url)
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self._init_collections()

    def _init_collections(self):
        """Initializes domain collections with Ollama embedding function."""
        self.collections: Dict[str, Any] = {}
        for col_name in self.COLLECTIONS:
            self.collections[col_name] = self.client.get_or_create_collection(
                name=col_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )

    def add_chunks(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]]
    ) -> int:
        """
        Adds structured medical chunks to the target collection.
        Each chunk must have: id, text, metadata.
        """
        if collection_name not in self.collections:
            collection_name = "clinical_guidelines"
            
        coll = self.collections[collection_name]
        
        ids = []
        documents = []
        metadatas = []
        
        for c in chunks:
            chunk_id = c["id"]
            doc_text = c["text"]
            meta = c.get("metadata", {})
            
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif v is None:
                    clean_meta[k] = ""
                else:
                    clean_meta[k] = str(v)
            
            ids.append(chunk_id)
            documents.append(doc_text)
            metadatas.append(clean_meta)

        if ids:
            coll.upsert(ids=ids, documents=documents, metadatas=metadatas)
            
        return len(ids)

    def search_semantic(
        self,
        query: str,
        collection_name: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Searches across one or all collections using semantic similarity.
        Applies metadata filters (e.g. {'is_active': 1, 'age_group': 'adult'}).
        """
        target_colls = [self.collections[collection_name]] if (collection_name and collection_name in self.collections) else list(self.collections.values())
        
        all_results = []
        where_clause = {}
        if metadata_filter:
            clean_items = []
            for k, v in metadata_filter.items():
                clean_items.append({k: {"$eq": v}})
            if len(clean_items) == 1:
                where_clause = {list(metadata_filter.keys())[0]: list(metadata_filter.values())[0]}
            elif len(clean_items) > 1:
                where_clause = {"$and": clean_items}

        for coll in target_colls:
            try:
                count = coll.count()
                if count == 0:
                    continue
                    
                n_res = min(top_k, count)
                kwargs = {
                    "query_texts": [query],
                    "n_results": n_res,
                    "include": ["documents", "metadatas", "distances"]
                }
                if where_clause:
                    kwargs["where"] = where_clause
                    
                res = coll.query(**kwargs)
                    
                if res and res.get("ids") and res["ids"][0]:
                    for i in range(len(res["ids"][0])):
                        doc_id = res["ids"][0][i]
                        doc_text = res["documents"][0][i]
                        meta = res["metadatas"][0][i]
                        dist = res["distances"][0][i] if res.get("distances") else 0.0
                        # In cosine space: dist = 1 - cosine_similarity. Similarity = max(0, 1 - dist)
                        similarity = max(0.0, min(1.0, 1.0 - float(dist)))
                        all_results.append({
                            "id": doc_id,
                            "text": doc_text,
                            "metadata": meta,
                            "score": round(float(similarity), 4),
                            "collection": coll.name
                        })
            except Exception as e:
                logger.warning(f"Error querying collection {coll.name}: {e}")

        # Sort combined results by similarity descending
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def deactivate_document(self, doc_id: str):
        """Marks all chunks belonging to doc_id as inactive (is_active = 0)."""
        for coll in self.collections.values():
            try:
                matching = coll.get(where={"doc_id": doc_id}, include=["metadatas", "documents"])
                if matching and matching.get("ids"):
                    updated_metas = []
                    for m in matching["metadatas"]:
                        m["is_active"] = 0
                        updated_metas.append(m)
                    coll.update(ids=matching["ids"], metadatas=updated_metas)
            except Exception as e:
                logger.error(f"Failed to deactivate doc_id {doc_id} in {coll.name}: {e}")

    def delete_document(self, doc_id: str):
        """Permanently deletes chunks belonging to doc_id across all collections."""
        for coll in self.collections.values():
            try:
                matching = coll.get(where={"doc_id": doc_id})
                if matching and matching.get("ids"):
                    coll.delete(ids=matching["ids"])
            except Exception as e:
                logger.error(f"Failed to delete doc_id {doc_id} in {coll.name}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Returns chunk counts and status across collections."""
        stats = {}
        total = 0
        for name, coll in self.collections.items():
            count = coll.count()
            stats[name] = count
            total += count
        stats["total_chunks"] = total
        stats["persist_dir"] = self.persist_dir
        return stats
