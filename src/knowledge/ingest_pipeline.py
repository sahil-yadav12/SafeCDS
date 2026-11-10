import os
import re
import time
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple

from src.knowledge.source_registry import SourceRegistry
from src.knowledge.vector_store import PersistentMedicalVectorStore

try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except ImportError:
    PDFIUM_AVAILABLE = False


class ClinicalDocumentIngestionPipeline:
    """
    Robust Clinical Document Processing & Ingestion Pipeline.
    Preserves:
    - Tables (dosing matrices, staging criteria, threshold grids)
    - Inequality signs (<, >, <=, >=, <=, >=)
    - Laboratory units (mL/min/1.73m², mg/g, mmol/L, %)
    - Clinical sections and recommendation contexts
    
    Enforces the full 6-stage lifecycle:
    Discovered -> Retrieved -> Extracted -> Validated -> Indexed -> Active.
    """

    def __init__(
        self,
        registry: Optional[SourceRegistry] = None,
        vector_store: Optional[PersistentMedicalVectorStore] = None
    ):
        self.registry = registry or SourceRegistry()
        self.vector_store = vector_store or PersistentMedicalVectorStore()

    def extract_text_from_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extracts text from PDF, text, or markdown file preserving structure and page numbers."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            if not PDFIUM_AVAILABLE:
                raise RuntimeError("pypdfium2 is required for PDF extraction.")
            pdf = pdfium.PdfDocument(file_path)
            num_pages = len(pdf)
            page_texts = []
            for p_idx in range(num_pages):
                page = pdf[p_idx]
                text_page = page.get_textpage()
                raw_text = text_page.get_text_range()
                if raw_text.strip():
                    page_texts.append(f"\n[Page {p_idx + 1}]\n{raw_text}")
            full_text = "\n".join(page_texts)
            return full_text, {"format": "pdf", "pages": num_pages}
        else:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return content, {"format": ext.lstrip(".")}

    @staticmethod
    def clean_text_preserve_clinical(text: str) -> str:
        """Cleans extraction artifacts without altering inequality signs, units, or table formatting."""
        # Normalize non-breaking spaces
        text = text.replace('\xa0', ' ')
        # Standardize inequality signs
        text = text.replace('≤', '<=').replace('≥', '>=')
        # Preserve table pipes and newlines while collapsing excessive spaces
        lines = [line.strip() for line in text.split('\n')]
        cleaned_lines = []
        for line in lines:
            if not line:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
            else:
                # Collapse horizontal tabs/spaces
                clean_line = re.sub(r'[ \t]+', ' ', line)
                cleaned_lines.append(clean_line)
        return "\n".join(cleaned_lines)

    def split_into_clinical_chunks(
        self,
        text: str,
        doc_metadata: Dict[str, Any],
        max_chunk_chars: int = 1200,
        overlap_chars: int = 150
    ) -> List[Dict[str, Any]]:
        """
        Splits medical document into semantically bounded chunks.
        Keeps table blocks and recommendation sections cohesive.
        """
        doc_id = doc_metadata.get("doc_id", "DOC_UNKNOWN")
        doc_title = doc_metadata.get("title", "")
        org_id = doc_metadata.get("org_id", "")
        specialty = doc_metadata.get("specialty", "General")
        evidence_grade = doc_metadata.get("evidence_grade", "")
        version = doc_metadata.get("guideline_version", "")
        url = doc_metadata.get("original_url", "")
        
        # Determine default age group from metadata or title
        is_peds = "pediatric" in doc_title.lower() or "child" in doc_title.lower() or "ispad" in org_id.lower()
        age_group = "pediatric" if is_peds else "adult"

        # Split on section headings or major paragraphs
        raw_sections = re.split(r'\n(?=#{1,4}\s+|\[Page\s+\d+\]|Recommendation\s+\d+|Clinical\s+Practice\s+Guideline)', text)
        
        chunks = []
        chunk_idx = 1

        for sec in raw_sections:
            sec_clean = sec.strip()
            if not sec_clean or len(sec_clean) < 40:
                continue

            # Identify topic or heading
            first_line = sec_clean.split('\n')[0].strip()
            topic = re.sub(r'^[#\s]+', '', first_line)[:120] if first_line else doc_title

            # If section fits within max size, keep it intact
            if len(sec_clean) <= max_chunk_chars:
                cid = f"{doc_id}_c{chunk_idx:03d}"
                chunks.append({
                    "id": cid,
                    "text": sec_clean,
                    "metadata": {
                        "doc_id": doc_id,
                        "chunk_id": cid,
                        "title": doc_title,
                        "org_id": org_id,
                        "topic": topic,
                        "specialty": specialty,
                        "age_group": age_group,
                        "evidence_grade": evidence_grade,
                        "guideline_version": version,
                        "original_url": url,
                        "is_active": 1,
                        "section": topic
                    }
                })
                chunk_idx += 1
            else:
                # Subdivide large section by paragraphs while preserving table structures
                paragraphs = sec_clean.split('\n\n')
                current_subchunk = []
                current_len = 0

                for para in paragraphs:
                    p = para.strip()
                    if not p:
                        continue
                    if current_len + len(p) > max_chunk_chars and current_subchunk:
                        sub_text = "\n\n".join(current_subchunk)
                        cid = f"{doc_id}_c{chunk_idx:03d}"
                        chunks.append({
                            "id": cid,
                            "text": sub_text,
                            "metadata": {
                                "doc_id": doc_id,
                                "chunk_id": cid,
                                "title": doc_title,
                                "org_id": org_id,
                                "topic": topic,
                                "specialty": specialty,
                                "age_group": age_group,
                                "evidence_grade": evidence_grade,
                                "guideline_version": version,
                                "original_url": url,
                                "is_active": 1,
                                "section": topic
                            }
                        })
                        chunk_idx += 1
                        current_subchunk = [p]
                        current_len = len(p)
                    else:
                        current_subchunk.append(p)
                        current_len += len(p)

                if current_subchunk:
                    sub_text = "\n\n".join(current_subchunk)
                    cid = f"{doc_id}_c{chunk_idx:03d}"
                    chunks.append({
                        "id": cid,
                        "text": sub_text,
                        "metadata": {
                            "doc_id": doc_id,
                            "chunk_id": cid,
                            "title": doc_title,
                            "org_id": org_id,
                            "topic": topic,
                            "specialty": specialty,
                            "age_group": age_group,
                            "evidence_grade": evidence_grade,
                            "guideline_version": version,
                            "original_url": url,
                            "is_active": 1,
                            "section": topic
                        }
                    })
                    chunk_idx += 1

        return chunks

    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validates that extracted chunks are clinically coherent and non-corrupted."""
        errors = []
        if not chunks:
            errors.append("No chunks were generated from the document.")
            return False, errors

        for idx, c in enumerate(chunks):
            txt = c.get("text", "")
            if len(txt) < 30:
                errors.append(f"Chunk {c.get('id', idx)} is critically truncated (< 30 characters).")
            # Quality check for corrupted binary / OCR garbage
            non_printable = len([char for char in txt if ord(char) < 32 and char not in ('\n', '\r', '\t')])
            if non_printable > 5:
                errors.append(f"Chunk {c.get('id', idx)} contains binary or corrupted OCR characters.")

        is_valid = len(errors) == 0
        return is_valid, errors

    def ingest_document(
        self,
        doc_metadata: Dict[str, Any],
        raw_text: Optional[str] = None,
        file_path: Optional[str] = None,
        target_collection: str = "clinical_guidelines"
    ) -> Dict[str, Any]:
        """
        Executes end-to-end ingestion through the 6 review states:
        Discovered -> Retrieved -> Extracted -> Validated -> Indexed -> Active.
        Returns IngestionReport.
        """
        doc_id = doc_metadata["doc_id"]
        title = doc_metadata.get("title", "Untitled Medical Document")
        org_id = doc_metadata.get("org_id", "AUTHORITY")
        
        # 1. State: Discovered -> Register in Catalog
        content_for_checksum = raw_text or (open(file_path, "r", encoding="utf-8", errors="ignore").read() if file_path else title)
        reg_result = self.registry.register_document(
            doc_id=doc_id,
            title=title,
            org_id=org_id,
            document_type=doc_metadata.get("document_type", "clinical_guideline"),
            original_url=doc_metadata.get("original_url", ""),
            publication_date=doc_metadata.get("publication_date", time.strftime("%Y-%m-%d")),
            guideline_version=doc_metadata.get("guideline_version", "1.0"),
            specialty=doc_metadata.get("specialty", "General"),
            content=content_for_checksum,
            jurisdiction=doc_metadata.get("jurisdiction", "Global"),
            evidence_grade=doc_metadata.get("evidence_grade", "Authoritative Guideline"),
            selection_rationale=doc_metadata.get("selection_rationale"),
            license=doc_metadata.get("license", "Open Access")
        )

        # 2. State: Retrieved
        self.registry.update_review_status(doc_id, "Retrieved")

        # 3. State: Extracted
        try:
            if file_path and os.path.exists(file_path):
                extracted_text, file_info = self.extract_text_from_file(file_path)
            elif raw_text:
                extracted_text = raw_text
                file_info = {"format": "raw_text"}
            else:
                raise ValueError("Neither file_path nor raw_text provided for ingestion.")

            cleaned_text = self.clean_text_preserve_clinical(extracted_text)
            chunks = self.split_into_clinical_chunks(cleaned_text, doc_metadata)
            self.registry.update_review_status(doc_id, "Extracted", total_chunks=len(chunks))
        except Exception as e:
            err_msg = f"Extraction failed: {str(e)}"
            self.registry.update_review_status(doc_id, "Rejected", error_message=err_msg)
            return {
                "doc_id": doc_id,
                "status": "REJECTED",
                "stage": "Extracted",
                "error": err_msg
            }

        # 4. State: Validated
        is_valid, validation_errors = self.validate_chunks(chunks)
        if not is_valid:
            err_msg = "; ".join(validation_errors)
            self.registry.update_review_status(doc_id, "Rejected", error_message=err_msg)
            return {
                "doc_id": doc_id,
                "status": "REJECTED",
                "stage": "Validated",
                "errors": validation_errors
            }
        self.registry.update_review_status(doc_id, "Validated", total_chunks=len(chunks))

        # 5. State: Indexed (Push to persistent ChromaDB)
        try:
            indexed_count = self.vector_store.add_chunks(
                collection_name=target_collection,
                chunks=chunks
            )
            self.registry.update_review_status(doc_id, "Indexed", total_chunks=indexed_count)
        except Exception as e:
            err_msg = f"Vector indexing failed: {str(e)}"
            self.registry.update_review_status(doc_id, "Rejected", error_message=err_msg)
            return {
                "doc_id": doc_id,
                "status": "REJECTED",
                "stage": "Indexed",
                "error": err_msg
            }

        # 6. State: Active
        self.registry.update_review_status(doc_id, "Active", total_chunks=indexed_count)

        return {
            "doc_id": doc_id,
            "title": title,
            "org_id": org_id,
            "status": "ACTIVE",
            "review_status": "Active",
            "chunks_indexed": indexed_count,
            "target_collection": target_collection,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def ingest_existing_corpus(self, guidelines_json_path: str = "data/guidelines.json") -> Dict[str, Any]:
        """
        Migrates and seeds the existing static corpus into the persistent registry and vector store.
        Assigns official documents, organizations, evidence grades, and collections.
        """
        if not os.path.exists(guidelines_json_path):
            return {"status": "SKIPPED", "message": f"{guidelines_json_path} does not exist."}

        with open(guidelines_json_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        # Group items by source organization & document
        docs_map = {
            "KDIGO": {
                "doc_id": "KDIGO_2023_CKD_DM",
                "title": "KDIGO 2023 Clinical Practice Guideline for Diabetes Management in Chronic Kidney Disease",
                "org_id": "KDIGO",
                "document_type": "clinical_guideline",
                "original_url": "https://kdigo.org/guidelines/diabetes-ckd/",
                "publication_date": "2023-11-01",
                "guideline_version": "2023.1",
                "specialty": "Nephrology",
                "evidence_grade": "Level 1A Recommendation",
                "collection": "clinical_guidelines"
            },
            "ADA": {
                "doc_id": "ADA_STANDARDS_2024",
                "title": "American Diabetes Association Standards of Care in Diabetes—2024",
                "org_id": "ADA",
                "document_type": "clinical_guideline",
                "original_url": "https://diabetesjournals.org/care/issue/47/Supplement_1",
                "publication_date": "2024-01-01",
                "guideline_version": "2024.1",
                "specialty": "Endocrinology",
                "evidence_grade": "Grade A Evidence",
                "collection": "clinical_guidelines"
            },
            "AHA/ACC": {
                "doc_id": "AHA_ACC_HF_2022",
                "title": "AHA/ACC/HFSA 2022 Guideline for the Management of Heart Failure",
                "org_id": "AHA_ACC",
                "document_type": "clinical_guideline",
                "original_url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063",
                "publication_date": "2022-04-01",
                "guideline_version": "2022.1",
                "specialty": "Cardiology",
                "evidence_grade": "Class 1 Level of Evidence A",
                "collection": "clinical_guidelines"
            },
            "ISPAD": {
                "doc_id": "ISPAD_2024_PEDIATRIC",
                "title": "ISPAD Clinical Practice Consensus Guidelines 2024: Pediatric Diabetes & Nephropathy",
                "org_id": "ISPAD",
                "document_type": "clinical_guideline",
                "original_url": "https://www.ispad.org/page/guidelines",
                "publication_date": "2024-03-01",
                "guideline_version": "2024.1",
                "specialty": "Pediatric Endocrinology",
                "evidence_grade": "ISPAD Consensus Recommendation",
                "collection": "clinical_guidelines"
            }
        }

        # Build combined text per document
        grouped_texts: Dict[str, List[str]] = {}
        for it in items:
            src = it.get("source", "KDIGO")
            match_key = None
            for k in docs_map:
                if k.lower() in src.lower():
                    match_key = k
                    break
            if not match_key:
                match_key = "KDIGO"
            
            if match_key not in grouped_texts:
                grouped_texts[match_key] = []
            grouped_texts[match_key].append(f"### {it.get('topic', 'Clinical Guidance')}\n{it.get('text', '')}")

        results = []
        for key, text_list in grouped_texts.items():
            meta = docs_map[key]
            full_doc_text = f"# {meta['title']}\n\n" + "\n\n".join(text_list)
            res = self.ingest_document(
                doc_metadata=meta,
                raw_text=full_doc_text,
                target_collection=meta["collection"]
            )
            results.append(res)

        return {
            "status": "COMPLETED",
            "documents_seeded": len(results),
            "details": results
        }
