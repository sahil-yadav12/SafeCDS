import os
import sqlite3
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple


class SourceRegistry:
    """
    Persistent SQLite Registry for Authoritative Medical Knowledge Sources.
    Tracks document versions, SHA-256 checksums, credibility tiers,
    evidence recommendation grades, and strict review states:
    Discovered -> Retrieved -> Extracted -> Validated -> Indexed -> Active.
    """

    REVIEW_STATES = [
        "Discovered",
        "Retrieved",
        "Extracted",
        "Validated",
        "Indexed",
        "Active",
        "Superseded",
        "Rejected"
    ]

    CREDIBILITY_TIERS = {
        "Tier_1": "Clinical Practice Guidelines & Official Health Authorities (KDIGO, ADA, AHA/ACC, NICE, WHO, FDA, EMA, ICMR, ISPAD)",
        "Tier_2": "Peer-Reviewed Systematic Reviews & Syntheses (PubMed/NCBI, Europe PMC, Cochrane, ClinicalTrials.gov)",
        "Tier_3": "Official Drug Labels & Pharmacological Databases (openFDA, DailyMed, RxNorm, LOINC)"
    }

    def __init__(self, db_path: str = "data/source_registry.db"):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Authoritative Organizations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS organizations (
                    org_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    credibility_tier TEXT NOT NULL,
                    base_url TEXT,
                    jurisdiction TEXT,
                    specialty TEXT,
                    api_endpoint TEXT,
                    rate_limit_per_min INTEGER DEFAULT 60
                )
            """)

            # 2. Medical Documents Catalog
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    org_id TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    original_url TEXT,
                    publication_date TEXT,
                    last_updated_date TEXT,
                    guideline_version TEXT,
                    retrieval_date TEXT NOT NULL,
                    jurisdiction TEXT DEFAULT 'Global',
                    specialty TEXT NOT NULL,
                    evidence_grade TEXT,
                    checksum_sha256 TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    superseded_by TEXT,
                    is_active INTEGER DEFAULT 0,
                    license TEXT DEFAULT 'Open Access',
                    selection_rationale TEXT,
                    rejection_reason TEXT,
                    total_chunks INTEGER DEFAULT 0,
                    FOREIGN KEY (org_id) REFERENCES organizations(org_id)
                )
            """)

            # 3. Document Version History
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_versions (
                    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    previous_version TEXT,
                    new_version TEXT,
                    superseded_date TEXT,
                    change_summary TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                )
            """)

            # 4. Ingestion & Audit Logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    doc_id TEXT,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chunks_created INTEGER DEFAULT 0,
                    error_message TEXT,
                    details TEXT
                )
            """)

            conn.commit()
            self._seed_default_organizations(conn)

    def _seed_default_organizations(self, conn: sqlite3.Connection):
        """Seeds canonical medical authorities if not already present."""
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM organizations")
        if cursor.fetchone()[0] > 0:
            return

        defaults = [
            ("KDIGO", "Kidney Disease: Improving Global Outcomes", "Tier_1", "https://kdigo.org", "Global", "Nephrology", "https://kdigo.org/guidelines", 30),
            ("ADA", "American Diabetes Association", "Tier_1", "https://diabetesjournals.org", "US/Global", "Endocrinology", "https://diabetesjournals.org/care", 30),
            ("AHA_ACC", "American Heart Association / American College of Cardiology", "Tier_1", "https://www.ahajournals.org", "US", "Cardiology", "https://www.ahajournals.org", 30),
            ("ISPAD", "International Society for Pediatric and Adolescent Diabetes", "Tier_1", "https://www.ispad.org", "Global", "Pediatric Endocrinology", "https://www.ispad.org/page/guidelines", 30),
            ("FDA", "U.S. Food and Drug Administration", "Tier_1", "https://www.fda.gov", "US", "Pharmacology", "https://api.fda.gov", 120),
            ("NICE", "National Institute for Health and Care Excellence", "Tier_1", "https://www.nice.org.uk", "UK", "Multidisciplinary", "https://api.nice.org.uk", 60),
            ("EUROPE_PMC", "Europe PubMed Central", "Tier_2", "https://europepmc.org", "Global", "Biomedical Research", "https://europepmc.org/RestAPI/vc/v1", 60),
            ("NCBI_PUBMED", "National Center for Biotechnology Information / PubMed", "Tier_2", "https://pubmed.ncbi.nlm.nih.gov", "US/Global", "Biomedical Research", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils", 60),
            ("OPENFDA", "openFDA Drug & Adverse Event Database", "Tier_3", "https://open.fda.gov", "US", "Pharmacology", "https://api.fda.gov/drug/label.json", 240)
        ]
        cursor.executemany("""
            INSERT INTO organizations (org_id, name, credibility_tier, base_url, jurisdiction, specialty, api_endpoint, rate_limit_per_min)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, defaults)
        conn.commit()

    @staticmethod
    def calculate_checksum(content: str) -> str:
        """Returns SHA-256 hex digest of document text."""
        return hashlib.sha256(content.strip().encode('utf-8')).hexdigest()

    def register_document(
        self,
        doc_id: str,
        title: str,
        org_id: str,
        document_type: str,
        original_url: str,
        publication_date: str,
        guideline_version: str,
        specialty: str,
        content: str,
        jurisdiction: str = "Global",
        evidence_grade: Optional[str] = None,
        selection_rationale: Optional[str] = None,
        license: str = "Open Access"
    ) -> Dict[str, Any]:
        """
        Registers a medical document into the catalog.
        Detects duplicates and previous versions by checksum and title similarity.
        Initial review status: 'Discovered'.
        """
        checksum = self.calculate_checksum(content)
        retrieval_date = time.strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if exact duplicate checksum already exists
            cursor.execute("SELECT doc_id, title, review_status, is_active FROM documents WHERE checksum_sha256 = ?", (checksum,))
            existing = cursor.fetchone()
            if existing:
                return {
                    "status": "DUPLICATE",
                    "doc_id": existing["doc_id"],
                    "title": existing["title"],
                    "review_status": existing["review_status"],
                    "is_active": bool(existing["is_active"]),
                    "message": f"Identical document checksum already registered under {existing['doc_id']}."
                }

            # Check if an existing version of this document exists
            cursor.execute("SELECT doc_id, guideline_version, is_active FROM documents WHERE title = ? AND org_id = ?", (title, org_id))
            prev_version = cursor.fetchone()

            cursor.execute("""
                INSERT OR REPLACE INTO documents (
                    doc_id, title, org_id, document_type, original_url,
                    publication_date, last_updated_date, guideline_version,
                    retrieval_date, jurisdiction, specialty, evidence_grade,
                    checksum_sha256, review_status, superseded_by, is_active,
                    license, selection_rationale, total_chunks
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id, title, org_id, document_type, original_url,
                publication_date, publication_date, guideline_version,
                retrieval_date, jurisdiction, specialty, evidence_grade,
                checksum, "Discovered", None, 0,
                license, selection_rationale or f"Selected as authoritative {org_id} evidence.", 0
            ))

            if prev_version and prev_version["guideline_version"] != guideline_version:
                cursor.execute("""
                    INSERT INTO document_versions (doc_id, previous_version, new_version, superseded_date, change_summary)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    doc_id,
                    prev_version["guideline_version"],
                    guideline_version,
                    retrieval_date,
                    f"Updated from {prev_version['guideline_version']} to {guideline_version}."
                ))

            conn.commit()

        return {
            "status": "REGISTERED",
            "doc_id": doc_id,
            "title": title,
            "review_status": "Discovered",
            "is_active": False,
            "checksum": checksum
        }

    def update_review_status(
        self,
        doc_id: str,
        new_status: str,
        error_message: Optional[str] = None,
        total_chunks: Optional[int] = None
    ) -> bool:
        """
        Transitions document review status along the lifecycle:
        Discovered -> Retrieved -> Extracted -> Validated -> Indexed -> Active.
        Documents failing validation or extraction remain inactive.
        """
        if new_status not in self.REVIEW_STATES:
            raise ValueError(f"Invalid review status '{new_status}'. Allowed: {self.REVIEW_STATES}")

        is_active = 1 if new_status == "Active" else 0
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            update_fields = ["review_status = ?", "is_active = ?"]
            params = [new_status, is_active]

            if total_chunks is not None:
                update_fields.append("total_chunks = ?")
                params.append(total_chunks)

            if error_message:
                update_fields.append("rejection_reason = ?")
                params.append(error_message)

            params.append(doc_id)
            cursor.execute(f"UPDATE documents SET {', '.join(update_fields)} WHERE doc_id = ?", params)

            # Ingestion audit log
            cursor.execute("""
                INSERT INTO ingestion_logs (timestamp, doc_id, stage, status, chunks_created, error_message, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(), doc_id, new_status,
                "ERROR" if error_message else "SUCCESS",
                total_chunks or 0, error_message,
                f"Transitioned to {new_status} at {now_str}"
            ))

            conn.commit()
            return cursor.rowcount > 0

    def supersede_document(self, old_doc_id: str, new_doc_id: str, reason: str = "Superseded by newer clinical practice guideline"):
        """Marks old document as Superseded and inactive, linking superseded_by."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE documents
                SET review_status = 'Superseded', is_active = 0, superseded_by = ?, rejection_reason = ?
                WHERE doc_id = ?
            """, (new_doc_id, reason, old_doc_id))
            conn.commit()

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_documents(
        self,
        active_only: bool = False,
        specialty: Optional[str] = None,
        org_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM documents WHERE 1=1"
            params = []

            if active_only:
                query += " AND is_active = 1"
            if specialty:
                query += " AND specialty = ?"
                params.append(specialty)
            if org_id:
                query += " AND org_id = ?"
                params.append(org_id)

            query += " ORDER BY publication_date DESC"
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def list_sources_with_stats(self) -> List[Dict[str, Any]]:
        """Returns comprehensive catalog of all indexed sources with chunk counts and tiers."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    d.doc_id,
                    d.title,
                    d.org_id,
                    o.name AS organization_name,
                    o.credibility_tier,
                    d.document_type,
                    d.guideline_version,
                    d.publication_date,
                    d.retrieval_date,
                    d.specialty,
                    d.evidence_grade,
                    d.review_status,
                    d.is_active,
                    d.total_chunks,
                    d.original_url,
                    d.superseded_by,
                    d.rejection_reason
                FROM documents d
                LEFT JOIN organizations o ON d.org_id = o.org_id
                ORDER BY d.is_active DESC, d.publication_date DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_audit_trail(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT log_id, datetime(timestamp, 'unixepoch') AS log_time, doc_id, stage, status, chunks_created, error_message, details
                FROM ingestion_logs
                ORDER BY log_id DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
