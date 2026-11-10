import os
import re
from typing import Dict, Any, List, Optional

try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except ImportError:
    PDFIUM_AVAILABLE = False


class ClinicalPDFParser:
    """
    Local PDF report parser for extracting clinical history, lab values, and symptoms.
    Uses pypdfium2 for zero-dependency local text extraction, coupled with
    specialized medical regex pattern matching.
    """

    def __init__(self):
        pass

    def extract_text_from_pdf(self, file_path_or_bytes: Any) -> str:
        """Extracts all text from a PDF file path or raw bytes."""
        if not PDFIUM_AVAILABLE:
            raise RuntimeError("pypdfium2 is required for PDF parsing.")

        pdf = pdfium.PdfDocument(file_path_or_bytes)
        full_text = []
        for page_idx in range(len(pdf)):
            page = pdf.get_page(page_idx)
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            if text:
                full_text.append(text)
        return "\n".join(full_text)

    def parse_clinical_entities(self, text: str) -> Dict[str, Any]:
        """
        Parses clinical variables from medical report text:
        - eGFR (Glomerular Filtration Rate)
        - HbA1c (Glycated Hemoglobin)
        - Blood Pressure (Systolic / Diastolic)
        - Serum Creatinine
        - Identified conditions
        - Symptoms
        - Current medications
        """
        result: Dict[str, Any] = {
            "patient_id": None,
            "patient_name": None,
            "age": None,
            "gender": None,
            "egfr": None,
            "hba1c": None,
            "sbp": None,
            "dbp": None,
            "creatinine": None,
            "potassium": None,
            "uacr": None,
            "sodium": None,
            "bicarbonate": None,
            "conditions": [],
            "symptoms": [],
            "current_medications": [],
            "raw_text_snippet": text[:600] if len(text) > 600 else text
        }

        # 1. Patient ID
        id_match = re.search(r'(?:Patient\s*ID|PID|MRN|Case\s*#?)[\s:]+([A-Za-z0-9_\-]+)', text, re.IGNORECASE)
        if id_match:
            result["patient_id"] = id_match.group(1).strip()

        # 2. Patient Name
        name_match = re.search(r'Patient\s*Name[\s:]+([A-Za-z\s\.\-]+?)(?:\s*\||\s*Age|\s*\n|$)', text, re.IGNORECASE)
        if name_match:
            result["patient_name"] = name_match.group(1).strip()

        # 3. Age & Gender
        age_match = re.search(r'(?:Age|Age/Sex)[\s:]+(\d{1,3})\s*(?:yo|years|yrs)?', text, re.IGNORECASE)
        if age_match:
            try:
                result["age"] = int(age_match.group(1))
            except ValueError:
                pass

        gender_match = re.search(r'(?:Gender|Sex)[\s:]+(Male|Female|M|F)\b', text, re.IGNORECASE)
        if gender_match:
            g = gender_match.group(1).upper()
            result["gender"] = "Male" if g in ["M", "MALE"] else "Female"

        # 4. eGFR (Estimated Glomerular Filtration Rate)
        # Handles: "eGFR: 22.0", "(eGFR): 22", "Estimated GFR: 22", "GFR = 14"
        egfr_match = re.search(
            r'(?:eGFR|GFR|Glomerular\s*Filtration\s*Rate)[^\d\n\r]{0,25}?([0-9]{1,3}(?:\.[0-9]+)?)',
            text,
            re.IGNORECASE
        )
        if egfr_match:
            try:
                val = float(egfr_match.group(1))
                if 2.0 <= val <= 180.0:
                    result["egfr"] = val
            except ValueError:
                pass

        # 5. HbA1c (Glycated Hemoglobin)
        # Handles: "HbA1c: 9.2%", "(HbA1c): 9.2 %", "Glycated Hemoglobin: 8.4 %"
        hba1c_match = re.search(
            r'(?:HbA1c|A1C|Glycated\s*H(?:a|e)moglobin)[^\d\n\r]{0,25}?([0-9]{1,2}(?:\.[0-9]+)?)\s*%',
            text,
            re.IGNORECASE
        )
        if not hba1c_match:
            hba1c_match = re.search(
                r'(?:HbA1c|A1C|Glycated\s*H(?:a|e)moglobin)[^\d\n\r]{0,25}?([0-9]{1,2}\.[0-9]+)',
                text,
                re.IGNORECASE
            )
        if hba1c_match:
            try:
                val = float(hba1c_match.group(1))
                if 3.0 <= val <= 20.0:
                    result["hba1c"] = val
            except ValueError:
                pass

        # 6. Blood Pressure (SBP / DBP)
        # Combined BP: e.g. "145/90 mmHg", "165 / 95"
        bp_match = re.search(
            r'(?:BP|Blood\s*Pressure|SBP/DBP)[^\d\n\r]{0,15}?([0-9]{2,3})\s*/\s*([0-9]{2,3})',
            text,
            re.IGNORECASE
        )
        if bp_match:
            try:
                result["sbp"] = int(bp_match.group(1))
                result["dbp"] = int(bp_match.group(2))
            except ValueError:
                pass
        else:
            # Check individual SBP and DBP
            sbp_match = re.search(r'(?:Systolic\s*BP|Systolic|SBP)[^\d\n\r]{0,15}?([0-9]{2,3})', text, re.IGNORECASE)
            if sbp_match:
                try:
                    result["sbp"] = int(sbp_match.group(1))
                except ValueError:
                    pass
            dbp_match = re.search(r'(?:Diastolic\s*BP|Diastolic|DBP)[^\d\n\r]{0,15}?([0-9]{2,3})', text, re.IGNORECASE)
            if dbp_match:
                try:
                    result["dbp"] = int(dbp_match.group(1))
                except ValueError:
                    pass

        # 7. Serum Creatinine
        creat_match = re.search(
            r'(?:Serum\s*Creatinine|Creatinine|Cr)[^\d\n\r]{0,15}?([0-9]{1,2}(?:\.[0-9]+)?)',
            text,
            re.IGNORECASE
        )
        if creat_match:
            try:
                val = float(creat_match.group(1))
                if 0.2 <= val <= 25.0:
                    result["creatinine"] = val
            except ValueError:
                pass

        # 8. Serum Potassium (K+)
        k_match = re.search(
            r'(?:Serum\s*Potassium|Potassium|K\+?)[^\d\n\r]{0,15}?([0-9]{1,2}(?:\.[0-9]+)?)',
            text,
            re.IGNORECASE
        )
        if k_match:
            try:
                val = float(k_match.group(1))
                if 1.5 <= val <= 10.0:
                    result["potassium"] = val
            except ValueError:
                pass

        # 9. Urine Albumin-to-Creatinine Ratio (UACR) / Albuminuria
        uacr_match = re.search(
            r'(?:Urine\s*(?:Albumin|ACR|Microalbumin)|UACR|Albumin-to-Creatinine|ACR)[^\d\n\r]{0,20}?([0-9]{1,5}(?:\.[0-9]+)?)',
            text,
            re.IGNORECASE
        )
        if uacr_match:
            try:
                val = float(uacr_match.group(1))
                if 0.0 <= val <= 10000.0:
                    result["uacr"] = val
            except ValueError:
                pass

        # 10. Serum Sodium (Na+)
        na_match = re.search(
            r'(?:Serum\s*Sodium|Sodium|Na\+?)[^\d\n\r]{0,15}?([0-9]{2,3}(?:\.[0-9]+)?)',
            text,
            re.IGNORECASE
        )
        if na_match:
            try:
                val = float(na_match.group(1))
                if 100.0 <= val <= 180.0:
                    result["sodium"] = val
            except ValueError:
                pass

        # 11. Serum Bicarbonate (HCO3 / CO2)
        hco3_match = re.search(
            r'(?:Serum\s*Bicarbonate|Bicarbonate|HCO3|CO2)[^\d\n\r]{0,15}?([0-9]{1,2}(?:\.[0-9]+)?)',
            text,
            re.IGNORECASE
        )
        if hco3_match:
            try:
                val = float(hco3_match.group(1))
                if 5.0 <= val <= 50.0:
                    result["bicarbonate"] = val
            except ValueError:
                pass

        # 12. Diagnosed Conditions Mapping (Accurate Staging)
        lower_text = text.lower()
        if "ckd" in lower_text or "chronic kidney disease" in lower_text or (result["egfr"] is not None and result["egfr"] < 60):
            if result["egfr"] is not None:
                if result["egfr"] < 15.0:
                    result["conditions"].append("CKD_Stage_5")
                elif result["egfr"] < 30.0:
                    result["conditions"].append("CKD_Stage_4")
                elif result["egfr"] < 45.0:
                    result["conditions"].append("CKD_Stage_3b")
                elif result["egfr"] < 60.0:
                    result["conditions"].append("CKD_Stage_3a")
                else:
                    result["conditions"].append("Chronic_Kidney_Disease")
            else:
                result["conditions"].append("Chronic_Kidney_Disease")

        if any(w in lower_text for w in ["type 2 diabetes", "t2dm", "t2d", "type ii diabetes", "diabetes mellitus"]) or (result["hba1c"] is not None and result["hba1c"] >= 6.5):
            result["conditions"].append("Type_2_Diabetes")

        if any(w in lower_text for w in ["type 1 diabetes", "t1d", "type i diabetes"]):
            result["conditions"].append("Type_1_Diabetes")

        if any(w in lower_text for w in ["hypertension", "htn", "high blood pressure"]) or (result["sbp"] is not None and result["sbp"] >= 130):
            result["conditions"].append("Hypertension")

        if any(w in lower_text for w in ["heart failure", "chf", "congestive heart failure", "cardiorenal"]):
            result["conditions"].append("Heart_Failure")

        # Deduplicate conditions
        result["conditions"] = list(dict.fromkeys(result["conditions"]))

        # 13. Clinical Symptoms Detection
        symptom_catalog = [
            ("bilateral pedal edema", "Bilateral pedal edema"),
            ("bilateral lower limb edema", "Bilateral lower limb edema"),
            ("pedal edema", "Pedal edema"),
            ("peripheral edema", "Peripheral edema"),
            ("breathlessness on exertion", "Breathlessness on exertion"),
            ("reduced exercise tolerance", "Reduced exercise tolerance"),
            ("fatigue", "Fatigue"),
            ("lethargy", "Severe lethargy"),
            ("nocturia", "Nocturia"),
            ("polydipsia", "Polydipsia"),
            ("polyuria", "Polyuria"),
            ("morning occipital headache", "Morning occipital headaches"),
            ("headache", "Headaches"),
            ("anorexia", "Anorexia"),
            ("pruritus", "Generalized pruritus"),
            ("shortness of breath", "Shortness of breath"),
            ("dyspnea", "Dyspnea"),
            ("orthopnea", "Orthopnea")
        ]
        for term, label in symptom_catalog:
            if term in lower_text and label not in result["symptoms"]:
                result["symptoms"].append(label)

        # 10. Documented Current Medications
        med_catalog = ["metformin", "lisinopril", "insulin", "amlodipine", "losartan", "glipizide", "atorvastatin"]
        for med in med_catalog:
            if re.search(rf'\b{med}\b', lower_text):
                result["current_medications"].append(med.capitalize())

        return result
