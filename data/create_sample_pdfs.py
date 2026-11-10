import os

def create_simple_pdf(filename: str, title: str, lines: list):
    """Generates a clean, valid standard PDF file containing structured medical text."""
    content_lines = [f"BT /F1 14 Tf 50 750 Td ({title}) Tj ET"]
    y = 720
    for line in lines:
        escaped_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_lines.append(f"BT /F1 10 Tf 50 {y} Td ({escaped_line}) Tj ET")
        y -= 18

    stream_content = "\n".join(content_lines)
    stream_len = len(stream_content.encode('latin1'))

    pdf_parts = []
    pdf_parts.append("%PDF-1.4\n")
    
    # 1 0 obj: Catalog
    offset_1 = sum(len(p.encode('latin1')) for p in pdf_parts)
    pdf_parts.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    
    # 2 0 obj: Pages
    offset_2 = sum(len(p.encode('latin1')) for p in pdf_parts)
    pdf_parts.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    
    # 3 0 obj: Page
    offset_3 = sum(len(p.encode('latin1')) for p in pdf_parts)
    pdf_parts.append("3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n")
    
    # 4 0 obj: Contents stream
    offset_4 = sum(len(p.encode('latin1')) for p in pdf_parts)
    pdf_parts.append(f"4 0 obj\n<< /Length {stream_len} >>\nstream\n{stream_content}\nendstream\nendobj\n")
    
    # 5 0 obj: Font
    offset_5 = sum(len(p.encode('latin1')) for p in pdf_parts)
    pdf_parts.append("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    
    # xref
    xref_offset = sum(len(p.encode('latin1')) for p in pdf_parts)
    pdf_parts.append("xref\n0 6\n")
    pdf_parts.append("0000000000 65535 f \n")
    pdf_parts.append(f"{offset_1:010d} 00000 n \n")
    pdf_parts.append(f"{offset_2:010d} 00000 n \n")
    pdf_parts.append(f"{offset_3:010d} 00000 n \n")
    pdf_parts.append(f"{offset_4:010d} 00000 n \n")
    pdf_parts.append(f"{offset_5:010d} 00000 n \n")
    
    # trailer
    pdf_parts.append(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n")

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        f.write("".join(pdf_parts).encode('latin1'))


def main():
    target_dir = os.path.join("data", "sample_reports")
    os.makedirs(target_dir, exist_ok=True)

    # 1. PT_001 Report (CKD4 + T2DM)
    create_simple_pdf(
        os.path.join(target_dir, "PT_001_Clinical_Lab_Report.pdf"),
        "ST. JUDE METABOLIC INSTITUTE - COMPREHENSIVE RENAL & DIABETIC PANEL",
        [
            "Patient ID: PT_001",
            "Patient Name: Robert Miller | Age: 68 | Gender: Male | Date: 2026-10-14",
            "--------------------------------------------------------------------------------",
            "CLINICAL PRESENTATION & SYMPTOMS:",
            "Patient presents with progressive bilateral pedal edema, severe fatigue, marked nocturia,",
            "and frequent thirst (polydipsia). Known history of Type 2 Diabetes for 14 years.",
            "--------------------------------------------------------------------------------",
            "LABORATORY BIOMARKERS:",
            "  * Estimated GFR (eGFR): 22.0 mL/min/1.73m2 (KDIGO Stage 4 - Severely Decreased)",
            "  * Glycated Hemoglobin (HbA1c): 9.2 % (Target < 7.0%)",
            "  * Serum Creatinine: 3.1 mg/dL (Reference: 0.7 - 1.3 mg/dL)",
            "  * Systolic BP: 138 mmHg | Diastolic BP: 84 mmHg",
            "  * Urine Albumin-to-Creatinine Ratio: 420 mg/g (Severely increased A3)",
            "--------------------------------------------------------------------------------",
            "CURRENT PRESCRIBED MEDICATIONS:",
            "  - Metformin 500mg (Prior home regimen - urgent safety review requested)",
            "  - Atorvastatin 20mg",
            "--------------------------------------------------------------------------------",
            "ASSESSMENT: Stage 4 Chronic Kidney Disease secondary to Diabetic Kidney Disease."
        ]
    )

    # 2. PT_002 Report (CKD5 + HTN)
    create_simple_pdf(
        os.path.join(target_dir, "PT_002_Renal_Panel_Discharge.pdf"),
        "METROPOLITAN NEPHROLOGY CENTER - DISCHARGE SUMMARY",
        [
            "Patient ID: PT_002",
            "Patient Name: Eleanor Vance | Age: 72 | Gender: Female | Date: 2026-10-18",
            "--------------------------------------------------------------------------------",
            "CLINICAL PRESENTATION & SYMPTOMS:",
            "72-year-old female admitted with severe lethargy, generalized pruritus, anorexia,",
            "and persistent morning occipital headaches. No history of diabetes mellitus.",
            "--------------------------------------------------------------------------------",
            "LABORATORY & HEMODYNAMIC FINDINGS:",
            "  * Estimated GFR (eGFR): 14.0 mL/min/1.73m2 (KDIGO Stage 5 - Kidney Failure)",
            "  * Blood Pressure (BP): 165/95 mmHg (Stage 2 Essential Hypertension)",
            "  * Serum Creatinine: 4.2 mg/dL",
            "  * Glycated Hemoglobin (HbA1c): 5.6 % (Normal glycemic profile)",
            "  * Serum Potassium: 5.3 mEq/L (Borderline hyperkalemia)",
            "--------------------------------------------------------------------------------",
            "CURRENT THERAPY EVALUATION:",
            "  - Lisinopril 20mg oral tablet (RAAS blockade - evaluate contraindication in CKD 5)",
            "--------------------------------------------------------------------------------",
            "NOTE: High risk of potassium elevation and acute decompensation with ACE inhibitors."
        ]
    )

    # 3. PT_004 Report (Hypertension)
    create_simple_pdf(
        os.path.join(target_dir, "PT_004_Cardiology_Summary.pdf"),
        "CARDIAC SPECIALTY CLINIC - OUTPATIENT EVALUATION",
        [
            "Patient ID: PT_004",
            "Patient Name: Margaret Thorne | Age: 61 | Gender: Female | Date: 2026-10-22",
            "--------------------------------------------------------------------------------",
            "CLINICAL PRESENTATION & SYMPTOMS:",
            "Presents for routine cardiovascular follow-up. Reports occasional occipital headaches",
            "and lightheadedness upon standing. Denies chest pain or shortness of breath.",
            "--------------------------------------------------------------------------------",
            "LABORATORY & VITAL SIGNS:",
            "  * Blood Pressure (BP): 152/94 mmHg (Stage 2 Essential Hypertension)",
            "  * Estimated GFR (eGFR): 78.0 mL/min/1.73m2 (Preserved renal function)",
            "  * Serum Creatinine: 1.0 mg/dL",
            "  * Glycated Hemoglobin (HbA1c): 5.4 % (Normal)",
            "--------------------------------------------------------------------------------",
            "DIAGNOSIS: Stage 2 Primary Hypertension with intact glomerular filtration."
        ]
    )

    # 4. PT_008 Report (Sarah Johnson - CKD 3b + T2D + UACR 420)
    create_simple_pdf(
        os.path.join(target_dir, "PT_008_Sarah_Johnson_Report.pdf"),
        "METABOLIC & NEPHROLOGY CLINIC - ROUTINE CKD EVALUATION",
        [
            "Patient ID: PT_008",
            "Patient Name: Sarah Johnson | Age: 62 | Gender: Female | Date: 2026-10-25",
            "--------------------------------------------------------------------------------",
            "CLINICAL PRESENTATION & SYMPTOMS:",
            "62-year-old female with longstanding Type 2 Diabetes and Hypertension presenting",
            "for routine review. Reports mild bilateral ankle edema, fatigue, and nocturia.",
            "--------------------------------------------------------------------------------",
            "LABORATORY BIOMARKERS:",
            "  * Estimated GFR (eGFR): 38.0 mL/min/1.73m2 (KDIGO Stage 3b - Moderate to Severe)",
            "  * Glycated Hemoglobin (HbA1c): 8.1 %",
            "  * Serum Creatinine: 1.6 mg/dL",
            "  * Blood Pressure (BP): 146/88 mmHg",
            "  * Serum Potassium (K+): 4.4 mmol/L (Normokalemia)",
            "  * Urine Albumin-to-Creatinine Ratio (UACR): 420 mg/g (Stage A3 Severely Increased)",
            "--------------------------------------------------------------------------------",
            "DIAGNOSTIC CODING: Chronic Kidney Disease Stage 3b (ICD-10: N18.32).",
            "GUIDELINE NOTE: Metformin is safe with dose reduction (max 1000mg/day); SGLT2i indicated."
        ]
    )

    # 5. PT_009 Report (David Anderson - CKD 4 + Heart Failure + Hyperkalemia K+ 5.3)
    create_simple_pdf(
        os.path.join(target_dir, "PT_009_David_Anderson_Report.pdf"),
        "UNIVERSITY CARDIOLOGY & RENAL CENTER - CLINICAL REPORT",
        [
            "Patient ID: PT_009",
            "Patient Name: David Anderson | Age: 69 | Gender: Male | Date: 2026-10-28",
            "--------------------------------------------------------------------------------",
            "CLINICAL PRESENTATION & SYMPTOMS:",
            "69-year-old male with Heart Failure and Type 2 Diabetes presenting with active",
            "fluid overload: bilateral pedal edema, breathlessness on exertion, and orthopnea.",
            "--------------------------------------------------------------------------------",
            "LABORATORY & HEMODYNAMIC FINDINGS:",
            "  * Estimated GFR (eGFR): 24.0 mL/min/1.73m2 (KDIGO Stage 4 - Severely Decreased)",
            "  * Serum Potassium (K+): 5.3 mmol/L (CRITICAL MILD HYPERKALEMIA ALERT)",
            "  * Glycated Hemoglobin (HbA1c): 7.4 %",
            "  * Serum Creatinine: 2.7 mg/dL",
            "  * Blood Pressure (BP): 154/92 mmHg (Stage 2 Hypertension)",
            "  * Urine Albumin-to-Creatinine Ratio (UACR): 780 mg/g (Stage A3 Macroalbuminuria)",
            "--------------------------------------------------------------------------------",
            "ASSESSMENT: Cardiorenal Syndrome Type 2/4 with CKD Stage 4 (ICD-10: N18.4).",
            "SAFETY ACTIONS: Metformin contraindicated (eGFR < 30); SGLT2i + Loop diuretic indicated."
        ]
    )

    print(f"Sample medical PDF reports generated in {target_dir}")

if __name__ == "__main__":
    main()
