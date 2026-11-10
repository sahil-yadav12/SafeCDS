"""
SafeCDS Deterministic Clinical Rules Engine
Implements evidence-based KDIGO 2023, ADA 2024, and ACC/AHA clinical practice guidelines.
Provides deterministic staging, laboratory validation, medication eligibility evaluation,
and clinical alert generation to eliminate hallucinated classifications.
"""

from typing import Dict, Any, List, Optional, Tuple


def classify_ckd(egfr: Optional[float], has_kidney_damage: bool = True) -> Dict[str, str]:
    """
    Deterministic KDIGO 2023 CKD Glomerular Filtration Rate (G-Stage) Staging.
    Returns:
        stage_code: G1, G2, G3a, G3b, G4, or G5
        stage_name: Full descriptive clinical name
        icd10: Exact ICD-10 clinical diagnostic code
        severity: NORMAL, MILD, MODERATE, SEVERE, or KIDNEY_FAILURE
        monitoring_interval: Recommended frequency of eGFR and electrolyte review
    """
    if egfr is None:
        return {
            "stage_code": "UNKNOWN",
            "stage_name": "Unclassified (eGFR Missing)",
            "icd10": "R69",
            "severity": "UNKNOWN",
            "monitoring_interval": "Immediate lab testing required"
        }

    egfr_val = float(egfr)

    if egfr_val >= 90.0:
        if has_kidney_damage:
            return {
                "stage_code": "G1",
                "stage_name": "Chronic Kidney Disease Stage 1 (Normal or High GFR with Kidney Damage)",
                "icd10": "N18.1",
                "severity": "MILD",
                "monitoring_interval": "Annual monitoring"
            }
        return {
            "stage_code": "G1",
            "stage_name": "Preserved Renal Function (eGFR >= 90 mL/min/1.73m²)",
            "icd10": "Z00.00",
            "severity": "NORMAL",
            "monitoring_interval": "Routine care"
        }
    elif egfr_val >= 60.0:
        if has_kidney_damage:
            return {
                "stage_code": "G2",
                "stage_name": "Chronic Kidney Disease Stage 2 (Mild GFR Reduction with Kidney Damage)",
                "icd10": "N18.2",
                "severity": "MILD",
                "monitoring_interval": "Every 6-12 months"
            }
        return {
            "stage_code": "G2",
            "stage_name": "Mildly Decreased eGFR without Established Damage",
            "icd10": "R94.4",
            "severity": "MILD",
            "monitoring_interval": "Annual review"
        }
    elif egfr_val >= 45.0:
        return {
            "stage_code": "G3a",
            "stage_name": "Chronic Kidney Disease Stage 3a (Mild-to-Moderate GFR Reduction)",
            "icd10": "N18.31",
            "severity": "MODERATE",
            "monitoring_interval": "Every 6 months"
        }
    elif egfr_val >= 30.0:
        return {
            "stage_code": "G3b",
            "stage_name": "Chronic Kidney Disease Stage 3b (Moderate-to-Severe GFR Reduction)",
            "icd10": "N18.32",
            "severity": "MODERATE_SEVERE",
            "monitoring_interval": "Every 3-6 months (Nephrology consultation recommended)"
        }
    elif egfr_val >= 15.0:
        return {
            "stage_code": "G4",
            "stage_name": "Chronic Kidney Disease Stage 4 (Severely Decreased GFR)",
            "icd10": "N18.4",
            "severity": "SEVERE",
            "monitoring_interval": "Every 1-3 months (Active nephrology co-management mandated)"
        }
    else:
        return {
            "stage_code": "G5",
            "stage_name": "Chronic Kidney Disease Stage 5 (Kidney Failure / End-Stage Renal Disease)",
            "icd10": "N18.5",
            "severity": "KIDNEY_FAILURE",
            "monitoring_interval": "Monthly or per dialysis/transplant protocol"
        }


def classify_albuminuria(uacr_mg_g: Optional[float]) -> Dict[str, str]:
    """
    Deterministic KDIGO 2023 Albuminuria (A-Stage) Classification based on Urine Albumin-to-Creatinine Ratio (UACR).
    A1: < 30 mg/g (Normal to mildly increased)
    A2: 30-300 mg/g (Moderately increased / microalbuminuria)
    A3: > 300 mg/g (Severely increased / macroalbuminuria / nephrotic range)
    """
    if uacr_mg_g is None:
        return {
            "stage_code": "A_UNSPECIFIED",
            "stage_name": "Urine ACR Not Documented",
            "category": "Pending UACR test",
            "risk_multiplier": "Unknown"
        }

    val = float(uacr_mg_g)
    if val < 30.0:
        return {
            "stage_code": "A1",
            "stage_name": "Albuminuria Stage A1 (Normal to Mildly Increased: < 30 mg/g)",
            "category": "Normal to mildly increased",
            "risk_multiplier": "Baseline"
        }
    elif val <= 300.0:
        return {
            "stage_code": "A2",
            "stage_name": "Albuminuria Stage A2 (Moderately Increased: 30-300 mg/g)",
            "category": "Microalbuminuria",
            "risk_multiplier": "High Cardiorenal Risk"
        }
    else:
        return {
            "stage_code": "A3",
            "stage_name": "Albuminuria Stage A3 (Severely Increased: > 300 mg/g)",
            "category": "Macroalbuminuria / Nephrotic Range",
            "risk_multiplier": "Very High Cardiorenal Risk"
        }


def evaluate_potassium(k_mmol_l: Optional[float]) -> Dict[str, Any]:
    """
    Evaluates serum potassium status per KDIGO / AHA guidelines.
    Normal: 3.5 - 5.0 mmol/L
    Mild Hyperkalemia: 5.1 - 5.4 mmol/L
    Moderate Hyperkalemia: 5.5 - 5.9 mmol/L
    Severe Hyperkalemia: >= 6.0 mmol/L
    Hypokalemia: < 3.5 mmol/L
    """
    if k_mmol_l is None:
        return {
            "status": "NOT_SUPPLIED",
            "level": None,
            "is_alert": False,
            "severity": "NORMAL",
            "message": "Serum potassium not supplied. Essential in CKD Stage 3b-5 and when using RAAS inhibitors."
        }

    val = float(k_mmol_l)
    if val < 3.5:
        return {
            "status": "HYPOKALEMIA",
            "level": val,
            "is_alert": True,
            "severity": "MODERATE",
            "message": f"Hypokalemia alert (K+: {val} mmol/L). Evaluate for diuretic-induced wasting or metabolic alkalosis."
        }
    elif val <= 5.0:
        return {
            "status": "NORMOKALEMIA",
            "level": val,
            "is_alert": False,
            "severity": "NORMAL",
            "message": f"Normokalemia (K+: {val} mmol/L, reference 3.5-5.0 mmol/L)."
        }
    elif val <= 5.4:
        return {
            "status": "MILD_HYPERKALEMIA",
            "level": val,
            "is_alert": True,
            "severity": "MILD_ALERT",
            "message": f"Hyperkalemia Alert: Mild potassium elevation (K+: {val} mmol/L). Monitor closely; review potassium-sparing drugs, dietary potassium, and repeat electrolyte panel within 1-2 weeks."
        }
    elif val <= 5.9:
        return {
            "status": "MODERATE_HYPERKALEMIA",
            "level": val,
            "is_alert": True,
            "severity": "MODERATE_ALERT",
            "message": f"Critical Hyperkalemia Alert: Potassium {val} mmol/L. Elevated cardiac arrhythmia risk. Hold/dose-reduce ACEi/ARB or potassium-sparing agents; consider oral potassium binder (Patiromer/SZC); repeat within 48-72 hours."
        }
    else:
        return {
            "status": "SEVERE_HYPERKALEMIA",
            "level": val,
            "is_alert": True,
            "severity": "SEVERE_CRITICAL",
            "message": f"Medical Emergency: Severe Hyperkalemia (K+: {val} mmol/L). Urgent ECG for peaked T waves/conduction blocks and prompt emergency management (calcium gluconate, insulin-dextrose, urgent nephrology review)."
        }


def evaluate_cardiorenal(
    has_hf_condition: bool,
    symptoms: List[str],
    clinical_narrative: str,
    egfr: Optional[float]
) -> Dict[str, Any]:
    """
    Evaluates for clinical Heart Failure and Cardiorenal Syndrome (CRS).
    """
    text = f"{' '.join(symptoms)} {clinical_narrative}".lower()
    has_hf = has_hf_condition or any(term in text for term in ["heart failure", "chf", "reduced ejection fraction", "cardiorenal"])
    has_fluid_overload = any(term in text for term in [
        "bilateral lower limb edema", "pedal edema", "pitting edema",
        "breathlessness on exertion", "dyspnea", "shortness of breath",
        "orthopnea", "paroxysmal nocturnal dyspnea", "reduced exercise tolerance"
    ])

    is_crs = has_hf and (egfr is not None and egfr < 60.0)

    return {
        "heart_failure_present": has_hf,
        "fluid_overload_present": has_fluid_overload,
        "cardiorenal_syndrome": is_crs,
        "crs_type": "Cardiorenal Syndrome Type 2/4 (Chronic Cardiorenal / Renocardiac)" if is_crs else None,
        "clinical_action": (
            "Evaluate for active fluid overload, loop diuretic requirement (Furosemide/Bumetanide), "
            "and prioritize SGLT2 inhibitor therapy for cardiorenal protection and HF hospitalization reduction."
        ) if has_hf else None
    }


def evaluate_medication_eligibility(
    egfr: Optional[float],
    uacr: Optional[float],
    k_val: Optional[float],
    hba1c: Optional[float],
    has_hf: bool,
    has_dm: bool,
    has_htn: bool,
    age: Optional[int]
) -> Dict[str, Dict[str, Any]]:
    """
    Detailed evidence-based evaluation of clinical medication eligibility across:
    - Metformin
    - SGLT2 Inhibitors (Empagliflozin, Dapagliflozin)
    - GLP-1 Receptor Agonists (Dulaglutide, Semaglutide)
    - Insulin
    - ACE Inhibitors / ARBs (Lisinopril, Losartan)
    - Calcium Channel Blockers (Amlodipine)
    - Loop Diuretics (Furosemide)
    """
    egfr_val = float(egfr) if egfr is not None else 60.0
    uacr_val = float(uacr) if uacr is not None else 0.0
    k = float(k_val) if k_val is not None else 4.2
    a1c = float(hba1c) if hba1c is not None else 7.0
    is_peds = (age is not None and age < 18)

    eligibility = {}

    # 1. METFORMIN
    if not has_dm:
        eligibility["Metformin"] = {
            "status": "NOT_INDICATED",
            "dosing": "Not indicated (No diabetes)",
            "rationale": "Patient does not have documented diabetes.",
            "guideline": "ADA 2024",
            "contraindicated": False
        }
    elif is_peds:
        eligibility["Metformin"] = {
            "status": "CONTRAINDICATED",
            "dosing": "Withheld",
            "rationale": "Contraindicated in pediatric nephropathy / type 1 diabetes (ISPAD 2024 / KDIGO Pediatric).",
            "guideline": "ISPAD 2024",
            "contraindicated": True
        }
    elif egfr_val < 30.0:
        eligibility["Metformin"] = {
            "status": "CONTRAINDICATED",
            "dosing": "Discontinue / Withhold",
            "rationale": f"Strictly contraindicated at eGFR < 30 mL/min/1.73m² (Current eGFR: {egfr_val}) due to high risk of fatal lactic acidosis.",
            "guideline": "KDIGO 2023 Recommendation 1.3 / ADA 2024 (Level 1A)",
            "contraindicated": True
        }
    elif egfr_val < 45.0:  # eGFR 30 - 44 (G3b)
        eligibility["Metformin"] = {
            "status": "DOSE_REDUCED_MONITOR",
            "dosing": "Dose reduce to maximum 1000 mg/day (e.g. 500 mg BID)",
            "rationale": (
                f"eGFR is {egfr_val} mL/min/1.73m² (Stage G3b). Metformin is NOT automatically contraindicated. "
                "If patient is already established on therapy, reduce maximum dose to 1000 mg/day and monitor renal function every 3 months. "
                "Do not initiate if treatment-naive unless clear benefit is documented."
            ),
            "guideline": "KDIGO 2023 Practice Guideline Section 1.3 / ADA 2024 Standards of Care",
            "contraindicated": False
        }
    else:  # eGFR >= 45 (G1, G2, G3a)
        eligibility["Metformin"] = {
            "status": "FIRST_LINE_RECOMMENDED",
            "dosing": "Standard dose up to 2000 mg/day",
            "rationale": f"First-line antidiabetic therapy. Preserved filtration (eGFR {egfr_val} >= 45 mL/min/1.73m²).",
            "guideline": "ADA 2024 / KDIGO 2023 (Level 1A)",
            "contraindicated": False
        }

    # 2. SGLT2 INHIBITORS (Empagliflozin, Dapagliflozin)
    if is_peds:
        eligibility["SGLT2_Inhibitor"] = {
            "status": "NOT_RECOMMENDED",
            "dosing": "Withheld",
            "rationale": "Safety and efficacy not established for pediatric nephropathy (ISPAD 2024).",
            "guideline": "ISPAD 2024",
            "contraindicated": True
        }
    elif has_dm or has_hf or (uacr_val >= 200.0 and egfr_val >= 20.0):
        if egfr_val >= 20.0:
            benefit_desc = []
            if has_dm: benefit_desc.append("slows CKD progression & glycemic support")
            if has_hf: benefit_desc.append("reduces heart failure hospitalizations & cardiovascular death")
            if uacr_val >= 300.0: benefit_desc.append("substantial reduction in albuminuria (A3)")

            eligibility["SGLT2_Inhibitor"] = {
                "status": "STRONGLY_RECOMMENDED",
                "drug": "Empagliflozin 10mg daily or Dapagliflozin 10mg daily",
                "dosing": "10 mg orally once daily",
                "rationale": (
                    f"Strongly indicated for cardiorenal protection (eGFR {egfr_val} >= 20 mL/min/1.73m²). "
                    f"Provides critical organ protection: {', '.join(benefit_desc)}. "
                    "Cardiovascular and renal benefits persist even though glycemic lowering effect is diminished at lower eGFR."
                ),
                "guideline": "KDIGO 2023 Recommendation 1.4 / ADA 2024 / AHA/ACC/HFSA 2022 (Level 1A)",
                "contraindicated": False
            }
        else:
            eligibility["SGLT2_Inhibitor"] = {
                "status": "WITHHOLD_LOW_GFR",
                "dosing": "Withhold initiation (eGFR < 20)",
                "rationale": f"Initiation generally not recommended when eGFR is under 20 mL/min/1.73m² (Current eGFR: {egfr_val}).",
                "guideline": "KDIGO 2023",
                "contraindicated": True
            }
    else:
        eligibility["SGLT2_Inhibitor"] = {
            "status": "NOT_INDICATED",
            "dosing": "Not indicated",
            "rationale": "Criteria for SGLT2i initiation (T2D, CKD with albuminuria, or HF) not met.",
            "guideline": "KDIGO 2023",
            "contraindicated": False
        }

    # 3. GLP-1 RECEPTOR AGONISTS (Dulaglutide, Semaglutide)
    if is_peds:
        eligibility["GLP1_RA"] = {
            "status": "NOT_RECOMMENDED",
            "dosing": "Withheld",
            "rationale": "Not indicated as primary treatment for adolescent Type 1 Diabetes.",
            "guideline": "ISPAD 2024",
            "contraindicated": False
        }
    elif has_dm:
        if egfr_val >= 15.0:
            eligibility["GLP1_RA"] = {
                "status": "RECOMMENDED_CONSIDERATION",
                "drug": "Dulaglutide 0.75-1.5mg weekly or Semaglutide 0.5-1.0mg weekly",
                "dosing": "Weekly subcutaneous injection with upward titration",
                "rationale": (
                    f"Recommended add-on or alternative for T2D in CKD (eGFR {egfr_val} >= 15). "
                    "Demonstrated cardiovascular event reduction, weight benefit, and slowing of kidney disease progression without hypoglycemia risk."
                ),
                "guideline": "KDIGO 2023 Recommendation 1.5 / ADA 2024 (Level 1A)",
                "contraindicated": False
            }
        else:
            eligibility["GLP1_RA"] = {
                "status": "WITHHOLD_ESRD",
                "dosing": "Withhold in ESRD (eGFR < 15)",
                "rationale": "Limited clinical evidence in ESRD Stage 5.",
                "guideline": "KDIGO 2023",
                "contraindicated": True
            }
    else:
        eligibility["GLP1_RA"] = {
            "status": "NOT_INDICATED",
            "dosing": "Not indicated",
            "rationale": "No diabetes documented.",
            "guideline": "ADA 2024",
            "contraindicated": False
        }

    # 4. INSULIN
    if is_peds:
        eligibility["Insulin"] = {
            "status": "MANDATORY_FIRST_LINE",
            "drug": "Pediatric Basal-Bolus Insulin Regimen",
            "dosing": "Weight-based basal-bolus subcutaneous insulin with specialist titration",
            "rationale": "Mandatory standard of care for pediatric/adolescent Type 1 Diabetes and diabetic nephropathy.",
            "guideline": "ISPAD 2024 / KDIGO Pediatric 2023",
            "contraindicated": False
        }
    elif has_dm:
        if a1c >= 10.0 or (a1c >= 9.0 and egfr_val < 30.0):
            eligibility["Insulin"] = {
                "status": "INDICATED_SEVERE_HYPERGLYCEMIA",
                "drug": "Basal Insulin (e.g. Glargine or Degludec)",
                "dosing": "Conservative initial dose (0.1 - 0.2 units/kg/day) with careful titration",
                "rationale": (
                    f"Indicated for severe hyperglycemia (HbA1c {a1c}%). "
                    f"In advanced CKD (eGFR {egfr_val}), insulin clearance is substantially reduced; dose must be reduced by 25-50% to prevent life-threatening hypoglycemia."
                ),
                "guideline": "ADA 2024 / KDIGO 2023",
                "contraindicated": False
            }
        else:
            # Crucial check: HbA1c 7.4% or 8.1% alone does NOT justify knee-jerk insulin!
            eligibility["Insulin"] = {
                "status": "NOT_AUTOMATICALLY_INDICATED",
                "drug": "Not indicated as default monotherapy",
                "dosing": "Withhold default insulinization; prioritize non-insulin cardiorenal protective agents (SGLT2i / GLP-1 RA)",
                "rationale": (
                    f"HbA1c is {a1c}%. Insulin is NOT automatically indicated. "
                    "Premature insulinization carries substantial hypoglycemia risk, especially in advanced CKD where renal insulin degradation is impaired. "
                    "Prioritize guideline-directed oral/incretin therapies with cardiorenal protection (SGLT2 inhibitors and/or GLP-1 receptor agonists)."
                ),
                "guideline": "KDIGO 2023 / ADA 2024 Standards of Care",
                "contraindicated": False
            }
    else:
        eligibility["Insulin"] = {
            "status": "NOT_INDICATED",
            "dosing": "Not indicated",
            "rationale": "No diabetes documented.",
            "guideline": "ADA 2024",
            "contraindicated": False
        }

    # 5. ACE INHIBITORS / ARBS (Lisinopril, Losartan)
    if is_peds:
        eligibility["ACEi_ARB"] = {
            "status": "SPECIALIST_CONSULT",
            "dosing": "Pediatric nephrologist titration",
            "rationale": "Requires pediatric nephrology oversight for microalbuminuria.",
            "guideline": "KDIGO Pediatric 2023",
            "contraindicated": False
        }
    elif has_htn or uacr_val >= 30.0:
        if k >= 5.5:
            eligibility["ACEi_ARB"] = {
                "status": "CONTRAINDICATED_HYPERKALEMIA",
                "drug": "Withhold / Discontinue ACEi or ARB",
                "dosing": "Withheld until K+ <= 5.0 mmol/L",
                "rationale": f"Contraindicated due to moderate-severe hyperkalemia (Potassium: {k} mmol/L >= 5.5). Cardiac arrhythmia risk.",
                "guideline": "KDIGO 2023 Blood Pressure Guideline / AHA 2022",
                "contraindicated": True
            }
        elif k > 5.0:  # 5.1 - 5.4 mmol/L
            eligibility["ACEi_ARB"] = {
                "status": "CAUTION_CLOSE_MONITORING",
                "drug": "Low-dose ACEi/ARB with intensive potassium monitoring",
                "dosing": "Reduce dose or maintain under close electrolyte surveillance; repeat K+ in 1-2 weeks",
                "rationale": (
                    f"Indicated for renoprotection (UACR {uacr_val} mg/g), but serum potassium is mildly elevated ({k} mmol/L). "
                    "Initiate dietary potassium restriction; consider oral potassium binder if RAAS optimization is crucial; monitor closely."
                ),
                "guideline": "KDIGO 2023 Blood Pressure Consensus",
                "contraindicated": False
            }
        elif egfr_val < 30.0:
            eligibility["ACEi_ARB"] = {
                "status": "CAUTION_ADVANCED_CKD",
                "drug": "Conservative dosing",
                "dosing": "Low dose with creatinine & K+ check at 2 weeks",
                "rationale": f"Indicated for albuminuria, but in Stage 4 CKD (eGFR {egfr_val}), acute GFR decline and hyperkalemia must be closely monitored.",
                "guideline": "KDIGO 2023 Blood Pressure Guideline",
                "contraindicated": False
            }
        else:
            eligibility["ACEi_ARB"] = {
                "status": "STRONGLY_RECOMMENDED",
                "drug": "Lisinopril 10-20mg daily or Losartan 50mg daily",
                "dosing": "Titrate to maximally tolerated dose for albuminuria reduction",
                "rationale": f"First-line therapy for hypertension with albuminuria (UACR: {uacr_val} mg/g). Proven renoprotection.",
                "guideline": "KDIGO 2023 / ACC/AHA 2017 (Level 1A)",
                "contraindicated": False
            }
    else:
        eligibility["ACEi_ARB"] = {
            "status": "NOT_INDICATED",
            "dosing": "Not indicated",
            "rationale": "Normotensive with normal albumin excretion.",
            "guideline": "ACC/AHA 2017",
            "contraindicated": False
        }

    # 6. CALCIUM CHANNEL BLOCKERS (Amlodipine)
    if has_htn:
        eligibility["Amlodipine"] = {
            "status": "RECOMMENDED",
            "drug": "Amlodipine 5-10mg daily",
            "dosing": "5 mg once daily, titrate to 10 mg as needed",
            "rationale": "Effective vasodilator for blood pressure control without affecting renal hemodynamics or serum potassium.",
            "guideline": "ACC/AHA 2017 / KDIGO 2023",
            "contraindicated": False
        }

    # 7. LOOP DIURETICS (Furosemide)
    if has_hf or egfr_val < 30.0:
        eligibility["Loop_Diuretic"] = {
            "status": "RECOMMENDED_FOR_VOLUME_OVERLOAD",
            "drug": "Furosemide 20-40mg daily or Torsemide 10-20mg daily",
            "dosing": "Adjust based on fluid status and edema",
            "rationale": "Indicated to manage volume overload, peripheral edema, and cardiorenal congestion in heart failure / advanced CKD.",
            "guideline": "AHA/ACC/HFSA 2022 Heart Failure Guidelines",
            "contraindicated": False
        }

    return eligibility
