"""
SafeCDS Deterministic Clinical Rules Engine & Multi-Condition Safety Validator
Implements evidence-based KDIGO 2023, ADA 2024, ISPAD 2024, and ACC/AHA/HFSA 2022 clinical practice guidelines.
Provides deterministic staging, pediatric safety validation, hyperkalemia stratification,
comprehensive individual medication eligibility, and conversational question coverage.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
from src.models.patient_context import format_lab_value


def classify_ckd(egfr: Optional[float], has_kidney_damage: bool = True) -> Dict[str, Any]:
    """
    Deterministic KDIGO 2023 CKD Glomerular Filtration Rate (G-Stage) Staging.
    Preserves exact eGFR; never substitutes missing values with defaults.
    """
    if egfr is None:
        return {
            "stage_code": "UNKNOWN",
            "stage_name": "Unclassified (eGFR Not Tested / Missing)",
            "icd10": "R69",
            "severity": "UNKNOWN",
            "monitoring_interval": "Baseline kidney function panel (serum creatinine & eGFR) required",
            "is_confirmed": False
        }

    egfr_val = float(egfr)

    if egfr_val >= 90.0:
        if has_kidney_damage:
            return {
                "stage_code": "G1",
                "stage_name": "Chronic Kidney Disease Stage 1 (Normal or High GFR with Confirmed Kidney Damage)",
                "icd10": "N18.1",
                "severity": "MILD",
                "monitoring_interval": "Annual monitoring of eGFR and UACR",
                "is_confirmed": True
            }
        return {
            "stage_code": "G1",
            "stage_name": "Preserved Renal Function (eGFR >= 90 mL/min/1.73m²)",
            "icd10": "Z00.00",
            "severity": "NORMAL",
            "monitoring_interval": "Routine clinical care",
            "is_confirmed": True
        }
    elif egfr_val >= 60.0:
        if has_kidney_damage:
            return {
                "stage_code": "G2",
                "stage_name": "Chronic Kidney Disease Stage 2 (Mild GFR Reduction with Confirmed Kidney Damage)",
                "icd10": "N18.2",
                "severity": "MILD",
                "monitoring_interval": "Every 6-12 months",
                "is_confirmed": True
            }
        return {
            "stage_code": "G2",
            "stage_name": "Mildly Decreased eGFR without Established Structural Damage",
            "icd10": "R94.4",
            "severity": "MILD",
            "monitoring_interval": "Annual review of renal parameters",
            "is_confirmed": True
        }
    elif egfr_val >= 45.0:
        return {
            "stage_code": "G3a",
            "stage_name": "Chronic Kidney Disease Stage 3a (Mild-to-Moderate GFR Reduction)",
            "icd10": "N18.31",
            "severity": "MODERATE",
            "monitoring_interval": "Every 6 months (electrolytes, eGFR, UACR)",
            "is_confirmed": True
        }
    elif egfr_val >= 30.0:
        return {
            "stage_code": "G3b",
            "stage_name": "Chronic Kidney Disease Stage 3b (Moderate-to-Severe GFR Reduction)",
            "icd10": "N18.32",
            "severity": "MODERATE_SEVERE",
            "monitoring_interval": "Every 3-6 months (Nephrology consultation recommended)",
            "is_confirmed": True
        }
    elif egfr_val >= 15.0:
        return {
            "stage_code": "G4",
            "stage_name": "Chronic Kidney Disease Stage 4 (Severely Decreased GFR)",
            "icd10": "N18.4",
            "severity": "SEVERE",
            "monitoring_interval": "Every 1-3 months (Active nephrology co-management mandated)",
            "is_confirmed": True
        }
    else:
        return {
            "stage_code": "G5",
            "stage_name": "Chronic Kidney Disease Stage 5 (Kidney Failure / End-Stage Renal Disease)",
            "icd10": "N18.5",
            "severity": "KIDNEY_FAILURE",
            "monitoring_interval": "Monthly or per dialysis/transplant protocol",
            "is_confirmed": True
        }


def evaluate_pediatric_schwartz(
    height_cm: Optional[float],
    serum_creatinine: Optional[float]
) -> Optional[float]:
    """
    Computes pediatric eGFR using the Bedside Schwartz equation:
    eGFR = 0.413 * (height in cm / serum creatinine in mg/dL).
    Applicable for ages 1 to 17 years.
    """
    if height_cm is not None and serum_creatinine is not None and serum_creatinine > 0:
        return round(0.413 * (float(height_cm) / float(serum_creatinine)), 1)
    return None


def classify_albuminuria(
    uacr_mg_g: Optional[float],
    repeat_confirmed: bool = False,
    is_pediatric: bool = False
) -> Dict[str, Any]:
    """
    Deterministic KDIGO 2023 / ISPAD 2024 Albuminuria (A-Stage) Classification.
    Crucial Safety Rule: A single elevated UACR indicates SUSPECTED microalbuminuria.
    Guidelines strictly require 2 of 3 positive first-morning specimens over 3-6 months
    to establish confirmed persistent diabetic kidney disease or initiate pediatric ACEi/ARB.
    """
    if uacr_mg_g is None:
        return {
            "stage_code": "A_UNSPECIFIED",
            "stage_name": "Urine ACR Not Documented / Not Tested",
            "category": "Pending UACR test",
            "risk_multiplier": "Unknown",
            "is_confirmed": False,
            "clinical_guidance": "Spot urine albumin-to-creatinine ratio (UACR) recommended to assess renal microvascular injury."
        }

    val = float(uacr_mg_g)
    if val < 30.0:
        return {
            "stage_code": "A1",
            "stage_name": "Albuminuria Stage A1 (Normal to Mildly Increased: < 30 mg/g)",
            "category": "Normal to mildly increased",
            "risk_multiplier": "Baseline",
            "is_confirmed": True,
            "clinical_guidance": "Annual screening recommended."
        }
    elif val <= 300.0:
        if not repeat_confirmed:
            guidance = (
                f"Single elevated UACR ({val} mg/g) indicates Suspected Microalbuminuria (Stage A2). "
                "Per ISPAD 2024 and ADA 2024 guidelines, a single test does NOT establish confirmed diabetic nephropathy; "
                "transient elevations occur with fever, exercise, or posture. "
                "Mandatory repeat testing: 2 out of 3 first-morning specimens over 3 to 6 months required before confirming diagnosis "
                "or initiating ACEi/ARB therapy."
            )
            return {
                "stage_code": "A2",
                "stage_name": f"Suspected Albuminuria Stage A2 (Moderately Increased: {val} mg/g)",
                "category": "Suspected Microalbuminuria (Awaiting Repeat Confirmation)",
                "risk_multiplier": "High Cardiorenal Risk (if persistent)",
                "is_confirmed": False,
                "clinical_guidance": guidance
            }
        else:
            guidance = (
                f"Confirmed Persistent Microalbuminuria (Stage A2: {val} mg/g verified across multiple samples). "
                "Indicates established diabetic kidney disease. Guideline-directed renoprotection indicated "
                "(ACEi or ARB titration with potassium surveillance)."
            )
            return {
                "stage_code": "A2",
                "stage_name": f"Confirmed Albuminuria Stage A2 (Moderately Increased: {val} mg/g)",
                "category": "Confirmed Persistent Microalbuminuria",
                "risk_multiplier": "High Cardiorenal Risk",
                "is_confirmed": True,
                "clinical_guidance": guidance
            }
    else:  # val > 300
        return {
            "stage_code": "A3",
            "stage_name": f"Albuminuria Stage A3 (Severely Increased / Macroalbuminuria: {val} mg/g)",
            "category": "Macroalbuminuria / Nephrotic Range",
            "risk_multiplier": "Very High Cardiorenal Risk",
            "is_confirmed": repeat_confirmed,
            "clinical_guidance": (
                f"Severe albuminuria (UACR {val} mg/g). Prompt nephrology evaluation indicated. "
                "Strong indication for SGLT2 inhibitor (in adults with eGFR >= 20) and maximally tolerated RAAS blockade."
            )
        }


def evaluate_potassium(k_mmol_l: Optional[float]) -> Dict[str, Any]:
    """
    Evaluates serum potassium status per KDIGO / AHA guidelines.
    Preserves exact potassium value.
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
            "message": "Serum potassium not tested. Essential monitoring parameter in CKD and when using RAAS inhibitors.",
            "repeat_interval": "Obtain baseline electrolyte panel"
        }

    val = float(k_mmol_l)
    if val < 3.5:
        return {
            "status": "HYPOKALEMIA",
            "level": val,
            "is_alert": True,
            "severity": "MODERATE",
            "message": f"Hypokalemia alert (K+: {val} mmol/L). Evaluate for diuretic-induced wasting or gastrointestinal loss.",
            "repeat_interval": "Recheck in 1-2 weeks after potassium repletion"
        }
    elif val <= 5.0:
        return {
            "status": "NORMOKALEMIA",
            "level": val,
            "is_alert": False,
            "severity": "NORMAL",
            "message": f"Normokalemia (K+: {val} mmol/L, reference 3.5-5.0 mmol/L).",
            "repeat_interval": "Routine monitoring per CKD stage"
        }
    elif val <= 5.4:
        return {
            "status": "MILD_HYPERKALEMIA",
            "level": val,
            "is_alert": True,
            "severity": "MILD_ALERT",
            "message": (
                f"Mild Hyperkalemia Alert: Serum potassium is {val} mmol/L (reference 3.5-5.0 mmol/L). "
                "Monitor closely; review dietary potassium intake, avoid NSAIDs and salt substitutes. "
                "Repeat electrolyte panel within 1-2 weeks. Do NOT reflexively discontinue indicated RAAS inhibitors."
            ),
            "repeat_interval": "Repeat electrolyte panel within 1-2 weeks"
        }
    elif val <= 5.9:
        return {
            "status": "MODERATE_HYPERKALEMIA",
            "level": val,
            "is_alert": True,
            "severity": "MODERATE_ALERT",
            "message": (
                f"Critical Moderate Hyperkalemia Alert: Serum potassium is {val} mmol/L. "
                "Significantly increased cardiac arrhythmia risk. "
                "Mandatory prompt clinical assessment within 24-72 hours. "
                "Obtain 12-lead ECG to rule out peaked T waves or PR prolongation. "
                "Review all medications contributing to hyperkalemia (RAASi, MRAs, NSAIDs). "
                "Consider temporary dose reduction or hold of ACEi/ARB while initiating dietary restriction and oral potassium binder (Patiromer / SZC). "
                "Crucial: Do NOT permanently discontinue ACEi/ARB solely because CKD is advanced if hyperkalemia can be medically managed."
            ),
            "repeat_interval": "Prompt assessment and repeat within 24-72 hours"
        }
    else:  # val >= 6.0
        return {
            "status": "SEVERE_HYPERKALEMIA",
            "level": val,
            "is_alert": True,
            "severity": "SEVERE_CRITICAL",
            "message": (
                f"Medical Emergency: Severe Hyperkalemia (Serum K+: {val} mmol/L). "
                "Immediate emergency department referral required. "
                "Continuous cardiac telemetry and urgent 12-lead ECG. "
                "Immediate myocardial membrane stabilization (IV Calcium Gluconate) and intracellular shift (IV Insulin + Dextrose, Nebulized Albuterol), "
                "followed by active potassium removal (Sodium Zirconium Cyclosilicate, Loop Diuretics, or Hemodialysis). "
                "Hold all RAAS inhibitors and potassium supplements immediately."
            ),
            "repeat_interval": "Immediate emergency evaluation"
        }


def identify_hyperkalemia_contributing_medications(medications: List[str]) -> List[str]:
    """Identifies medications known to elevate serum potassium."""
    known_offenders = [
        ("lisinopril", "Lisinopril (ACE inhibitor - decreases aldosterone, reduces potassium excretion)"),
        ("enalapril", "Enalapril (ACE inhibitor - impairs kaliuresis)"),
        ("ramipril", "Ramipril (ACE inhibitor - impairs kaliuresis)"),
        ("losartan", "Losartan (ARB - blocks AT1 receptor, decreases potassium excretion)"),
        ("valsartan", "Valsartan (ARB - decreases potassium excretion)"),
        ("spironolactone", "Spironolactone (Potassium-sparing MRA - directly blocks aldosterone in distal tubule)"),
        ("eplerenone", "Eplerenone (MRA - increases hyperkalemia risk)"),
        ("finerenone", "Finerenone (Non-steroidal MRA - dose-dependent serum potassium increase)"),
        ("amiloride", "Amiloride (Potassium-sparing diuretic - blocks ENaC channel)"),
        ("triamterene", "Triamterene (Potassium-sparing diuretic)"),
        ("ibuprofen", "Ibuprofen (NSAID - inhibits prostaglandins, reduces renin and renal potassium clearance)"),
        ("naproxen", "Naproxen (NSAID - suppresses renal potassium secretion)"),
        ("bactrim", "Trimethoprim-Sulfamethoxazole (Amiloride-like ENaC blockage)"),
        ("trimethoprim", "Trimethoprim (Inhibits renal tubular potassium secretion)")
    ]
    detected = []
    meds_str = " ".join(medications).lower()
    for drug, explanation in known_offenders:
        if drug in meds_str:
            detected.append(explanation)
    return detected


def evaluate_cardiorenal(
    has_hf_condition: bool,
    symptoms: List[str],
    clinical_narrative: str,
    egfr: Optional[float]
) -> Dict[str, Any]:
    """Evaluates for clinical Heart Failure and Cardiorenal Syndrome (CRS)."""
    text = f"{' '.join(symptoms)} {clinical_narrative}".lower()
    has_hf = has_hf_condition or any(term in text for term in ["heart failure", "chf", "reduced ejection fraction", "cardiorenal", "hfpef", "hfref"])
    has_fluid_overload = any(term in text for term in [
        "bilateral lower limb edema", "pedal edema", "pitting edema",
        "breathlessness on exertion", "dyspnea", "shortness of breath",
        "orthopnea", "paroxysmal nocturnal dyspnea", "reduced exercise tolerance", "fluid overload"
    ])

    is_crs = has_hf and (egfr is not None and egfr < 60.0)

    return {
        "heart_failure_present": has_hf,
        "fluid_overload_present": has_fluid_overload,
        "cardiorenal_syndrome": is_crs,
        "crs_type": "Cardiorenal Syndrome Type 2/4 (Chronic Cardiorenal / Renocardiac)" if is_crs else None,
        "clinical_action": (
            "Evaluate for active fluid overload, loop diuretic requirement (Furosemide/Torsemide), "
            "and prioritize SGLT2 inhibitor therapy (Empagliflozin 10mg) for cardiorenal protection and HF hospitalization reduction."
        ) if has_hf else None
    }


def evaluate_individual_medication(
    med_name: str,
    patient_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates an individual medication for safety, contraindications, and guideline recommendations
    based on the patient's age, diabetes type, eGFR, potassium, albuminuria, heart failure, and current medications.
    Does NOT prescribe exact doses conversational; frames as clinical safety consideration.
    """
    med_key = med_name.strip().lower()
    age = patient_context.get("age")
    is_pediatric = (age is not None and age < 18)
    is_t1d = patient_context.get("is_type_1_diabetes", False)
    is_t2d = patient_context.get("is_type_2_diabetes", False)
    has_dm = is_t1d or is_t2d or ("diabetes" in str(patient_context.get("conditions", [])).lower())
    egfr = patient_context.get("lab_values", {}).get("egfr")
    k_val = patient_context.get("lab_values", {}).get("potassium")
    uacr = patient_context.get("lab_values", {}).get("uacr")
    has_hf = patient_context.get("cardiorenal_status", {}).get("heart_failure_present", False)
    current_meds = patient_context.get("current_medications", [])

    # 1. METFORMIN
    if "metformin" in med_key:
        if is_t1d:
            return {
                "medication": "Metformin",
                "status": "CONTRAINDICATED_AS_INSULIN_SUBSTITUTE",
                "safety_verdict": "Unsafe to Replace Insulin",
                "contraindicated": True,
                "rationale": (
                    "In Type 1 Diabetes, Metformin CANNOT replace essential insulin therapy. "
                    "Type 1 Diabetes involves absolute autoimmune insulin deficiency. "
                    "Attempting to treat T1D with oral metformin without insulin results in life-threatening Diabetic Ketoacidosis (DKA)."
                ),
                "guideline": "ISPAD 2024 / ADA 2024 Standards of Care",
                "action": "Ensure basal-bolus insulin is maintained; do not substitute with metformin."
            }
        elif is_pediatric:
            if is_t2d and age is not None and age >= 10:
                return {
                    "medication": "Metformin",
                    "status": "APPROVED_PEDIATRIC_T2D",
                    "safety_verdict": "FDA-Approved in Youth >= 10 with T2D",
                    "contraindicated": False,
                    "rationale": (
                        f"Metformin is FDA-approved for youth aged >= 10 years with Type 2 Diabetes. "
                        "It is NOT a blanket contraindication in children, provided renal function is normal (eGFR >= 60). "
                        "Requires titration to mitigate gastrointestinal side effects."
                    ),
                    "guideline": "ADA 2024 Children & Adolescents / ISPAD 2024",
                    "action": "Eligible for consideration by pediatric endocrinologist for Type 2 Diabetes."
                }
            else:
                return {
                    "medication": "Metformin",
                    "status": "NOT_RECOMMENDED_PEDIATRIC",
                    "safety_verdict": "Not Recommended in Pediatric Context",
                    "contraindicated": True,
                    "rationale": "Safety and efficacy not established for this pediatric presentation.",
                    "guideline": "ISPAD 2024",
                    "action": "Withhold in pediatric patient without confirmed Type 2 Diabetes."
                }
        elif egfr is not None and egfr < 30.0:
            return {
                "medication": "Metformin",
                "status": "STRICTLY_CONTRAINDICATED",
                "safety_verdict": "Contraindicated (High Lactic Acidosis Risk)",
                "contraindicated": True,
                "rationale": (
                    f"Strictly contraindicated at eGFR < 30 mL/min/1.73m² (Current eGFR: {egfr}). "
                    "Renal clearance of metformin is severely impaired, markedly elevating the risk of fatal lactic acidosis."
                ),
                "guideline": "KDIGO 2023 Recommendation 1.3 / ADA 2024 (Level 1A)",
                "action": "Discontinue or withhold metformin; transition to renal-safe alternatives."
            }
        elif egfr is not None and egfr < 45.0:
            return {
                "medication": "Metformin",
                "status": "DOSE_REDUCTION_AND_MONITORING",
                "safety_verdict": "Eligible with Dose Halving & Close Monitoring",
                "contraindicated": False,
                "rationale": (
                    f"eGFR is {egfr} mL/min/1.73m² (Stage G3b). Metformin is NOT automatically contraindicated. "
                    "KDIGO 2023 recommends reducing the maximum daily dose to 1000 mg (e.g. 500 mg twice daily) "
                    "with eGFR surveillance every 3 months. Do not initiate in treatment-naive patients unless benefit outweighs risk."
                ),
                "guideline": "KDIGO 2023 Practice Guideline 1.3 / ADA 2024",
                "action": "Consult physician for dose reduction to max 1000 mg/day and 3-month renal monitoring."
            }
        elif egfr is not None and egfr >= 45.0:
            return {
                "medication": "Metformin",
                "status": "FIRST_LINE_RECOMMENDED",
                "safety_verdict": "First-Line Recommended",
                "contraindicated": False,
                "rationale": f"First-line antidiabetic therapy with preserved filtration (eGFR {egfr} >= 45 mL/min/1.73m²).",
                "guideline": "ADA 2024 / KDIGO 2023",
                "action": "Standard dosing under routine glycemic and renal review."
            }
        else:
            return {
                "medication": "Metformin",
                "status": "PENDING_EGFR_EVALUATION",
                "safety_verdict": "Pending eGFR Lab Confirmation",
                "contraindicated": False,
                "rationale": "Metformin eligibility requires documented eGFR to rule out renal contraindication (< 30 mL/min/1.73m²).",
                "guideline": "KDIGO 2023",
                "action": "Check baseline eGFR prior to continuing or initiating."
            }

    # 2. SGLT2 INHIBITORS (Empagliflozin, Dapagliflozin)
    elif any(term in med_key for term in ["sglt2", "empagliflozin", "dapagliflozin", "canagliflozin"]):
        if is_t1d:
            return {
                "medication": "SGLT2 Inhibitor",
                "status": "NOT_RECOMMENDED_T1D_DKA_RISK",
                "safety_verdict": "High Risk of Euglycemic DKA in T1D",
                "contraindicated": True,
                "rationale": (
                    "In Type 1 Diabetes, SGLT2 inhibitors carry an unacceptable risk of euglycemic Diabetic Ketoacidosis (euDKA), "
                    "where ketoacidosis occurs with normal blood glucose levels. Not FDA approved for T1D."
                ),
                "guideline": "ADA 2024 / ISPAD 2024 Consensus",
                "action": "Avoid SGLT2 inhibitors in Type 1 Diabetes."
            }
        elif is_pediatric:
            if is_t2d and age is not None and age >= 10 and "empagliflozin" in med_key:
                return {
                    "medication": "Empagliflozin",
                    "status": "APPROVED_PEDIATRIC_T2D",
                    "safety_verdict": "FDA-Approved in Youth >= 10 with T2D",
                    "contraindicated": False,
                    "rationale": "Empagliflozin is approved as an adjunct to diet/exercise in pediatric T2D >= 10 years.",
                    "guideline": "FDA 2023 / ADA 2024",
                    "action": "Pediatric endocrinology evaluation recommended."
                }
            return {
                "medication": "SGLT2 Inhibitor",
                "status": "NOT_ESTABLISHED_PEDIATRIC",
                "safety_verdict": "Limited Evidence in Pediatrics",
                "contraindicated": True,
                "rationale": "Safety and efficacy not established for pediatric nephropathy in Type 1 Diabetes.",
                "guideline": "ISPAD 2024",
                "action": "Withhold in pediatric presentation."
            }
        elif egfr is not None and egfr < 20.0:
            return {
                "medication": "SGLT2 Inhibitor",
                "status": "WITHHOLD_INITIATION_LOW_GFR",
                "safety_verdict": "Withhold Initiation (eGFR < 20)",
                "contraindicated": True,
                "rationale": f"Initiation is not recommended when eGFR is under 20 mL/min/1.73m² (Current eGFR: {egfr}).",
                "guideline": "KDIGO 2023",
                "action": "Withhold initiation; alternative cardioprotective agents indicated."
            }
        elif egfr is not None and egfr >= 20.0:
            indications = []
            if is_t2d: indications.append("Type 2 Diabetes glycemic and renal protection")
            if has_hf: indications.append("Heart Failure with reduced or preserved ejection fraction")
            if uacr is not None and uacr >= 200: indications.append("Albuminuric Chronic Kidney Disease")
            ind_str = ", ".join(indications) if indications else "Cardiorenal protection"
            return {
                "medication": "SGLT2 Inhibitor (Empagliflozin 10mg / Dapagliflozin 10mg)",
                "status": "STRONGLY_RECOMMENDED",
                "safety_verdict": "Strong Guideline Recommendation (Level 1A)",
                "contraindicated": False,
                "rationale": (
                    f"Strongly indicated for {ind_str} (eGFR {egfr} >= 20 mL/min/1.73m²). "
                    "Demonstrates landmark reductions in CKD progression, cardiovascular mortality, and heart failure hospitalizations. "
                    "Organ protection persists down to eGFR 20 even when glycemic lowering diminishes."
                ),
                "guideline": "KDIGO 2023 Recommendation 1.4 / ADA 2024 / AHA/ACC/HFSA 2022",
                "action": "Physician consideration for initiation of Empagliflozin 10mg or Dapagliflozin 10mg daily."
            }
        else:
            return {
                "medication": "SGLT2 Inhibitor",
                "status": "EVALUATE_BASELINE_EGFR",
                "safety_verdict": "Confirm eGFR >= 20 prior to initiation",
                "contraindicated": False,
                "rationale": "SGLT2 inhibitors require confirmed eGFR >= 20 for initiation.",
                "guideline": "KDIGO 2023",
                "action": "Obtain baseline renal function."
            }

    # 3. GLP-1 RECEPTOR AGONISTS (Semaglutide, Dulaglutide)
    elif any(term in med_key for term in ["glp1", "glp-1", "semaglutide", "dulaglutide", "liraglutide"]):
        if is_t1d:
            return {
                "medication": "GLP-1 Receptor Agonist",
                "status": "NOT_INDICATED_T1D",
                "safety_verdict": "Not Approved for Type 1 Diabetes",
                "contraindicated": True,
                "rationale": "GLP-1 receptor agonists are not indicated for glycemic control in Type 1 Diabetes.",
                "guideline": "ISPAD 2024 / ADA 2024",
                "action": "Maintain insulin therapy; avoid GLP-1 RA in T1D."
            }
        elif egfr is not None and egfr < 15.0:
            return {
                "medication": "GLP-1 Receptor Agonist",
                "status": "WITHHOLD_ESRD",
                "safety_verdict": "Withhold in Stage 5 ESRD",
                "contraindicated": True,
                "rationale": "Insufficient safety data in dialysis-dependent Stage 5 CKD.",
                "guideline": "KDIGO 2023",
                "action": "Withhold in advanced kidney failure."
            }
        else:
            return {
                "medication": "GLP-1 Receptor Agonist (Semaglutide / Dulaglutide)",
                "status": "RECOMMENDED_CONSIDERATION",
                "safety_verdict": "Recommended for Cardiorenal & Metabolic Benefit",
                "contraindicated": False,
                "rationale": (
                    "Strongly supported by the FLOW trial and KDIGO 2023 for slowing kidney disease progression, "
                    "reducing major adverse cardiovascular events (MACE), and providing glycemic control without hypoglycemia."
                ),
                "guideline": "KDIGO 2023 Recommendation 1.5 / ADA 2024 (Level 1A)",
                "action": "Consider addition by clinician with gradual upward titration."
            }

    # 4. INSULIN
    elif "insulin" in med_key:
        if is_t1d:
            return {
                "medication": "Insulin",
                "status": "MANDATORY_LIFE_ESSENTIAL",
                "safety_verdict": "Mandatory Life-Sustaining Therapy",
                "contraindicated": False,
                "rationale": (
                    "In Type 1 Diabetes, subcutaneous insulin (basal-bolus or insulin pump) is life-essential and non-negotiable. "
                    "Autoimmune destruction of pancreatic beta cells prevents endogenous insulin production. "
                    "Insulin must NEVER be stopped or substituted with oral antidiabetic agents."
                ),
                "guideline": "ISPAD 2024 / ADA 2024 Standards of Care (Level 1A)",
                "action": "Maintain individualized basal-bolus or pump insulin regimen under endocrinology care."
            }
        else:
            hba1c = patient_context.get("lab_values", {}).get("hba1c")
            if egfr is not None and egfr < 30.0:
                return {
                    "medication": "Insulin",
                    "status": "CAUTION_ADVANCED_CKD_HYPOGLYCEMIA",
                    "safety_verdict": "Use with Extreme Caution in CKD G4/G5",
                    "contraindicated": False,
                    "rationale": (
                        f"In advanced CKD (eGFR {egfr}), renal clearance and degradation of insulin are markedly reduced, "
                        "substantially increasing the risk of prolonged, life-threatening hypoglycemia. "
                        "If insulin is necessary, conservative initial dosing (reduced by 25-50%) and intensive glucose monitoring are required. "
                        "Do not reflexively start insulin for moderate HbA1c without evaluating safer incretin/SGLT2 therapies."
                    ),
                    "guideline": "KDIGO 2023 Practice Guideline / ADA 2024",
                    "action": "Physician-supervised titration with mandatory dose reduction and frequent glucose checks."
                }
            else:
                return {
                    "medication": "Insulin",
                    "status": "RESERVED_FOR_INDICATION",
                    "safety_verdict": "Reserved for Severe Hyperglycemia or Failed Oral Therapy",
                    "contraindicated": False,
                    "rationale": (
                        "In Type 2 Diabetes, insulin is reserved for severe symptomatic hyperglycemia (HbA1c > 10%) "
                        "or when combination non-insulin therapies fail. Not indicated as automatic first-line therapy."
                    ),
                    "guideline": "ADA 2024 / KDIGO 2023",
                    "action": "Prioritize guideline-directed oral/incretin agents with proven cardiorenal benefit."
                }

    # 5. ACE INHIBITORS / ARBS (Lisinopril, Losartan, Enalapril)
    elif any(term in med_key for term in ["lisinopril", "losartan", "enalapril", "ramipril", "ace", "arb"]):
        if is_pediatric:
            uacr_stage = classify_albuminuria(uacr, is_pediatric=True)
            if not uacr_stage["is_confirmed"]:
                return {
                    "medication": "ACE Inhibitor / ARB",
                    "status": "DEFER_UNTIL_REPEAT_CONFIRMATION",
                    "safety_verdict": "Defer Until Persistent Albuminuria Confirmed",
                    "contraindicated": False,
                    "rationale": (
                        "In pediatric diabetes, a single elevated UACR does NOT justify starting an ACE inhibitor or ARB. "
                        "Guidelines mandate confirming persistent microalbuminuria (2 of 3 positive tests over 3-6 months) "
                        "before initiating therapy, with teratogenicity counseling and pediatric nephrology consultation."
                    ),
                    "guideline": "ISPAD 2024 / ADA 2024 Pediatric Section",
                    "action": "Order 2 repeat morning spot UACR tests over next 3-6 months before initiating RAAS blockade."
                }
            else:
                return {
                    "medication": "ACE Inhibitor / ARB",
                    "status": "RECOMMENDED_CONFIRMED_ALBUMINURIA",
                    "safety_verdict": "Indicated for Confirmed Persistent Albuminuria",
                    "contraindicated": False,
                    "rationale": "Indicated for confirmed persistent microalbuminuria under pediatric nephrology supervision.",
                    "guideline": "ISPAD 2024",
                    "action": "Pediatric nephrology consultation for titration and electrolyte monitoring."
                }

        # Adult ACEi/ARB evaluation
        if k_val is not None and k_val >= 6.0:
            return {
                "medication": "ACE Inhibitor / ARB",
                "status": "URGENTLY_WITHHOLD_SEVERE_HYPERKALEMIA",
                "safety_verdict": "Urgently Withhold (Severe Hyperkalemia >= 6.0)",
                "contraindicated": True,
                "rationale": f"Severe hyperkalemia (Potassium: {k_val} mmol/L). Extreme cardiac risk; withhold immediately.",
                "guideline": "KDIGO 2023 Blood Pressure Guideline",
                "action": "Withhold immediately; refer for emergency management."
            }
        elif k_val is not None and k_val >= 5.5:
            return {
                "medication": "ACE Inhibitor / ARB",
                "status": "MODERATE_HYPERKALEMIA_MANAGEMENT",
                "safety_verdict": "Manage Hyperkalemia; Do Not Automatically Discontinue Permanently",
                "contraindicated": False,
                "rationale": (
                    f"Serum potassium is {k_val} mmol/L (moderate hyperkalemia). ACE inhibitors decrease aldosterone and potassium excretion. "
                    "However, KDIGO 2023 practice guidelines emphasize that ACEi/ARBs should NOT be permanently discontinued solely because CKD is advanced. "
                    "Recommended clinical strategy: Obtain 12-lead ECG, review dietary potassium, consider temporary dose reduction or temporary hold, "
                    "and initiate oral potassium binders (Patiromer or Sodium Zirconium Cyclosilicate) or loop diuretics to allow continuation of RAAS cardiorenal benefit."
                ),
                "guideline": "KDIGO 2023 Blood Pressure Consensus / AHA 2022",
                "action": "Prompt clinical evaluation within 24-72 hours. Consider potassium binder before permanent discontinuation."
            }
        elif k_val is not None and k_val > 5.0:
            return {
                "medication": "ACE Inhibitor / ARB",
                "status": "CAUTION_MILD_HYPERKALEMIA",
                "safety_verdict": "Continue with Dietary Review & 1-2 Week Repeat",
                "contraindicated": False,
                "rationale": (
                    f"Mild potassium elevation ({k_val} mmol/L). Continue therapy with dietary counseling and repeat K+ in 1-2 weeks. "
                    "Do NOT reflexively stop therapy."
                ),
                "guideline": "KDIGO 2023 Blood Pressure Guideline",
                "action": "Dietary review and repeat electrolyte panel in 1-2 weeks."
            }
        elif egfr is not None and egfr < 30.0:
            return {
                "medication": "ACE Inhibitor / ARB",
                "status": "CAUTION_ADVANCED_CKD",
                "safety_verdict": "Eligible in Stage 4 CKD with Close Monitoring",
                "contraindicated": False,
                "rationale": (
                    f"In Stage 4 CKD (eGFR {egfr}), ACEi/ARBs provide ongoing cardiorenal protection. "
                    "KDIGO recommends continuing them unless hyperkalemia cannot be controlled or symptomatic uremia occurs. "
                    "Do NOT discontinue solely based on low eGFR."
                ),
                "guideline": "KDIGO 2023 Blood Pressure Consensus / STOP-ACEi Trial",
                "action": "Maintain under routine creatinine and potassium surveillance."
            }
        else:
            return {
                "medication": "ACE Inhibitor / ARB (Lisinopril / Losartan)",
                "status": "STRONGLY_RECOMMENDED",
                "safety_verdict": "First-Line Recommended for Albuminuria & Hypertension",
                "contraindicated": False,
                "rationale": "First-line agent for blood pressure control and renoprotection in patients with albuminuria.",
                "guideline": "KDIGO 2023 / ACC/AHA 2017 (Level 1A)",
                "action": "Standard titration to maximum tolerated dose."
            }

    # 6. MINERALOCORTICOID RECEPTOR ANTAGONISTS (Finerenone, Spironolactone)
    elif any(term in med_key for term in ["finerenone", "spironolactone", "eplerenone", "mra"]):
        if k_val is not None and k_val > 5.0:
            return {
                "medication": "Mineralocorticoid Receptor Antagonist",
                "status": "CONTRAINDICATED_HYPERKALEMIA",
                "safety_verdict": "Contraindicated if Potassium > 5.0 mmol/L",
                "contraindicated": True,
                "rationale": (
                    f"Serum potassium is {k_val} mmol/L (> 5.0 mmol/L). "
                    "Non-steroidal MRAs (Finerenone) and steroidal MRAs (Spironolactone) are contraindicated "
                    "when baseline serum potassium exceeds 5.0 mmol/L due to steep hyperkalemia risk."
                ),
                "guideline": "KDIGO 2023 / FIDELIO-DKD Trial Criteria",
                "action": "Withhold MRA initiation until potassium is reliably <= 4.8-5.0 mmol/L."
            }
        elif egfr is not None and egfr < 25.0:
            return {
                "medication": "Finerenone",
                "status": "WITHHOLD_LOW_GFR",
                "safety_verdict": "Withhold Initiation (eGFR < 25)",
                "contraindicated": True,
                "rationale": "Finerenone initiation is not recommended when eGFR is under 25 mL/min/1.73m².",
                "guideline": "KDIGO 2023 / FDA Prescribing Information",
                "action": "Withhold in severe renal impairment."
            }
        elif is_t2d and (uacr is not None and uacr >= 30.0) and (egfr is not None and egfr >= 25.0):
            return {
                "medication": "Finerenone (Non-steroidal MRA)",
                "status": "STRONGLY_RECOMMENDED",
                "safety_verdict": "Indicated for CKD in Type 2 Diabetes with Albuminuria",
                "contraindicated": False,
                "rationale": (
                    "Finerenone reduces CKD progression and cardiovascular events in T2D patients on maximum tolerated RAASi. "
                    "Requires baseline K+ <= 5.0 mmol/L and regular potassium checks at 1 and 4 months."
                ),
                "guideline": "KDIGO 2023 Recommendation 1.6 / ADA 2024",
                "action": "Eligible for consideration with scheduled potassium surveillance."
            }
        else:
            return {
                "medication": "Mineralocorticoid Receptor Antagonist",
                "status": "EVALUATE_CRITERIA",
                "safety_verdict": "Assess Potassium & eGFR Thresholds",
                "contraindicated": False,
                "rationale": "Requires confirmed potassium <= 5.0 and eGFR >= 25 for safe initiation.",
                "guideline": "KDIGO 2023",
                "action": "Verify baseline potassium and renal parameters."
            }

    # 7. LOOP DIURETICS (Furosemide, Torsemide)
    elif any(term in med_key for term in ["furosemide", "torsemide", "bumetanide", "loop diuretic"]):
        if has_hf or (egfr is not None and egfr < 30.0) or ("fluid overload" in str(patient_context).lower()):
            return {
                "medication": "Loop Diuretic (Furosemide / Torsemide)",
                "status": "STRONGLY_INDICATED_FOR_VOLUME_OVERLOAD",
                "safety_verdict": "Indicated for Decongestion & Volume Overload",
                "contraindicated": False,
                "rationale": (
                    "Indicated for managing fluid retention, peripheral edema, and cardiorenal congestion in heart failure and advanced CKD. "
                    "Also promotes urinary potassium excretion, which may assist with borderline elevated potassium."
                ),
                "guideline": "AHA/ACC/HFSA 2022 Heart Failure Guidelines / KDIGO 2023",
                "action": "Physician evaluation for titration based on fluid volume status and dry weight."
            }
        else:
            return {
                "medication": "Loop Diuretic",
                "status": "NOT_ROUTINELY_INDICATED",
                "safety_verdict": "Reserved for Hypervolemia or Edema",
                "contraindicated": False,
                "rationale": "Loop diuretics are indicated for fluid overload and edema; not recommended as routine monotherapy without fluid retention.",
                "guideline": "ACC/AHA 2017",
                "action": "Reserve for clinical signs of congestion."
            }

    # 8. CALCIUM CHANNEL BLOCKERS (Amlodipine)
    elif any(term in med_key for term in ["amlodipine", "nifedipine", "diltiazem"]):
        return {
            "medication": "Calcium Channel Blocker (Amlodipine)",
            "status": "RECOMMENDED_HYPERTENSION",
            "safety_verdict": "Safe Antihypertensive in Renal Disease",
            "contraindicated": False,
            "rationale": (
                "Potent blood pressure reduction without affecting renal hemodynamics or serum potassium. "
                "Safe across all stages of CKD (G1-G5) and does not increase hyperkalemia risk."
            ),
            "guideline": "ACC/AHA 2017 / KDIGO 2023",
            "action": "Standard dosing (5-10 mg daily) under routine BP monitoring."
        }

    # Default fallback
    return {
        "medication": med_name,
        "status": "CLINICAL_REVIEW_REQUIRED",
        "safety_verdict": "Requires Individual Physician Review",
        "contraindicated": False,
        "rationale": f"Safety assessment for '{med_name}' requires comprehensive evaluation of patient comorbidities and lab values.",
        "guideline": "Standard Clinical Practice",
        "action": "Consult treating physician for clinical determination."
    }


def extract_inquired_medications(text: str) -> List[str]:
    """
    Extracts all distinct medications explicitly mentioned in conversational text.
    Ensures that every medication inquired about by the user receives independent coverage.
    """
    catalog = [
        "metformin", "lisinopril", "losartan", "enalapril", "ramipril",
        "empagliflozin", "dapagliflozin", "canagliflozin", "sglt2 inhibitor", "sglt2",
        "semaglutide", "dulaglutide", "liraglutide", "glp-1", "glp1",
        "insulin", "furosemide", "torsemide", "bumetanide",
        "spironolactone", "eplerenone", "finerenone", "amlodipine",
        "aspirin", "atorvastatin", "glipizide"
    ]
    detected = []
    lower = text.lower()
    for med in catalog:
        # Match as whole word or boundary
        pattern = rf'\b{re.escape(med)}\b'
        if re.search(pattern, lower):
            canonical = med.title()
            if canonical == "Sglt2" or canonical == "Sglt2 Inhibitor":
                canonical = "SGLT2 Inhibitor"
            elif canonical == "Glp-1" or canonical == "Glp1":
                canonical = "GLP-1 Receptor Agonist"
            if canonical not in detected:
                detected.append(canonical)
    return detected


def evaluate_inquired_medications(
    inquired_meds: List[str],
    patient_context: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates every medication explicitly mentioned by the user independently.
    Guarantees that no medication inquiry is omitted from the response.
    """
    results = {}
    for med in inquired_meds:
        results[med] = evaluate_individual_medication(med, patient_context)
    return results


def categorize_clinical_diagnoses(
    patient_context: Dict[str, Any]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Strictly separates confirmed clinical diagnoses from suspected conditions and missing evidence.
    Returns: (confirmed_diagnoses, suspected_conditions, missing_evidence)
    """
    confirmed: List[str] = []
    suspected: List[str] = []
    missing: List[str] = []

    conds = patient_context.get("conditions", [])
    labs = patient_context.get("lab_values", {})
    egfr = labs.get("egfr")
    uacr = labs.get("uacr")
    k_val = labs.get("potassium")
    hba1c = labs.get("hba1c")
    age = patient_context.get("age")
    is_pediatric = (age is not None and age < 18)

    # 1. Diabetes
    if patient_context.get("is_type_1_diabetes"):
        confirmed.append("Type 1 Diabetes Mellitus (ICD-10: E10.9) - Insulin Dependent")
    elif patient_context.get("is_type_2_diabetes"):
        confirmed.append("Type 2 Diabetes Mellitus (ICD-10: E11.9)")

    # 2. CKD Staging
    ckd = classify_ckd(egfr)
    if ckd["is_confirmed"]:
        confirmed.append(f"{ckd['stage_name']} (ICD-10: {ckd['icd10']})")
    else:
        missing.append("Baseline renal panel (eGFR & serum creatinine) to determine kidney disease stage")

    # 3. Albuminuria
    if uacr is not None:
        alb = classify_albuminuria(uacr, repeat_confirmed=False, is_pediatric=is_pediatric)
        if alb["is_confirmed"]:
            confirmed.append(f"{alb['stage_name']}")
        else:
            suspected.append(
                f"Suspected Microalbuminuria ({uacr} mg/g) - Requires 2 of 3 repeat tests over 3-6 months to establish confirmed diabetic kidney disease"
            )
            missing.append("Repeat morning spot UACR tests (2 of 3 over 3-6 months) for clinical confirmation")
    else:
        missing.append("Urine Albumin-to-Creatinine Ratio (UACR) test")

    # 4. Potassium
    if k_val is not None:
        k_eval = evaluate_potassium(k_val)
        if k_eval["is_alert"]:
            suspected.append(f"{k_eval['status']}: {k_eval['message']}")
            if k_eval["severity"] in ["MODERATE_ALERT", "SEVERE_CRITICAL"]:
                missing.append("Immediate 12-lead ECG to rule out hyperkalemic cardiac conduction delay (peaked T waves)")
    else:
        missing.append("Serum potassium test (essential in CKD and RAAS therapy)")

    # 5. Heart Failure
    cr = patient_context.get("cardiorenal_status", {})
    if cr.get("heart_failure_present"):
        confirmed.append("Clinical Heart Failure (ICD-10: I50.9)")
    if cr.get("fluid_overload_present"):
        suspected.append("Active Cardiorenal Fluid Congestion / Volume Overload")

    # Clean duplicates
    confirmed = list(dict.fromkeys(confirmed))
    suspected = list(dict.fromkeys(suspected))
    missing = list(dict.fromkeys(missing))

    return confirmed, suspected, missing
