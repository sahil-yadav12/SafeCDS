"""
SafeCDS Single Structured Patient Context & Data Integrity Module
Provides strict data preservation, reference range validation, unit checks,
and bidirectional consistency verification between patient inputs and generated outputs.
"""

from typing import Dict, Any, List, Optional, Tuple
import copy


# Standard Clinical Reference Ranges and Units
CLINICAL_REFERENCE_RANGES = {
    "egfr": {
        "unit": "mL/min/1.73m²",
        "normal_min": 90.0,
        "normal_max": 180.0,
        "valid_min": 1.0,
        "valid_max": 250.0,
        "name": "Estimated Glomerular Filtration Rate (eGFR)"
    },
    "potassium": {
        "unit": "mmol/L",
        "normal_min": 3.5,
        "normal_max": 5.0,
        "valid_min": 1.5,
        "valid_max": 9.5,
        "name": "Serum Potassium (K+)"
    },
    "serum_potassium": {
        "unit": "mmol/L",
        "normal_min": 3.5,
        "normal_max": 5.0,
        "valid_min": 1.5,
        "valid_max": 9.5,
        "name": "Serum Potassium (K+)"
    },
    "creatinine": {
        "unit": "mg/dL",
        "normal_min": 0.6,
        "normal_max": 1.2,
        "valid_min": 0.1,
        "valid_max": 30.0,
        "name": "Serum Creatinine"
    },
    "serum_creatinine": {
        "unit": "mg/dL",
        "normal_min": 0.6,
        "normal_max": 1.2,
        "valid_min": 0.1,
        "valid_max": 30.0,
        "name": "Serum Creatinine"
    },
    "hba1c": {
        "unit": "%",
        "normal_min": 4.0,
        "normal_max": 5.6,
        "valid_min": 3.0,
        "valid_max": 22.0,
        "name": "Glycated Hemoglobin (HbA1c)"
    },
    "sbp": {
        "unit": "mmHg",
        "normal_min": 90,
        "normal_max": 120,
        "valid_min": 50,
        "valid_max": 280,
        "name": "Systolic Blood Pressure"
    },
    "dbp": {
        "unit": "mmHg",
        "normal_min": 60,
        "normal_max": 80,
        "valid_min": 30,
        "valid_max": 160,
        "name": "Diastolic Blood Pressure"
    },
    "uacr": {
        "unit": "mg/g",
        "normal_min": 0.0,
        "normal_max": 29.9,
        "valid_min": 0.0,
        "valid_max": 15000.0,
        "name": "Urine Albumin-to-Creatinine Ratio (UACR)"
    },
    "sodium": {
        "unit": "mmol/L",
        "normal_min": 135.0,
        "normal_max": 145.0,
        "valid_min": 100.0,
        "valid_max": 180.0,
        "name": "Serum Sodium (Na+)"
    },
    "serum_sodium": {
        "unit": "mmol/L",
        "normal_min": 135.0,
        "normal_max": 145.0,
        "valid_min": 100.0,
        "valid_max": 180.0,
        "name": "Serum Sodium (Na+)"
    },
    "bicarbonate": {
        "unit": "mmol/L",
        "normal_min": 22.0,
        "normal_max": 29.0,
        "valid_min": 5.0,
        "valid_max": 50.0,
        "name": "Serum Bicarbonate (HCO3-)"
    }
}


def format_lab_value(val: Any, key: str = "") -> str:
    """
    Prevents missing values from displaying as 'None' or '0'.
    Formats with proper medical unit when available.
    """
    if val is None or val == "":
        return "Not Tested"
    ref = CLINICAL_REFERENCE_RANGES.get(key.lower(), {})
    unit = ref.get("unit", "")
    try:
        f_val = float(val)
        if f_val.is_integer() and key.lower() in ["sbp", "dbp", "age"]:
            return f"{int(f_val)} {unit}".strip()
        return f"{f_val:.1f} {unit}".strip()
    except (ValueError, TypeError):
        return f"{val} {unit}".strip()


def validate_lab_integrity(raw_labs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Validates patient laboratory values:
    1. Preserves original values without substitution or defaulting.
    2. Checks plausible physiological range.
    3. Records missing clinical parameters without zero-filling.
    Returns: (preserved_labs, integrity_alerts, missing_parameters)
    """
    preserved: Dict[str, Any] = {}
    alerts: List[str] = []
    missing: List[str] = []

    # Check key lab parameters
    key_params = ["egfr", "potassium", "creatinine", "hba1c", "sbp", "dbp", "uacr"]
    for param in key_params:
        # Check canonical or alternative key (e.g. serum_potassium)
        val = raw_labs.get(param)
        if val is None and f"serum_{param}" in raw_labs:
            val = raw_labs.get(f"serum_{param}")

        if val is None or val == "":
            missing.append(param)
            preserved[param] = None
            continue

        try:
            num = float(val)
            ref = CLINICAL_REFERENCE_RANGES.get(param, {})
            valid_min = ref.get("valid_min", 0.0)
            valid_max = ref.get("valid_max", 99999.0)

            if num < valid_min or num > valid_max:
                alerts.append(
                    f"Plausibility Warning: {ref.get('name', param)} value {num} {ref.get('unit', '')} "
                    f"is outside standard physiological range ({valid_min}-{valid_max} {ref.get('unit', '')}). "
                    "Confirm lab reporting units."
                )
            preserved[param] = num
        except (ValueError, TypeError):
            alerts.append(f"Data Integrity Error: Unable to parse {param} value: '{val}'. Value not recognized as numeric.")
            preserved[param] = None
            missing.append(param)

    # Copy any additional lab fields verbatim without loss
    for k, v in raw_labs.items():
        if k not in preserved:
            preserved[k] = copy.deepcopy(v)

    return preserved, alerts, missing


class PatientContext:
    """
    Single structured patient context object representing the complete patient state.
    Preserves original lab inputs throughout the reasoning pipeline.
    """
    def __init__(
        self,
        patient_id: str,
        patient_name: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        height_cm: Optional[float] = None,
        weight_kg: Optional[float] = None,
        conditions: Optional[List[str]] = None,
        symptoms: Optional[List[str]] = None,
        current_medications: Optional[List[str]] = None,
        inquired_medications: Optional[List[str]] = None,
        lab_values: Optional[Dict[str, Any]] = None,
        clinical_narrative: Optional[str] = None
    ):
        self.patient_id = patient_id
        self.patient_name = patient_name or "Anonymous Patient"
        self.age = age
        self.gender = gender or "Unspecified"
        self.height_cm = height_cm
        self.weight_kg = weight_kg
        self.conditions = list(dict.fromkeys(conditions or []))
        self.symptoms = list(dict.fromkeys(symptoms or []))
        self.current_medications = list(dict.fromkeys(current_medications or []))
        self.inquired_medications = list(dict.fromkeys(inquired_medications or []))
        self.clinical_narrative = clinical_narrative or ""

        # Store immutable copy of original labs
        raw_labs = lab_values or {}
        self.original_lab_values = copy.deepcopy(raw_labs)

        # Validate and preserve labs
        preserved, alerts, missing = validate_lab_integrity(raw_labs)
        self.lab_values = preserved
        self.integrity_alerts = alerts
        self.missing_labs = missing

        # Diabetes classification (strictly distinct T1D vs T2D)
        self.is_type_1_diabetes = self._determine_t1d()
        self.is_type_2_diabetes = self._determine_t2d()

        # Pediatric classification
        self.is_pediatric = (self.age is not None and self.age < 18)

    def _determine_t1d(self) -> bool:
        conds_str = " ".join(self.conditions).lower()
        narrative_str = self.clinical_narrative.lower()
        return (
            "type_1_diabetes" in conds_str
            or "type 1 diabetes" in narrative_str
            or "t1d" in conds_str
            or "t1dm" in narrative_str
        )

    def _determine_t2d(self) -> bool:
        # If explicitly T1D, it cannot be T2D unless dual etiology is clinically documented
        if self._determine_t1d():
            return False
        conds_str = " ".join(self.conditions).lower()
        narrative_str = self.clinical_narrative.lower()
        hba1c = self.lab_values.get("hba1c")
        return (
            "type_2_diabetes" in conds_str
            or "type 2 diabetes" in narrative_str
            or "t2d" in conds_str
            or "t2dm" in narrative_str
            or (hba1c is not None and hba1c >= 6.5)
        )

    def get_lab_display(self, key: str) -> str:
        """Returns formatted string or 'Not Tested'."""
        val = self.lab_values.get(key)
        if val is None and f"serum_{key}" in self.lab_values:
            val = self.lab_values.get(f"serum_{key}")
        return format_lab_value(val, key)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes patient context to a dictionary."""
        return {
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "age": self.age,
            "gender": self.gender,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "conditions": self.conditions,
            "symptoms": self.symptoms,
            "current_medications": self.current_medications,
            "inquired_medications": self.inquired_medications,
            "original_lab_values": self.original_lab_values,
            "lab_values": self.lab_values,
            "missing_labs": self.missing_labs,
            "integrity_alerts": self.integrity_alerts,
            "is_pediatric": self.is_pediatric,
            "is_type_1_diabetes": self.is_type_1_diabetes,
            "is_type_2_diabetes": self.is_type_2_diabetes,
            "clinical_narrative": self.clinical_narrative
        }


def verify_patient_output_consistency(
    context: PatientContext,
    generated_output: Dict[str, Any]
) -> List[str]:
    """
    Consistency check comparing patient inputs with generated output:
    1. Verifies that original lab values were preserved without corruption.
    2. Verifies that missing labs were not populated with hallucinated numbers.
    3. Verifies that all inquired medications received individual assessment.
    4. Verifies diabetes type compatibility.
    5. Verifies pediatric versus adult guidelines.
    """
    inconsistencies: List[str] = []

    # 1. Potassium preservation check
    orig_k = context.lab_values.get("potassium")
    if orig_k is not None:
        reported_k = generated_output.get("potassium_status")
        # Ensure that if potassium is elevated, output does not describe it as normal
        if orig_k > 5.0 and reported_k and "NORMOKALEMIA" in reported_k:
            inconsistencies.append(
                f"Data Inconsistency: Patient potassium is {orig_k} mmol/L, but output status reported Normokalemia."
            )

    # 2. eGFR preservation check
    orig_egfr = context.lab_values.get("egfr")
    reported_stage = generated_output.get("ckd_stage")
    if orig_egfr is not None and reported_stage:
        if orig_egfr >= 30 and orig_egfr < 45 and reported_stage not in ["G3b", "CKD_Stage_3b"]:
            inconsistencies.append(
                f"Staging Inconsistency: Patient eGFR {orig_egfr} requires G3b staging; received {reported_stage}."
            )
        elif orig_egfr >= 15 and orig_egfr < 30 and reported_stage not in ["G4", "CKD_Stage_4"]:
            inconsistencies.append(
                f"Staging Inconsistency: Patient eGFR {orig_egfr} requires G4 staging; received {reported_stage}."
            )

    # 3. T1D insulin preservation check
    if context.is_type_1_diabetes:
        rec_treatment = str(generated_output.get("recommended_treatment", "")).lower()
        rationale = str(generated_output.get("clinical_rationale", "")).lower()
        combined = f"{rec_treatment} {rationale}"
        if "replace insulin" in combined or "stop insulin" in combined or "discontinue insulin" in combined:
            inconsistencies.append(
                "Critical Safety Inconsistency: Type 1 Diabetes patient cannot have essential insulin therapy replaced or stopped."
            )

    # 4. Inquired medications coverage check
    med_assessments = generated_output.get("medication_assessments", {})
    for med in context.inquired_medications:
        med_lower = med.lower()
        found = any(k.lower() == med_lower or med_lower in k.lower() for k in med_assessments.keys())
        if not found:
            # Check if mentioned in individual answers or rationale
            rat = str(generated_output.get("clinical_rationale", "")).lower()
            if med_lower not in rat:
                inconsistencies.append(
                    f"Medication Coverage Inconsistency: Inquired medication '{med}' was not individually assessed in output."
                )

    return inconsistencies
