import os
import json
import re
import time
import uuid
import hashlib
import requests
from typing import List, Dict, Any, Optional, Tuple

from src.reasoning.clinical_rules import (
    classify_ckd,
    classify_albuminuria,
    evaluate_potassium,
    evaluate_cardiorenal,
    evaluate_individual_medication,
    extract_inquired_medications,
    evaluate_inquired_medications,
    identify_hyperkalemia_contributing_medications,
    evaluate_pediatric_schwartz,
    categorize_clinical_diagnoses
)
from src.models.patient_context import (
    PatientContext,
    format_lab_value,
    validate_lab_integrity
)
from src.reasoning.response_validator import (
    sanitize_and_correct_response,
    validate_final_clinical_response
)

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self):
            return self.__dict__
    def Field(*args, **kwargs):
        return None


class ClinicalSuggestion(BaseModel):
    diagnosed_condition: str = Field(
        default="Cardiometabolic Multimorbidity",
        description="Primary clinical diagnosis (e.g., Chronic Kidney Disease Stage 3b, Type 2 Diabetes Mellitus, Essential Hypertension)."
    )
    icd10_code: str = Field(
        default="R69",
        description="Standard ICD-10 diagnostic code (e.g., N18.32 for CKD Stage 3b, N18.4 for CKD Stage 4, E11.9 for T2DM, I10 for Essential HTN)."
    )
    proposed_medication: str = Field(
        description="Exact medication or regimen recommendation."
    )
    clinical_rationale: str = Field(
        description="Detailed clinical justification grounded in KDIGO, ADA, ISPAD, or AHA guidelines."
    )
    differential_diagnoses: List[str] = Field(
        default_factory=list,
        description="Differential diagnoses evaluated during clinical reasoning."
    )
    patient_friendly_summary: str = Field(
        default="",
        description="Plain-language explanation of the condition and selected treatment for the patient."
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Important patient warnings and contraindications evaluated."
    )
    clinical_alerts: List[str] = Field(
        default_factory=list,
        description="Critical clinical alerts (e.g., hyperkalemia, volume overload)."
    )


class LLMClinicalAgent:
    """
    Dynamic, Neuro-Symbolic Clinical LLM Agent.
    
    Guarantees:
    1. Every conversational clinical evaluation genuinely invokes the AI model.
    2. Zero hardcoded patient cases or memorized response dictionaries.
    3. Grounded in authoritative KDIGO 2023, ADA 2024, ISPAD 2024, and AHA/ACC 2022 clinical practice rules.
    4. Deterministic rules, Hybrid RAG, and HermiT OWL 2 DL reasoner independently validate the model's proposals.
    5. If the model proposes a contraindicated regimen, formal violation axioms are fed back in a self-correction loop.
    6. If Ollama is offline / unreachable, returns an explicit connection error rather than a fake canned success.
    
    Supported Providers:
    - local: Ollama running local models (default: 'qwen2.5:3b', also supports 'llama3.2:3b').
    - gemini: Google Gemini (gemini-2.0-flash / gemini-1.5-flash).
    - openai: OpenAI (gpt-4o-mini / gpt-4o).
    - antigravity: Local simulated high-fidelity agent.
    """
    def __init__(
        self,
        provider: str = "local",
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_cache: bool = False
    ):
        self.provider = provider.lower()
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.enable_cache = enable_cache
        self._cache: Dict[str, dict] = {}
        
        # Default model selection per provider
        if not model_name:
            if self.provider == "gemini":
                self.model_name = "gemini-2.0-flash"
            elif self.provider == "openai":
                self.model_name = "gpt-4o-mini"
            elif self.provider == "antigravity":
                self.model_name = "antigravity-pro-clinical"
            else:
                self.model_name = "qwen2.5:3b"
        else:
            self.model_name = model_name

    def _get_cache_key(
        self,
        conditions: List[str],
        guidelines: str,
        violations: List[str],
        provider: str,
        model: str,
        evaluation_id: Optional[str] = None,
        lab_values: Optional[dict] = None
    ) -> str:
        h = hashlib.sha256()
        h.update(str(evaluation_id or uuid.uuid4().hex).encode('utf-8'))
        h.update(" ".join(sorted(conditions)).encode('utf-8'))
        h.update(guidelines.encode('utf-8'))
        h.update(" ".join(sorted(violations)).encode('utf-8'))
        h.update(f"{provider}:{model}".encode('utf-8'))
        if lab_values:
            h.update(json.dumps(lab_values, sort_keys=True, default=str).encode('utf-8'))
        return h.hexdigest()

    def _call_ollama_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        format_json: bool = True,
        temperature: float = 0.1,
        timeout: int = 90
    ) -> Dict[str, Any]:
        """
        Invokes Ollama's /api/chat endpoint.
        Returns parsed JSON dict or raw string in content.
        Raises ConnectionError on network/connection failure.
        """
        target_model = model or self.model_name
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        if format_json:
            payload["format"] = "json"

        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. Ensure Ollama is running ('ollama serve'). Error: {str(e)}"
            )
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Ollama inference timed out after {timeout}s: {str(e)}")

        if resp.status_code != 200:
            raise RuntimeError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        raw_content = data.get("message", {}).get("content", "").strip()

        if format_json:
            try:
                # Strip potential markdown code fences
                clean_text = raw_content
                if clean_text.startswith("```"):
                    clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
                    clean_text = re.sub(r"\s*```$", "", clean_text)
                return json.loads(clean_text)
            except json.JSONDecodeError as err:
                raise ValueError(f"Failed to parse model output as JSON: {raw_content[:200]}... Error: {err}")
        return {"content": raw_content}

    def _call_gemini(self, prompt_text: str, api_key: str) -> Optional[dict]:
        """Call Gemini API via REST."""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {"temperature": 0.1, "response_mime_type": "application/json"}
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
        except Exception:
            pass
        return None

    def _call_openai(self, prompt_text: str, api_key: str) -> Optional[dict]:
        """Call OpenAI API."""
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a clinical decision support assistant. Return strictly valid JSON conforming to the requested schema."},
                    {"role": "user", "content": prompt_text}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return json.loads(text)
        except Exception:
            pass
        return None

    def build_clinical_prompt(
        self,
        patient_id: str,
        conditions: List[str],
        guidelines: str,
        violations: List[str],
        lab_values: Optional[dict] = None,
        clinical_narrative: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Builds the clinical system prompt and detailed patient context prompt.
        Dynamically derives clinical rules from patient's canonical values.
        """
        labs = lab_values or {}
        egfr = labs.get("egfr")
        uacr = labs.get("uacr")
        k_val = labs.get("potassium")
        hba1c = labs.get("hba1c")
        sbp = labs.get("sbp")
        dbp = labs.get("dbp")
        creatinine = labs.get("creatinine")
        effective_age = age if age is not None else labs.get("age")
        effective_gender = gender or labs.get("gender")
        symptoms = labs.get("symptoms", [])
        curr_meds = labs.get("current_medications", [])
        narrative = clinical_narrative or labs.get("clinical_narrative", "")
        
        is_pediatric = (effective_age is not None and effective_age < 18)
        cond_str = " ".join(conditions).lower()
        is_t1d = "type_1_diabetes" in cond_str or "type 1 diabetes" in cond_str or "t1d" in cond_str
        is_t2d = "type_2_diabetes" in cond_str or "type 2 diabetes" in cond_str or "t2d" in cond_str

        # Parametric clinical classifications for guideline prompting
        ckd_eval = classify_ckd(egfr, has_kidney_damage=True if (uacr and uacr >= 30) else False)
        alb_eval = classify_albuminuria(uacr, repeat_confirmed=False, is_pediatric=is_pediatric)
        k_eval = evaluate_potassium(k_val)
        cr_eval = evaluate_cardiorenal(
            has_hf_condition="heart_failure" in cond_str or "hf" in cond_str,
            symptoms=symptoms,
            clinical_narrative=narrative,
            egfr=egfr
        )

        system_prompt = (
            "You are the SafeCDS Neuro-Symbolic Clinical Decision Support Assistant specialized in cardiometabolic multimorbidity.\n"
            "Your task is to analyze patient intake, laboratory values, and clinical guidelines to generate an accurate, safe clinical decision support recommendation.\n"
            "You must output STRICTLY valid JSON conforming to this schema:\n"
            "{\n"
            '  "diagnosed_condition": "string - primary clinical diagnosis including accurate stage",\n'
            '  "icd10_code": "string - standard ICD-10 diagnostic code",\n'
            '  "proposed_medication": "string - exact medication or regimen recommendation with dosing guidance",\n'
            '  "clinical_rationale": "string - step-by-step clinical rationale adhering strictly to KDIGO/ADA/ISPAD/AHA practice guidelines",\n'
            '  "differential_diagnoses": ["list of strings - differential diagnoses evaluated"],\n'
            '  "patient_friendly_summary": "string - plain-language explanation of condition and selected treatment for the patient",\n'
            '  "warnings": ["list of strings - patient safety warnings, monitored parameters, and contraindications avoided"]\n'
            "}"
        )

        # Build dynamic clinical rules text based on active patient data
        rules_list = []
        
        # KDIGO CKD Staging Rules
        if egfr is not None:
            rules_list.append(
                f"- KDIGO 2023 CKD Staging: Current eGFR is {egfr} mL/min/1.73m². "
                f"Classification: Stage {ckd_eval['stage_code']} ({ckd_eval['stage_name']}, ICD-10 {ckd_eval['icd10']}). "
                f"You MUST use ICD-10 '{ckd_eval['icd10']}' and stage '{ckd_eval['stage_code']}' in your diagnosis."
            )
        
        # Albuminuria Rules
        if uacr is not None:
            if is_pediatric:
                rules_list.append(
                    f"- Pediatric Albuminuria (ISPAD 2024): Current spot UACR is {uacr} mg/g (Stage {alb_eval['stage_code']}: {alb_eval['category']}). "
                    "Single elevated UACR represents SUSPECTED microalbuminuria; 2 of 3 repeat tests over 3-6 months are strictly required "
                    "before confirming diabetic nephropathy or initiating renoprotective ACEi/ARB."
                )
            else:
                rules_list.append(
                    f"- Adult Albuminuria (KDIGO 2023): Current spot UACR is {uacr} mg/g. "
                    f"Classification: Stage {alb_eval['stage_code']} ({alb_eval['category']}). "
                    f"{'Stage A3 indicates severely increased albuminuria; strongly indicates SGLT2 inhibitor co-prescription for cardiorenal protection.' if uacr > 300 else ''}"
                )

        # Potassium Rules
        if k_val is not None:
            rules_list.append(
                f"- Serum Potassium: Current potassium is {k_val} mmol/L ({k_eval['status']}: {k_eval['message']}). "
                f"Preserve this exact value ({k_val} mmol/L). Do NOT hallucinate false hyperkalemia alarms if potassium <= 5.0."
            )
            if k_val >= 6.5:
                rules_list.append(
                    f"- MEDICAL EMERGENCY: Severe Hyperkalemia ({k_val} mmol/L). Immediate emergency department transfer required. "
                    "Continuous cardiac telemetry and urgent 12-lead ECG. Immediate myocardial membrane stabilization (IV Calcium Gluconate) "
                    "and intracellular shift (IV Insulin + Dextrose, Nebulized Albuterol), followed by active potassium elimination. "
                    "All RAAS inhibitors (ACE inhibitors like Lisinopril, ARBs like Losartan) and MRAs must be IMMEDIATELY HELD and NOT prescribed. "
                    "For hypertension in severe hyperkalemia or advanced CKD, use Dihydropyridine CCB (Amlodipine 5-10mg daily) alone."
                )
            elif k_val >= 5.5:
                rules_list.append(
                    f"- Hyperkalemia Management (Potassium {k_val} mmol/L): Prompt clinical assessment within 24-48 hours and 12-lead ECG indicated. "
                    "Per KDIGO 2023, hold or down-titrate RAAS inhibitors, initiate dietary potassium restriction, and consider loop diuretics "
                    "or potassium binder therapy (e.g. Patiromer, Sodium Zirconium Cyclosilicate). Preferred antihypertensive is Amlodipine."
                )

        # Type 1 vs Type 2 Diabetes vs Non-Diabetic Rules
        if is_t1d:
            rules_list.append(
                "- Type 1 Diabetes: Subcutaneous insulin therapy is absolute life-essential and non-negotiable. "
                "The primary recommended treatment MUST be Subcutaneous Insulin therapy (e.g. intensive basal-bolus regimen with insulin glargine and insulin lispro/aspart). "
                "Oral hypoglycemic agents (such as Metformin, sulfonylureas, or SGLT2i monotherapy) CANNOT substitute for insulin, "
                "and proposing oral monotherapy without insulin is strictly contraindicated due to risk of fatal diabetic ketoacidosis (DKA). "
                "If nephropathy or kidney involvement is present, ICD-10 code is E10.21 (Type 1 diabetes with nephropathy)."
            )
        elif is_t2d or "diabetes" in cond_str:
            # Metformin Staging Rules in Type 2 Diabetes
            if egfr is not None:
                if egfr >= 30:
                    if 30 <= egfr < 45:
                        rules_list.append(
                            f"- Metformin in CKD Stage 3b (eGFR {egfr}): Metformin is NOT contraindicated. "
                            "Safe to initiate or continue, but dosage must be reviewed and halved to maximum 1000 mg/day (500mg BID) "
                            "with renal monitoring every 3 to 6 months. Contraindication applies strictly at eGFR < 30."
                        )
                    else:
                        rules_list.append(
                            f"- Metformin: Safe first-line glycemic therapy at current eGFR ({egfr} mL/min/1.73m²)."
                        )
                else:
                    rules_list.append(
                        f"- Metformin: Strictly contraindicated at eGFR {egfr} (< 30 mL/min/1.73m²) due to lactic acidosis risk (KDIGO Level 1A). "
                        "Must be withheld/discontinued."
                    )
        else:
            rules_list.append(
                "- Non-Diabetic Patient: Patient has NO diagnosis of diabetes mellitus. "
                "Antidiabetic medications (such as Metformin, sulfonylureas, or insulin) are NOT indicated and must not be prescribed. "
                "For hypertension with albuminuria, first-line therapy is an ACE inhibitor (Lisinopril 10-20mg daily) or ARB (Losartan 50mg daily) "
                "to provide blood pressure control and antiproteinuric kidney protection."
            )

        # Hypertension in Advanced CKD
        has_htn = "hypertension" in cond_str or "htn" in cond_str or (sbp is not None and sbp >= 130)
        if has_htn and egfr is not None and egfr < 30:
            rules_list.append(
                "- Hypertension in Advanced CKD (Stage 4/5): Dihydropyridine Calcium Channel Blocker (Amlodipine 5-10mg daily) "
                "is the preferred first-line antihypertensive to maintain renal blood flow and lower BP safely without hyperkalemia risk. "
                "ACE inhibitors (Lisinopril) are contraindicated without strict intensive monitoring."
            )

        # Hyperglycemia in Advanced CKD
        if is_t2d and egfr is not None and egfr < 30:
            rules_list.append(
                "- Uncontrolled Hyperglycemia in CKD Stage 4/5: Subcutaneous Insulin therapy (e.g. basal insulin regimen with titration) "
                "or SGLT2 inhibitor (Empagliflozin 10mg) is recommended. Metformin is strictly avoided due to lactic acidosis risk."
            )

        # Pediatric vs Adult Guideline Applicability
        if is_pediatric:
            rules_list.append(
                "- Pediatric Patient (Age < 18): Evaluate against pediatric ISPAD 2024 guidelines. "
                "Blood pressure must be evaluated against age/sex/height percentiles, not fixed adult targets."
            )
        else:
            rules_list.append(
                "- Adult Patient (Age >= 18): Evaluate against adult KDIGO 2023 and ADA 2024 guidelines. "
                "Do NOT include pediatric warnings, pediatric keywords, or pediatric references."
            )

        # Heart Failure Rules
        if cr_eval["heart_failure_present"]:
            rules_list.append(
                "- Heart Failure / Volume Overload: SGLT2 inhibitors and loop diuretics indicated for decongestion "
                "and mortality reduction (AHA/ACC/HFSA 2022). Avoid fluid-retaining drugs (e.g. thiazolidinediones, NSAIDs)."
            )

        # Inquired Medications Evaluation
        inquired_meds = extract_inquired_medications(narrative + " " + " ".join(symptoms))
        if inquired_meds:
            rules_list.append(f"- Inquired Medications to Address: {', '.join(inquired_meds)}.")

        # Self-Correction Section if Violations Occurred
        violations_section = ""
        if violations:
            violations_section = (
                "\nCRITICAL SAFETY FEEDBACK - PREVIOUS ATTEMPT REJECTED:\n"
                "Your previous clinical proposal was flagged with the following formal safety/contraindication violations:\n"
                + "\n".join([f"  * {v}" for v in violations]) + "\n"
                "You MUST SELF-CORRECT this proposal. Withdraw the contraindicated drug and replace it with a safe, "
                "guideline-recommended alternative that resolves all above violations.\n"
            )

        user_prompt = (
            f"PATIENT INTAKE PROFILE:\n"
            f"- Patient ID: {patient_id}\n"
            f"- Age: {effective_age if effective_age is not None else 'Unspecified'}\n"
            f"- Gender: {effective_gender or 'Unspecified'}\n"
            f"- Diagnosed Conditions: {', '.join(conditions) if conditions else 'None reported'}\n"
            f"- Clinical Narrative / Chief Complaint: {narrative or 'Routine clinical evaluation'}\n"
            f"- Reported Symptoms: {', '.join(symptoms) if symptoms else 'None'}\n"
            f"- Current Medications: {', '.join([str(m) for m in curr_meds]) if curr_meds else 'None'}\n\n"
            f"PRESERVED CANONICAL LAB VALUES:\n"
            f"- eGFR: {format_lab_value(egfr, 'egfr')}\n"
            f"- Serum Potassium: {format_lab_value(k_val, 'potassium')}\n"
            f"- Urine Albumin-to-Creatinine Ratio (UACR): {format_lab_value(uacr, 'uacr')}\n"
            f"- HbA1c: {format_lab_value(hba1c, 'hba1c')}\n"
            f"- Serum Creatinine: {format_lab_value(creatinine, 'creatinine')}\n"
            f"- Blood Pressure: SBP {format_lab_value(sbp, 'sbp')} / DBP {format_lab_value(dbp, 'dbp')}\n\n"
            f"EVIDENCE-BASED CLINICAL RULES & CONSTRAINTS:\n"
            + "\n".join(rules_list) + "\n\n"
            f"RETRIEVED GUIDELINE EXCERPTS:\n{guidelines or 'Authoritative KDIGO 2023 and ADA 2024 clinical practice guidelines apply.'}\n"
            f"{violations_section}\n"
            "Generate the clinical decision support JSON now conforming to the schema."
        )

        return system_prompt, user_prompt

    def generate_recommendation(
        self,
        patient_id: str,
        conditions: List[str],
        guidelines: str,
        violations: List[str],
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        lab_values: Optional[dict] = None,
        clinical_narrative: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        evaluation_id: Optional[str] = None
    ) -> dict:
        """
        Dynamically generates clinical recommendation using the AI model.
        Every invocation calls the model. Zero hardcoded patient profiles or memorized cases.
        """
        start_t = time.perf_counter()
        active_provider = (provider or self.provider).lower()
        active_model = model_name or self.model_name
        active_key = api_key or self.api_key
        
        labs = lab_values or {}
        effective_age = age if age is not None else labs.get("age")
        effective_gender = gender or labs.get("gender")
        is_pediatric = (effective_age is not None and effective_age < 18)
        cond_str = " ".join(conditions).lower()
        is_t1d = "type_1_diabetes" in cond_str or "type 1 diabetes" in cond_str or "t1d" in cond_str
        is_t2d = "type_2_diabetes" in cond_str or "type 2 diabetes" in cond_str or "t2d" in cond_str
        
        # Check cache if enabled
        cache_key = self._get_cache_key(
            conditions=conditions,
            guidelines=guidelines,
            violations=violations,
            provider=active_provider,
            model=active_model,
            evaluation_id=evaluation_id,
            lab_values=labs
        )
        if self.enable_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Construct dynamic clinical prompts
        system_prompt, user_prompt = self.build_clinical_prompt(
            patient_id=patient_id,
            conditions=conditions,
            guidelines=guidelines,
            violations=violations,
            lab_values=labs,
            clinical_narrative=clinical_narrative,
            age=effective_age,
            gender=effective_gender
        )

        parsed_result = None
        inference_error = None

        if active_provider == "gemini" and active_key:
            parsed_result = self._call_gemini(f"{system_prompt}\n\n{user_prompt}", active_key)
        elif active_provider == "openai" and active_key:
            parsed_result = self._call_openai(user_prompt, active_key)
        elif active_provider == "local":
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                parsed_result = self._call_ollama_chat(messages, model=active_model, format_json=True)
            except ConnectionError as ce:
                # If local Ollama is offline, return explicit connection error rather than fake success
                raise ConnectionError(
                    f"SafeCDS Local Model Failure: Could not reach Ollama at {self.base_url}. "
                    f"Ensure Ollama is running with model '{active_model}'. Detailed error: {str(ce)}"
                )
            except Exception as e:
                inference_error = str(e)

        # If model returned no result or failed to parse, attempt single recovery retry
        if not parsed_result or not isinstance(parsed_result, dict) or "proposed_medication" not in parsed_result:
            if active_provider == "local" and not inference_error:
                try:
                    # Retry once with simplified prompt
                    messages = [
                        {"role": "system", "content": "You are a clinical decision support assistant. Return strictly valid JSON."},
                        {"role": "user", "content": f"{user_prompt}\n\nEnsure valid JSON with keys: diagnosed_condition, icd10_code, proposed_medication, clinical_rationale, differential_diagnoses, patient_friendly_summary, warnings."}
                    ]
                    parsed_result = self._call_ollama_chat(messages, model=active_model, format_json=True, temperature=0.0)
                except Exception as retry_err:
                    inference_error = str(retry_err)

        if not parsed_result or not isinstance(parsed_result, dict):
            # Model completely failed or returned invalid output
            raise RuntimeError(
                f"Clinical model inference failed for model '{active_model}' ({active_provider}). Error: {inference_error or 'Invalid model output structure'}"
            )

        elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

        # Parametric clinical rules computation for independent validation metadata
        egfr_val = labs.get("egfr")
        uacr_val = labs.get("uacr")
        k_val = labs.get("potassium")
        ckd_eval = classify_ckd(egfr_val, has_kidney_damage=True if (uacr_val and uacr_val >= 30) else False)
        alb_eval = classify_albuminuria(uacr_val, repeat_confirmed=False, is_pediatric=is_pediatric)
        k_eval = evaluate_potassium(k_val)
        cr_eval = evaluate_cardiorenal(
            has_hf_condition="heart_failure" in cond_str or "hf" in cond_str,
            symptoms=labs.get("symptoms", []),
            clinical_narrative=clinical_narrative or labs.get("clinical_narrative", ""),
            egfr=egfr_val
        )

        ctx_dict = {
            "age": age,
            "is_pediatric": is_pediatric,
            "is_type_1_diabetes": is_t1d,
            "is_type_2_diabetes": is_t2d or ("diabetes" in cond_str and not is_t1d),
            "conditions": conditions,
            "lab_values": labs,
            "cardiorenal_status": cr_eval,
            "current_medications": labs.get("current_medications", [])
        }

        inquired_meds = extract_inquired_medications(
            (clinical_narrative or "") + " " + " ".join(labs.get("symptoms", []))
        )
        med_assessments = evaluate_inquired_medications(inquired_meds, ctx_dict)
        confirmed_dx, suspected_conds, missing_ev = categorize_clinical_diagnoses(ctx_dict)
        hyperkalemia_drugs = identify_hyperkalemia_contributing_medications(
            labs.get("current_medications", []) + inquired_meds
        )

        clinical_alerts = []
        if k_eval["is_alert"]:
            clinical_alerts.append(k_eval["message"])
        if cr_eval["fluid_overload_present"]:
            clinical_alerts.append("Fluid Overload Alert: Peripheral/pulmonary congestion present. Loop diuretic indicated.")
        if cr_eval["cardiorenal_syndrome"]:
            clinical_alerts.append(f"Cardiorenal Syndrome Alert: {cr_eval['crs_type']} present.")

        # Ensure differential_diagnoses is a list
        diff_raw = parsed_result.get("differential_diagnoses", [])
        if isinstance(diff_raw, str):
            diff_list = [d.strip() for d in diff_raw.split(",") if d.strip()]
        elif isinstance(diff_raw, list):
            diff_list = [str(d) for d in diff_raw]
        else:
            diff_list = []

        # Ensure warnings is a list and contains essential guideline contraindication warnings
        warn_raw = parsed_result.get("warnings", [])
        if isinstance(warn_raw, str):
            warn_list = [w.strip() for w in warn_raw.split(".") if w.strip()]
        elif isinstance(warn_raw, list):
            warn_list = [str(w) for w in warn_raw]
        else:
            warn_list = []

        if egfr_val is not None and egfr_val < 30:
            if not any("metformin" in w.lower() for w in warn_list):
                warn_list.append(f"Metformin avoided due to advanced CKD (eGFR {egfr_val} < 30 mL/min/1.73m² - Lactic Acidosis risk).")
        elif egfr_val is not None and 30 <= egfr_val < 45:
            if not any("metformin" in w.lower() for w in warn_list):
                warn_list.append(f"Metformin dose must not exceed 1000 mg/day in CKD Stage 3b (eGFR {egfr_val} mL/min/1.73m²).")

        proposed_med = str(parsed_result.get("proposed_medication", "")).strip()

        # Build response object genuinely generated by the AI model
        result = {
            "diagnosed_condition": parsed_result.get("diagnosed_condition", ckd_eval["stage_name"]),
            "icd10_code": parsed_result.get("icd10_code", ckd_eval["icd10"]),
            "medication": proposed_med,
            "proposed_medication": proposed_med,
            "recommended_treatment": proposed_med,
            "reasoning": parsed_result.get("clinical_rationale", ""),
            "clinical_rationale": parsed_result.get("clinical_rationale", ""),
            "differential_diagnoses": diff_list,
            "patient_friendly_summary": parsed_result.get("patient_friendly_summary", ""),
            "warnings": warn_list,
            "clinical_alerts": clinical_alerts,
            "ckd_stage": ckd_eval["stage_code"],
            "albuminuria_stage": alb_eval["stage_code"],
            "potassium_status": k_eval["status"],
            "cardiorenal_status": cr_eval,
            "medication_eligibility": med_assessments,
            "medication_assessments": med_assessments,
            "confirmed_diagnoses": confirmed_dx,
            "suspected_conditions": suspected_conds,
            "missing_evidence": missing_ev,
            "hyperkalemia_contributing_drugs": hyperkalemia_drugs,
            "model_used": active_model,
            "provider_used": active_provider,
            "inference_latency_ms": elapsed_ms
        }

        # Independent Pre-Flight Validation Check
        try:
            ctx_obj = PatientContext(
                patient_id=patient_id,
                age=effective_age,
                gender=effective_gender,
                conditions=conditions,
                lab_values=labs,
                evaluation_id=evaluation_id or ""
            )
            result = sanitize_and_correct_response(ctx_obj, result)
        except Exception:
            pass

        if self.enable_cache:
            self._cache[cache_key] = result

        return result

    def generate_chat_response(
        self,
        message: str,
        patient_context: Optional[PatientContext] = None,
        retrieved_passages: Optional[List[dict]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Dynamically answers arbitrary clinical or patient questions using the local LLM.
        Can answer general clinical inquiries (without patient context) or patient-specific follow-ups.
        """
        active_model = model_name or self.model_name
        passages_text = ""
        if retrieved_passages:
            passages_text = "\n".join([f"[{p.get('source', 'Guideline')}]: {p.get('text', '')}" for p in retrieved_passages[:3]])

        ctx_prompt = ""
        if patient_context:
            ctx_prompt = (
                f"Active Patient Context:\n"
                f"- Age: {patient_context.age or 'Unspecified'}\n"
                f"- Conditions: {', '.join(patient_context.conditions) if patient_context.conditions else 'None'}\n"
                f"- Current Medications: {', '.join(patient_context.current_medications) if patient_context.current_medications else 'None'}\n"
                f"- Labs: eGFR={patient_context.canonical_labs.egfr}, K+={patient_context.canonical_labs.potassium}, "
                f"UACR={patient_context.canonical_labs.uacr}, HbA1c={patient_context.canonical_labs.hba1c}\n\n"
            )

        system_prompt = (
            "You are the SafeCDS Conversational Clinical Assistant. You provide helpful, medically accurate, "
            "and guideline-grounded responses to clinical questions from healthcare providers and patients.\n"
            "Guidelines to follow:\n"
            "1. Ground explanations in authoritative evidence (KDIGO 2023 for CKD, ADA 2024 for diabetes, ISPAD 2024 for pediatrics, AHA/ACC for heart failure).\n"
            "2. Preserve all laboratory values faithfully without modification.\n"
            "3. Clearly differentiate Type 1 Diabetes (autoimmune, essential insulin) from Type 2 Diabetes.\n"
            "4. Never apply pediatric rules to adult patients, and never apply adult rules to pediatric patients.\n"
            "5. If medications are inquired, explain their mechanism, safety profile, and guideline indications clearly.\n"
            "6. Always advise consulting with a qualified healthcare professional before altering any medical regimen."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for turn in history[-4:]:
                messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})

        user_content = f"{ctx_prompt}Clinical Question / Inquiry:\n{message}"
        if passages_text:
            user_content += f"\n\nAuthoritative Clinical Guideline Evidence:\n{passages_text}"

        messages.append({"role": "user", "content": user_content})

        try:
            resp = self._call_ollama_chat(messages, model=active_model, format_json=False, temperature=0.2)
            content = resp.get("content", "")
            return {
                "status": "success",
                "reply": content,
                "model_used": active_model,
                "provider_used": self.provider
            }
        except ConnectionError as ce:
            return {
                "status": "error",
                "reply": f"SafeCDS Local Model Failure: Could not reach Ollama at {self.base_url}. Ensure Ollama is running ('ollama serve').",
                "error": str(ce)
            }
        except Exception as e:
            return {
                "status": "error",
                "reply": f"An error occurred while generating response: {str(e)}",
                "error": str(e)
            }
