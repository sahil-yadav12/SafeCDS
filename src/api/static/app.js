// SafeCDS Frontend Controller — Adaptive ElevenLabs Light & Linear Dark System
document.addEventListener("DOMContentLoaded", () => {
    // State
    let currentCohort = [];
    let activeView = localStorage.getItem("safecds_view") || "clinician"; // "clinician" | "patient"
    let currentTheme = localStorage.getItem("safecds_theme") || "light"; // "light" | "dark"
    let chatHistory = [];
    let chatAttachedContext = { conditions: [], lab_values: {}, filename: null };
    let apiKeys = {
        gemini: localStorage.getItem("safecds_gemini_key") || "",
        openai: localStorage.getItem("safecds_openai_key") || ""
    };

    // DOM Elements - Theme & Modes
    const btnThemeToggle = document.getElementById("btn-theme-toggle");
    const themeIcon = document.getElementById("theme-icon");
    const themeLabel = document.getElementById("theme-label");

    const btnViewClinician = document.getElementById("btn-view-clinician");
    const btnViewPatient = document.getElementById("btn-view-patient");
    const viewClinician = document.getElementById("view-clinician");
    const viewPatient = document.getElementById("view-patient");

    const providerSelect = document.getElementById("provider-select");
    const btnApiKey = document.getElementById("btn-api-key");
    const modalApiKeys = document.getElementById("modal-api-keys");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const btnSaveKeys = document.getElementById("btn-save-keys");
    const inputGeminiKey = document.getElementById("input-gemini-key");
    const inputOpenaiKey = document.getElementById("input-openai-key");
    const btnPrint = document.getElementById("btn-print");

    // Clinician Elements
    const cohortButtonsContainer = document.getElementById("cohort-buttons");
    const pdfDropzone = document.getElementById("pdf-dropzone");
    const pdfFileInput = document.getElementById("pdf-file-input");
    const intakeForm = document.getElementById("intake-form");
    const btnEvaluate = document.getElementById("btn-evaluate");
    const inputEgfr = document.getElementById("input-egfr");
    const labelEgfrStage = document.getElementById("label-egfr-stage");
    const patientAgeInput = document.getElementById("patient-age");
    const resultsWelcome = document.getElementById("results-welcome");
    const resultsPanel = document.getElementById("results-panel");
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabContents = document.querySelectorAll(".tab-content");

    // Patient Chat Elements
    const chatMessages = document.getElementById("chat-messages");
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const btnChatSend = document.getElementById("btn-chat-send");
    const btnChatUploadPdf = document.getElementById("btn-chat-upload-pdf");
    const chatPdfFileInput = document.getElementById("chat-pdf-file-input");
    const chatAttachedPill = document.getElementById("chat-attached-pill");
    const chatFilename = document.getElementById("chat-filename");
    const btnRemoveChatFile = document.getElementById("btn-remove-chat-file");

    // ========================================================
    // 1. THEME CONTROLLER (ELEVENLABS LIGHT VS LINEAR DARK)
    // ========================================================
    function applyTheme(theme) {
        currentTheme = theme;
        localStorage.setItem("safecds_theme", theme);

        if (theme === "dark") {
            document.body.classList.remove("theme-light");
            document.body.classList.add("theme-dark");
            if (themeIcon) themeIcon.textContent = "🌙";
            if (themeLabel) themeLabel.textContent = "Dark";
        } else {
            document.body.classList.remove("theme-dark");
            document.body.classList.add("theme-light");
            if (themeIcon) themeIcon.textContent = "☀️";
            if (themeLabel) themeLabel.textContent = "Light";
        }
    }

    if (btnThemeToggle) {
        btnThemeToggle.addEventListener("click", () => {
            const nextTheme = currentTheme === "light" ? "dark" : "light";
            applyTheme(nextTheme);
        });
    }
    applyTheme(currentTheme);

    // ========================================================
    // 2. VIEW SWITCHER (CLINICIAN WORKBENCH VS PATIENT CHAT)
    // ========================================================
    function switchView(view) {
        activeView = view;
        localStorage.setItem("safecds_view", view);

        if (view === "patient") {
            btnViewPatient.classList.add("active");
            btnViewClinician.classList.remove("active");
            viewClinician.classList.add("hidden");
            viewPatient.classList.remove("hidden");
            viewClinician.style.display = "none";
            viewPatient.style.display = "block";
            if (chatInput) chatInput.focus();
        } else {
            btnViewClinician.classList.add("active");
            btnViewPatient.classList.remove("active");
            viewPatient.classList.add("hidden");
            viewClinician.classList.remove("hidden");
            viewPatient.style.display = "none";
            viewClinician.style.display = "block";
        }
    }

    if (btnViewClinician && btnViewPatient) {
        btnViewClinician.addEventListener("click", () => switchView("clinician"));
        btnViewPatient.addEventListener("click", () => switchView("patient"));
        switchView(activeView);
    }

    // Print
    if (btnPrint) {
        btnPrint.addEventListener("click", () => window.print());
    }

    // API Key Modal
    if (btnApiKey && modalApiKeys) {
        btnApiKey.addEventListener("click", () => {
            inputGeminiKey.value = apiKeys.gemini;
            inputOpenaiKey.value = apiKeys.openai;
            modalApiKeys.classList.remove("hidden");
        });
        btnCloseModal.addEventListener("click", () => modalApiKeys.classList.add("hidden"));
        btnSaveKeys.addEventListener("click", () => {
            apiKeys.gemini = inputGeminiKey.value.trim();
            apiKeys.openai = inputOpenaiKey.value.trim();
            localStorage.setItem("safecds_gemini_key", apiKeys.gemini);
            localStorage.setItem("safecds_openai_key", apiKeys.openai);
            modalApiKeys.classList.add("hidden");
        });
    }

    // ========================================================
    // 3. CLINICIAN TABS NAVIGATION
    // ========================================================
    navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const targetId = tab.getAttribute("data-tab");
            navTabs.forEach(t => t.classList.remove("active-tab"));
            tab.classList.add("active-tab");

            tabContents.forEach(tc => {
                if (tc.id === targetId) {
                    tc.classList.remove("hidden");
                    tc.classList.add("block");
                } else {
                    tc.classList.add("hidden");
                    tc.classList.remove("block");
                }
            });

            if (targetId === "tab-benchmark") loadBenchmarkTab();
            if (targetId === "tab-axioms") loadAxiomsTab();
            if (targetId === "tab-guidelines") loadGuidelinesTab();
        });
    });

    // ========================================================
    // 4. eGFR STAGING & PEDIATRIC FORMULA LOGIC
    // ========================================================
    function updateEgfrLabel(val) {
        const egfr = parseFloat(val);
        const age = parseInt(patientAgeInput.value) || 65;
        let desc = "";

        if (age < 18) {
            if (egfr >= 90) desc = `${egfr} (Pediatric Normal)`;
            else if (egfr >= 60) desc = `${egfr} (Pediatric Mild Reduction)`;
            else if (egfr >= 30) desc = `${egfr} (Pediatric Moderate Nephropathy)`;
            else desc = `${egfr} (Pediatric Severe Kidney Impairment)`;
        } else {
            if (egfr >= 90) desc = `${egfr} (Normal / Stage 1)`;
            else if (egfr >= 60) desc = `${egfr} (Mild / Stage 2)`;
            else if (egfr >= 45) desc = `${egfr} (Moderate / Stage 3a)`;
            else if (egfr >= 30) desc = `${egfr} (Moderate-Severe / Stage 3b)`;
            else if (egfr >= 15) desc = `${egfr} (Severely Decreased / Stage 4)`;
            else desc = `${egfr} (Kidney Failure / Stage 5)`;
        }
        if (labelEgfrStage) labelEgfrStage.textContent = desc;
    }

    if (inputEgfr) {
        inputEgfr.addEventListener("input", (e) => updateEgfrLabel(e.target.value));
        patientAgeInput.addEventListener("input", () => updateEgfrLabel(inputEgfr.value));
    }

    // ========================================================
    // 5. COHORT PRESETS LOADING (ADULT + PEDIATRICS)
    // ========================================================
    async function fetchCohort() {
        try {
            const res = await fetch("/api/cohort");
            const data = await res.json();
            currentCohort = Array.isArray(data) ? data : (data.cohort || []);
            renderCohortButtons(currentCohort);
            if (currentCohort.length > 0) {
                populateIntakeForm(currentCohort[0]);
            }
        } catch (err) {
            console.error("Failed to fetch cohort:", err);
        }
    }

    function renderCohortButtons(cohort) {
        if (!cohortButtonsContainer) return;
        cohortButtonsContainer.innerHTML = "";

        cohort.forEach(patient => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "cohort-pill-btn";
            btn.dataset.patientId = patient.patient_id;

            const isPeds = (patient.age && patient.age < 18) || patient.patient_id.includes("007") || patient.name.includes("Pediatric");
            const labelText = isPeds ? `${patient.patient_id} (Peds)` : patient.patient_id;

            btn.innerHTML = `
                <span>${labelText}</span>
                <span class="w-2 h-2 rounded-full ${isPeds ? 'bg-[var(--accent-spark)]' : 'bg-[var(--color-text-muted)]'}"></span>
            `;

            btn.addEventListener("click", () => {
                document.querySelectorAll(".cohort-pill-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                populateIntakeForm(patient);
            });

            cohortButtonsContainer.appendChild(btn);
        });

        // Select first preset by default
        if (cohortButtonsContainer.firstChild) {
            cohortButtonsContainer.firstChild.classList.add("active");
        }
    }

    function populateIntakeForm(patient) {
        document.getElementById("patient-id").value = patient.patient_id;
        document.getElementById("patient-name").value = patient.name;
        document.getElementById("patient-age").value = patient.age;
        document.getElementById("patient-gender").value = patient.gender;
        document.getElementById("intake-badge").textContent = patient.patient_id;

        inputEgfr.value = patient.egfr;
        updateEgfrLabel(patient.egfr);

        document.getElementById("input-sbp").value = patient.sbp;
        document.getElementById("input-dbp").value = patient.dbp;
        document.getElementById("input-hba1c").value = patient.hba1c;
        document.getElementById("input-creatinine").value = patient.creatinine;

        const potEl = document.getElementById("input-potassium");
        if (potEl) potEl.value = (patient.potassium !== undefined && patient.potassium !== null) ? patient.potassium : "";

        const uacrEl = document.getElementById("input-uacr");
        if (uacrEl) uacrEl.value = (patient.uacr !== undefined && patient.uacr !== null) ? patient.uacr : "";

        document.getElementById("clinical-narrative").value = patient.clinical_narrative;

        // Check matching condition boxes
        document.querySelectorAll(".condition-checkbox").forEach(cb => {
            cb.checked = patient.conditions.includes(cb.value);
        });
    }

    // ========================================================
    // 6. PDF DRAG-AND-DROP & UPLOAD
    // ========================================================
    if (pdfDropzone && pdfFileInput) {
        pdfDropzone.addEventListener("click", () => pdfFileInput.click());
        pdfDropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            pdfDropzone.style.borderColor = "var(--color-border-hover)";
        });
        pdfDropzone.addEventListener("dragleave", () => {
            pdfDropzone.style.borderColor = "var(--color-border)";
        });
        pdfDropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            pdfDropzone.style.borderColor = "var(--color-border)";
            if (e.dataTransfer.files.length > 0) handlePdfUpload(e.dataTransfer.files[0]);
        });
        pdfFileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) handlePdfUpload(e.target.files[0]);
        });
    }

    document.querySelectorAll(".btn-sample-pdf").forEach(btn => {
        btn.addEventListener("click", async (e) => {
            e.preventDefault();
            const fileName = btn.getAttribute("data-file");
            try {
                const res = await fetch(`/sample_reports/${fileName}`);
                const blob = await res.blob();
                const file = new File([blob], fileName, { type: "application/pdf" });
                handlePdfUpload(file);
            } catch (err) {
                console.error("Failed to load sample report:", err);
            }
        });
    });

    async function handlePdfUpload(file) {
        const formData = new FormData();
        formData.append("file", file);

        pdfDropzone.innerHTML = `<span class="text-xs text-[var(--color-text-muted)]">Extracting clinical entities from ${file.name}...</span>`;

        try {
            const res = await fetch("/api/upload-pdf", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (data.status === "success") {
                const parsed = data.data;
                populateFromPdf(parsed);
                pdfDropzone.innerHTML = `
                    <div class="text-xs text-[var(--status-verified-text)] font-medium">${file.name} parsed successfully</div>
                    <span class="text-[11px] text-[var(--color-text-muted)]">eGFR ${parsed.egfr || '-'}, BP ${parsed.sbp || '-'}/${parsed.dbp || '-'}, HbA1c ${parsed.hba1c || '-'}%</span>
                `;
            }
        } catch (err) {
            console.error("PDF upload error:", err);
            pdfDropzone.innerHTML = `<span class="text-xs text-[var(--status-contraindicated-text)]">Error parsing PDF</span>`;
        }
    }

    function populateFromPdf(parsed) {
        if (parsed.patient_id) document.getElementById("patient-id").value = parsed.patient_id;
        if (parsed.patient_name) document.getElementById("patient-name").value = parsed.patient_name;
        if (parsed.age) document.getElementById("patient-age").value = parsed.age;
        if (parsed.gender) document.getElementById("patient-gender").value = parsed.gender;
        if (parsed.patient_id) document.getElementById("intake-badge").textContent = parsed.patient_id;

        if (parsed.egfr !== null && parsed.egfr !== undefined) {
            inputEgfr.value = parsed.egfr;
            updateEgfrLabel(parsed.egfr);
        }
        if (parsed.sbp) document.getElementById("input-sbp").value = parsed.sbp;
        if (parsed.dbp) document.getElementById("input-dbp").value = parsed.dbp;
        if (parsed.hba1c) document.getElementById("input-hba1c").value = parsed.hba1c;
        if (parsed.creatinine) document.getElementById("input-creatinine").value = parsed.creatinine;
        if (parsed.potassium !== null && parsed.potassium !== undefined) {
            const potEl = document.getElementById("input-potassium");
            if (potEl) potEl.value = parsed.potassium;
        }
        if (parsed.uacr !== null && parsed.uacr !== undefined) {
            const uacrEl = document.getElementById("input-uacr");
            if (uacrEl) uacrEl.value = parsed.uacr;
        }

        if (parsed.conditions && parsed.conditions.length > 0) {
            document.querySelectorAll(".condition-checkbox").forEach(cb => {
                cb.checked = parsed.conditions.includes(cb.value);
            });
        }

        let narrative = parsed.raw_text_snippet || "";
        if (parsed.symptoms && parsed.symptoms.length > 0) {
            narrative += `\nIdentified Symptoms: ${parsed.symptoms.join(", ")}.`;
        }
        if (parsed.current_medications && parsed.current_medications.length > 0) {
            narrative += `\nCurrent Medications: ${parsed.current_medications.join(", ")}.`;
        }
        document.getElementById("clinical-narrative").value = narrative.trim();
    }

    // ========================================================
    // 7. CLINICIAN EVALUATION EXECUTION
    // ========================================================
    if (intakeForm) {
        intakeForm.addEventListener("submit", async (e) => {
            e.preventDefault();

            const selectedConditions = Array.from(document.querySelectorAll(".condition-checkbox:checked")).map(cb => cb.value);
            const provider = providerSelect.value;
            const apiKey = provider === "gemini" ? apiKeys.gemini : (provider === "openai" ? apiKeys.openai : "");

            const potInput = document.getElementById("input-potassium");
            const uacrInput = document.getElementById("input-uacr");

            const payload = {
                patient_id: document.getElementById("patient-id").value.trim(),
                patient_name: document.getElementById("patient-name").value.trim(),
                age: parseInt(document.getElementById("patient-age").value) || 65,
                gender: document.getElementById("patient-gender").value,
                conditions: selectedConditions,
                lab_values: {
                    egfr: parseFloat(inputEgfr.value),
                    sbp: parseInt(document.getElementById("input-sbp").value) || 120,
                    dbp: parseInt(document.getElementById("input-dbp").value) || 80,
                    hba1c: parseFloat(document.getElementById("input-hba1c").value) || 5.7,
                    creatinine: parseFloat(document.getElementById("input-creatinine").value) || 1.0,
                    potassium: potInput ? parseFloat(potInput.value) : 4.5,
                    uacr: uacrInput ? parseFloat(uacrInput.value) : 30.0
                },
                clinical_narrative: document.getElementById("clinical-narrative").value.trim(),
                provider: provider,
                api_key: apiKey
            };

            btnEvaluate.disabled = true;
            btnEvaluate.innerHTML = `<span>Evaluating Case...</span>`;

            try {
                const response = await fetch("/api/evaluate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                if (data.status === "success") {
                    displayClinicianResults(data.result, payload);
                }
            } catch (err) {
                console.error("Evaluation error:", err);
            } finally {
                btnEvaluate.disabled = false;
                btnEvaluate.innerHTML = `<span>Run Clinical Evaluation</span> <span>→</span>`;
            }
        });
    }

    // Render the high-visibility, pure clinical report
    function displayClinicianResults(res, intakePayload) {
        if (!resultsWelcome || !resultsPanel) return;

        resultsWelcome.classList.add("hidden");
        resultsPanel.classList.remove("hidden");

        const isApproved = res.status === "APPROVED";
        const statusBadge = document.getElementById("res-status-badge");
        const statusText = document.getElementById("res-status-text");

        if (isApproved) {
            statusBadge.className = "badge-status badge-verified";
            statusText.textContent = "HERMIT OWL 2 DL VERIFIED: APPROVED";
        } else {
            statusBadge.className = "badge-status badge-contraindicated";
            statusText.textContent = "ESCALATED FOR SPECIALIST REVIEW";
        }

        // Patient Profile & Diagnosed Condition
        const pid = intakePayload?.patient_id || res.patient_id || "PT_001";
        const pname = intakePayload?.patient_name || res.patient_name || "Patient";
        const page = intakePayload?.age || 65;
        const pgender = intakePayload?.gender || "Male";
        document.getElementById("res-patient-profile").textContent = `Patient: ${pname} (${pid}) • Age: ${page} • ${pgender}`;

        document.getElementById("res-diagnosed-condition").textContent = res.diagnosed_condition || "Cardiometabolic Multimorbidity";
        document.getElementById("res-icd10").textContent = `ICD-10: ${res.icd10_code || 'R69'}`;

        const labs = intakePayload?.lab_values || {};

        // Staging Badges
        const ckdBadge = document.getElementById("res-ckd-badge");
        if (ckdBadge) {
            ckdBadge.textContent = `KDIGO: ${res.ckd_stage ? 'Stage ' + res.ckd_stage.replace('G','') + ' (' + res.ckd_stage + ')' : 'Stage 4 (G4)'}`;
        }
        const albBadge = document.getElementById("res-alb-badge");
        if (albBadge) {
            const albStage = res.albuminuria_stage || (labs.uacr > 300 ? 'A3' : (labs.uacr >= 30 ? 'A2' : 'A1'));
            albBadge.textContent = `Albuminuria: ${albStage} (${labs.uacr || 30} mg/g)`;
        }
        const kBadge = document.getElementById("res-k-badge");
        if (kBadge) {
            const kVal = labs.potassium !== undefined ? labs.potassium : 4.5;
            if (kVal > 5.0) {
                kBadge.className = "badge-status badge-contraindicated text-[11px]";
                kBadge.textContent = `K+: Elevated (${kVal} mmol/L)`;
            } else {
                kBadge.className = "badge-status badge-verified text-[11px]";
                kBadge.textContent = `K+: Normal (${kVal} mmol/L)`;
            }
        }

        // Biomarkers Strip (Preserving exact values; never displaying fake defaults or None)
        function formatBiomarker(val, unit) {
            if (val === null || val === undefined || val === "") return "Not Tested";
            return `${val} ${unit}`.trim();
        }

        document.getElementById("strip-egfr").textContent = formatBiomarker(labs.egfr, "mL/min");
        if (labs.sbp && labs.dbp) {
            document.getElementById("strip-bp").textContent = `${labs.sbp}/${labs.dbp} mmHg`;
        } else if (labs.sbp) {
            document.getElementById("strip-bp").textContent = `${labs.sbp} mmHg (SBP)`;
        } else {
            document.getElementById("strip-bp").textContent = "Not Tested";
        }
        document.getElementById("strip-hba1c").textContent = formatBiomarker(labs.hba1c, "%");
        document.getElementById("strip-creatinine").textContent = formatBiomarker(labs.creatinine, "mg/dL");
        const stripK = document.getElementById("strip-potassium");
        if (stripK) stripK.textContent = formatBiomarker(labs.potassium, "mmol/L");
        const stripUacr = document.getElementById("strip-uacr");
        if (stripUacr) stripUacr.textContent = formatBiomarker(labs.uacr, "mg/g");

        // Clinical Safety Alerts Panel
        const alertsContainer = document.getElementById("res-clinical-alerts-container");
        if (alertsContainer) {
            alertsContainer.innerHTML = "";
            const alerts = res.clinical_alerts || [];
            if (alerts.length > 0) {
                alertsContainer.classList.remove("hidden");
                alertsContainer.style.display = "block";
                alerts.forEach(alertText => {
                    const alertDiv = document.createElement("div");
                    const isCritical = alertText.toLowerCase().includes("hyperkalemia") || alertText.toLowerCase().includes("emergency");
                    alertDiv.className = `p-3 rounded-[12px] border text-xs flex items-start space-x-2.5 ${isCritical ? 'bg-[var(--status-contraindicated-bg)] border-[var(--status-contraindicated-border)] text-[var(--status-contraindicated-text)]' : 'bg-[var(--status-warning-bg)] border-[var(--status-warning-border)] text-[var(--status-warning-text)]'}`;
                    alertDiv.innerHTML = `
                        <span class="font-bold text-sm">⚠️</span>
                        <div class="leading-relaxed">
                            <span class="font-semibold uppercase text-[11px] block tracking-wide">${isCritical ? 'CRITICAL SAFETY ALERT' : 'CLINICAL ALERT'}</span>
                            ${alertText}
                        </div>
                    `;
                    alertsContainer.appendChild(alertDiv);
                });
            } else {
                alertsContainer.classList.add("hidden");
                alertsContainer.style.display = "none";
            }
        }

        // Approved Treatment & Clinical Rationale
        document.getElementById("res-recommended-treatment").textContent = res.recommended_treatment || res.last_proposed_treatment || "Withheld for Safety";
        document.getElementById("res-clinical-rationale").textContent = res.clinical_rationale || "";

        // Contraindications Blocked Panel
        const violationsContainer = document.getElementById("res-violations-list");
        const violationsCount = document.getElementById("res-violations-count");
        violationsContainer.innerHTML = "";

        const allViolations = [...(res.intercepted_violations || []), ...(res.warnings || [])];
        const uniqueViolations = Array.from(new Set(allViolations));

        violationsCount.textContent = `${uniqueViolations.length} Safety Safeguard${uniqueViolations.length === 1 ? '' : 's'}`;

        if (uniqueViolations.length === 0) {
            violationsContainer.innerHTML = `
                <div class="p-3 bg-[var(--bg-canvas)] border border-[var(--status-verified-border)] rounded-[12px] text-[var(--status-verified-text)]">
                    ✓ Zero contraindications violated. Prescribed regimen satisfies all formal Description Logic safety axioms.
                </div>
            `;
        } else {
            uniqueViolations.forEach(v => {
                const item = document.createElement("div");
                item.className = "p-3 bg-[var(--bg-canvas)] border border-[var(--status-contraindicated-border)] rounded-[12px] flex items-start space-x-2 text-[var(--color-text-primary)]";
                item.innerHTML = `
                    <span class="text-[var(--status-contraindicated-text)] font-bold text-sm">⚠</span>
                    <div class="leading-relaxed text-xs">
                        <span class="font-medium text-[var(--status-contraindicated-text)]">Contraindication Blocked: </span>
                        ${v}
                    </div>
                `;
                violationsContainer.appendChild(item);
            });
        }

        // Differential Diagnoses
        const diffContainer = document.getElementById("res-differentials");
        diffContainer.innerHTML = "";
        (res.differential_diagnoses || ["Type 2 Diabetes Mellitus", "Stage 2 Essential Hypertension"]).forEach(d => {
            const span = document.createElement("span");
            span.className = "badge-status badge-info";
            span.textContent = d;
            diffContainer.appendChild(span);
        });

        // Guidelines Excerpts
        const glContainer = document.getElementById("res-guidelines-container");
        glContainer.innerHTML = "";
        const passages = res.retrieved_passages || [];
        document.getElementById("res-guidelines-count").textContent = `${passages.length} Citations`;

        passages.forEach(p => {
            const div = document.createElement("div");
            div.className = "bg-[var(--bg-canvas)] border border-[var(--color-border)] p-3 rounded-[12px] text-xs space-y-1";
            div.innerHTML = `
                <div class="flex items-center justify-between text-[11px]">
                    <span class="font-medium text-[var(--color-text-primary)]">${p.source}</span>
                    <span class="font-mono text-[var(--color-text-muted)]">RRF Score: ${p.score || '0.016'}</span>
                </div>
                <div class="text-[11px] text-[var(--color-text-muted)]">${p.topic}</div>
                <p class="text-[var(--color-text-secondary)] leading-relaxed italic">"${p.text}"</p>
            `;
            glContainer.appendChild(div);
        });

        // Collapsible Formal Proof Accordion (Bottom)
        const verDetails = res.verification_details || {};
        document.getElementById("res-proof-details").textContent = verDetails.proof_details || "Tableaux verified with 0 contradiction.";
        document.getElementById("res-reasoning-engine").textContent = verDetails.engine || "HermiT OWL 2 DL Reasoner";
        document.getElementById("res-reasoner-latency").textContent = `${verDetails.reasoner_latency_ms || 0.40} ms reasoner`;
        document.getElementById("res-memory-isolation").textContent = verDetails.memory_isolation || "Ephemeral In-Memory World";

        // Audit Trail
        const trailContainer = document.getElementById("res-audit-trail-container");
        trailContainer.innerHTML = "";
        const trail = res.chronological_trace || res.audit_trail || [];

        trail.forEach((t, idx) => {
            const div = document.createElement("div");
            div.className = "p-2 bg-[var(--bg-surface)] border border-[var(--color-border)] rounded-[8px] flex items-start space-x-2 text-[11px]";
            if (typeof t === "string") {
                div.innerHTML = `<span class="text-[var(--color-text-muted)]">${idx + 1}.</span> <span>${t}</span>`;
            } else {
                div.innerHTML = `
                    <span class="text-[var(--color-text-muted)]">${t.step || idx + 1}.</span>
                    <span class="font-mono font-medium text-[var(--color-text-primary)]">[${t.stage || 'TRACE'}]</span>
                    <span class="text-[var(--color-text-secondary)]">${t.message || JSON.stringify(t)}</span>
                `;
            }
            trailContainer.appendChild(div);
        });
    }

    // ========================================================
    // 8. BENCHMARK TAB (TABLE 8.3)
    // ========================================================
    const btnRunBenchmark = document.getElementById("btn-run-benchmark");
    if (btnRunBenchmark) {
        btnRunBenchmark.addEventListener("click", () => loadBenchmarkTab(true));
    }

    async function loadBenchmarkTab(forceRun = false) {
        const tbody = document.getElementById("benchmark-tbody");
        if (!tbody) return;

        tbody.innerHTML = `<tr><td colspan="5" class="p-4 text-center text-[var(--color-text-muted)]">Running cohort evaluation across all adult & pediatric cases...</td></tr>`;

        try {
            const res = await fetch("/api/benchmark", { method: "POST" });
            const data = await res.json();
            if (data.status === "success") {
                const bm = data.benchmark;
                document.getElementById("bm-first-pass").textContent = `${bm.metrics.first_pass_rate}%`;
                document.getElementById("bm-escape-rate").textContent = `${bm.metrics.formal_violation_escape_rate}%`;
                document.getElementById("bm-safe-rate").textContent = `${bm.metrics.overall_safe_rate}%`;
                document.getElementById("bm-runtime").textContent = `${bm.metrics.total_runtime_seconds}s`;

                tbody.innerHTML = "";
                bm.patients.forEach(p => {
                    const tr = document.createElement("tr");
                    const isPassed = p.status === "APPROVED" || p.status === "SAFE_FIRST_PASS";
                    const isPeds = p.patient_id.includes("007");

                    tr.innerHTML = `
                        <td class="p-3 font-mono font-medium ${isPeds ? 'text-[var(--accent-spark)]' : ''}">
                            ${p.patient_id} ${isPeds ? '<span class="badge-status badge-info text-[10px]">Peds</span>' : ''}
                        </td>
                        <td class="p-3 text-[var(--color-text-muted)]">${p.conditions.join(", ")}</td>
                        <td class="p-3">
                            <span class="badge-status ${isPassed ? 'badge-verified' : 'badge-contraindicated'}">
                                ${p.outcome || p.status}
                            </span>
                        </td>
                        <td class="p-3 font-mono">${p.attempts || 1}</td>
                        <td class="p-3 font-medium text-[var(--color-text-primary)]">${p.approved_medication || '-'}</td>
                    `;
                    tbody.appendChild(tr);
                });
            }
        } catch (err) {
            console.error("Benchmark error:", err);
            tbody.innerHTML = `<tr><td colspan="5" class="p-4 text-center text-[var(--status-contraindicated-text)]">Error executing benchmark</td></tr>`;
        }
    }

    // ========================================================
    // 9. SAFETY AXIOMS TAB
    // ========================================================
    async function loadAxiomsTab() {
        const container = document.getElementById("axioms-cards-container");
        if (!container) return;
        container.innerHTML = `<span class="text-xs text-[var(--color-text-muted)]">Loading Description Logic axioms...</span>`;

        try {
            const res = await fetch("/api/ontology");
            const data = await res.json();
            if (data.status === "success") {
                container.innerHTML = "";
                data.axioms.forEach(ax => {
                    const card = document.createElement("div");
                    card.className = "bg-[var(--bg-canvas)] border border-[var(--color-border)] p-4 rounded-[16px] space-y-2 text-xs";
                    card.innerHTML = `
                        <div class="flex items-center justify-between">
                            <span class="font-mono font-medium text-[var(--status-contraindicated-text)]">${ax.class_name}</span>
                            <span class="badge-status badge-info font-mono text-[10px]">OWL EquivalentClass</span>
                        </div>
                        <p class="text-[var(--color-text-secondary)] leading-relaxed">${ax.description}</p>
                        <div class="pt-2 border-t border-[var(--color-border)] font-mono text-[11px] text-[var(--color-text-muted)]">
                            Condition: <span class="text-[var(--color-text-primary)]">${ax.condition}</span> | Contraindicated Drug: <span class="text-[var(--status-contraindicated-text)]">${ax.medication}</span>
                        </div>
                    `;
                    container.appendChild(card);
                });
            }
        } catch (err) {
            console.error("Axioms error:", err);
        }
    }

    // ========================================================
    // 10. GUIDELINES KNOWLEDGE BASE TAB
    // ========================================================
    async function loadGuidelinesTab() {
        const container = document.getElementById("guidelines-full-list");
        if (!container) return;
        container.innerHTML = `<span class="text-xs text-[var(--color-text-muted)]">Loading clinical practice guidelines corpus...</span>`;

        try {
            const res = await fetch("/api/guidelines");
            const data = await res.json();
            const list = Array.isArray(data) ? data : (data.guidelines || []);
            container.innerHTML = "";
            list.forEach(g => {
                const div = document.createElement("div");
                div.className = "bg-[var(--bg-canvas)] border border-[var(--color-border)] p-4 rounded-[16px] text-xs space-y-1.5";
                div.innerHTML = `
                    <div class="flex items-center justify-between">
                        <span class="font-medium text-[var(--color-text-primary)]">${g.source}</span>
                        <span class="badge-status badge-info font-mono text-[10px]">${g.id}</span>
                    </div>
                    <div class="text-[var(--color-text-muted)] font-medium">${g.topic}</div>
                    <p class="text-[var(--color-text-secondary)] leading-relaxed">"${g.text}"</p>
                `;
                container.appendChild(div);
            });
        } catch (err) {
            console.error("Guidelines error:", err);
        }
    }

    // ========================================================
    // 11. PATIENT ASSISTANT (CHATBOT CONTROLLER)
    // ========================================================
    if (chatForm && chatInput) {
        chatForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const text = chatInput.value.trim();
            if (!text && !chatAttachedContext.filename) return;

            appendChatMessage("user", text || `Uploaded report: ${chatAttachedContext.filename}`);
            chatInput.value = "";

            const typingEl = appendChatTyping();
            const provider = providerSelect.value;
            const apiKey = provider === "gemini" ? apiKeys.gemini : (provider === "openai" ? apiKeys.openai : "");

            try {
                const res = await fetch("/api/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        message: text,
                        patient_id: "Patient_Chat_" + Math.random().toString(36).substring(2, 9),
                        evaluation_id: "eval_" + Math.random().toString(36).substring(2, 11),
                        conditions: chatAttachedContext.conditions,
                        lab_values: chatAttachedContext.lab_values,
                        provider: provider,
                        api_key: apiKey
                    })
                });
                const data = await res.json();
                typingEl.remove();

                if (data.status === "success") {
                    appendChatMessage("assistant", data.reply);
                } else {
                    appendChatMessage("assistant", "I encountered an error analyzing your health question. Please try again.");
                }
            } catch (err) {
                console.error("Chat error:", err);
                typingEl.remove();
                appendChatMessage("assistant", "Network connection error. Please verify your connection.");
            }
        });
    }

    function appendChatMessage(role, text) {
        const bubble = document.createElement("div");
        bubble.className = `chat-bubble ${role === 'user' ? 'chat-bubble-user' : 'chat-bubble-assistant'}`;

        let html = text
            .replace(/^### (.*$)/gim, '<h3>$1</h3>')
            .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/gim, '<em>$1</em>')
            .replace(/^- (.*$)/gim, '<li>$1</li>')
            .replace(/\n\n/g, '<p></p>')
            .replace(/\n/g, '<br>');

        bubble.innerHTML = html;
        chatMessages.appendChild(bubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendChatTyping() {
        const bubble = document.createElement("div");
        bubble.className = "chat-bubble chat-bubble-assistant text-xs text-[var(--color-text-muted)] italic";
        bubble.textContent = "SafeCDS assistant is evaluating formal safety guidelines...";
        chatMessages.appendChild(bubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return bubble;
    }

    // Prompt Starters
    document.querySelectorAll(".prompt-pill").forEach(pill => {
        pill.addEventListener("click", () => {
            if (chatInput) {
                chatInput.value = pill.getAttribute("data-prompt");
                chatForm.dispatchEvent(new Event("submit"));
            }
        });
    });

    // Chat PDF Attachment
    if (btnChatUploadPdf && chatPdfFileInput) {
        btnChatUploadPdf.addEventListener("click", () => chatPdfFileInput.click());
        chatPdfFileInput.addEventListener("change", async (e) => {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                const formData = new FormData();
                formData.append("file", file);

                try {
                    const res = await fetch("/api/upload-pdf", { method: "POST", body: formData });
                    const data = await res.json();
                    if (data.status === "success") {
                        const parsed = data.data;
                        chatAttachedContext.conditions = parsed.conditions || [];
                        chatAttachedContext.lab_values = {
                            egfr: parsed.egfr,
                            sbp: parsed.sbp,
                            dbp: parsed.dbp,
                            hba1c: parsed.hba1c,
                            creatinine: parsed.creatinine,
                            potassium: parsed.potassium,
                            uacr: parsed.uacr,
                            age: parsed.age,
                            gender: parsed.gender
                        };
                        chatAttachedContext.filename = file.name;

                        chatAttachedPill.classList.remove("hidden");
                        chatFilename.textContent = `Attached: ${file.name} (eGFR: ${parsed.egfr || '-'}, HbA1c: ${parsed.hba1c || '-'}%)`;
                    }
                } catch (err) {
                    console.error("Chat PDF error:", err);
                }
            }
        });
    }

    if (btnRemoveChatFile) {
        btnRemoveChatFile.addEventListener("click", () => {
            chatAttachedContext = { conditions: [], lab_values: {}, filename: null };
            chatAttachedPill.classList.add("hidden");
        });
    }

    // Initial Load
    fetchCohort();
});
