"""Generates the official B.Tech 3rd Year SRS Document (.docx) conforming to IEEE Std 830-1998.

Populates all academic sections, styled tables, and embedded high-resolution diagrams.
"""

import os
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from twin.paths import PROJECT_ROOT, DOCS_DIR

DOCX_OUTPUT = PROJECT_ROOT / "BTech_3rd_Year_SRS_Final.docx"
FIG_DIR = DOCS_DIR / "phase3_demo" / "figures"


def set_cell_background(cell, fill_hex):
    """Set cell shading color."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set internal cell padding."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{m}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def style_table(table, col_widths=None):
    """Applies clean engineering table borders and alignment."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table.rows):
        # Header row formatting
        if i == 0:
            for cell in row.cells:
                set_cell_background(cell, "2B6CB0")  # Professional Blue
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r in p.runs:
                        r.font.bold = True
                        r.font.color.rgb = RGBColor(255, 255, 255)
                        r.font.size = Pt(9.5)
        else:
            # Alternating zebra striping
            bg_color = "F7FAFC" if i % 2 == 1 else "FFFFFF"
            for cell in row.cells:
                set_cell_background(cell, bg_color)
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(9.0)
                        r.font.color.rgb = RGBColor(45, 55, 72)

        for j, cell in enumerate(row.cells):
            set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if col_widths and j < len(col_widths):
                cell.width = col_widths[j]


def build_srs_document():
    doc = Document()

    # Page Margins: Standard 1 inch
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Styles
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(45, 55, 72)

    # --------------------------------------------------------------------------
    # COVER PAGE
    # --------------------------------------------------------------------------
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_inst = p_inst.add_run("DEPARTMENT OF MECHANICAL ENGINEERING\nABES ENGINEERING COLLEGE / AKTU, LUCKNOW\n")
    r_inst.font.bold = True
    r_inst.font.size = Pt(13)
    r_inst.font.color.rgb = RGBColor(43, 108, 176)

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(30)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_type = p_title.add_run("SOFTWARE REQUIREMENTS SPECIFICATION (SRS)\n")
    r_type.font.bold = True
    r_type.font.size = Pt(18)
    r_type.font.color.rgb = RGBColor(26, 32, 44)

    r_proj = p_title.add_run("Knowledge-Driven IoT Fault Diagnosis Assistant:\nAI-Driven Digital Twin for Smart Factory Operations\n")
    r_proj.font.bold = True
    r_proj.font.size = Pt(16)
    r_proj.font.color.rgb = RGBColor(49, 130, 206)

    r_std = p_title.add_run("Standard Academic Compliance: IEEE Std 830-1998\n")
    r_std.font.italic = True
    r_std.font.size = Pt(11)
    r_std.font.color.rgb = RGBColor(113, 128, 150)

    p_divider = doc.add_paragraph()
    p_divider.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_divider.paragraph_format.space_after = Pt(40)
    r_div = p_divider.add_run("—" * 45)
    r_div.font.color.rgb = RGBColor(203, 213, 225)

    # Submission Box
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.line_spacing = 1.3
    r_sub = p_meta.add_run("SUBMITTED BY:\n")
    r_sub.font.bold = True
    r_sub.font.size = Pt(11)
    r_sub.font.color.rgb = RGBColor(43, 108, 176)

    p_meta.add_run("Candidate: ")
    r_name = p_meta.add_run("Nishchay Arora\n")
    r_name.font.bold = True
    p_meta.add_run("University Roll No: ")
    r_roll = p_meta.add_run("2400320400036\n")
    r_roll.font.bold = True
    p_meta.add_run("Degree: Bachelor of Technology (B.Tech) in Mechanical Engineering\n")
    p_meta.add_run("Academic Year: 2026 – 2027\n\n")

    r_gui = p_meta.add_run("UNDER THE GUIDANCE OF:\n")
    r_gui.font.bold = True
    r_gui.font.size = Pt(11)
    r_gui.font.color.rgb = RGBColor(43, 108, 176)
    p_meta.add_run("Supervisor: ")
    r_guide = p_meta.add_run("Prof. / Dr. Tanvi Saxena\n")
    r_guide.font.bold = True
    p_meta.add_run("Department: Department of Mechanical Engineering\n")
    p_meta.add_run("Status: 3rd Year Mid-Term Project Specification\n")
    p_meta.add_run("Date of Submission: 30-September-2026\n")

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # REVISION HISTORY TABLE
    # --------------------------------------------------------------------------
    h_rev = doc.add_heading("Document Revision & Approval History", level=2)
    h_rev.style.font.color.rgb = RGBColor(43, 108, 176)

    table_rev = doc.add_table(rows=3, cols=5)
    headers = ["Version", "Date", "Description / Major Changes", "Prepared By", "Reviewed By"]
    for j, h in enumerate(headers):
        table_rev.rows[0].cells[j].paragraphs[0].text = h

    rev_data = [
        ("0.1", "15-Sep-2026", "Initial Draft, Problem Definition & Scope Boundary", "Nishchay Arora", "Dr. Tanvi Saxena"),
        ("1.0", "30-Sep-2026", "Final 3rd Year Mid-Term SRS Baseline (IEEE 830-1998)", "Nishchay Arora", "HOD / Review Committee"),
    ]
    for i, row in enumerate(rev_data):
        for j, val in enumerate(row):
            table_rev.rows[i + 1].cells[j].paragraphs[0].text = val
    style_table(table_rev, [Inches(0.8), Inches(1.1), Inches(2.7), Inches(1.1), Inches(1.3)])

    doc.add_paragraph().paragraph_format.space_after = Pt(15)

    # --------------------------------------------------------------------------
    # 1. INTRODUCTION
    # --------------------------------------------------------------------------
    h1 = doc.add_heading("1. Introduction", level=1)
    h1.style.font.color.rgb = RGBColor(26, 32, 44)

    doc.add_heading("1.1 Purpose", level=2).style.font.color.rgb = RGBColor(43, 108, 176)
    doc.add_paragraph(
        "The purpose of this Software Requirements Specification (SRS) document is to establish a comprehensive, "
        "rigorous, and unambiguous formal definition of the software architecture, functional capabilities, performance metrics, "
        "and operational boundaries for the 'Knowledge-Driven IoT Fault Diagnosis Assistant: AI-Driven Digital Twin for Smart Factory Operations'. "
        "Conforming strictly to IEEE Std 830-1998 recommendations, this document serves as the formal academic baseline for the B.Tech "
        "3rd-year Mechanical Engineering technical evaluation at ABES Engineering College / AKTU Lucknow."
    )

    doc.add_heading("1.2 Scope of the System", level=2).style.font.color.rgb = RGBColor(43, 108, 176)
    doc.add_paragraph(
        "In modern automated manufacturing, unplanned production downtime and delayed quality defect discovery result in substantial scrap costs and "
        "process bottlenecks. The proposed platform addresses this industrial challenge by engineering an end-to-end Cyber-Physical Digital Twin "
        "grounded in the real-world Bosch Production Line Performance dataset (1,183,747 parts across multi-station manufacturing lines)."
    )

    p_scope = doc.add_paragraph()
    r = p_scope.add_run("In-Scope Capabilities:\n")
    r.font.bold = True
    p_scope.add_run(
        "• Deterministic shop-floor telemetry replay over Eclipse Mosquitto MQTT with speed scaling (1x to 120x, burst >7,000 events/s).\n"
        "• Real-time topological tracking of Line 3 core flow (S29 -> S30 -> S33 -> S34 -> (S35 | S36) -> S37) handling 89.02% of factory throughput.\n"
        "• Stateful multi-factor Station Health Scoring (Hs in [0, 100]) combining sensor Z-score drift, pacing delay, and local defect rates.\n"
        "• PyTorch Multilayer Perceptron (MLP) defect risk prediction evaluated at the S34 fork with Platt probability calibration.\n"
        "• 1-layer PyTorch LSTM Autoencoder modeling 24-hour multivariate window sequences for unsupervised process anomaly detection.\n"
        "• Sub-second time-series persistence in InfluxDB v2 and live operational visualization via Grafana 11 with declarative alerting."
    )

    p_out = doc.add_paragraph()
    r_out = p_out.add_run("Out-of-Scope Boundaries:\n")
    r_out.font.bold = True
    p_out.add_run(
        "• Closed-loop mechanical PLC actuation or pneumatic diverter hardware control (open-loop predictive advisory mode).\n"
        "• Commercial enterprise ERP/billing integration (SAP/Oracle).\n"
        "• Fabricated machine types or unverified physical tooling names (retaining strictly anonymized empirical nomenclature S29-S37)."
    )

    p_ben = doc.add_paragraph()
    r_ben = p_ben.add_run("Expected Benefits:\n")
    r_ben.font.bold = True
    p_ben.add_run(
        "• Early Detection Lead Time: Provides +30 to 75 minutes of proactive quality warning before exit QA at S37.\n"
        "• Inspection Efficiency: Delivers Lift@1% of 5.8x to 7.26x, allowing QA inspectors to capture >7% of defects by inspecting only the top 1% highest-risk parts.\n"
        "• Zero Data Fabrication: 100% grounded in empirical manufacturing data without synthetic anomaly injection."
    )

    doc.add_heading("1.3 Definitions, Acronyms, and Abbreviations", level=2).style.font.color.rgb = RGBColor(43, 108, 176)
    table_def = doc.add_table(rows=8, cols=3)
    def_headers = ["Category", "Term / Acronym", "Definition / Standard Industrial Context"]
    for j, h in enumerate(def_headers):
        table_def.rows[0].cells[j].paragraphs[0].text = h
    defs = [
        ("Standard", "SRS", "Software Requirements Specification conforming to IEEE Std 830-1998 format."),
        ("Architecture", "Digital Twin", "Virtual software representation of physical machines and factory processes updated in real time."),
        ("Network", "MQTT", "Message Queuing Telemetry Transport (ISO/IEC 20922); industrial lightweight pub/sub protocol."),
        ("Database", "TSM / Flux", "Time-Structured Merge tree storage engine in InfluxDB v2; Flux functional data query language."),
        ("Deep Learning", "MLP / LSTM-AE", "Multilayer Perceptron for defect classification; LSTM Autoencoder for temporal anomaly detection."),
        ("Statistics", "PR-AUC", "Precision-Recall Area Under Curve; robust performance metric under severe class imbalance (0.51% defect rate)."),
        ("Calibration", "Platt Scaling", "Logistic calibration mapping neural network logits into true posterior Bayesian probabilities."),
    ]
    for i, row in enumerate(defs):
        for j, val in enumerate(row):
            table_def.rows[i + 1].cells[j].paragraphs[0].text = val
    style_table(table_def, [Inches(1.2), Inches(1.3), Inches(4.5)])

    # --------------------------------------------------------------------------
    # 2. OVERALL DESCRIPTION
    # --------------------------------------------------------------------------
    doc.add_page_break()
    h2 = doc.add_heading("2. Overall Description", level=1)
    h2.style.font.color.rgb = RGBColor(26, 32, 44)

    doc.add_heading("2.1 Product Perspective & Context Architecture", level=2).style.font.color.rgb = RGBColor(43, 108, 176)
    doc.add_paragraph(
        "The digital twin functions as an edge-native industrial intelligence platform. Raw historical sensor visits are replayed "
        "by the Replay Engine over MQTT topics. The Ingestion Service transforms telemetry into nanosecond InfluxDB points, while the Scoring Service "
        "computes composite health and triggers GPU-accelerated PyTorch defect and anomaly inferences. Operational stakeholders view live dashboards "
        "and receive provisioned alerts via Grafana 11."
    )

    # Embed Figure 1
    fig1_path = FIG_DIR / "figure1_line3_topology.png"
    if fig1_path.exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(fig1_path), width=Inches(6.2))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure 2.1: Bosch Line 3 Core Manufacturing Topology & Informative Feature Allocation")
        r_cap.font.italic = True
        r_cap.font.size = Pt(9.5)

    doc.add_heading("2.2 User Classes and Characteristics", level=2).style.font.color.rgb = RGBColor(43, 108, 176)
    table_users = doc.add_table(rows=5, cols=3)
    u_headers = ["User Role / Persona", "Technical Proficiency", "Core Responsibilities & Access Rights"]
    for j, h in enumerate(u_headers):
        table_users.rows[0].cells[j].paragraphs[0].text = h
    users_data = [
        ("Plant Operations Engineer", "High (Industrial Engg)", "Monitors Line Overview; tracks plant WIP, station health matrix, and line throughput."),
        ("Quality Control Specialist", "Intermediate (QA / QC)", "Monitors Parts & Risk and AI Insights; inspects parts with risk > 0.70; reviews early warnings."),
        ("Maintenance Technician", "Intermediate (Shop Floor)", "Responds to station health warnings (Hs < 50); diagnoses sensor feature drift and transit delays."),
        ("Academic Evaluator / Guide", "High (Software / AI Research)", "Audits system architecture, causal validation integrity, ML metrics, and test coverage."),
    ]
    for i, row in enumerate(users_data):
        for j, val in enumerate(row):
            table_users.rows[i + 1].cells[j].paragraphs[0].text = val
    style_table(table_users, [Inches(1.8), Inches(1.5), Inches(3.7)])

    doc.add_heading("2.3 Operating Environment & Technical Stack", level=2).style.font.color.rgb = RGBColor(43, 108, 176)
    doc.add_paragraph(
        "• Host Hardware: Intel Core i5/i7/i9 (x86_64) CPU, 16 GB RAM minimum.\n"
        "• Hardware Acceleration: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) with CUDA 12.4 support (automatic CPU fallback supported).\n"
        "• Host Operating System: Microsoft Windows 11 / Windows 10 (PowerShell 7+) or Ubuntu 22.04 LTS.\n"
        "• Virtualization: Docker Desktop 4.28+ with Linux container subsystem and Intel VT-x.\n"
        "• Core Software Frameworks: Python 3.12, PyTorch 2.6.0+cu124, InfluxDB v2.7, Eclipse Mosquitto 2.0, Grafana 11.1.0."
    )

    # --------------------------------------------------------------------------
    # 3. SYSTEM FEATURES AND FUNCTIONAL REQUIREMENTS
    # --------------------------------------------------------------------------
    doc.add_page_break()
    h3 = doc.add_heading("3. System Features and Functional Requirements", level=1)
    h3.style.font.color.rgb = RGBColor(26, 32, 44)

    doc.add_heading("3.1 Module 1: Ingestion, Streaming & Discrete-Event Replay", level=2).style.font.color.rgb = RGBColor(43, 108, 176)
    table_m1 = doc.add_table(rows=5, cols=4)
    m_headers = ["Req ID", "Feature Description", "Input / Validation Rule", "Priority"]
    for j, h in enumerate(m_headers):
        table_m1.rows[0].cells[j].paragraphs[0].text = h
    m1_data = [
        ("FR-1.1", "Deterministic Replay", "Streams historical visits chronologically from pre-compiled Parquet scenario datasets.", "High"),
        ("FR-1.2", "Simulation Speed Scaling", "Configurable speed multiplier (1x-120x, burst mode >7,000 events/s).", "High"),
        ("FR-1.3", "Production Gap Handling", "Detects line stoppages (Delta t > 0.02 units); fast-forwards idle periods in 5.0s.", "Medium"),
        ("FR-1.4", "ISA-95 Topic Serialization", "Publishes JSON StationTelemetryEvent to factory/line3/{station_id}/telemetry via MQTT.", "High"),
    ]
    for i, row in enumerate(m1_data):
        for j, val in enumerate(row):
            table_m1.rows[i + 1].cells[j].paragraphs[0].text = val
    style_table(table_m1, [Inches(0.9), Inches(1.8), Inches(3.4), Inches(0.9)])

    # Embed Figure 2
    fig2_path = FIG_DIR / "figure2_scenario_timelines.png"
    if fig2_path.exists():
        p_img = doc.add_paragraph()
        p_img.paragraph_format.space_before = Pt(10)
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(fig2_path), width=Inches(6.2))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure 3.1: Leave-Scenario-Out Replay Scenarios on Continuous Production Timeline")
        r_cap.font.italic = True
        r_cap.font.size = Pt(9.5)

    doc.add_heading("3.2 Module 2: Edge Analytics, Multi-Factor Health Scoring & AI Inference", level=2).style.font.color.rgb = RGBColor(43, 108, 176)
    table_m2 = doc.add_table(rows=6, cols=4)
    for j, h in enumerate(m_headers):
        table_m2.rows[0].cells[j].paragraphs[0].text = h
    m2_data = [
        ("FR-2.1", "Baseline Normalization", "Loads 39 feature distributions; computes sensor Z-scores (Z = (x - mean)/std).", "High"),
        ("FR-2.2", "Composite Health Scoring", "Computes Hs = 100 - (0.45*Drift + 0.30*Pacing + 0.25*Defect) bounded in [0, 100].", "High"),
        ("FR-2.3", "PyTorch Defect MLP", "2-layer MLP (128->64) evaluates 85 features at S34 fork before physical entry to S35/S36.", "High"),
        ("FR-2.4", "Platt Calibration", "Calibrates raw neural logits into true Bayesian probabilities matching 0.51% base rate.", "High"),
        ("FR-2.5", "LSTM Autoencoder Anomaly", "1-layer LSTM autoencoder models 24h multivariate sequences; flags errors > p99.", "Medium"),
    ]
    for i, row in enumerate(m2_data):
        for j, val in enumerate(row):
            table_m2.rows[i + 1].cells[j].paragraphs[0].text = val
    style_table(table_m2, [Inches(0.9), Inches(1.8), Inches(3.4), Inches(0.9)])

    # Embed Figure 3
    fig3_path = FIG_DIR / "figure3_lead_time_comparison.png"
    if fig3_path.exists():
        p_img = doc.add_paragraph()
        p_img.paragraph_format.space_before = Pt(10)
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(fig3_path), width=Inches(5.8))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure 3.2: Operational Lead Time Comparison: Proactive In-Line AI vs Lagging Exit QC")
        r_cap.font.italic = True
        r_cap.font.size = Pt(9.5)

    doc.add_heading("3.3 Module 3: Time-Series Storage, Alerting, Visualization & Control", level=2).style.font.color.rgb = RGBColor(43, 108, 176)
    table_m3 = doc.add_table(rows=5, cols=4)
    for j, h in enumerate(m_headers):
        table_m3.rows[0].cells[j].paragraphs[0].text = h
    m3_data = [
        ("FR-3.1", "Time-Series Persistence", "Consumes MQTT events; commits Line Protocol points in batches of 50 to InfluxDB v2.", "High"),
        ("FR-3.2", "Grafana Dashboards", "Provisioned dashboards: Line Overview, Station Detail, Parts & Risk, AI Insights.", "High"),
        ("FR-3.3", "Declarative Alerting", "YAML rules trigger on high-risk part surges, station starvation, and LSTM anomaly spikes.", "High"),
        ("FR-3.4", "Unified Demo Orchestrator", "PowerShell script (demo.ps1) manages Docker, background daemons, and scenario replay.", "Medium"),
    ]
    for i, row in enumerate(m3_data):
        for j, val in enumerate(row):
            table_m3.rows[i + 1].cells[j].paragraphs[0].text = val
    style_table(table_m3, [Inches(0.9), Inches(1.8), Inches(3.4), Inches(0.9)])

    # --------------------------------------------------------------------------
    # 4. EXTERNAL INTERFACE & NON-FUNCTIONAL REQUIREMENTS
    # --------------------------------------------------------------------------
    doc.add_page_break()
    h4 = doc.add_heading("4. External Interfaces & Non-Functional Requirements", level=1)
    h4.style.font.color.rgb = RGBColor(26, 32, 44)

    doc.add_heading("4.1 External Interfaces", level=2).style.font.color.rgb = RGBColor(43, 108, 176)
    doc.add_paragraph(
        "• User Interface: Responsive web interface delivered via Grafana 11 at http://localhost:3000.\n"
        "• IoT Protocol Interface: Eclipse Mosquitto MQTT broker on TCP port 1883.\n"
        "• Database Storage Interface: InfluxDB v2 API on TCP port 8086 with Flux scripting engine.\n"
        "• Compute Interface: NVIDIA Ampere GPU via PyTorch CUDA 12.4 tensor backend."
    )

    doc.add_heading("4.2 Non-Functional Requirements (NFRs)", level=2).style.font.color.rgb = RGBColor(43, 108, 176)
    table_nfr = doc.add_table(rows=6, cols=3)
    nfr_headers = ["Attribute", "Target Metric / Standard", "Verification & Validation Method"]
    for j, h in enumerate(nfr_headers):
        table_nfr.rows[0].cells[j].paragraphs[0].text = h
    nfr_data = [
        ("Performance (Throughput)", "Replay throughput >= 5,000 events/sec in burst mode.", "Achieved 7,405 events/sec via run.py --speed 0."),
        ("Performance (Latency)", "Neural inference latency < 25ms per part on GPU.", "Achieved 11.4ms batch inference on RTX 3050 GPU."),
        ("Security", "Token-based InfluxDB access; Grafana RBAC authentication.", "Enforced via twin_config.yaml and Docker networking."),
        ("Reliability", "Continuous uptime >= 99.0%; zero memory leakage across long runs.", "Enforced with bounded sliding deques (maxlen=10000)."),
        ("ML Predictive Lift", "Defect model Lift@1% >= 5.0x over random baseline.", "Achieved 5.8x to 7.26x Lift@1% in forward chaining."),
    ]
    for i, row in enumerate(nfr_data):
        for j, val in enumerate(row):
            table_nfr.rows[i + 1].cells[j].paragraphs[0].text = val
    style_table(table_nfr, [Inches(1.8), Inches(2.5), Inches(2.7)])

    # Embed Figure 4
    fig4_path = FIG_DIR / "figure4_risk_bands_overlay.png"
    if fig4_path.exists():
        p_img = doc.add_paragraph()
        p_img.paragraph_format.space_before = Pt(10)
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(fig4_path), width=Inches(5.8))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure 4.1: Part Risk Categorization Bands vs Empirical Quality Outcomes")
        r_cap.font.italic = True
        r_cap.font.size = Pt(9.5)

    # --------------------------------------------------------------------------
    # 5. SYSTEM DESIGN AND ANALYSIS MODELS
    # --------------------------------------------------------------------------
    doc.add_page_break()
    h5 = doc.add_heading("5. System Design and Analysis Models", level=1)
    h5.style.font.color.rgb = RGBColor(26, 32, 44)

    doc.add_heading("5.1 Use Case Specifications", level=2).style.font.color.rgb = RGBColor(43, 108, 176)
    doc.add_paragraph(
        "The system accommodates plant engineers, quality specialists, and evaluators through well-defined interactions:"
    )
    table_uc = doc.add_table(rows=6, cols=4)
    uc_headers = ["Use Case ID", "Use Case Title", "Primary Actor", "Operational Description"]
    for j, h in enumerate(uc_headers):
        table_uc.rows[0].cells[j].paragraphs[0].text = h
    uc_data = [
        ("UC-01", "Launch Replay & Set Speed", "Plant Operations Eng.", "Initiates scenario streaming at configurable speed multiplier."),
        ("UC-02", "Monitor Line Overview", "Plant Operations Eng.", "Observes throughput, WIP, and real-time Station Status Matrix."),
        ("UC-03", "Inspect Station Health & Drift", "Maintenance Tech.", "Analyzes feature Z-score deviations and transit pacing percentiles."),
        ("UC-04", "Predictive Defect Screening", "Quality Specialist", "Evaluates parts at S34 fork; flags high-risk units (+30-75m lead time)."),
        ("UC-05", "LSTM Anomaly Investigation", "Quality Specialist", "Drills into reconstruction error spikes and station contribution radar."),
    ]
    for i, row in enumerate(uc_data):
        for j, val in enumerate(row):
            table_uc.rows[i + 1].cells[j].paragraphs[0].text = val
    style_table(table_uc, [Inches(1.0), Inches(1.8), Inches(1.6), Inches(2.6)])

    doc.add_heading("5.2 Data Flow Architecture (DFD Levels 0 and 1)", level=2).style.font.color.rgb = RGBColor(43, 108, 176)
    doc.add_paragraph(
        "• DFD Level 0 (Context Level): Historical shop-floor data feeds into the Digital Twin boundary; the twin emits real-time "
        "telemetry, health status, and predictive risk alerts to plant engineers, quality teams, and maintenance crews.\n"
        "• DFD Level 1 (Decomposition): Ingestion Daemon serializes events into InfluxDB Line Protocol points; Scoring Engine processes "
        "sliding windows through Z-score normalizers, Platt-calibrated Defect MLPs, and LSTM Autoencoders before committing to InfluxDB and Grafana."
    )

    # --------------------------------------------------------------------------
    # 6. ACADEMIC REVIEW & SIGN-OFF
    # --------------------------------------------------------------------------
    doc.add_page_break()
    h6 = doc.add_heading("6. Academic Review & Sign-Off", level=1)
    h6.style.font.color.rgb = RGBColor(26, 32, 44)

    doc.add_paragraph(
        "This Software Requirements Specification document has been submitted and reviewed by the academic supervisory "
        "panel at ABES Engineering College / AKTU Lucknow as the binding technical specification for the B.Tech 3rd-year engineering project:\n\n"
        "Project Title: Knowledge-Driven IoT Fault Diagnosis Assistant: AI-Driven Digital Twin for Smart Factory Operations\n"
        "Candidate: Nishchay Arora (University Roll No: 2400320400036)\n"
        "Degree: Bachelor of Technology (B.Tech) in Mechanical Engineering\n"
        "Academic Year: 2026 – 2027\n"
    )

    doc.add_paragraph().paragraph_format.space_before = Pt(20)

    # Signature Table
    sig_table = doc.add_table(rows=4, cols=3)
    sig_headers = ["Project Supervisor / Guide", "Internal Examiner", "Head of Department (HOD)"]
    for j, h in enumerate(sig_headers):
        sig_table.rows[0].cells[j].paragraphs[0].text = h

    sig_data = [
        ("Prof. / Dr. Tanvi Saxena\nDept. of Mechanical Engineering", "Evaluation Committee\nDept. of Mechanical Engineering", "Head of Department\nDept. of Mechanical Engineering"),
        ("\nSignature: __________________\n", "\nSignature: __________________\n", "\nSignature: __________________\n"),
        ("Date: ________________________", "Date: ________________________", "Date: ________________________"),
    ]
    for i, row in enumerate(sig_data):
        for j, val in enumerate(row):
            sig_table.rows[i + 1].cells[j].paragraphs[0].text = val

    style_table(sig_table, [Inches(2.3), Inches(2.3), Inches(2.3)])

    # Save to destination
    doc.save(str(DOCX_OUTPUT))
    print(f"Document successfully created: {DOCX_OUTPUT}")


if __name__ == "__main__":
    build_srs_document()
