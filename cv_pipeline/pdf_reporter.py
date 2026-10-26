"""
STRUCTVISION - Executive PDF Report Generator.
Compiles OpenSeesPy C++ FEA audit results, incremental pushover curves,
damage statistics, and 3D visual field screenshots into a publication-ready PDF deliverable.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass ReportLab canvas to compute total page count and draw running
    headers, footers, and dynamic 'Page X of Y' numbering across all pages.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        
        width, height = letter
        margin = 36  # 0.5 inch margins

        # --- RUNNING HEADER ---
        self.setStrokeColor(colors.HexColor('#CBD5E1'))
        self.setLineWidth(0.75)
        self.line(margin, height - 42, width - margin, height - 42)

        self.setFont('Helvetica-Bold', 8)
        self.setFillColor(colors.HexColor('#64748B'))
        self.drawString(margin, height - 36, "STRUCTVISION | FEA STRUCTURAL AUDIT REPORT")

        # --- RUNNING FOOTER ---
        self.line(margin, 42, width - margin, 42)

        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#64748B'))
        self.drawString(margin, 28, "CONFIDENTIAL & PROPRIETARY — FOR PROFESSIONAL USE ONLY")

        # Dynamic Page X of Y (Bottom Right)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(width - margin, 28, page_str)

        self.restoreState()


class StructuralReportPDFGenerator:
    def __init__(self, output_pdf_path: str = "output/Executive_Structural_Audit_Report.pdf"):
        self.output_pdf_path = output_pdf_path
        os.makedirs(os.path.dirname(self.output_pdf_path), exist_ok=True)

    def generate(
        self, 
        audit_results: dict, 
        image_path: str = "output/structural_report.png",
        curve_image_path: str = None
    ):
        """Generates a structured, multi-panel executive PDF report with embedded pushover curves."""
        
        doc = SimpleDocTemplate(
            self.output_pdf_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=54,
            bottomMargin=54
        )
        
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#1A365D'),
            spaceAfter=4
        )
        
        subtitle_style = ParagraphStyle(
            'DocSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            textColor=colors.HexColor('#4A5568'),
            spaceAfter=12
        )
        
        section_heading = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=15,
            textColor=colors.HexColor('#2B6CB0'),
            spaceBefore=8,
            spaceAfter=6
        )
        
        body_style = ParagraphStyle(
            'BodyDark',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor('#2D3748')
        )

        table_header_style = ParagraphStyle(
            'TableHeader',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=8.5,
            leading=11,
            textColor=colors.white
        )

        elements = []

        # --- REPORT HEADER ---
        elements.append(Paragraph("STRUCTVISION: FEA STRUCTURAL AUDIT REPORT", title_style))
        date_str = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
        elements.append(Paragraph(f"<b>Automated Inspection & Non-Linear OpenSees Audit Deliverable</b> | Generated: {date_str}", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1A365D'), spaceAfter=10))

        # --- EXECUTIVE SUMMARY TABLE ---
        elements.append(Paragraph("1. Audit & Solver Key Metrics", section_heading))
        
        curve_path = curve_image_path or audit_results.get("curve_plot_path", "output/pushover_capacity_curve.png")
        
        table_data = [
            [Paragraph("<b>Metric Parameter</b>", body_style), Paragraph("<b>Audit Value</b>", body_style)],
            [Paragraph("Solver Status", body_style), Paragraph(f"<font color='green'><b>{audit_results.get('status', 'N/A')}</b></font>", body_style)],
            [Paragraph("FEA Engine Backend", body_style), Paragraph(str(audit_results.get('backend', 'N/A')), body_style)],
            [Paragraph("Newton-Raphson Steps Completed", body_style), Paragraph(f"{audit_results.get('completed_steps', 'N/A')} / {audit_results.get('total_steps', 'N/A')} steps", body_style)],
            [Paragraph("Peak Applied Load (kN)", body_style), Paragraph(f"{audit_results.get('max_load_kN', 0.0):.2f} kN", body_style)],
            [Paragraph("Max Displ. Magnitude (mm)", body_style), Paragraph(f"{audit_results.get('max_displacement_mm', 0.0):.4f} mm", body_style)],
            [Paragraph("Mesh Damage Ratio (%)", body_style), Paragraph(f"{audit_results.get('mesh_damage_ratio_percent', 0.0):.2f}%", body_style)],
            [Paragraph("Estimated Capacity Reduction (%)", body_style), Paragraph(f"<b><font color='red'>{audit_results.get('capacity_reduction_percent', 0.0):.2f}%</font></b>", body_style)],
            [Paragraph("Cracked Nodal Sites / Total", body_style), Paragraph(f"{audit_results.get('damaged_nodes_count', 0)} / {audit_results.get('total_nodes_count', 0)} nodes", body_style)],
        ]
        
        summary_table = Table(table_data, colWidths=[240, 300], repeatRows=1)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EDF2F7')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1A365D')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 10))

        # --- NON-LINEAR PUSHOVER CAPACITY CURVE ---
        elements.append(Paragraph("2. Non-Linear Incremental Pushover Capacity Curve", section_heading))
        if os.path.exists(curve_path):
            img_curve = Image(curve_path, width=520, height=210)
            elements.append(img_curve)
        else:
            elements.append(Paragraph("<i>Pushover capacity plot missing. Ensure run_pushover_analysis() has executed.</i>", body_style))
            
        elements.append(Spacer(1, 10))

        # --- VISUALIZATION FIELD PANEL ---
        elements.append(Paragraph("3. 3D Damage Map & Stress Field Rendering", section_heading))
        if os.path.exists(image_path):
            img_field = Image(image_path, width=520, height=210)
            elements.append(img_field)
        else:
            elements.append(Paragraph("<i>Visual field rendering missing. Ensure pipeline.py has executed.</i>", body_style))
            
        elements.append(Spacer(1, 10))

        # --- TECHNICAL METHODOLOGY NOTES ---
        elements.append(Paragraph("4. Compliance & Solver Standards", section_heading))
        compliance_text = (
            "This structural capacity audit was performed using non-linear static incremental pushover analysis "
            "under Berkeley OpenSees research standards. Material behavior at concrete continuum sites was modeled "
            "using Concrete01 (Kent-Scott-Park) parameters with spatial stiffness degradation. System response was "
            "evaluated incrementally using Newton-Raphson equilibrium iterations across single-point quadrilateral "
            "(SSPquad) plane stress finite elements."
        )
        elements.append(Paragraph(compliance_text, body_style))
        elements.append(Spacer(1, 10))

        # --- 5. CRITICAL NODAL SITES SUMMARY (TOP 20 - OPTION B) ---
        elements.append(Paragraph("5. Critical Nodal Stress & Strain Summary (Top 20 Sites)", section_heading))

        all_nodal_records = audit_results.get("nodal_data", [])
        total_count = len(all_nodal_records)

        # Sort all mesh nodes descending by stress magnitude and take the Top 20
        top_critical_nodes = sorted(
            all_nodal_records, 
            key=lambda item: item.get("stress_MPa", 0.0), 
            reverse=True
        )[:20]

        if top_critical_nodes:
            elements.append(Paragraph(
                f"<i>Displaying top {len(top_critical_nodes)} critical nodal sites out of {total_count} total mesh nodes evaluated (ranked by peak Von Mises stress). Full mesh dataset exported to <b>full_nodal_stress_report.csv</b>.</i>", 
                body_style
            ))
            elements.append(Spacer(1, 6))

            long_table_data = [
                [
                    Paragraph("Node ID", table_header_style),
                    Paragraph("Location (X, Y)", table_header_style),
                    Paragraph("Von Mises Stress", table_header_style),
                    Paragraph("Strain Value", table_header_style),
                    Paragraph("Damage Status", table_header_style)
                ]
            ]

            for node in top_critical_nodes:
                node_id = node.get("id", "N/A")
                x, y = node.get("coords", (0.0, 0.0))
                stress = node.get("stress_MPa", 0.0)
                strain = node.get("strain_micro", 0.0)
                is_cracked = node.get("is_cracked", False)

                status_text = (
                    "<font color='red'><b>CRACKED</b></font>" 
                    if is_cracked else 
                    "<font color='#D97706'><b>HIGH STRESS</b></font>"
                )

                long_table_data.append([
                    Paragraph(f"Node #{node_id}", body_style),
                    Paragraph(f"({x:.2f}m, {y:.2f}m)", body_style),
                    Paragraph(f"{stress:.2f} MPa", body_style),
                    Paragraph(f"{strain:.1f} µε", body_style),
                    Paragraph(status_text, body_style)
                ])

            long_table = Table(long_table_data, colWidths=[65, 115, 130, 110, 100], repeatRows=1)
            long_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A365D')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
            ]))

            elements.append(long_table)
        else:
            elements.append(Paragraph("<i>No nodal stress data recorded for this run.</i>", body_style))

        # Build document using NumberedCanvas
        doc.build(elements, canvasmaker=NumberedCanvas)
        print(f"Executive PDF report successfully generated: {self.output_pdf_path}")


if __name__ == "__main__":
    sample_metrics = {
        "status": "CONVERGED",
        "backend": "OpenSeesPy C++ (Incremental Newton-Raphson)",
        "total_steps": 50,
        "completed_steps": 50,
        "max_load_kN": 100.0,
        "max_displacement_mm": 0.0136,
        "curve_plot_path": "output/pushover_capacity_curve.png",
        "damaged_nodes_count": 1,
        "total_nodes_count": 121,
        "mesh_damage_ratio_percent": 0.83,
        "capacity_reduction_percent": 1.53
    }
    generator = StructuralReportPDFGenerator()
    generator.generate(sample_metrics)