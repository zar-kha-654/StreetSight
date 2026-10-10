
from io import BytesIO
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle,
)


def generate_evidence_pdf(analysis, decisions):
    """Create a downloadable PDF from analyzed image evidence."""

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="StreetSight Evidence Report",
        author="StreetSight",
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="StreetSightTitle",
            parent=styles["Title"],
            fontSize=23,
            leading=28,
            textColor=colors.HexColor("#10233F"),
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="StreetSightSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#566477"),
            alignment=TA_CENTER,
        )
    )

    story = []

    report_id = f"SS-{analysis['image_hash'].upper()}"

    story.append(
        Paragraph("StreetSight", styles["StreetSightTitle"])
    )
    story.append(
        Paragraph(
            "See the damage. Verify the evidence. Report the problem.",
            styles["StreetSightSubtitle"],
        )
    )
    story.append(Spacer(1, 8 * mm))

    story.append(
        Paragraph("Evidence report", styles["Heading1"])
    )

    
    metadata = [
        ["Report reference", report_id],
        ["Analysis time (UTC)", analysis["timestamp"]],
        ["Road or street", analysis.get("road_name") or "Not provided"],
        ["Area or neighborhood", analysis.get("area_name") or "Not provided"],
        ["City", analysis.get("city_name") or "Not provided"],
        ["Nearby landmark", analysis.get("landmark") or "Not provided"],
        ["Detector", "Pretrained pothole object detector"],
        ["Predictions", str(len(analysis["detections"]))],
    ]

    metadata_table = Table(
        metadata,
        colWidths=[48 * mm, 112 * mm],
    )

    metadata_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1),
                 colors.HexColor("#E9EEF5")),
                ("TEXTCOLOR", (0, 0), (-1, -1),
                 colors.HexColor("#10233F")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4,
                 colors.HexColor("#D4DCE7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(metadata_table)
    story.append(Spacer(1, 8 * mm))

    # Images are resized to fit the printable page width.
    available_width = A4[0] - 36 * mm
    max_image_height = 105 * mm

    for title, image_key in [
        ("Original photograph", "original"),
        ("AI-annotated photograph", "annotated"),
    ]:
        story.append(
            Paragraph(title, styles["Heading2"])
        )

        image_buffer = BytesIO()
        analysis[image_key].convert("RGB").save(
            image_buffer,
            format="JPEG",
            quality=90,
        )
        image_buffer.seek(0)

        pdf_image = PDFImage(
            image_buffer,
            width=available_width,
            height=max_image_height,
            kind="proportional",
        )

        story.append(pdf_image)
        story.append(Spacer(1, 5 * mm))

    story.append(
        Paragraph("Prediction review", styles["Heading1"])
    )

    review_data = [
        ["#", "Predicted class", "Confidence", "User decision"]
    ]

    for detection in analysis["detections"]:
        detection_id = detection["id"]

        review_data.append(
            [
                str(detection_id + 1),
                str(detection["class"]),
                f"{detection['confidence']:.1%}",
                decisions.get(detection_id, "Pending"),
            ]
        )

    review_table = Table(
        review_data,
        colWidths=[12 * mm, 47 * mm, 32 * mm, 55 * mm],
        repeatRows=1,
    )

    review_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0),
                 colors.HexColor("#10233F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4,
                 colors.HexColor("#D4DCE7")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(review_table)
    story.append(Spacer(1, 8 * mm))

    accepted = sum(
        decisions.get(d["id"], "Pending") == "Accepted"
        for d in analysis["detections"]
    )
    rejected = sum(
        decisions.get(d["id"], "Pending") == "Rejected"
        for d in analysis["detections"]
    )
    pending = len(analysis["detections"]) - accepted - rejected

    story.append(
        Paragraph(
            f"<b>Verification summary:</b> "
            f"{accepted} accepted, {rejected} rejected, "
            f"{pending} pending.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 5 * mm))

    story.append(
        Paragraph("Important limitations", styles["Heading2"])
    )
    story.append(
        Paragraph(
            "This report records predictions from a pretrained "
            "computer-vision model and review decisions entered by "
            "a user. Predictions can be incorrect or incomplete. "
            "User review is not independent verification of actual "
            "road conditions. A lack of detections does not mean "
            "that a road is safe. The location, if supplied, is "
            "user-entered and has not been independently verified. "
            "This report is not an official inspection or government "
            "case record.",
            styles["Normal"],
        )
    )

    story.append(Spacer(1, 5 * mm))
    story.append(
        Paragraph(
            f"PDF generated at "
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}.",
            styles["Italic"],
        )
    )

    doc.build(story)

    buffer.seek(0)
    return buffer.getvalue()
