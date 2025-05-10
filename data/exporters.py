import streamlit as st
import pandas as pd
from datetime import datetime
from fpdf import FPDF


def generate_pdf_report(title, sections):
    """
    Generate a PDF report with multiple sections.

    Args:
        title (str): The title of the report
        sections (list): List of dictionaries containing section data
                         Each dict should have 'title' and 'content' keys

    Returns:
        bytes: PDF data that can be used with st.download_button
    """
    pdf = FPDF()
    pdf.add_page()

    # Add title
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, title, 0, 1, 'C')
    pdf.ln(5)

    # Add each section
    for section in sections:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, section.get('title', 'Section'), 0, 1)

        pdf.set_font("Arial", size=10)
        content = section.get('content', '')
        # Encode to latin-1 to handle most common characters
        # This is a limitation of FPDF, consider using FPDF2 for better unicode support
        safe_content = content.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 6, safe_content)
        pdf.ln(5)

    # Return PDF as bytes
    return pdf.output(dest='S').encode('latin-1')


def download_pdf_button(title, sections, filename=None):
    """
    Create a download button for a PDF report.

    Args:
        title (str): The title of the report
        sections (list): List of dictionaries containing section data
        filename (str, optional): Custom filename. If None, a timestamped name will be used.
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{title.lower().replace(' ', '_')}_{timestamp}.pdf"

    try:
        pdf_data = generate_pdf_report(title, sections)
        st.download_button(
            label="Click to Download PDF",
            data=pdf_data,
            file_name=filename,
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Error generating PDF: {e}")


def download_csv_button(df, filename=None):
    """
    Create a download button for a CSV file from a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to export
        filename (str, optional): Custom filename. If None, a timestamped name will be used.
    """
    if df is None or df.empty:
        st.warning("No data available to download.")
        return

    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"exported_data_{timestamp}.csv"

    try:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error generating CSV: {e}")