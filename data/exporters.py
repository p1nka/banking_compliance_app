"""
Data export utilities for the banking compliance application.
"""
import streamlit as st
import pandas as pd
from fpdf2 import FPDF
from io import BytesIO
import base64
from datetime import datetime


def download_csv_button(df, filename="data_export.csv", button_text="Download CSV"):
    """
    Create a download button for CSV data.
    
    Args:
        df (pd.DataFrame): Data to export
        filename (str): Name of the downloaded file
        button_text (str): Text to display on the button
    """
    if df is not None and not df.empty:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=button_text,
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )
    else:
        st.warning("No data available for CSV export")


def download_pdf_button(title, sections, filename="report.pdf", button_text="Download PDF Report"):
    """
    Create a download button for PDF reports.
    
    Args:
        title (str): Report title
        sections (list): List of dictionaries with 'title' and 'content' keys
        filename (str): Name of the downloaded file
        button_text (str): Text to display on the button
    """
    try:
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        
        # Add title
        pdf.cell(0, 10, title, ln=True, align='C')
        pdf.ln(10)
        
        # Add sections
        for section in sections:
            # Section title
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, section.get('title', 'Section'), ln=True)
            pdf.ln(5)
            
            # Section content
            pdf.set_font('Arial', '', 10)
            content = section.get('content', '')
            
            # Handle long content by splitting into lines
            lines = content.split('\n')
            for line in lines:
                # Wrap long lines
                if len(line) > 80:
                    words = line.split(' ')
                    current_line = ''
                    for word in words:
                        if len(current_line + word) < 80:
                            current_line += word + ' '
                        else:
                            if current_line:
                                pdf.cell(0, 5, current_line.strip(), ln=True)
                            current_line = word + ' '
                    if current_line:
                        pdf.cell(0, 5, current_line.strip(), ln=True)
                else:
                    pdf.cell(0, 5, line, ln=True)
            
            pdf.ln(5)
        
        # Generate PDF data
        pdf_data = pdf.output(dest='S').encode('latin1')
        
        # Create download button
        st.download_button(
            label=button_text,
            data=pdf_data,
            file_name=filename,
            mime="application/pdf"
        )
        
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        st.warning("PDF generation failed. You can copy the text content instead.")


def create_excel_download(df, filename="data_export.xlsx", sheet_name="Data"):
    """
    Create an Excel download button.
    
    Args:
        df (pd.DataFrame): Data to export
        filename (str): Name of the downloaded file
        sheet_name (str): Name of the Excel sheet
    """
    if df is not None and not df.empty:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        excel_data = output.getvalue()
        
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("No data available for Excel export")


def export_data_summary(data_dict, title="Data Summary"):
    """
    Export a summary of multiple datasets.
    
    Args:
        data_dict (dict): Dictionary with dataset names as keys and DataFrames as values
        title (str): Title for the summary
    """
    summary_sections = []
    
    # Add overview section
    overview = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    overview += f"Number of datasets: {len(data_dict)}\n\n"
    
    for name, df in data_dict.items():
        if df is not None and not df.empty:
            overview += f"- {name}: {len(df)} records, {len(df.columns)} columns\n"
        else:
            overview += f"- {name}: No data\n"
    
    summary_sections.append({
        "title": "Overview",
        "content": overview
    })
    
    # Add detailed sections for each dataset
    for name, df in data_dict.items():
        if df is not None and not df.empty:
            content = f"Records: {len(df)}\n"
            content += f"Columns: {', '.join(df.columns)}\n\n"
            
            # Add basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                content += "Numeric Summary:\n"
                for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                    content += f"- {col}: Mean={df[col].mean():.2f}, "
                    content += f"Min={df[col].min():.2f}, Max={df[col].max():.2f}\n"
            
            summary_sections.append({
                "title": f"Dataset: {name}",
                "content": content
            })
    
    # Create download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        download_pdf_button(title, summary_sections, f"{title.replace(' ', '_').lower()}_summary.pdf")
    
    with col2:
        if data_dict:
            # Create a combined CSV with all datasets
            combined_data = []
            for name, df in data_dict.items():
                if df is not None and not df.empty:
                    df_copy = df.copy()
                    df_copy['Dataset'] = name
                    combined_data.append(df_copy)
            
            if combined_data:
                combined_df = pd.concat(combined_data, ignore_index=True)
                download_csv_button(combined_df, f"{title.replace(' ', '_').lower()}_combined.csv", "Download Combined CSV")
