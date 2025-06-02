"""
Data export utilities for the banking compliance application.
Cloud-compatible version without PDF dependencies.
"""
import streamlit as st
import pandas as pd
from io import BytesIO
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
    Create a download button for reports.
    Since PDF libraries may not be available, we create text reports.
    
    Args:
        title (str): Report title
        sections (list): List of dictionaries with 'title' and 'content' keys
        filename (str): Name of the downloaded file
        button_text (str): Text to display on the button
    """
    try:
        # Create text report instead of PDF
        report_content = f"{title}\n"
        report_content += "=" * len(title) + "\n"
        report_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add sections
        for section in sections:
            section_title = section.get('title', 'Section')
            section_content = section.get('content', '')
            
            report_content += f"\n{section_title}\n"
            report_content += "-" * len(section_title) + "\n"
            report_content += f"{section_content}\n\n"
        
        # Create download button for text file
        text_filename = filename.replace('.pdf', '.txt')
        st.download_button(
            label=f"{button_text} (Text Format)",
            data=report_content.encode('utf-8'),
            file_name=text_filename,
            mime="text/plain"
        )
        
        # Also show the content in an expandable section
        with st.expander("📄 View Report Content"):
            st.text_area("Report Content", report_content, height=300, key=f"report_{hash(title)}")
        
        st.info("💡 Report available as text file download.")
        
    except Exception as e:
        st.error(f"Error generating report: {e}")
        # Fallback: show content directly
        st.subheader(title)
        for section in sections:
            st.write(f"**{section.get('title', 'Section')}**")
            st.write(section.get('content', ''))


def create_excel_download(df, filename="data_export.xlsx", sheet_name="Data"):
    """
    Create an Excel download button.
    
    Args:
        df (pd.DataFrame): Data to export
        filename (str): Name of the downloaded file
        sheet_name (str): Name of the Excel sheet
    """
    if df is not None and not df.empty:
        try:
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
        except Exception as e:
            st.warning(f"Excel export failed: {e}. Using CSV instead.")
            download_csv_button(df, filename.replace('.xlsx', '.csv'), "Download CSV")
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
                    try:
                        content += f"- {col}: Mean={df[col].mean():.2f}, "
                        content += f"Min={df[col].min():.2f}, Max={df[col].max():.2f}\n"
                    except:
                        content += f"- {col}: Statistics unavailable\n"
            
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
                    try:
                        df_copy = df.copy()
                        df_copy['Dataset'] = name
                        combined_data.append(df_copy)
                    except Exception as e:
                        st.warning(f"Could not process dataset {name}: {e}")
            
            if combined_data:
                try:
                    combined_df = pd.concat(combined_data, ignore_index=True)
                    download_csv_button(combined_df, f"{title.replace(' ', '_').lower()}_combined.csv", "Download Combined CSV")
                except Exception as e:
                    st.warning(f"Could not create combined dataset: {e}")
