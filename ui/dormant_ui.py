import streamlit as st
from datetime import datetime, timedelta, date
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import SESSION_COLUMN_MAPPING
from agents.dormant import (
    check_safe_deposit, check_investment_inactivity,
    check_fixed_deposit_inactivity, check_general_inactivity,
    check_unreachable_dormant, run_all_dormant_checks,
    convert_foreign_currencies, prepare_central_bank_transfer
)
from database.operations import save_summary_to_db
from data.exporters import download_pdf_button, download_csv_button
from ai.llm import (

    DORMANT_SUMMARY_PROMPT,
    OBSERVATION_PROMPT,
    TREND_PROMPT,
    NARRATION_PROMPT,
    ACTION_PROMPT
)


def preprocess_dataset(df):
    """
    Preprocess the dataset to ensure it has the required columns with proper formats.

    Args:
        df (pandas.DataFrame): The original dataset

    Returns:
        pandas.DataFrame: Processed dataset ready for dormant account analysis
    """
    processed_df = df.copy()

    # Create standardized column mappings based on your CSV structure
    if 'Date_Last_Customer_Communication_Any_Type' in processed_df.columns:
        processed_df['Last_Transaction_Date'] = processed_df['Date_Last_Customer_Communication_Any_Type']
    elif 'Date_Last_Cust_Initiated_Activity' in processed_df.columns:
        processed_df['Last_Transaction_Date'] = processed_df['Date_Last_Cust_Initiated_Activity']

    # Map contact attempt columns
    if 'Bank_Contact_Attempted_Post_Dormancy_Trigger' in processed_df.columns:
        # Create Yes/No contact attempt columns needed by dormant.py functions
        processed_df['Email_Contact_Attempt'] = processed_df['Bank_Contact_Attempted_Post_Dormancy_Trigger'].apply(
            lambda x: 'Yes' if x == 'Yes' else 'No')
        processed_df['SMS_Contact_Attempt'] = processed_df['Bank_Contact_Attempted_Post_Dormancy_Trigger'].apply(
            lambda x: 'Yes' if x == 'Yes' else 'No')
        processed_df['Phone_Call_Attempt'] = processed_df['Bank_Contact_Attempted_Post_Dormancy_Trigger'].apply(
            lambda x: 'Yes' if x == 'Yes' else 'No')

    # If there's a status column, map it to Account_Status
    if 'Expected_Account_Dormant' in processed_df.columns:
        processed_df['Account_Status'] = processed_df['Expected_Account_Dormant'].apply(
            lambda x: 'dormant' if x == 'Yes' else 'active')

    # Map balance for high-value detection
    if 'Current_Balance' in processed_df.columns:
        processed_df['Balance'] = processed_df['Current_Balance']
    elif 'Unclaimed_Item_Amount' in processed_df.columns:
        processed_df['Balance'] = processed_df['Unclaimed_Item_Amount']

    # Convert date strings to datetime objects
    date_columns = [
        'Last_Transaction_Date',
        'Date_Last_Customer_Communication_Any_Type',
        'Date_Last_Cust_Initiated_Activity',
        'Date_Last_Bank_Contact_Attempt',
        'Account_Creation_Date'
    ]

    for col in date_columns:
        if col in processed_df.columns:
            try:
                processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
            except Exception as e:
                st.warning(f"Failed to convert {col} to datetime: {e}")

    return processed_df


def render_dormant_analyzer(df, llm):
    """
    Render the Dormant Account Analyzer UI.

    Args:
        df (pandas.DataFrame): The account data to analyze
        llm: The LLM model for generating insights
    """
    st.subheader("🏦 Dormant Account Analysis")

    # Preprocess dataset to ensure it has the required columns
    processed_df = preprocess_dataset(df)

    # Store processed DataFrame in session state
    st.session_state.processed_df = processed_df

    # Create both 3-year and 5-year thresholds
    threshold_3y = datetime.now() - timedelta(days=3 * 365)  # 3 years inactivity threshold
    threshold_5y = datetime.now() - timedelta(days=5 * 365)  # 5 years for Central Bank transfer

    st.info(
        "Analysis uses 3-year threshold for dormancy detection and 5-year threshold for Central Bank transfer eligibility.")

    agent_option = st.selectbox(
        "🧭 Choose Dormant Detection Agent",
        [
            "📊 Summarized Dormant Analysis",
            "🔐 Safe Deposit Box Agent",
            "💼 Investment Inactivity Agent",
            "🏦 Fixed Deposit Agent",
            "📉 3-Year General Inactivity Agent",
            "📵 Unreachable + No Active Accounts Agent",
            "🏛️ Central Bank Transfer Report"
        ],
        key="dormant_agent_selector"
    )

    # Handle the summarized dormant analysis option
    if agent_option == "📊 Summarized Dormant Analysis":
        render_summarized_dormant_analysis(processed_df, threshold_3y, threshold_5y, llm)
    elif agent_option == "🏛️ Central Bank Transfer Report":
        render_central_bank_transfer_report(processed_df, threshold_3y, threshold_5y, llm)
    else:
        # Handle individual agent options
        render_individual_dormant_agent(processed_df, agent_option, threshold_3y, threshold_5y, llm)


def render_summarized_dormant_analysis(df, threshold_3y, threshold_5y, llm):
    """Render the summarized dormant analysis UI."""
    st.subheader("📈 Summarized Dormant Analysis Results")

    if st.button("📊 Run Summarized Dormant Analysis", key="run_summary_dormant_button"):
        with st.spinner("Running all dormant checks..."):
            try:
                # Run the analysis
                results = run_all_dormant_checks(df, threshold_3y, threshold_5y)

                # Store results in session state for later reference
                st.session_state.dormant_summary_results = results

                # Display the numerical summary
                st.subheader("🔢 Dormant Account Summary (3-Year Threshold)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Uncontacted Safe Deposit (>3y)",
                        results["sd"]["count"],
                        help=results["sd"]["desc"]
                    )
                    st.metric(
                        "General Inactivity (>3y)",
                        results["gen"]["count"],
                        help=results["gen"]["desc"]
                    )
                with col2:
                    st.metric(
                        "Uncontacted Investment (>3y)",
                        results["inv"]["count"],
                        help=results["inv"]["desc"]
                    )
                    st.metric(
                        "Unreachable & 'Dormant'",
                        results["unr"]["count"],
                        help=results["unr"]["desc"]
                    )
                with col3:
                    st.metric(
                        "Fixed Deposit Inactivity (>3y)",
                        results["fd"]["count"],
                        help=results["fd"]["desc"]
                    )

                # Display Central Bank transfer eligibility (5-year threshold)
                st.subheader("🏛️ Central Bank Transfer Eligibility (5-Year Threshold)")
                transfer_col1, transfer_col2, transfer_col3 = st.columns(3)

                with transfer_col1:
                    sd_transfer_count = len(results["sd"]["transfer_df"]) if results["sd"][
                                                                                 "transfer_df"] is not None else 0
                    st.metric(
                        "Safe Deposit Transfer-Eligible",
                        sd_transfer_count,
                        help="Accounts eligible for Central Bank transfer (>5y)"
                    )

                    gen_transfer_count = len(results["gen"]["transfer_df"]) if results["gen"][
                                                                                   "transfer_df"] is not None else 0
                    st.metric(
                        "General Accounts Transfer-Eligible",
                        gen_transfer_count,
                        help="Accounts eligible for Central Bank transfer (>5y)"
                    )

                with transfer_col2:
                    inv_transfer_count = len(results["inv"]["transfer_df"]) if results["inv"][
                                                                                   "transfer_df"] is not None else 0
                    st.metric(
                        "Investment Transfer-Eligible",
                        inv_transfer_count,
                        help="Accounts eligible for Central Bank transfer (>5y)"
                    )

                    unr_transfer_count = len(results["unr"]["transfer_df"]) if results["unr"][
                                                                                   "transfer_df"] is not None else 0
                    st.metric(
                        "Unreachable Transfer-Eligible",
                        unr_transfer_count,
                        help="Accounts eligible for Central Bank transfer (>5y)"
                    )

                with transfer_col3:
                    fd_transfer_count = len(results["fd"]["transfer_df"]) if results["fd"][
                                                                                 "transfer_df"] is not None else 0
                    st.metric(
                        "Fixed Deposit Transfer-Eligible",
                        fd_transfer_count,
                        help="Accounts eligible for Central Bank transfer (>5y)"
                    )

                    total_transfer_eligible = sd_transfer_count + inv_transfer_count + fd_transfer_count + gen_transfer_count + unr_transfer_count
                    st.metric(
                        "Total Transfer-Eligible",
                        total_transfer_eligible,
                        help="Total accounts eligible for Central Bank transfer (>5y)"
                    )

                # Prepare input text for AI summary
                summary_input_text = (
                    f"Dormant Analysis Findings ({results['total_accounts']} total accounts analyzed):\n\n"
                    f"3-Year Dormancy Threshold Results:\n"
                    f"- {results['sd']['desc']}\n"
                    f"- {results['inv']['desc']}\n"
                    f"- {results['fd']['desc']}\n"
                    f"- {results['gen']['desc']}\n"
                    f"- {results['unr']['desc']}\n\n"
                    f"5-Year Central Bank Transfer Eligibility:\n"
                    f"- Safe Deposit Transfer-Eligible: {sd_transfer_count} accounts\n"
                    f"- Investment Transfer-Eligible: {inv_transfer_count} accounts\n"
                    f"- Fixed Deposit Transfer-Eligible: {fd_transfer_count} accounts\n"
                    f"- General Accounts Transfer-Eligible: {gen_transfer_count} accounts\n"
                    f"- Unreachable Transfer-Eligible: {unr_transfer_count} accounts\n"
                    f"- Total Transfer-Eligible: {total_transfer_eligible} accounts"
                )

                st.subheader("📝 Narrative Summary")
                narrative_summary = summary_input_text  # Default to raw data in case AI fails

                if llm:
                    try:
                        with st.spinner("Generating AI Summary..."):
                            # Use the predefined summary prompt template
                            summary_prompt_template = PromptTemplate.from_template(DORMANT_SUMMARY_PROMPT)
                            summary_chain = summary_prompt_template | llm | StrOutputParser()
                            narrative_summary = summary_chain.invoke({
                                "analysis_details": summary_input_text
                            })
                            st.markdown(narrative_summary)
                            st.session_state.dormant_narrative_summary = narrative_summary  # Store for PDF
                    except Exception as llm_e:
                        st.error(f"AI summary generation failed: {llm_e}")
                        fallback_summary = get_fallback_response("dormant_summary")
                        st.warning(fallback_summary)
                        st.text_area("Raw Findings:", summary_input_text, height=150)
                        st.session_state.dormant_narrative_summary = f"{fallback_summary}\n\nRaw Findings:\n{summary_input_text}"
                else:
                    fallback_summary = get_fallback_response("dormant_summary")
                    st.warning(fallback_summary)
                    st.text_area("Raw Findings:", summary_input_text, height=150)
                    st.session_state.dormant_narrative_summary = f"{fallback_summary}\n\nRaw Findings:\n{summary_input_text}"

                # High-Value Dormant Account section
                st.subheader("💰 High-Value Dormant Accounts (≥ AED 25,000)")

                # Combine all dormant accounts
                all_dormant = pd.concat([
                    results["sd"]["df"] if not results["sd"]["df"].empty else pd.DataFrame(),
                    results["inv"]["df"] if not results["inv"]["df"].empty else pd.DataFrame(),
                    results["fd"]["df"] if not results["fd"]["df"].empty else pd.DataFrame(),
                    results["gen"]["df"] if not results["gen"]["df"].empty else pd.DataFrame(),
                    results["unr"]["df"] if not results["unr"]["df"].empty else pd.DataFrame()
                ])

                # Check if 'Balance' column exists for high-value filtering
                if 'Balance' in all_dormant.columns:
                    high_value = all_dormant[all_dormant['Balance'] >= 25000]
                    st.metric(
                        "High-Value Dormant Accounts",
                        len(high_value),
                        help="Dormant accounts with balance ≥ AED 25,000"
                    )

                    if not high_value.empty and st.checkbox("View High-Value Dormant Accounts"):
                        st.dataframe(high_value.head(15))
                else:
                    st.warning("Balance information not available to identify high-value accounts.")

                # Export options
                st.subheader("⬇️ Export Summary")

                # Create report sections for PDF
                sections = [
                    {
                        "title": "Dormant Account Summary (3-Year Threshold)",
                        "content": f"- {results['sd']['desc']}\n- {results['inv']['desc']}\n- {results['fd']['desc']}\n- {results['gen']['desc']}\n- {results['unr']['desc']}"
                    },
                    {
                        "title": "Central Bank Transfer Eligibility (5-Year Threshold)",
                        "content": f"- Safe Deposit Transfer-Eligible: {sd_transfer_count} accounts\n- Investment Transfer-Eligible: {inv_transfer_count} accounts\n- Fixed Deposit Transfer-Eligible: {fd_transfer_count} accounts\n- General Accounts Transfer-Eligible: {gen_transfer_count} accounts\n- Unreachable Transfer-Eligible: {unr_transfer_count} accounts\n- Total Transfer-Eligible: {total_transfer_eligible} accounts"
                    },
                    {
                        "title": "Narrative Summary (AI Generated or Raw Findings)",
                        "content": st.session_state.get('dormant_narrative_summary',
                                                        "Summary not generated or AI failed.")
                    }
                ]

                # Add download button
                download_pdf_button(
                    "Dormant Account Analysis Summary Report",
                    sections,
                    f"dormant_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                )

            except Exception as e:
                st.error(f"Error running dormant account analysis: {e}")
                st.exception(e)


def render_central_bank_transfer_report(df, threshold_3y, threshold_5y, llm):
    """Render a dedicated UI for Central Bank transfer reporting."""
    st.subheader("🏛️ Central Bank Transfer Report")

    if st.button("Generate Central Bank Transfer Report", key="central_bank_report_button"):
        with st.spinner("Analyzing accounts eligible for Central Bank transfer..."):
            try:
                results = run_all_dormant_checks(df, threshold_3y, threshold_5y)

                # Collect all transfer-eligible accounts
                transfer_accounts = pd.DataFrame()

                for key in ["sd", "inv", "fd", "gen", "unr"]:
                    if results[key]["transfer_df"] is not None and not results[key]["transfer_df"].empty:
                        category_df = results[key]["transfer_df"].copy()
                        category_df['Category'] = key.upper()  # Add category identifier
                        transfer_accounts = pd.concat([transfer_accounts, category_df])

                total_transfer_eligible = len(transfer_accounts)

                if total_transfer_eligible > 0:
                    # Convert foreign currencies to AED
                    try:
                        if 'Currency' in transfer_accounts.columns and not all(transfer_accounts['Currency'] == 'AED'):
                            transfer_accounts = convert_foreign_currencies(transfer_accounts)
                    except Exception as conv_err:
                        st.warning(f"Currency conversion warning: {conv_err}")

                    # Prepare for Central Bank submission format
                    cb_formatted = prepare_central_bank_transfer(transfer_accounts)

                    st.success(f"Found {total_transfer_eligible} accounts eligible for Central Bank transfer.")

                    # Display statistics
                    if 'Balance' in transfer_accounts.columns:
                        total_balance = transfer_accounts['Balance'].sum()
                        avg_balance = transfer_accounts['Balance'].mean()
                        max_balance = transfer_accounts['Balance'].max()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Transfer Balance", f"AED {total_balance:,.2f}")
                        with col2:
                            st.metric("Average Account Balance", f"AED {avg_balance:,.2f}")
                        with col3:
                            st.metric("Maximum Account Balance", f"AED {max_balance:,.2f}")

                    # Display formatted data
                    if st.checkbox("View Central Bank Report Data (First 15 rows)"):
                        st.dataframe(cb_formatted.head(15))

                    # Breakdown by account type
                    st.subheader("Breakdown by Account Type")

                    if not transfer_accounts.empty and 'Category' in transfer_accounts.columns:
                        category_counts = transfer_accounts['Category'].value_counts()
                        st.bar_chart(category_counts)

                    # CBUAE format note
                    st.info(
                        "The exported file will follow the Central Bank of UAE formatting requirements per Article 8 of the Dormant Accounts Regulation.")

                    # Download options
                    st.subheader("⬇️ Export Central Bank Report")

                    download_csv_button(
                        cb_formatted,
                        f"central_bank_transfer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )

                    # PDF summary report
                    sections = [
                        {
                            "title": "Central Bank Transfer Summary",
                            "content": f"- Total Accounts: {total_transfer_eligible}\n- Total Balance: AED {total_balance if 'Balance' in transfer_accounts.columns else 'N/A'}\n- Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        },
                        {
                            "title": "Account Type Breakdown",
                            "content": "\n".join([f"- {cat}: {count} accounts" for cat, count in
                                                  category_counts.items()]) if 'Category' in transfer_accounts.columns else "Category breakdown not available"
                        },
                        {
                            "title": "CBUAE Compliance Note",
                            "content": "This report follows the Central Bank of UAE requirements per Article 8 of the Dormant Accounts Regulation. Accounts included have been inactive for 5+ years with no customer contact."
                        }
                    ]

                    download_pdf_button(
                        "Central Bank Transfer Report Summary",
                        sections,
                        f"cb_transfer_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    )

                else:
                    st.info("No accounts are currently eligible for Central Bank transfer (>5 years dormant).")

            except Exception as e:
                st.error(f"Error generating Central Bank transfer report: {e}")
                st.exception(e)


def render_individual_dormant_agent(df, agent_option, threshold_3y, threshold_5y, llm):
    """Render the UI for an individual dormant agent."""
    st.subheader(f"Agent Task Results: {agent_option}")
    data_filtered = pd.DataFrame()
    data_transfer = pd.DataFrame()  # For 5-year threshold
    agent_desc = "Select an agent above."
    agent_executed = False

    agent_mapping = {
        "🔐 Safe Deposit Box Agent": check_safe_deposit,
        "💼 Investment Inactivity Agent": check_investment_inactivity,
        "🏦 Fixed Deposit Agent": check_fixed_deposit_inactivity,
        "📉 3-Year General Inactivity Agent": check_general_inactivity,
        "📵 Unreachable + No Active Accounts Agent": check_unreachable_dormant
    }

    if selected_agent := agent_mapping.get(agent_option):
        with st.spinner(f"Running {agent_option}..."):
            try:
                # Pass necessary args based on agent
                if agent_option in [
                    "🔐 Safe Deposit Box Agent",
                    "💼 Investment Inactivity Agent",
                    "🏦 Fixed Deposit Agent",
                    "📉 3-Year General Inactivity Agent"
                ]:
                    data_filtered, count, agent_desc, data_transfer = selected_agent(df, threshold_3y, threshold_5y)
                else:  # "📵 Unreachable + No Active Accounts Agent"
                    data_filtered, count, agent_desc, data_transfer = selected_agent(df)
                agent_executed = True

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Dormant Accounts (>3y)", count, help=agent_desc)
                with col2:
                    transfer_count = len(data_transfer) if data_transfer is not None else 0
                    st.metric("Transfer-Eligible (>5y)", transfer_count,
                              help="Accounts eligible for Central Bank transfer")

                st.markdown(f"**Agent Description:** {agent_desc}")

            except Exception as agent_error:
                st.error(f"Error running agent: {agent_error}")
                st.exception(agent_error)
                agent_executed = False

    if agent_executed:
        # Create tabs for 3-year and 5-year results
        dormant_tab, transfer_tab = st.tabs(["3-Year Dormancy", "5-Year Transfer Eligible"])

        with dormant_tab:
            if not data_filtered.empty:
                st.success(f"{len(data_filtered)} accounts identified as dormant (>3 years).")
                if st.checkbox(f"View first 15 dormant accounts for '{agent_option}'",
                               key=f"view_detected_{agent_option.replace(' ', '_')}"):
                    # Display the DataFrame with original column names if available
                    display_df = data_filtered.head(15).copy()
                    if SESSION_COLUMN_MAPPING in st.session_state and st.session_state[SESSION_COLUMN_MAPPING]:
                        try:
                            # Create a display mapping that only includes columns present in the data
                            display_columns_mapping = {
                                std_col: st.session_state[SESSION_COLUMN_MAPPING].get(std_col, std_col)
                                for std_col in display_df.columns if std_col in st.session_state[SESSION_COLUMN_MAPPING]
                            }
                            display_df.rename(columns=display_columns_mapping, inplace=True)
                        except Exception as e:
                            st.warning(f"Could not display original column names: {e}")

                    st.dataframe(display_df)
            elif len(data_filtered) == 0:
                st.info("No accounts matching the 3-year dormancy criteria were found.")

        with transfer_tab:
            if data_transfer is not None and not data_transfer.empty:
                st.success(f"{len(data_transfer)} accounts eligible for Central Bank transfer (>5 years).")
                if st.checkbox(f"View first 15 transfer-eligible accounts for '{agent_option}'",
                               key=f"view_transfer_{agent_option.replace(' ', '_')}"):
                    # Display the DataFrame with original column names if available
                    transfer_display_df = data_transfer.head(15).copy()
                    if SESSION_COLUMN_MAPPING in st.session_state and st.session_state[SESSION_COLUMN_MAPPING]:
                        try:
                            # Create a display mapping that only includes columns present in the data
                            transfer_display_columns_mapping = {
                                std_col: st.session_state[SESSION_COLUMN_MAPPING].get(std_col, std_col)
                                for std_col in transfer_display_df.columns if
                                std_col in st.session_state[SESSION_COLUMN_MAPPING]
                            }
                            transfer_display_df.rename(columns=transfer_display_columns_mapping, inplace=True)
                        except Exception as e:
                            st.warning(f"Could not display original column names: {e}")

                    st.dataframe(transfer_display_df)
            elif data_transfer is None or len(data_transfer) == 0:
                st.info("No accounts matching the 5-year transfer eligibility criteria were found.")

        # Generate insights with LLM if available
        if llm and not data_filtered.empty:
            sample_size = min(15, len(data_filtered))
            sample_data_csv = data_filtered.sample(n=sample_size).to_csv(index=False)

            # Initialize the prompt templates with our predefined prompts
            observation_prompt = PromptTemplate.from_template(OBSERVATION_PROMPT)
            trend_prompt = PromptTemplate.from_template(TREND_PROMPT)

            if st.button(f"Generate Insights for '{agent_option}'",
                         key=f"generate_insights_{agent_option.replace(' ', '_')}"):
                try:
                    with st.spinner("Running insight agents..."):
                        # Setup the LLM chains
                        output_parser = StrOutputParser()
                        observation_chain = observation_prompt | llm | output_parser
                        trend_chain = trend_prompt | llm | output_parser

                        # Generate observations and trends
                        obs_output = observation_chain.invoke({"data": sample_data_csv})
                        trend_output = trend_chain.invoke({"data": sample_data_csv})

                        # Generate summary and actions
                        narration_prompt = PromptTemplate.from_template(NARRATION_PROMPT)
                        action_prompt = PromptTemplate.from_template(ACTION_PROMPT)

                        narration_chain = narration_prompt | llm | output_parser
                        action_chain = action_prompt | llm | output_parser

                        final_insight = narration_chain.invoke({"observation": obs_output, "trend": trend_output})
                        action_output = action_chain.invoke({"observation": obs_output, "trend": trend_output})

                    # Store insights in session state
                    st.session_state[f'{agent_option}_insights'] = {
                        'observation': obs_output,
                        'trend': trend_output,
                        'summary': final_insight,
                        'actions': action_output
                    }

                    # Save to DB log
                    try:
                        if save_summary_to_db(obs_output, trend_output, final_insight, action_output):
                            st.success("Insights saved to insight log.")
                        else:
                            st.error("Failed to save insights to DB log.")
                    except Exception as e:
                        st.error(f"Error saving insights to DB: {e}")

                except Exception as insight_error:
                    st.error(f"Error generating insights: {insight_error}")
                    # Use fallback responses
                    obs_output = get_fallback_response("observation")
                    trend_output = get_fallback_response("trend")
                    final_insight = get_fallback_response("narration")
                    action_output = get_fallback_response("action")

                    # Store fallback insights
                    st.session_state[f'{agent_option}_insights'] = {
                        'observation': obs_output,
                        'trend': trend_output,
                        'summary': final_insight,
                        'actions': action_output
                    }

                    st.warning("Using fallback insights due to AI error.")

            # Display insights if they exist in session state
            if f'{agent_option}_insights' in st.session_state:
                insights = st.session_state[f'{agent_option}_insights']
                with st.expander("🔍 Observation Insight"):
                    st.markdown(insights['observation'])
                with st.expander("📊 Trend Insight"):
                    st.markdown(insights['trend'])
                with st.expander("📌 CXO Summary"):
                    st.markdown(insights['summary'])
                with st.expander("🚀 Recommended Actions"):
                    st.markdown(insights['actions'])

                # PDF export for individual agent insights
                st.subheader("⬇️ Export Insights")
                sections = [
                    {"title": "Observations", "content": insights['observation']},
                    {"title": "Trends", "content": insights['trend']},
                    {"title": "Executive Summary", "content": insights['summary']},
                    {"title": "Recommended Actions", "content": insights['actions']}
                ]

                download_pdf_button(
                    f"{agent_option} - Analysis Report",
                    sections,
                    f"{agent_option.replace(' ', '_').replace(':', '')}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                )

                # Add CSV download for identified accounts
                st.markdown("#### Download Dormant Account Data")
                download_csv_button(
                    data_filtered,
                    f"{agent_option.replace(' ', '_').replace(':', '')}_dormant_accounts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )

                # Add CSV download for transfer-eligible accounts
                if data_transfer is not None and not data_transfer.empty:
                    st.markdown("#### Download Transfer-Eligible Account Data")
                    download_csv_button(
                        data_transfer,
                        f"{agent_option.replace(' ', '_').replace(':', '')}_transfer_accounts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )

                elif len(data_filtered) > 0:
                    st.info(f"AI Assistant not available to generate insights.")

                    # Still provide CSV download even without AI
                    st.subheader("⬇️ Export Data")

                    st.markdown("#### Download Dormant Account Data")
                    download_csv_button(
                        data_filtered,
                        f"{agent_option.replace(' ', '_').replace(':', '')}_dormant_accounts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )

                    # Add CSV download for transfer-eligible accounts
                    if data_transfer is not None and not data_transfer.empty:
                        st.markdown("#### Download Transfer-Eligible Account Data")
                        download_csv_button(
                            data_transfer,
                            f"{agent_option.replace(' ', '_').replace(':', '')}_transfer_accounts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        )