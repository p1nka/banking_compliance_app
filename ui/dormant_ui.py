import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import SESSION_COLUMN_MAPPING
from agents.dormant import (
    check_safe_deposit, check_investment_inactivity,
    check_fixed_deposit_inactivity, check_demand_deposit_inactivity,
    check_bankers_cheques, check_transfer_to_central_bank,
    check_art3_process_required, check_contact_attempts_needed,
    run_all_dormant_checks
)
from database.operations import save_summary_to_db
from data.exporters import download_pdf_button, download_csv_button
from ai.llm import (
    get_fallback_response,
    DORMANT_SUMMARY_PROMPT,
    OBSERVATION_PROMPT,
    TREND_PROMPT,
    NARRATION_PROMPT,
    ACTION_PROMPT
)


def render_dormant_analyzer(df, llm):
    """
    Render the Dormant Account Analyzer UI based on  the UAE Central Bank regulations.

    Args:
        df (pandas.DataFrame): The account data to analyze
        llm: The LLM model for generating insights
    """
    st.subheader("🏦 UAE Dormant Account Analysis")

    # Set up column filters for necessary fields
    if st.checkbox("⚙️ Configure Data Fields", key="config_dormant_fields"):
        with st.expander("Column Mapping - Verify Required Fields"):
            column_display = pd.DataFrame({
                'Required Field': [
                    'Account_ID', 'Customer_ID', 'Account_Type', 'Currency','Account_Creation_Date', 'Current_Balance',
                    'Date_Last_Bank_Initiated_Activity', 'Date_Last_Customer_Communication_Any_Type',
                    'FTD_Maturity_Date', 'FTD_Auto_Renewal', 'Date_Last_FTD_Renewal_Claim_Request',
                    'Inv_Maturity_Redemption_Date', 'SDB_Charges_Outstanding',
                    'Date_SDB_Charges_Became_Outstanding', 'SDB_Tenant_Communication_Received',
                    'Unclaimed_Item_Trigger_Date', 'Unclaimed_Item_Amount',
                    'Date_Last_Cust_Initiated_Activity', 'Bank_Contact_Attempted_Post_Dormancy_Trigger',
                    'Date_Last_Bank_Contact_Attempt', 'Customer_Responded_to_Bank_Contact',
                    'Customer_Address_Known', 'Customer_Has_Active_Liability_Account',
                    'Customer_Has_Litigation_Regulatory_Reqs'
                ],
                'Present in Data': [col in df.columns for col in [
                    'Account_ID', 'Customer_ID', 'Account_Type', 'Currency','Account_Creation_Date', 'Current_Balance',
                    'Date_Last_Bank_Initiated_Activity', 'Date_Last_Customer_Communication_Any_Type',
                    'FTD_Maturity_Date', 'FTD_Auto_Renewal', 'Date_Last_FTD_Renewal_Claim_Request',
                    'Inv_Maturity_Redemption_Date', 'SDB_Charges_Outstanding',
                    'Date_SDB_Charges_Became_Outstanding', 'SDB_Tenant_Communication_Received',
                    'Unclaimed_Item_Trigger_Date', 'Unclaimed_Item_Amount',
                    'Date_Last_Cust_Initiated_Activity', 'Bank_Contact_Attempted_Post_Dormancy_Trigger',
                    'Date_Last_Bank_Contact_Attempt', 'Customer_Responded_to_Bank_Contact',
                    'Customer_Address_Known', 'Customer_Has_Active_Liability_Account',
                    'Customer_Has_Litigation_Regulatory_Reqs'
                ]]
            })
            st.dataframe(column_display)

            missing_columns = [col for col in column_display['Required Field']
                               if col not in df.columns]
            if missing_columns:
                st.warning(f"⚠️ Missing recommended columns: {', '.join(missing_columns)}")
                st.info("The analysis will still work but some UAE regulation checks may be skipped.")

    # Display information about UAE regulations
    with st.expander("ℹ️ UAE Central Bank Dormant Account Regulation Info"):
        st.markdown("""
        ### UAE Central Bank Dormant Account Regulation Summary

        This analysis follows the UAE Central Bank's Dormant Accounts Regulation which establishes requirements for:

        1. **Dormancy Criteria**:
           - Demand Deposit Accounts (Current/Savings/Call): 3 years without transactions or customer contact
           - Fixed Term Deposits: 3 years after maturity without renewal or customer contact
           - Investment Accounts: 3 years from maturity/redemption without customer contact
           - Safe Deposit Boxes: 3+ years with charges unpaid and no tenant response
           - Bankers Cheques/Drafts: 1 year unclaimed from issue date

        2. **Required Actions**:
           - Contact attempts (written, electronic, verbal)
           - 3-month waiting period for response
           - Transfer to dormant accounts ledger
           - After 5 years, transfer to Central Bank if no address known and no other active accounts

        3. **Customer Rights**:
           - Funds remain property of account holder/heirs indefinitely
           - Interest accrues until transfer to Central Bank
           - No fees should be charged on dormant accounts

        For complete details, refer to the UAE Central Bank's Dormant Accounts Regulation.
        """)

    # Calculate thresholds as per UAE Central Bank regulations
    report_date = datetime.now()
    three_year_threshold = report_date - timedelta(days=3 * 365)
    one_year_threshold = report_date - timedelta(days=365)
    five_year_threshold = report_date - timedelta(days=5 * 365)

    # Date ranges
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📅 3-Year Threshold (Standard Dormancy): {three_year_threshold.strftime('%Y-%m-%d')}")
        st.info(f"📅 1-Year Threshold (Bankers Cheques): {one_year_threshold.strftime('%Y-%m-%d')}")
    with col2:
        st.info(f"📅 5-Year Threshold (Central Bank Transfer): {five_year_threshold.strftime('%Y-%m-%d')}")
        st.info(f"📅 Report Date: {report_date.strftime('%Y-%m-%d')}")

    agent_option = st.selectbox(
        "🧭 Choose Dormant Detection Agent",
        [
            "📊 UAE Regulatory Dormant Analysis",
            "📉 Demand Deposit Accounts (3+ years)",
            "🏦 Fixed Deposit Accounts (3+ years)",
            "💼 Investment Accounts (3+ years)",
            "🔐 Safe Deposit Boxes (3+ years)",
            "🧾 Bankers Cheques & Drafts (1+ year)",
            "🏛️ Central Bank Transfer Eligible (5+ years)",
            "📋 Article 3 Process Required",
            "📱 Contact Attempts Needed"
        ],
        key="dormant_agent_selector"
    )

    # Handle the summarized dormant analysis option
    if agent_option == "📊 UAE Regulatory Dormant Analysis":
        render_summarized_dormant_analysis(df, report_date, llm)
    else:
        # Handle individual agent options
        render_individual_dormant_agent(df, agent_option, report_date, llm)


def render_summarized_dormant_analysis(df, report_date, llm):
    """
    Render the summarized dormant analysis UI based on UAE regulations.

    Args:
        df: DataFrame containing account data
        report_date: Current report date
        llm: Language model for generating insights
    """
    st.subheader("📈 UAE Regulatory Dormant Analysis Results")

    if st.button("📊 Run Comprehensive UAE Dormant Analysis", key="run_summary_dormant_button"):
        with st.spinner("Running all dormant checks as per UAE Central Bank regulations..."):
            results = run_all_dormant_checks(df, report_date)

        # Store results in session state for later reference
        st.session_state.dormant_summary_results = results

        # Display overall statistics
        st.subheader("📊 Overall Statistics")
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric(
                "Total Accounts Analyzed",
                f"{results['total_accounts']:,}",
                help="Total number of accounts in the dataset"
            )
        with stat_col2:
            st.metric(
                "Total Dormant Accounts",
                f"{results['statistics']['total_dormant']:,}",
                help="All accounts meeting dormancy criteria across all types"
            )
        with stat_col3:
            st.metric(
                "Dormancy Rate",
                f"{results['statistics']['dormant_percentage']}%",
                help="Percentage of total accounts that are dormant"
            )

        # Display the numerical summary
        st.subheader("🔢 Dormancy Categories")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Demand Deposits (3+ years)",
                results["dd"]["count"],
                help=results["dd"]["desc"]
            )
            st.metric(
                "Safe Deposit Boxes (3+ years)",
                results["sd"]["count"],
                help=results["sd"]["desc"]
            )
        with col2:
            st.metric(
                "Fixed Deposits (3+ years)",
                results["fd"]["count"],
                help=results["fd"]["desc"]
            )
            st.metric(
                "Investment Accounts (3+ years)",
                results["inv"]["count"],
                help=results["inv"]["desc"]
            )
        with col3:
            st.metric(
                "Bankers Cheques (1+ year)",
                results["chq"]["count"],
                help=results["chq"]["desc"]
            )
            st.metric(
                "Central Bank Transfer (5+ years)",
                results["cb"]["count"],
                help=results["cb"]["desc"]
            )

        # Display action required stats
        st.subheader("🔔 Action Required")
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            st.metric(
                "Article 3 Process Required",
                results["art3"]["count"],
                help=results["art3"]["desc"]
            )
        with action_col2:
            st.metric(
                "Contact Attempts Needed",
                results["con"]["count"],
                help=results["con"]["desc"]
            )

        # Prepare input text for AI summary
        summary_input_text = (
            f"UAE Central Bank Dormant Account Analysis as of {results['statistics']['report_date']}:\n\n"
            f"Overall Statistics:\n"
            f"- Total accounts analyzed: {results['total_accounts']:,}\n"
            f"- Total dormant accounts: {results['statistics']['total_dormant']:,} ({results['statistics']['dormant_percentage']}%)\n"
            f"- Eligible for Central Bank transfer: {results['cb']['count']:,} ({results['statistics']['cb_transfer_percentage']}% of dormant)\n\n"
            f"Dormancy Categories:\n"
            f"- {results['dd']['desc']}\n"
            f"- {results['fd']['desc']}\n"
            f"- {results['inv']['desc']}\n"
            f"- {results['sd']['desc']}\n"
            f"- {results['chq']['desc']}\n\n"
            f"Action Required:\n"
            f"- {results['art3']['desc']}\n"
            f"- {results['con']['desc']}\n\n"
            f"UAE Regulatory Requirements Reference:\n"
            f"- Accounts dormant after 3 years of inactivity with no customer contact\n"
            f"- Banks must attempt to contact dormant account holders\n"
            f"- After 5 years, accounts must be transferred to Central Bank if customer address unknown\n"
            f"- Bankers cheques/drafts become unclaimed after 1 year"
        )

        st.subheader("📝 Regulatory Compliance Summary")
        narrative_summary = summary_input_text  # Default to raw data in case AI fails

        if llm:
            try:
                with st.spinner("Generating AI Regulatory Summary..."):
                    # Use a modified prompt for UAE regulatory context
                    uae_summary_prompt = DORMANT_SUMMARY_PROMPT + "\n\nFocus on UAE regulatory compliance aspects, highlighting actions needed to comply with Central Bank requirements."
                    summary_prompt_template = PromptTemplate.from_template(uae_summary_prompt)
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

        # Export options
        st.subheader("⬇️ Export Summary")

        # Create report sections for PDF
        sections = [
            {
                "title": "UAE Central Bank Dormant Accounts Regulation",
                "content": "This report is generated based on the United Arab Emirates Central Bank's Dormant Accounts Regulation issued on January 15, 2020. It identifies accounts meeting dormancy criteria and required actions."
            },
            {
                "title": "Overall Statistics",
                "content": f"• Total accounts analyzed: {results['total_accounts']:,}\n• Total dormant accounts: {results['statistics']['total_dormant']:,} ({results['statistics']['dormant_percentage']}%)\n• Eligible for Central Bank transfer: {results['cb']['count']:,} ({results['statistics']['cb_transfer_percentage']}% of dormant)"
            },
            {
                "title": "Dormancy Categories",
                "content": f"• Demand Deposits (Current/Savings/Call): {results['dd']['count']} accounts\n• Fixed Deposits: {results['fd']['count']} accounts\n• Investment Accounts: {results['inv']['count']} accounts\n• Safe Deposit Boxes: {results['sd']['count']} boxes\n• Bankers Cheques/Drafts: {results['chq']['count']} instruments"
            },
            {
                "title": "Action Required",
                "content": f"• Article 3 Process Required: {results['art3']['count']} accounts\n• Contact Attempts Needed: {results['con']['count']} accounts\n• Central Bank Transfer Eligible: {results['cb']['count']} accounts"
            },
            {
                "title": "Regulatory Compliance Summary",
                "content": st.session_state.get('dormant_narrative_summary', "Summary not generated or AI failed.")
            },
            {
                "title": "Report Generation Information",
                "content": f"Report Date: {results['statistics']['report_date']}\nGenerated in compliance with UAE Central Bank Dormant Accounts Regulation (2020)"
            }
        ]

        # Add download button
        download_pdf_button(
            "UAE Dormant Account Regulatory Compliance Report",
            sections,
            f"uae_dormant_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

        # Offer CSV downloads for each account category
        with st.expander("Download Account Lists"):
            st.subheader("Export Specific Account Categories")

            csv_cols = st.columns(3)
            with csv_cols[0]:
                if results["dd"]["count"] > 0:
                    download_csv_button(
                        results["dd"]["df"],
                        f"dormant_demand_deposits_{datetime.now().strftime('%Y%m%d')}.csv",

                    )
                if results["inv"]["count"] > 0:
                    download_csv_button(
                        results["inv"]["df"],
                        f"dormant_investment_accounts_{datetime.now().strftime('%Y%m%d')}.csv"

                    )

            with csv_cols[1]:
                if results["fd"]["count"] > 0:
                    download_csv_button(
                        results["fd"]["df"],
                        f"dormant_fixed_deposits_{datetime.now().strftime('%Y%m%d')}.csv"

                    )
                if results["sd"]["count"] > 0:
                    download_csv_button(
                        results["sd"]["df"],
                        f"dormant_safe_deposit_boxes_{datetime.now().strftime('%Y%m%d')}.csv"

                    )

            with csv_cols[2]:
                if results["chq"]["count"] > 0:
                    download_csv_button(
                        results["chq"]["df"],
                        f"unclaimed_bankers_cheques_{datetime.now().strftime('%Y%m%d')}.csv"

                    )
                if results["cb"]["count"] > 0:
                    download_csv_button(
                        results["cb"]["df"],
                        f"central_bank_transfer_accounts_{datetime.now().strftime('%Y%m%d')}.csv"

                    )


def render_individual_dormant_agent(df, agent_option, report_date, llm):
    """
    Render the UI for an individual dormant agent based on UAE regulations.

    Args:
        df: DataFrame containing account data
        agent_option: Selected agent option from dropdown
        report_date: Current report date
        llm: Language model for generating insights
    """
    st.subheader(f"Agent Task Results: {agent_option}")

    # Calculate thresholds
    three_year_threshold = report_date - timedelta(days=3 * 365)
    one_year_threshold = report_date - timedelta(days=365)
    five_year_threshold = report_date - timedelta(days=5 * 365)

    data_filtered = pd.DataFrame()
    agent_desc = "Select an agent above."
    agent_executed = False

    # Define mapping for agents and their corresponding functions/thresholds
    agent_mapping = {
        "📉 Demand Deposit Accounts (3+ years)": (check_demand_deposit_inactivity, three_year_threshold),
        "🏦 Fixed Deposit Accounts (3+ years)": (check_fixed_deposit_inactivity, three_year_threshold),
        "💼 Investment Accounts (3+ years)": (check_investment_inactivity, three_year_threshold),
        "🔐 Safe Deposit Boxes (3+ years)": (check_safe_deposit, three_year_threshold),
        "🧾 Bankers Cheques & Drafts (1+ year)": (check_bankers_cheques, one_year_threshold),
        "🏛️ Central Bank Transfer Eligible (5+ years)": (check_transfer_to_central_bank, five_year_threshold),
        "📋 Article 3 Process Required": (check_art3_process_required, three_year_threshold),
        "📱 Contact Attempts Needed": (check_contact_attempts_needed, three_year_threshold)
    }

    # Regulatory information for each agent
    regulatory_info = {
        "📉 Demand Deposit Accounts (3+ years)": """
        **UAE Regulation Article 2(1)**: An Individual/Corporate savings, call, or current account becomes dormant when:
        - No transactions (withdrawals or deposits) for 3 years
        - No non-financial actions (service requests, due diligence, particulars update) for 3 years
        - No written or electronic communication from the customer
        - The customer has no other active liability account with the bank

        **Required Action (Article 3)**: Bank must attempt to contact the dormant customer, wait 3 months for response, then transfer to dormant accounts ledger.
        """,

        "🏦 Fixed Deposit Accounts (3+ years)": """
        **UAE Regulation Article 2(2)**: A Fixed Term Deposit Account becomes dormant when:
        - There is no automatic renewal clause and the deposit has matured
        - Neither renewal nor claim request has been made in the past 3 years since maturity
        - OR where there is an automatic renewal clause, but there is no communication from the customer for 3 years from the first maturity

        **Required Action (Article 3)**: Bank must attempt to contact the dormant customer, wait 3 months for response, then transfer to dormant accounts ledger.
        """,

        "💼 Investment Accounts (3+ years)": """
        **UAE Regulation Article 2(3)**: An Investment Account becomes dormant when:
        - For closed-ended accounts: no communication from customer for 3 years from final maturity
        - For redeemable accounts: no communication for 3 years from redemption date
        - For open-ended accounts: when customer's other accounts are classified as dormant

        **Required Action (Article 3)**: Bank must contact the dormant customer. However, the bank shall liquidate or dispose of assets in open-ended investment accounts only as per stated terms and conditions.
        """,

        "🔐 Safe Deposit Boxes (3+ years)": """
        **UAE Regulation Article 2(6)**: A Safe Deposit Box becomes dormant when:
        - Charges remain outstanding for more than 3 years
        - The bank has not received a reply from the tenant
        - The tenant has not made alternative arrangements for the Safe Deposit Box

        **Required Action (Article 3)**: Bank must send final notice to tenant's last known address. If no response, the bank should apply to Court to appoint a person to supervise opening of the box.
        """,

        "🧾 Bankers Cheques & Drafts (1+ year)": """
        **UAE Regulation Article 2(4)**: Bankers cheques, bank drafts, or cashier orders become unclaimed when:
        - Issued at customer request by debiting their account
        - Not claimed by the beneficiary for 1 year
        - Despite bank efforts to contact the customer

        **Required Action (Article 3)**: Bank must initiate communication to issuers notifying them of non-encashment. After 3 months waiting period, transfer to "unclaimed balances account."
        """,

        "🏛️ Central Bank Transfer Eligible (5+ years)": """
        **UAE Regulation Article 8(1-4)**: Accounts must be transferred to the Central Bank when:
        - Account remains dormant for 5 years from the last transaction
        - The customer has no other active accounts with the same bank
        - The current address of the account owner is unknown

        **Central Bank Role (Article 9)**: The Central Bank will retain these funds until claimed by the beneficiary. No interest is accrued after transfer to the Central Bank.
        """,

        "📋 Article 3 Process Required": """
        **UAE Regulation Article 3**: Required actions for newly dormant accounts:
        1. Bank must attempt to contact the dormant customer through written, electronic, or verbal channels
        2. A communication must be sent to issuers of unclaimed instruments
        3. A final notice must be sent to dormant Safe Deposit Box tenant's last known address
        4. Bank must wait 3 months for response
        5. After 3 months with no response, transfer money to "dormant accounts ledger"

        This analysis identifies accounts that meet dormancy criteria but haven't yet gone through this required process.
        """,

        "📱 Contact Attempts Needed": """
        **UAE Regulation Article 5**: Banks' responsibilities:
        - Banks must ensure customer profiles are always updated
        - Banks must periodically advise customers to update their IDs, address, and telephone numbers
        - Banks may introduce "inactive" status before accounts become dormant for enhanced monitoring
        - Banks must carry out annual reviews of dormant accounts and contact customers

        This analysis identifies accounts approaching dormancy where proactive contact should be made to prevent dormancy.
        """
    }

    if agent_info := agent_mapping.get(agent_option):
        selected_agent, threshold = agent_info

        # Display regulatory information first
        st.info(regulatory_info.get(agent_option, "Regulatory information not available."))

        # Run the selected agent
        with st.spinner(f"Running {agent_option}..."):
            data_filtered, count, agent_desc = selected_agent(df, threshold)
            agent_executed = True

        st.metric("Accounts/Instruments Identified", count, help=agent_desc)
        st.markdown(f"**Agent Description:** {agent_desc}")

    if agent_executed:
        if not data_filtered.empty:
            st.success(f"{len(data_filtered)} accounts/instruments identified per UAE regulations.")

            # Add a summary of key characteristics
            with st.expander("📊 Quick Statistics", expanded=True):
                try:
                    # Create summary stats based on agent type
                    if agent_option == "📉 Demand Deposit Accounts (3+ years)":
                        account_types = data_filtered['Account_Type'].value_counts().head(5)
                        st.bar_chart(account_types)

                        if 'Current_Balance' in data_filtered.columns:
                            total_balance = data_filtered['Current_Balance'].sum()
                            avg_balance = data_filtered['Current_Balance'].mean()
                            st.metric("Total Dormant Balance", f"{total_balance:,.2f}")
                            st.metric("Average Dormant Balance", f"{avg_balance:,.2f}")

                    elif agent_option == "🧾 Bankers Cheques & Drafts (1+ year)":
                        if 'Unclaimed_Item_Amount' in data_filtered.columns:
                            total_amount = data_filtered['Unclaimed_Item_Amount'].sum()
                            avg_amount = data_filtered['Unclaimed_Item_Amount'].mean()
                            st.metric("Total Unclaimed Amount", f"{total_amount:,.2f}")
                            st.metric("Average Unclaimed Amount", f"{avg_amount:,.2f}")

                    elif agent_option == "🏛️ Central Bank Transfer Eligible (5+ years)":
                        if 'Currency' in data_filtered.columns:
                            currency_breakdown = data_filtered['Currency'].value_counts()
                            st.bar_chart(currency_breakdown)

                        if 'Current_Balance' in data_filtered.columns:
                            total_transfer = data_filtered['Current_Balance'].sum()
                            st.metric("Total Central Bank Transfer Amount", f"{total_transfer:,.2f}")
                except Exception as stats_error:
                    st.warning(f"Could not generate statistics: {stats_error}")

            if st.checkbox(f"View detected accounts for '{agent_option}'",
                           key=f"view_detected_{agent_option.replace(' ', '_')}"):
                # Allow selecting number of rows to display
                num_rows = st.slider("Number of rows to display", 5, 100, 15,
                                     key=f"rows_slider_{agent_option.replace(' ', '_')}")

                # Display the DataFrame with original column names if available
                display_df = data_filtered.head(num_rows).copy()
                if SESSION_COLUMN_MAPPING in st.session_state and st.session_state[SESSION_COLUMN_MAPPING]:
                    try:
                        # Create a display mapping that only includes columns present in the data
                        display_columns_mapping = {
                            std_col: st.session_state[SESSION_COLUMN_MAPPING].get(std_col, std_col)
                            for std_col in display_df.columns
                        }
                        display_df.rename(columns=display_columns_mapping, inplace=True)
                    except Exception as e:
                        st.warning(f"Could not display original column names: {e}")

                st.dataframe(display_df)

            # Generate insights with LLM if available
            if llm:
                sample_size = min(30, len(data_filtered))
                sample_data_csv = data_filtered.sample(n=sample_size).to_csv(index=False)

                # Initialize the prompt templates with our predefined prompts
                observation_prompt = PromptTemplate.from_template(
                    OBSERVATION_PROMPT + "\nFocus on UAE regulatory compliance aspects.")
                trend_prompt = PromptTemplate.from_template(
                    TREND_PROMPT + "\nInclude analysis of compliance with UAE Central Bank requirements.")

                if st.button(f"Generate UAE Compliance Insights",
                             key=f"generate_insights_{agent_option.replace(' ', '_')}"):
                    try:
                        with st.spinner("Running regulatory insight agents..."):
                            # Setup the LLM chains
                            output_parser = StrOutputParser()
                            observation_chain = observation_prompt | llm | output_parser
                            trend_chain = trend_prompt | llm | output_parser

                            # Generate observations and trends
                            obs_output = observation_chain.invoke({"data": sample_data_csv})
                            trend_output = trend_chain.invoke({"data": sample_data_csv})

                            # Generate summary and actions
                            narration_prompt = PromptTemplate.from_template(
                                NARRATION_PROMPT + "\nInclude UAE regulatory compliance aspects and Central Bank requirements.")
                            action_prompt = PromptTemplate.from_template(
                                ACTION_PROMPT + "\nEnsure actions align with UAE Central Bank Dormant Accounts Regulation requirements.")

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
                                st.success("Regulatory insights saved to insight log.")
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

                        st.subheader("UAE Regulatory Compliance Insights")

                        with st.expander("🔍 Regulatory Observations", expanded=True):
                            st.markdown(insights['observation'])
                        with st.expander("📊 Trend Analysis"):
                            st.markdown(insights['trend'])
                        with st.expander("📌 Executive Summary"):
                            st.markdown(insights['summary'])
                        with st.expander("🚀 Recommended Compliance Actions", expanded=True):
                            st.markdown(insights['actions'])

                        # PDF export for individual agent insights
                        st.subheader("⬇️ Export Compliance Report")
                        sections = [
                            {"title": "UAE Regulatory Framework", "content": regulatory_info.get(agent_option, "")},
                            {"title": "Account Overview",
                             "content": f"Identified {len(data_filtered)} accounts/instruments requiring action under the UAE Central Bank Dormant Accounts Regulation."},
                            {"title": "Regulatory Observations", "content": insights['observation']},
                            {"title": "Trend Analysis", "content": insights['trend']},
                            {"title": "Executive Summary", "content": insights['summary']},
                            {"title": "Recommended Compliance Actions", "content": insights['actions']},
                            {"title": "Report Information",
                             "content": f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nAgent: {agent_option}\nRegulatory Framework: UAE Central Bank Dormant Accounts Regulation"}
                        ]

                        download_pdf_button(
                            f"UAE Dormant Accounts - {agent_option.replace('📉 ', '').replace('🏦 ', '').replace('💼 ', '').replace('🔐 ', '').replace('🧾 ', '').replace('🏛️ ', '').replace('📋 ', '').replace('📱 ', '')} Compliance Report",
                            sections,
                            f"{agent_option.replace(' ', '_').replace(':', '')}_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        )

                        # Add CSV download for identified accounts
                        download_csv_button(
                            data_filtered,
                            f"{agent_option.replace(' ', '_').replace(':', '')}_accounts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                        )

                    elif len(data_filtered) > 0:
                        st.info(f"AI Assistant not available to generate regulatory compliance insights.")

                        # Still provide CSV download even without AI
                        st.subheader("⬇️ Export Data")
                        download_csv_button(
                            data_filtered,
                            f"{agent_option.replace(' ', '_').replace(':', '')}_accounts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        )

                        # Provide a basic PDF report option without AI insights
                        basic_sections = [
                            {"title": "UAE Regulatory Framework", "content": regulatory_info.get(agent_option, "")},
                            {"title": "Account Overview",
                             "content": f"Identified {len(data_filtered)} accounts/instruments requiring action under the UAE Central Bank Dormant Accounts Regulation."},
                            {"title": "Regulatory Requirements",
                             "content": "This report identifies accounts or instruments meeting the criteria specified in the UAE Central Bank Dormant Accounts Regulation. Follow the required procedures outlined in Articles 3 and 8 of the regulation."},
                            {"title": "Report Information",
                             "content": f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nAgent: {agent_option}"}
                        ]

                        download_pdf_button(
                            f"UAE Dormant Accounts - Basic Compliance Report",
                            basic_sections,
                            f"{agent_option.replace(' ', '_').replace(':', '')}_basic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        )

                    elif len(data_filtered) == 0:
                        st.info("No accounts/instruments matching the UAE regulatory criteria were found.")

                        # Display suggestion based on the selected agent
                        if agent_option == "🏛️ Central Bank Transfer Eligible (5+ years)":
                            st.success(
                                "No accounts need to be transferred to the Central Bank at this time. Continue monitoring dormant accounts for the 5-year threshold.")
                        elif agent_option == "📱 Contact Attempts Needed":
                            st.success(
                                "No accounts are approaching dormancy currently. Continue with regular monitoring.")
                        elif agent_option == "📋 Article 3 Process Required":
                            st.success(
                                "All dormant accounts have completed the required Article 3 process or no accounts are dormant.")
                        else:
                            st.success(
                                "No dormant accounts of this type were detected. Continue with regular monitoring as per UAE Central Bank requirements.")