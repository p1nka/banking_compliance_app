# data_sync_component.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from data_sync import sync_data, get_data_stats
from ui.sqlbot_ui import get_sqlite_schema_for_bot, get_recent_sql_history, execute_nl_query


def render_data_sync_ui():
    """Render the data synchronization UI component."""
    st.subheader("Data Synchronization")

    # Get current data stats
    stats = get_data_stats()

    # Create two columns for stats
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Azure SQL Database")
        st.markdown(f"**Connection:** {'✅ Connected' if stats['azure']['connection'] else '❌ Not Connected'}")
        st.markdown(f"**Records:** {stats['azure']['record_count']}")

    with col2:
        st.markdown("### Local SQLite Database")
        st.markdown(f"**Connection:** {'✅ Connected' if stats['sqlite']['connection'] else '❌ Not Connected'}")
        st.markdown(f"**Records:** {stats['sqlite']['record_count']}")

        if stats['sqlite']['last_sync']:
            last_sync = stats['sqlite']['last_sync']
            time_diff = datetime.now() - last_sync

            if time_diff < timedelta(minutes=5):
                sync_status = "✅ Just synced"
            elif time_diff < timedelta(hours=1):
                sync_status = "✅ Recent"
            elif time_diff < timedelta(days=1):
                sync_status = "⚠️ Today"
            else:
                sync_status = "❌ Outdated"

            st.markdown(f"**Last Sync:** {sync_status} ({last_sync.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            st.markdown("**Last Sync:** ❌ Never")

    # Sync button with progress tracking
    if st.button("Synchronize Data Now"):
        with st.spinner("Synchronizing data from Azure to local database..."):
            success = sync_data(force=True)

            if success:
                st.success("Data synchronized successfully!")
                # Refresh stats after sync
                stats = get_data_stats()
                st.experimental_rerun()
            else:
                st.error("Data synchronization failed. Check the logs for details.")

    # Show schema information
    with st.expander("Database Schema"):
        schema = get_sqlite_schema_for_bot()
        if schema:
            for table_name, columns in schema.items():
                st.markdown(f"### {table_name}")

                # Create a DataFrame to display the columns
                df_columns = pd.DataFrame(columns, columns=["Column Name", "Data Type"])
                st.dataframe(df_columns, use_container_width=True)
        else:
            st.warning("Schema information not available.")


def render_sqlbot_ui():
    """Render the SQL Bot UI component."""
    st.subheader("SQL Bot")

    # Get schema for context
    schema = get_sqlite_schema_for_bot()

    if not schema:
        st.warning("Cannot access database schema. Please ensure the database is properly set up.")
        return

    # Input for natural language query
    nl_query = st.text_input("Ask a question about your data",
                             placeholder="e.g., 'Show me all dormant accounts' or 'Count accounts by type'")

    col1, col2 = st.columns([1, 3])

    with col1:
        execute_button = st.button("Execute Query", type="primary")
        suggest_button = st.button("Suggest SQL")

    # Handle query execution
    if nl_query and (execute_button or suggest_button):
        if suggest_button:
            # Just suggest SQL without executing
            _, sql_query = execute_nl_query(nl_query, suggest_sql=True)

            if sql_query:
                st.code(sql_query, language="sql")
            else:
                st.error("Could not generate SQL for this query.")

        elif execute_button:
            # Execute the query and show results
            with st.spinner("Executing query..."):
                results, sql_query = execute_nl_query(nl_query, suggest_sql=False)

                if sql_query:
                    st.code(sql_query, language="sql")

                if results is not None:
                    if len(results) > 0:
                        st.dataframe(results, use_container_width=True)
                        st.write(f"Found {len(results)} results.")
                    else:
                        st.info("Query executed successfully, but no results were found.")
                else:
                    st.error("Query execution failed.")

    # Show recent query history
    with st.expander("Recent Queries"):
        history = get_recent_sql_history()
        if history is not None and not history.empty:
            st.dataframe(history, use_container_width=True, column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Natural Language Query": st.column_config.TextColumn("Question"),
                "SQL Query": st.column_config.TextColumn("Generated SQL"),
                "Execution Time (s)": st.column_config.NumberColumn("Time (s)", format="%.2f"),
                "Results": st.column_config.NumberColumn("Results"),
                "Timestamp": st.column_config.DatetimeColumn("When")
            })
        else:
            st.info("No query history available.")