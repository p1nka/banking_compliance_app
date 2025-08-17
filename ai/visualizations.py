# --- START OF FILE visualizations.py ---

import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def fix_duplicate_columns(df):
    """Fix duplicate column names by adding suffixes."""
    if df.empty: return df
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [f"{dup}.{i}" if i != 0 else dup for i in
                                                         range(sum(cols == dup))]
    df.columns = cols
    return df


def generate_plot(llm_output, current_data, column_mapping=None):
    """Generate a plotly chart based on LLM output."""
    try:
        current_data = fix_duplicate_columns(current_data)
        plot_type = llm_output.get("plot_type")
        x_col = llm_output.get("x_column")
        y_col = llm_output.get("y_column")
        color_col = llm_output.get("color_column")
        names_col = llm_output.get("names_column")
        title = llm_output.get("title", "Generated Plot")

        def validate_col(col_name):
            return col_name if col_name and col_name in current_data.columns else None

        x_col, y_col, color_col, names_col = map(validate_col, [x_col, y_col, color_col, names_col])

        chart, response_text = None, "I've generated a visualization for you."

        if plot_type == 'pie':
            if not names_col: return None, "Cannot create pie chart: missing column for names."
            if current_data[
                names_col].nunique() > 25: return None, f"Too many unique values in '{names_col}' for a pie chart."
            chart = px.pie(current_data, names=names_col, title=title, hole=0.3)
            response_text = f"Generated pie chart for '{names_col}'."

        elif plot_type == 'bar':
            if not x_col: return None, "Cannot create bar chart: missing x-axis column."
            chart = px.bar(current_data, x=x_col, y=y_col, color=color_col, title=title)
            response_text = f"Generated bar chart for '{x_col}'."

        elif plot_type == 'histogram':
            if not x_col: return None, "Cannot create histogram: missing x-axis column."
            chart = px.histogram(current_data, x=x_col, color=color_col, title=title)
            response_text = f"Generated histogram for '{x_col}'."

        elif plot_type == 'box':
            if not y_col: return None, "Cannot create box plot: missing y-axis column."
            chart = px.box(current_data, x=x_col, y=y_col, color=color_col, title=title)
            response_text = f"Generated box plot for '{y_col}'."

        elif plot_type == 'scatter':
            if not x_col or not y_col: return None, "Cannot create scatter plot: missing x or y column."
            chart = px.scatter(current_data, x=x_col, y=y_col, color=color_col, title=title)
            response_text = f"Generated scatter plot of '{x_col}' vs '{y_col}'."

        else:
            return None, f"Unsupported plot type: {plot_type}."

        return chart, response_text

    except Exception as e:
        return None, f"Error generating visualization: {e}"


def create_insights_chart(data, labels=None, values=None, chart_type='pie', title=None):
    """Create a chart for analytics insights with better defaults."""
    try:
        data = fix_duplicate_columns(data)
        fig = None
        if chart_type == 'pie' and labels:
            # Aggregate data if not already aggregated
            if values and values in data.columns:
                df_agg = data.groupby(labels)[values].sum().reset_index()
                fig = px.pie(df_agg, names=labels, values=values, title=title, hole=0.3)
            else:
                counts = data[labels].value_counts().reset_index()
                counts.columns = [labels, 'count']
                fig = px.pie(counts, names=labels, values='count', title=title, hole=0.3)
            if fig: fig.update_traces(textposition='inside', textinfo='percent+label')

        elif chart_type == 'bar' and labels:
            if values and values in data.columns:
                fig = px.bar(data, x=labels, y=values, title=title, text_auto=True)
            else:
                counts = data[labels].value_counts().reset_index()
                counts.columns = [labels, 'count']
                fig = px.bar(counts, x=labels, y='count', title=title, text_auto=True)
            if fig: fig.update_layout(xaxis={'categoryorder': 'total descending'})

        elif chart_type == 'histogram' and labels:
            fig = px.histogram(data, x=labels, title=title)

        elif chart_type == 'line' and labels and values:
            fig = px.line(data, x=labels, y=values, title=title, markers=True)

        elif chart_type == 'scatter' and labels and values:
            fig = px.scatter(data, x=labels, y=values, title=title)

        return fig

    except Exception as e:
        st.warning(f"Could not create '{chart_type}' chart: {e}")
        return None


def smart_detect_chart_type(df: pd.DataFrame):
    """Intelligently detect the best chart type based on query results."""
    if df.empty or len(df.columns) == 0: return None, None, None, None

    df = fix_duplicate_columns(df)
    cols = df.columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if df[c].nunique() <= 25]
    date_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

    # IMPROVEMENT: Prefer bar charts for GROUP BY results as they are more versatile than pie charts.
    if len(cols) == 2 and categorical_cols and numeric_cols:
        cat_col, num_col = (categorical_cols[0], numeric_cols[0])
        return 'bar', cat_col, num_col, f"{num_col} by {cat_col}"

    if date_cols and numeric_cols:
        return 'line', date_cols[0], numeric_cols[0], f"{numeric_cols[0]} over Time"

    if len(numeric_cols) >= 2:
        return 'scatter', numeric_cols[0], numeric_cols[1], f"{numeric_cols[1]} vs {numeric_cols[0]}"

    if categorical_cols:
        return 'bar', categorical_cols[0], None, f"Count by {categorical_cols[0]}"

    if numeric_cols:
        return 'histogram', numeric_cols[0], None, f"Distribution of {numeric_cols[0]}"

    return None, None, None, None


def auto_visualize(results_df: pd.DataFrame):
    """Automatically generates a relevant visualization based on the DataFrame."""
    if results_df is None or results_df.empty:
        st.info("💡 No data available for visualization.")
        return

    st.subheader("📊 Automatic Visualization")

    try:
        results_df = fix_duplicate_columns(results_df)
        chart_type, x_col, y_col, title = smart_detect_chart_type(results_df)

        if chart_type:
            chart = create_insights_chart(data=results_df, labels=x_col, values=y_col, chart_type=chart_type,
                                          title=title)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("Could not automatically generate a suitable chart for this data.")
        else:
            st.info("Could not automatically detect a suitable chart type. Displaying data table instead.")
            st.dataframe(results_df)

    except Exception as e:
        st.warning(f"Could not create automatic visualization: {e}")
        st.dataframe(results_df)