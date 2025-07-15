import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def fix_duplicate_columns(df):
    """
    Fix duplicate column names by adding suffixes.
    """
    if df.empty:
        return df

    # Create a copy to avoid modifying the original
    df = df.copy()

    # Get column names
    columns = list(df.columns)

    # Check for duplicates
    if len(columns) != len(set(columns)):
        # There are duplicates, fix them
        new_columns = []
        column_counts = {}

        for col in columns:
            if col in column_counts:
                column_counts[col] += 1
                new_columns.append(f"{col}_{column_counts[col]}")
            else:
                column_counts[col] = 0
                new_columns.append(col)

        df.columns = new_columns
        print(f"Fixed duplicate columns: {columns} → {new_columns}")  # Debug print

    return df


def generate_plot(llm_output, current_data, column_mapping=None):
    """
    Generate a plotly chart based on LLM output and return both the chart object and response text.

    Args:
        llm_output (dict): Dictionary with plot specifications from LLM
        current_data (pd.DataFrame): The DataFrame to plot
        column_mapping (dict): Optional mapping from standardized column names to original ones

    Returns:
        tuple: (chart, response_text) where chart is a plotly figure or None if error
    """
    try:
        # Fix duplicate columns first
        current_data = fix_duplicate_columns(current_data)

        plot_type = llm_output.get("plot_type")
        x_col = llm_output.get("x_column")
        y_col = llm_output.get("y_column")
        color_col = llm_output.get("color_column")
        names_col = llm_output.get("names_column")
        title = llm_output.get("title", f"Plot based on user query")

        # Validate columns exist in the DataFrame
        all_cols = list(current_data.columns)

        def validate_col(col_name):
            return col_name if col_name is not None and col_name in all_cols else None

        x_col_valid = validate_col(x_col)
        y_col_valid = validate_col(y_col)
        color_col_valid = validate_col(color_col)
        names_col_valid = validate_col(names_col)

        def get_original_name(col):
            if column_mapping and col in column_mapping:
                return column_mapping.get(col, col)
            return col

        chart = None
        response_text = "I've generated a visualization for you."

        if plot_type == 'pie':
            if not names_col_valid:
                return None, "Cannot create pie chart: missing column for names."
            if not pd.api.types.is_string_dtype(
                    current_data[names_col_valid]) and not pd.api.types.is_categorical_dtype(
                current_data[names_col_valid]):
                return None, f"Cannot create pie chart: '{get_original_name(names_col_valid)}' is not a categorical column."
            unique_count = current_data[names_col_valid].nunique()
            if unique_count > 25:
                return None, f"Too many unique values ({unique_count}) in '{get_original_name(names_col_valid)}' for a pie chart. Try a bar chart instead."
            counts = current_data[names_col_valid].value_counts().reset_index()
            counts.columns = [names_col_valid, 'count']
            chart = px.pie(counts, names=names_col_valid, values='count', title=title, hole=0.3)
            response_text = f"Generated pie chart showing distribution of '{get_original_name(names_col_valid)}'."

        elif plot_type == 'bar':
            if not x_col_valid:
                return None, "Cannot create bar chart: missing x-axis column."
            if pd.api.types.is_numeric_dtype(current_data[x_col_valid]):
                return None, f"Bar chart x-axis ('{get_original_name(x_col_valid)}') is numeric. Consider using a histogram instead."
            counts = current_data[x_col_valid].value_counts().reset_index()
            counts.columns = [x_col_valid, 'count']
            chart = px.bar(counts, x=x_col_valid, y='count', color=color_col_valid, title=title)
            response_text = f"Generated bar chart showing counts for '{get_original_name(x_col_valid)}'."
            if color_col_valid:
                response_text += f" colored by '{get_original_name(color_col_valid)}'."

        elif plot_type == 'histogram':
            if not x_col_valid:
                return None, "Cannot create histogram: missing x-axis column."
            if not pd.api.types.is_numeric_dtype(
                    current_data[x_col_valid]) and not pd.api.types.is_datetime64_any_dtype(current_data[x_col_valid]):
                return None, f"Histogram requires a numeric or date column. '{get_original_name(x_col_valid)}' is not. Try a bar chart instead."
            chart = px.histogram(current_data, x=x_col_valid, color=color_col_valid, title=title)
            response_text = f"Generated histogram for '{get_original_name(x_col_valid)}'."
            if color_col_valid:
                response_text += f" colored by '{get_original_name(color_col_valid)}'."

        elif plot_type == 'box':
            if not y_col_valid:
                return None, "Cannot create box plot: missing y-axis column (numeric)."
            if not pd.api.types.is_numeric_dtype(current_data[y_col_valid]):
                return None, f"Box plot requires a numeric y-axis. '{get_original_name(y_col_valid)}' is not numeric."
            chart = px.box(current_data, x=x_col_valid, y=y_col_valid, color=color_col_valid, title=title,
                           points="outliers")
            response_text = f"Generated box plot for '{get_original_name(y_col_valid)}'."
            if x_col_valid:
                response_text += f" grouped by '{get_original_name(x_col_valid)}'"
            if color_col_valid:
                response_text += f" colored by '{get_original_name(color_col_valid)}'."

        elif plot_type == 'scatter':
            if not x_col_valid or not y_col_valid:
                return None, f"Cannot create scatter plot: need both x and y columns."
            if not (pd.api.types.is_numeric_dtype(current_data[x_col_valid]) or pd.api.types.is_datetime64_any_dtype(
                    current_data[x_col_valid])):
                return None, f"Scatter plot x-axis ('{get_original_name(x_col_valid)}') must be numeric or date."
            if not (pd.api.types.is_numeric_dtype(current_data[y_col_valid]) or pd.api.types.is_datetime64_any_dtype(
                    current_data[y_col_valid])):
                return None, f"Scatter plot y-axis ('{get_original_name(y_col_valid)}') must be numeric or date."
            chart = px.scatter(current_data, x=x_col_valid, y=y_col_valid, color=color_col_valid, title=title,
                               hover_data=current_data.columns)
            response_text = f"Generated scatter plot of '{get_original_name(x_col_valid)}' vs '{get_original_name(y_col_valid)}'."
            if color_col_valid:
                response_text += f" colored by '{get_original_name(color_col_valid)}'."
        else:
            return None, f"Unsupported plot type: {plot_type}. Please try a different visualization."

        return chart, response_text

    except Exception as e:
        return None, f"Error generating visualization: {e}"


def create_insights_chart(data, labels=None, values=None, chart_type='pie', title=None):
    """
    Create a chart for analytics insights with proper handling of pre-aggregated data and hover information.
    """
    try:
        # Fix duplicate columns first
        data = fix_duplicate_columns(data)

        if chart_type == 'pie' and labels:
            # Check if we have a pre-aggregated values column
            if values and values in data.columns:
                # Data is already aggregated (e.g., from GROUP BY query)
                # Filter out zero or negative values for pie charts
                filtered_data = data[data[values] > 0] if data[values].dtype in ['int64', 'float64'] else data
                if filtered_data.empty:
                    return None
                fig = px.pie(
                    filtered_data,
                    names=labels,
                    values=values,
                    title=title,
                    hover_data=[values]  # Show values on hover
                )
                # Customize hover template to show actual numbers
                fig.update_traces(
                    hovertemplate="<b>%{label}</b><br>" +
                                  f"{values}: %{{value}}<br>" +
                                  "Percentage: %{percent}<br>" +
                                  "<extra></extra>"
                )
            else:
                # Need to count occurrences for raw data
                counts = data[labels].value_counts().reset_index()
                counts.columns = [labels, 'count']
                fig = px.pie(
                    counts,
                    names=labels,
                    values='count',
                    title=title,
                    hover_data=['count']
                )
                fig.update_traces(
                    hovertemplate="<b>%{label}</b><br>" +
                                  "Count: %{value}<br>" +
                                  "Percentage: %{percent}<br>" +
                                  "<extra></extra>"
                )
            return fig

        elif chart_type == 'bar':
            if values and values in data.columns:
                # Use provided values column (pre-aggregated data)
                # Ensure categorical data is on x-axis, numerical on y-axis
                fig = px.bar(
                    data,
                    x=labels,
                    y=values,
                    title=title,
                    hover_data=[values]
                )
                # Customize hover template
                fig.update_traces(
                    hovertemplate="<b>%{x}</b><br>" +
                                  f"{values}: %{{y:,.0f}}<br>" +
                                  "<extra></extra>"
                )
                y_title = values
            else:
                # Count occurrences for raw data
                if pd.api.types.is_numeric_dtype(data[labels]):
                    # If labels column is numeric, treat it as categorical for better visualization
                    # Convert to string to treat as categories
                    data_copy = data.copy()
                    data_copy[labels] = data_copy[labels].astype(str)
                    counts = data_copy[labels].value_counts().reset_index()
                    counts.columns = [labels, 'count']
                else:
                    counts = data[labels].value_counts().reset_index()
                    counts.columns = [labels, 'count']

                fig = px.bar(
                    counts,
                    x=labels,
                    y='count',
                    title=title,
                    hover_data=['count']
                )
                fig.update_traces(
                    hovertemplate="<b>%{x}</b><br>" +
                                  "Count: %{y:,.0f}<br>" +
                                  "<extra></extra>"
                )
                y_title = 'Count'

            # Improve bar chart appearance
            fig.update_layout(
                xaxis_title=labels if labels else 'Category',
                yaxis_title=y_title,
                showlegend=False,
                xaxis={'categoryorder': 'total descending'}  # Sort bars by value
            )
            return fig

        elif chart_type == 'histogram' and labels:
            # Check if we should treat this as categorical instead of continuous
            unique_vals = data[labels].nunique()
            total_rows = len(data)

            if unique_vals <= 20 and unique_vals < total_rows * 0.1:
                # Treat as categorical - create a bar chart instead
                data_copy = data.copy()
                data_copy[labels] = data_copy[labels].astype(str)
                counts = data_copy[labels].value_counts().reset_index()
                counts.columns = [labels, 'count']
                counts = counts.sort_values(labels)  # Sort by category name

                fig = px.bar(
                    counts,
                    x=labels,
                    y='count',
                    title=title.replace('Distribution', 'Count by Value'),
                )
                fig.update_traces(
                    hovertemplate="<b>%{x}</b><br>" +
                                  "Count: %{y:,.0f}<br>" +
                                  "<extra></extra>"
                )
                fig.update_layout(
                    xaxis_title=f"{labels} (Values)",
                    yaxis_title='Count',
                    showlegend=False
                )
            else:
                # True histogram for continuous data
                fig = px.histogram(data, x=labels, title=title, nbins=min(30, unique_vals))
                fig.update_traces(
                    hovertemplate="<b>%{x}</b><br>" +
                                  "Count: %{y:,.0f}<br>" +
                                  "<extra></extra>"
                )
                fig.update_layout(
                    xaxis_title=labels,
                    yaxis_title='Frequency'
                )
            return fig

        elif chart_type == 'line' and labels and values:
            fig = px.line(data, x=labels, y=values, title=title, markers=True)
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>" +
                              f"{values}: %{{y:,.2f}}<br>" +
                              "<extra></extra>"
            )
            return fig

        elif chart_type == 'scatter' and labels and values:
            fig = px.scatter(data, x=labels, y=values, title=title)
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>" +
                              f"{values}: %{{y:,.2f}}<br>" +
                              "<extra></extra>"
            )
            return fig

        else:
            return None
    except Exception as e:
        print(f"Error creating chart: {e}")  # Debug print
        return None


def smart_detect_chart_type(results_df: pd.DataFrame):
    """
    Intelligently detect the best chart type based on query results structure.
    RULE: Categorical columns on X-axis, Numerical columns on Y-axis for better readability.
    """
    if results_df.empty:
        return None, None, None, None

    # IMPORTANT: Fix duplicate columns FIRST before any analysis
    results_df = fix_duplicate_columns(results_df)

    columns = list(results_df.columns)

    # Identify column types
    numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = results_df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = [col for col in results_df.columns if pd.api.types.is_datetime64_any_dtype(results_df[col])]

    # Filter categorical columns (avoid IDs and high cardinality)
    good_categorical_cols = [
        col for col in categorical_cols
        if not col.lower().endswith(('_id', 'id')) and 1 < results_df[col].nunique() <= 25
    ]

    # Filter numeric columns (avoid ID-like columns)
    good_numeric_cols = [
        col for col in numeric_cols
        if not col.lower().endswith(('_id', 'id'))
    ]

    # PRIORITY 1: GROUP BY results (categorical + numeric columns)
    if good_categorical_cols and good_numeric_cols:
        cat_col = good_categorical_cols[0]  # X-axis (categorical)
        num_col = good_numeric_cols[0]  # Y-axis (numerical)

        # Choose chart type based on number of categories
        if results_df[cat_col].nunique() <= 8:
            return 'pie', cat_col, num_col, f"Distribution of {num_col} by {cat_col}"
        else:
            return 'bar', cat_col, num_col, f"{num_col} by {cat_col}"

    # PRIORITY 2: Time series data (date + numeric)
    elif date_cols and good_numeric_cols:
        return 'line', date_cols[0], good_numeric_cols[0], f"{good_numeric_cols[0]} over time"

    # PRIORITY 3: Two numeric columns - scatter plot
    elif len(good_numeric_cols) >= 2:
        return 'scatter', good_numeric_cols[0], good_numeric_cols[
            1], f"{good_numeric_cols[1]} vs {good_numeric_cols[0]}"

    # PRIORITY 4: Single categorical column - count occurrences
    elif good_categorical_cols:
        cat_col = good_categorical_cols[0]
        if results_df[cat_col].nunique() <= 8:
            return 'pie', cat_col, None, f"Distribution of {cat_col}"
        else:
            return 'bar', cat_col, None, f"Count by {cat_col}"

    # PRIORITY 5: Single numeric column - create bins/ranges for better visualization
    elif good_numeric_cols:
        num_col = good_numeric_cols[0]

        # Check if the numeric data has reasonable distribution for histogram
        unique_values = results_df[num_col].nunique()
        total_rows = len(results_df)

        # If too few unique values relative to total rows, treat as categorical
        if unique_values <= 20 and unique_values < total_rows * 0.1:
            return 'bar', num_col, None, f"Distribution of {num_col} values"
        else:
            return 'histogram', num_col, None, f"Distribution of {num_col}"

    return None, None, None, None


def auto_visualize(results_df: pd.DataFrame):
    """
    Automatically generates a set of relevant visualizations based on the DataFrame's structure.
    This function is a powerful tool for quick data exploration.
    """
    if results_df is None or results_df.empty:
        st.info("💡 No data available for visualization.")
        return

    # Fix duplicate columns first
    try:
        results_df = fix_duplicate_columns(results_df)
    except Exception as e:
        st.error(f"Error fixing column names: {e}")
        return

    st.subheader("📊 Automatic Visualizations & Insights")

    # Debug information to help understand the data structure
    with st.expander("🔍 Data Structure Debug Info"):
        st.write("**DataFrame Shape:**", results_df.shape)
        st.write("**Column Names:**", list(results_df.columns))
        st.write("**Column Types:**")
        for col in results_df.columns:
            st.write(f"- {col}: {results_df[col].dtype} (unique values: {results_df[col].nunique()})")
        st.write("**First few rows:**")
        st.dataframe(results_df.head(3))

    # Use smart detection for the main visualization
    try:
        chart_type, x_col, y_col, title = smart_detect_chart_type(results_df)

        st.write(f"**Detected chart type:** {chart_type}")
        st.write(f"**X column:** {x_col}")
        st.write(f"**Y column:** {y_col}")

    except Exception as e:
        st.error(f"Error detecting chart type: {e}")
        return

    if chart_type:
        try:
            chart = create_insights_chart(
                data=results_df,
                labels=x_col,
                values=y_col,
                chart_type=chart_type,
                title=title
            )

            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("Could not generate primary visualization.")
        except Exception as e:
            st.warning(f"Could not create primary visualization: {e}")
            # Show the error details
            st.error(f"Full error: {str(e)}")

    # Additional exploratory visualizations
    numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = results_df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = results_df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

    # Filter out high-cardinality categorical columns to avoid cluttered charts
    filtered_categorical_cols = [
        col for col in categorical_cols
        if 1 < results_df[col].nunique() <= 30
    ]

    charts_created = 1 if chart_type else 0
    max_charts = 4

    # Time series analysis (if not already shown)
    if date_cols and numeric_cols and charts_created < max_charts and chart_type != 'line':
        date_col = date_cols[0]
        num_col = numeric_cols[0]
        try:
            st.write(f"**Time Series Analysis: `{num_col}` over `{date_col}`**")
            chart = px.line(results_df, x=date_col, y=num_col, title=f"Trend of {num_col} over time", markers=True)
            st.plotly_chart(chart, use_container_width=True)
            charts_created += 1
        except Exception as e:
            st.warning(f"Could not create time series plot: {e}")

    # Categorical distribution (if not already shown)
    if filtered_categorical_cols and charts_created < max_charts and chart_type not in ['pie', 'bar']:
        cat_col = filtered_categorical_cols[0]
        try:
            if results_df[cat_col].nunique() <= 10:
                st.write(f"**Distribution by `{cat_col}`**")
                chart = create_insights_chart(results_df, labels=cat_col, chart_type='pie',
                                              title=f"Distribution of {cat_col}")
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                    charts_created += 1
            else:
                st.write(f"**Count by `{cat_col}`**")
                chart = create_insights_chart(results_df, labels=cat_col, chart_type='bar', title=f"Count by {cat_col}")
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                    charts_created += 1
        except Exception as e:
            st.warning(f"Could not create categorical plot: {e}")

    # Numeric distribution (if not already shown)
    if numeric_cols and charts_created < max_charts and chart_type != 'histogram':
        num_col = numeric_cols[0]
        try:
            st.write(f"**Distribution of `{num_col}`**")
            chart = create_insights_chart(results_df, labels=num_col, chart_type='histogram',
                                          title=f"Distribution of {num_col}")
            if chart:
                st.plotly_chart(chart, use_container_width=True)
                charts_created += 1
        except Exception as e:
            st.warning(f"Could not create histogram: {e}")

    # Scatter plot for relationship between two numeric variables (if not already shown)
    if len(numeric_cols) >= 2 and charts_created < max_charts and chart_type != 'scatter':
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        try:
            st.write(f"**Relationship between `{x_col}` and `{y_col}`**")
            chart = px.scatter(results_df, x=x_col, y=y_col, title=f"{x_col} vs. {y_col}",
                               hover_data=results_df.columns)
            st.plotly_chart(chart, use_container_width=True)
            charts_created += 1
        except Exception as e:
            st.warning(f"Could not create scatter plot: {e}")

    if charts_created == 0:
        st.info(
            "Could not automatically generate visualizations for this query result. The data structure may be too complex or not suitable for standard charts.")

    # Always show summary statistics for numeric columns
    if numeric_cols:
        st.write("**Summary Statistics**")
        st.dataframe(results_df[numeric_cols].describe(), use_container_width=True)
