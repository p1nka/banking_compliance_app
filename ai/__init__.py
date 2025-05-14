# ai/__init__.py
"""
AI Package for Banking Compliance App

This package contains modules for AI and LLM functionality.
"""

# Import essential functions to make them available at package level
try:
    from .llm import get_llm, get_fallback_response
    from .llm import (
        SQL_GENERATION_PROMPT,
        SQL_EXPLANATION_PROMPT,
        DORMANT_SUMMARY_PROMPT,
        COMPLIANCE_SUMMARY_PROMPT,
        OBSERVATION_PROMPT,
        TREND_PROMPT,
        NARRATION_PROMPT,
        ACTION_PROMPT
    )
except ImportError as e:
    print(f"Warning: Could not import from ai.llm module: {e}")


    # Define empty/default values for essential functions
    def get_llm():
        """Fallback get_llm function that returns None."""
        import streamlit as st
        st.warning("AI functionality not available.")
        return None


    def get_fallback_response(response_type):
        """Fallback response function."""
        return "AI functionality not available."


    # Define empty strings for prompts
    SQL_GENERATION_PROMPT = ""
    SQL_EXPLANATION_PROMPT = ""
    DORMANT_SUMMARY_PROMPT = ""
    COMPLIANCE_SUMMARY_PROMPT = ""
    OBSERVATION_PROMPT = ""
    TREND_PROMPT = ""
    NARRATION_PROMPT = ""
    ACTION_PROMPT = ""