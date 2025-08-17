# --- START OF FILE dormant.py ---

from enum import Enum
import os
import csv
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import logging

from langchain_core.language_models import BaseChatModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Constants and Enums
class ActivityStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DORMANT = "dormant"
    UNCLAIMED = "unclaimed"
    PENDING_REVIEW = "pending_review"


class AccountType(Enum):
    SAFE_DEPOSIT = "safe_deposit"
    INVESTMENT = "investment"
    FIXED_DEPOSIT = "fixed_deposit"
    DEMAND_DEPOSIT = "demand_deposit"
    UNCLAIMED_INSTRUMENT = "unclaimed_instrument"


class CustomerTier(Enum):
    STANDARD = "standard"
    HIGH_VALUE = "high_value"  # ≥25K AED
    PREMIUM = "premium"  # ≥100K AED
    VIP = "vip"  # ≥500K AED
    PRIVATE_BANKING = "private_banking"  # ≥1M AED


class ContactMethod(Enum):
    EMAIL = "email"
    SMS = "sms"
    PHONE = "phone"
    LETTER = "letter"


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    IMMEDIATE = "immediate"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Base Agent Class
class DormantAgent:
    def __init__(self, llm_client: Any = None, config: Any = None):
        self.llm_client = llm_client
        self.config = config
        self.llm_available = bool(llm_client)

        self.llm_provider = 'unknown'
        self.llm_model_name = 'unknown'

        if self.llm_available and config and hasattr(config, 'rag_system') and hasattr(config.rag_system,
                                                                                       'llm_for_generation'):
            self.llm_config_gen = config.rag_system.llm_for_generation
            if hasattr(self.llm_config_gen, 'provider'):
                self.llm_provider = self.llm_config_gen.provider.lower()
            if hasattr(self.llm_config_gen, 'model_name'):
                self.llm_model_name = self.llm_config_gen.model_name
            # If the LLM client is directly passed and it's a LangChain model, try to infer provider
            elif isinstance(llm_client, BaseChatModel):
                # This is a bit of a heuristic but can help identify common LLM types
                if "groq" in str(type(llm_client)).lower():
                    self.llm_provider = "groq"
                elif "openai" in str(type(llm_client)).lower():
                    self.llm_provider = "openai"  # or azure_openai
                elif "ollama" in str(type(llm_client)).lower():
                    self.llm_provider = "ollama"
                self.llm_model_name = getattr(llm_client, 'model_name', getattr(llm_client, 'model', 'unknown_model'))
            elif hasattr(llm_client, 'model') and isinstance(llm_client.model,
                                                             str):  # For non-Langchain clients like openai.OpenAI()
                if "gpt" in llm_client.model: self.llm_provider = "openai"
                if "llama" in llm_client.model: self.llm_provider = "groq"  # Heuristic for Groq
                self.llm_model_name = llm_client.model

    def _call_generative_llm(self, user_prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 1000,
                             temperature: float = 0.1) -> str:
        """Helper to call the configured generative LLM synchronously."""
        if not self.llm_available or self.llm_client is None:
            logger.error("Generative LLM client not initialized for DormantAgent or LLM client is None.")
            return "LLM client not available."

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            # Check for OpenAI/Groq compatible client (uses .chat.completions.create)
            if hasattr(self.llm_client, 'chat') and hasattr(self.llm_client.chat, 'completions') and hasattr(
                    self.llm_client.chat.completions, 'create'):
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model_name if self.llm_model_name != 'unknown' else "llama3-8b-8192",
                    # Provide a default if unknown
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            # Check for LangChain BaseChatModel (uses .invoke)
            elif isinstance(self.llm_client, BaseChatModel) and hasattr(self.llm_client, 'invoke') and callable(
                    self.llm_client.invoke):
                response = self.llm_client.invoke(
                    messages,
                    config={'max_tokens': max_tokens, 'temperature': temperature}  # Pass config to invoke
                )
                return response.content
            # Check for Ollama client (uses .generate with prompt string and options)
            elif self.llm_provider == "ollama" and hasattr(self.llm_client, 'generate') and callable(
                    self.llm_client.generate):
                response = self.llm_client.generate(
                    model=self.llm_model_name if self.llm_model_name != 'unknown' else "llama3",
                    # Provide a default if unknown
                    prompt=f"{system_prompt}\n{user_prompt}" if system_prompt else user_prompt,
                    stream=False,
                    options={"num_predict": max_tokens, "temperature": temperature}
                )
                return response['response']
            else:
                logger.warning(
                    f"Unsupported LLM client type/provider ({self.llm_provider}) or missing expected methods. Attempting generic call (may fail).")
                # Fallback for other potential LLM client types, if they exist and use 'messages'
                # This is the problematic 'else' branch from your traceback.
                # The best fix is to correctly identify the provider and use its specific API.
                # If it still hits here, it implies a client not covered above.
                # Attempt to pass messages to a generic 'generate' but it's less reliable.
                try:
                    # Some legacy LangChain LLM models use generate(prompt=...)
                    # Some custom clients might expect messages directly.
                    response = self.llm_client.generate(messages=messages,
                                                        **{"max_tokens": max_tokens, "temperature": temperature})
                    return response.generations[0].text  # Common LangChain BaseChatModel.generate() response format
                except TypeError as te:
                    logger.error(
                        f"Generic LLM client.generate() failed with TypeError for {type(self.llm_client)}: {te}. Likely wrong argument signature.")
                    return "Failed to generate response: LLM client signature mismatch."
                except Exception as gen_e:
                    logger.error(f"Generic LLM client.generate() failed: {gen_e}")
                    return "Failed to generate response from LLM."


        except Exception as e:
            logger.error(f"Critical error calling generative LLM ({self.llm_provider}/{self.llm_model_name}): {e}",
                         exc_info=True)
            return f"Failed to generate response from LLM: {str(e)}"

    def get_llm_recommendation(self, context: str, prompt_type: str = "banking_compliance") -> str:
        """Get AI-generated recommendation using the provided LLM client."""
        if not self.llm_available:
            return "LLM not available for recommendations."

        system_prompts = {
            "banking_compliance": """You are a CBUAE banking compliance expert with deep knowledge of UAE banking regulations. 
            Provide specific, actionable recommendations for dormancy management and compliance issues. 
            Focus on regulatory requirements, risk mitigation, and operational efficiency.""",
            "dormancy_analysis": """You are a banking dormancy specialist. Analyze dormant account situations and provide 
            comprehensive recommendations including customer contact strategies, regulatory compliance steps, and risk assessments.""",
            "risk_assessment": """You are a banking risk management expert. Evaluate dormancy-related risks and provide 
            detailed mitigation strategies, focusing on operational, regulatory, and financial risks.""",
            "customer_engagement": """You are a customer relationship specialist in banking. Provide recommendations for 
            re-engaging dormant account holders, including communication strategies and reactivation approaches."""
        }
        system_prompt = system_prompts.get(prompt_type, system_prompts["banking_compliance"])
        user_prompt = f"Banking compliance context: {context}\n\nProvide specific recommendations with:\n1. Immediate actions\n2. Regulatory considerations\n3. Risk mitigation steps\n4. Timeline for implementation\n5. Success metrics"
        return self._call_generative_llm(user_prompt, system_prompt, max_tokens=1000, temperature=0.1)

    def generate_summary(self, data: Dict) -> str:
        """Generate AI-powered summary of dormancy analysis using the provided LLM client."""
        if not self.llm_available:
            return "LLM not available for summary."

        context = f"""
        Dormancy Analysis Data:
        - Account Type: {data.get('account_type', 'Unknown')}
        - Dormancy Days: {data.get('dormancy_days', 0)}
        - Account Balance: {data.get('account_balance', 0)} AED
        - Contact Attempts: {data.get('contact_attempts', 0)}
        - Customer Tier: {data.get('customer_tier', 'Standard')}
        - Last Activity: {data.get('last_activity', 'Unknown')}
        - Risk Factors: {data.get('risk_factors', [])}
        """
        system_prompt = "You are a banking analyst. Create concise, professional summaries of dormancy analysis results focusing on key insights and implications."
        user_prompt = f"Create a professional summary of this dormancy analysis:\n{context}\n\nProvide a concise summary highlighting key findings and implications."
        return self._call_generative_llm(user_prompt, system_prompt, max_tokens=300, temperature=0.2)

    def execute(self, account_data: Dict, report_date: datetime) -> Dict:
        raise NotImplementedError


# =============================================================================
# DORMANT ACCOUNT ANALYSIS AGENTS (As per image specification)
# =============================================================================

# 1. Safe Deposit Box Agent
class SafeDepositBoxAgent(DormantAgent):
    """Art. 2.6 CBUAE compliance for Safe Deposit Box dormancy."""

    def execute(self, account_data: Dict, report_date: datetime) -> Dict:
        agent_name = self.__class__.__name__
        last_activity_date = account_data.get('last_activity_date')
        if isinstance(last_activity_date, str):
            last_activity_date = datetime.fromisoformat(last_activity_date)
        elif not isinstance(last_activity_date, datetime):
            return {'agent': agent_name, 'error': 'Invalid last_activity_date format',
                    'status': ActivityStatus.PENDING_REVIEW.value, 'regulatory_citation': 'CBUAE Art. 2.6'}

        dormancy_days = (report_date - last_activity_date).days
        result = {'agent': agent_name, 'status': ActivityStatus.ACTIVE.value, 'dormancy_days': dormancy_days,
                  'regulatory_citation': 'CBUAE Art. 2.6'}

        if dormancy_days >= 1095 and account_data.get('account_type') == AccountType.SAFE_DEPOSIT.value:
            result.update({
                'status': ActivityStatus.DORMANT.value,
                'action': "Flag for Safe Deposit Box dormancy detection and next steps",
                'priority': Priority.HIGH.value,
                'risk_level': RiskLevel.HIGH.value
            })
            context = f"Safe Deposit Box dormant for {dormancy_days} days. Exceeded 3-year threshold for dormancy as per CBUAE Art. 2.6."

        return result


# 2. Investment Account Agent
class InvestmentAccountAgent(DormantAgent):
    """Art. 2.3 maturity-based dormancy for Investment Accounts."""

    def execute(self, account_data: Dict, report_date: datetime) -> Dict:
        agent_name = self.__class__.__name__
        maturity_date = account_data.get('maturity_date')

        if isinstance(maturity_date, str):
            try:
                maturity_date = datetime.fromisoformat(maturity_date)
            except ValueError:
                maturity_date = None
        elif not isinstance(maturity_date, datetime):
            maturity_date = None

        if not maturity_date:
            return {'agent': agent_name, 'error': 'Missing or invalid maturity date for maturity-based rule',
                    'status': ActivityStatus.PENDING_REVIEW.value, 'regulatory_citation': 'CBUAE Art. 2.3'}

        dormancy_days = (report_date - maturity_date).days
        result = {'agent': agent_name, 'status': ActivityStatus.ACTIVE.value, 'dormancy_days': dormancy_days,
                  'regulatory_citation': 'CBUAE Art. 2.3'}

        if dormancy_days >= 1095 and account_data.get('account_type') == AccountType.INVESTMENT.value:
            result.update({
                'status': ActivityStatus.DORMANT.value,
                'action': "Flag for Investment account dormancy detection (post-maturity)",
                'priority': Priority.HIGH.value,
                'risk_level': RiskLevel.HIGH.value
            })
            context = f"Investment account matured on {maturity_date.date()} and is unclaimed for {dormancy_days} days. Review required due to CBUAE Art. 2.3."

        return result


# 3. Fixed Deposit Agent
class FixedDepositAgent(DormantAgent):
    """Art. 2.2 renewal/claim monitoring for Fixed Deposits."""

    def execute(self, account_data: Dict, report_date: datetime) -> Dict:
        agent_name = self.__class__.__name__
        maturity_date = account_data.get('maturity_date')
        if isinstance(maturity_date, str):
            maturity_date = datetime.fromisoformat(maturity_date)
        elif not isinstance(maturity_date, datetime):
            return {'agent': agent_name, 'error': 'Missing or invalid maturity date',
                    'status': ActivityStatus.PENDING_REVIEW.value, 'regulatory_citation': 'CBUAE Art. 2.2'}

        dormancy_days = (report_date - maturity_date).days
        result = {'agent': agent_name, 'status': ActivityStatus.ACTIVE.value, 'dormancy_days': dormancy_days,
                  'regulatory_citation': 'CBUAE Art. 2.2'}

        if dormancy_days >= 1095 and account_data.get('account_type') == AccountType.FIXED_DEPOSIT.value:
            result.update({
                'status': ActivityStatus.DORMANT.value,
                'action': "Flag for Fixed/Term deposit dormancy detection",
                'priority': Priority.MEDIUM.value,
                'risk_level': RiskLevel.MEDIUM.value
            })
            context = f"Fixed Deposit matured {dormancy_days} days ago. Unclaimed funds as per CBUAE Art. 2.2."

        return result


# 4. Demand Deposit Agent
class DemandDepositAgent(DormantAgent):
    """Art. 2.1.1 activity tracking for Demand Deposits."""

    def execute(self, account_data: Dict, report_date: datetime) -> Dict:
        agent_name = self.__class__.__name__
        last_activity_date = account_data.get('last_activity_date')
        if isinstance(last_activity_date, str):
            last_activity_date = datetime.fromisoformat(last_activity_date)
        elif not isinstance(last_activity_date, datetime):
            return {'agent': agent_name, 'error': 'Invalid last_activity_date format',
                    'status': ActivityStatus.PENDING_REVIEW.value, 'regulatory_citation': 'CBUAE Art. 2.1.1'}

        dormancy_days = (report_date - last_activity_date).days
        result = {'agent': agent_name, 'status': ActivityStatus.ACTIVE.value, 'dormancy_days': dormancy_days,
                  'regulatory_citation': 'CBUAE Art. 2.1.1'}

        if dormancy_days >= 1095 and account_data.get('account_type') == AccountType.DEMAND_DEPOSIT.value:
            result.update({
                'status': ActivityStatus.DORMANT.value,
                'action': "Flag as dormant and initiate contact for Demand Deposit",
                'priority': Priority.MEDIUM.value,
                'risk_level': RiskLevel.MEDIUM.value
            })
            context = f"Demand Deposit account dormant for {dormancy_days} days. Flagging and contact initiation as per CBUAE Art. 2.1.1."

        return result


# 5. Payment Instruments Agent
class PaymentInstrumentsAgent(DormantAgent):
    """Art. 2.4 unclaimed item detection for Payment Instruments."""

    def execute(self, account_data: Dict, report_date: datetime) -> Dict:
        agent_name = self.__class__.__name__
        last_activity_date = account_data.get('last_activity_date')
        if isinstance(last_activity_date, str):
            last_activity_date = datetime.fromisoformat(last_activity_date)
        elif not isinstance(last_activity_date, datetime):
            return {'agent': agent_name, 'error': 'Invalid last_activity_date format',
                    'status': ActivityStatus.PENDING_REVIEW.value, 'regulatory_citation': 'CBUAE Art. 2.4'}

        dormancy_days = (report_date - last_activity_date).days
        result = {'agent': agent_name, 'status': ActivityStatus.ACTIVE.value, 'dormancy_days': dormancy_days,
                  'regulatory_citation': 'CBUAE Art. 2.4'}

        if dormancy_days >= 365 and account_data.get('account_type') == AccountType.UNCLAIMED_INSTRUMENT.value:
            result.update({
                'status': ActivityStatus.UNCLAIMED.value,
                'action': "Flag for Unclaimed instruments detection and process for ledger transfer",
                'priority': Priority.CRITICAL.value,
                'risk_level': RiskLevel.CRITICAL.value
            })
            context = f"Unclaimed payment instrument dormant for {dormancy_days} days. Requires ledger transfer as per CBUAE Art. 2.4."

        return result


# 6. CB Transfer Agent
class CBTransferAgent(DormantAgent):
    """Art. 8.1-8.2 eligibility assessment for Central Bank transfer."""

    def execute(self, account_data: Dict, report_date: datetime) -> Dict:
        agent_name = self.__class__.__name__
        last_activity_date = account_data.get('last_activity_date')
        if isinstance(last_activity_date, str):
            last_activity_date = datetime.fromisoformat(last_activity_date)
        elif not isinstance(last_activity_date, datetime):
            return {'agent': agent_name, 'error': 'Invalid last_activity_date format',
                    'status': ActivityStatus.PENDING_REVIEW.value, 'regulatory_citation': 'CBUAE Art. 8.1-8.2'}

        dormancy_days = (report_date - last_activity_date).days
        result = {'agent': agent_name, 'eligible': False, 'dormancy_days': dormancy_days,
                  'regulatory_citation': 'CBUAE Art. 8.1-8.2'}

        if dormancy_days >= 1825:  # 5 years is the common threshold for transfer eligibility
            result.update({
                'eligible': True,
                'action': "Initiate CBUAE transfer",
                'priority': Priority.HIGH.value,
                'risk_level': RiskLevel.CRITICAL.value
            })
            context = f"Account dormant for {dormancy_days} days. Eligible for CBUAE transfer as per Art. 8.1-8.2."

        return result


# 7. Article 3 Process Agent
class Article3ProcessAgent(DormantAgent):
    """Contact requirement monitoring based on CBUAE Article 3."""

    def execute(self, account_data: Dict, report_date: datetime) -> Dict:
        agent_name = self.__class__.__name__
        dormancy_status = account_data.get('dormancy_status')
        contact_attempts = account_data.get('contact_attempts_made', 0)
        result = {'agent': agent_name, 'status': 'N/A', 'action': None, 'regulatory_citation': 'CBUAE Art. 3'}

        # Only process if the account is already 'dormant' based on other agents/data
        if dormancy_status == 'dormant':
            # Check if contact attempts are less than 3 (assuming 3 attempts are required per CBUAE Art. 3)
            if contact_attempts < 3:
                result.update({
                    'status': 'Process Pending',
                    'action': f"Initiate/Review CBUAE Article 3 Contact Process (Attempts made: {contact_attempts})",
                    'priority': Priority.MEDIUM.value,
                    'risk_level': RiskLevel.MEDIUM.value
                })
                context = f"Dormant account with {contact_attempts} contact attempts. CBUAE Article 3 process for contact is needed."

            else:
                # If 3 or more attempts are made and it's still dormant, it might be ready for next stage (e.g., transfer)
                result.update({
                    'status': 'Contact Attempts Exhausted',
                    'action': f"Review for CBUAE Article 3 completion/escalation (Attempts made: {contact_attempts})",
                    'priority': Priority.LOW.value,  # Priority might drop as next steps are different
                    'risk_level': RiskLevel.LOW.value
                })
        return result


# 8. High-Value Account Agent
class HighValueAccountAgent(DormantAgent):
    """Balance-based prioritization for high-value accounts."""

    def execute(self, account_data: Dict, report_date: datetime) -> Dict:
        agent_name = self.__class__.__name__
        balance = account_data.get('balance', 0)
        dormancy_status = account_data.get('dormancy_status')
        high_value_threshold = 100000  # 100,000 AED
        result = {'agent': agent_name, 'status': 'Standard Value', 'action': None, 'regulatory_citation': 'Internal'}

        if dormancy_status == 'dormant' and balance >= high_value_threshold:
            result.update({
                'status': 'High Value Dormant',
                'action': "Escalate high-value dormant account for manual review",
                'priority': Priority.CRITICAL.value,
                'risk_level': RiskLevel.HIGH.value
            })
            context = f"High-value dormant account with balance {balance}. Requires immediate special handling."

        return result


# 9. Transition Detection Agent
class TransitionDetectionAgent(DormantAgent):
    """Dormant-to-active identification."""

    def execute(self, account_data: Dict, report_date: datetime) -> Dict:
        agent_name = self.__class__.__name__
        previous_status = account_data.get('previous_dormancy_status')
        current_status = account_data.get('current_activity_status')
        result = {'agent': agent_name, 'status': 'No Transition', 'action': None, 'regulatory_citation': 'Internal'}

        # This agent checks for reactivation from 'dormant' to 'active'
        if previous_status == 'dormant' and current_status == 'active':
            result.update({
                'status': 'Reactivated',
                'action': "Verify reactivation process and update internal records",
                'priority': Priority.LOW.value,
                'risk_level': RiskLevel.LOW.value
            })
            context = f"Account transitioned from dormant to active. Verify compliance with reactivation procedures."

        return result


# =============================================================================
# Master Orchestrator for Dormancy Analyzers
# =============================================================================

class DormantAccountOrchestrator:
    AGENT_CLASS_MAP = {
        'safe_deposit_box_agent': SafeDepositBoxAgent,
        'investment_account_agent': InvestmentAccountAgent,
        'fixed_deposit_agent': FixedDepositAgent,
        'demand_deposit_agent': DemandDepositAgent,
        'payment_instruments_agent': PaymentInstrumentsAgent,
        'cb_transfer_agent': CBTransferAgent,
        'article_3_process_agent': Article3ProcessAgent,
        'high_value_account_agent': HighValueAccountAgent,
        'transition_detection_agent': TransitionDetectionAgent,
    }

    def __init__(self, llm_client: Any = None, config: Any = None):
        self.agents = {
            name: agent_class(llm_client, config)
            for name, agent_class in self.AGENT_CLASS_MAP.items()
        }

    def run_all_agents(self, account_data: Dict, report_date: datetime) -> Dict:
        results = {}
        for agent_name, agent in self.agents.items():
            try:
                # The 'report_date' parameter is now correctly passed to all agent execute methods.
                results[agent_name] = agent.execute(account_data, report_date)
            except Exception as e:
                logger.error(f"Error executing agent {agent_name}: {e}", exc_info=True)
                results[agent_name] = {'error': str(e)}
        return results