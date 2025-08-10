

### **2. Updated `DormantAgent.py` (Synchronous Version)**
# --- START OF FILE DormantAgent.py ---

from enum import Enum
import os
import csv
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import logging



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
    def __init__(self, llm_client: Any, config: Any):  # Accept config here
        self.llm_client = llm_client
        # Use the config passed to the agent, not global import for LLM config
        self.llm_config_gen = config.rag_system.llm_for_generation
        self.llm_provider = self.llm_config_gen.provider.lower()
        self.llm_model_name = self.llm_config_gen.model_name
        self.config = config  # Store config for other potential uses within agent

    def _call_generative_llm(self, user_prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 1000,
                             temperature: float = 0.1) -> str:
        """Helper to call the configured generative LLM synchronously."""
        if not self.llm_client:
            logger.error("Generative LLM client not initialized for DormantAgent.")
            return "LLM client not available."

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            if self.llm_provider in ["openai", "azure_openai", "groq"]:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            elif self.llm_provider == "ollama":
                response = self.llm_client.generate(
                    model=self.llm_model_name,
                    prompt=f"{system_prompt}\n{user_prompt}" if system_prompt else user_prompt,
                    stream=False,
                    options={"num_predict": max_tokens}
                )
                return response['response']
            else:
                logger.warning(
                    f"Unsupported LLM provider for direct sync call: {self.llm_provider}. Using generic call.")
                response = self.llm_client.generate(
                    prompt=f"{system_prompt}\n{user_prompt}" if system_prompt else user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response['text'] if 'text' in response else str(response)

        except Exception as e:
            logger.error(f"Error calling generative LLM ({self.llm_provider}/{self.llm_model_name}): {e}",
                         exc_info=True)
            return f"Failed to generate response from LLM: {str(e)}"

    def get_llm_recommendation(self, context: str, prompt_type: str = "banking_compliance") -> str:
        """Get AI-generated recommendation using the provided LLM client."""
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

    def execute(self, account_data: Dict) -> Dict:
        raise NotImplementedError


# =============================================================================
# DORMANT ACCOUNT ANALYSIS AGENTS (As per image specification)
# =============================================================================

# 1. Safe Deposit Box Agent
class SafeDepositBoxAgent(DormantAgent):
    """Art. 2.6 CBUAE compliance for Safe Deposit Box dormancy."""

    def execute(self, account_data: Dict) -> Dict:
        agent_name = 'safe_deposit_box_agent'
        last_activity_date = account_data.get('last_activity_date')
        if isinstance(last_activity_date, str):
            last_activity_date = datetime.fromisoformat(last_activity_date)
        elif not isinstance(last_activity_date, datetime):
            return {'agent': agent_name, 'error': 'Invalid last_activity_date format',
                    'status': ActivityStatus.PENDING_REVIEW.value, 'regulatory_citation': 'CBUAE Art. 2.6'}

        dormancy_days = (datetime.now() - last_activity_date).days
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
            result['recommendation'] = self.get_llm_recommendation(context, "dormancy_analysis")
            result['ai_summary'] = self.generate_summary(
                {'account_type': 'Safe Deposit Box', 'dormancy_days': dormancy_days})
        return result


# 2. Investment Account Agent
class InvestmentAccountAgent(DormantAgent):
    """Art. 2.3 maturity-based dormancy for Investment Accounts."""

    def execute(self, account_data: Dict) -> Dict:
        agent_name = 'investment_account_agent'
        last_activity_date = account_data.get('last_activity_date')
        if isinstance(last_activity_date, str):
            last_activity_date = datetime.fromisoformat(last_activity_date)
        elif not isinstance(last_activity_date, datetime):
            return {'agent': agent_name, 'error': 'Invalid last_activity_date format',
                    'status': ActivityStatus.PENDING_REVIEW.value, 'regulatory_citation': 'CBUAE Art. 2.3'}

        dormancy_days = (datetime.now() - last_activity_date).days
        result = {'agent': agent_name, 'status': ActivityStatus.ACTIVE.value, 'dormancy_days': dormancy_days,
                  'regulatory_citation': 'CBUAE Art. 2.3'}

        if dormancy_days >= 1095 and account_data.get('account_type') == AccountType.INVESTMENT.value:
            result.update({
                'status': ActivityStatus.DORMANT.value,
                'action': "Flag for Investment account dormancy detection",
                'priority': Priority.HIGH.value,
                'risk_level': RiskLevel.HIGH.value
            })
            context = f"Investment account dormant for {dormancy_days} days. Review required due to CBUAE Art. 2.3."
            result['recommendation'] = self.get_llm_recommendation(context, "risk_assessment")
            result['ai_summary'] = self.generate_summary(
                {'account_type': 'Investment Account', 'dormancy_days': dormancy_days})
        return result


# 3. Fixed Deposit Agent
class FixedDepositAgent(DormantAgent):
    """Art. 2.2 renewal/claim monitoring for Fixed Deposits."""

    def execute(self, account_data: Dict) -> Dict:
        agent_name = 'fixed_deposit_agent'
        maturity_date = account_data.get('maturity_date')
        if isinstance(maturity_date, str):
            maturity_date = datetime.fromisoformat(maturity_date)
        elif not isinstance(maturity_date, datetime):
            return {'agent': agent_name, 'error': 'Missing or invalid maturity date',
                    'status': ActivityStatus.PENDING_REVIEW.value, 'regulatory_citation': 'CBUAE Art. 2.2'}

        dormancy_days = (datetime.now() - maturity_date).days
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
            result['recommendation'] = self.get_llm_recommendation(context, "customer_engagement")
            result['ai_summary'] = self.generate_summary(
                {'account_type': 'Fixed Deposit (Post-Maturity)', 'dormancy_days': dormancy_days})
        return result


# 4. Demand Deposit Agent
class DemandDepositAgent(DormantAgent):
    """Art. 2.1.1 activity tracking for Demand Deposits."""

    def execute(self, account_data: Dict) -> Dict:
        agent_name = 'demand_deposit_agent'
        last_activity_date = account_data.get('last_activity_date')
        if isinstance(last_activity_date, str):
            last_activity_date = datetime.fromisoformat(last_activity_date)
        elif not isinstance(last_activity_date, datetime):
            return {'agent': agent_name, 'error': 'Invalid last_activity_date format',
                    'status': ActivityStatus.PENDING_REVIEW.value, 'regulatory_citation': 'CBUAE Art. 2.1.1'}

        dormancy_days = (datetime.now() - last_activity_date).days
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
            result['recommendation'] = self.get_llm_recommendation(context, "customer_engagement")
            result['ai_summary'] = self.generate_summary(
                {'account_type': 'Demand Deposit', 'dormancy_days': dormancy_days})
        return result


# 5. Payment Instruments Agent
class PaymentInstrumentsAgent(DormantAgent):
    """Art. 2.4 unclaimed item detection for Payment Instruments."""

    def execute(self, account_data: Dict) -> Dict:
        agent_name = 'payment_instruments_agent'
        last_activity_date = account_data.get('last_activity_date')
        if isinstance(last_activity_date, str):
            last_activity_date = datetime.fromisoformat(last_activity_date)
        elif not isinstance(last_activity_date, datetime):
            return {'agent': agent_name, 'error': 'Invalid last_activity_date format',
                    'status': ActivityStatus.PENDING_REVIEW.value, 'regulatory_citation': 'CBUAE Art. 2.4'}

        dormancy_days = (datetime.now() - last_activity_date).days
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
            result['recommendation'] = self.get_llm_recommendation(context, "banking_compliance")
            result['ai_summary'] = self.generate_summary(
                {'account_type': 'Unclaimed Payment Instrument', 'dormancy_days': dormancy_days})
        return result


# 6. CB Transfer Agent
class CBTransferAgent(DormantAgent):
    """Art. 8.1-8.2 eligibility assessment for Central Bank transfer."""

    def execute(self, account_data: Dict) -> Dict:
        agent_name = 'cb_transfer_agent'
        last_activity_date = account_data.get('last_activity_date')
        if isinstance(last_activity_date, str):
            last_activity_date = datetime.fromisoformat(last_activity_date)
        elif not isinstance(last_activity_date, datetime):
            return {'agent': agent_name, 'error': 'Invalid last_activity_date format',
                    'status': ActivityStatus.PENDING_REVIEW.value, 'regulatory_citation': 'CBUAE Art. 8.1-8.2'}

        dormancy_days = (datetime.now() - last_activity_date).days
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
            result['recommendation'] = self.get_llm_recommendation(context, "banking_compliance")
            result['ai_summary'] = self.generate_summary(
                {'account_type': account_data.get('account_type'), 'dormancy_days': dormancy_days})
        return result


# 7. Article 3 Process Agent
class Article3ProcessAgent(DormantAgent):
    """Contact requirement monitoring based on CBUAE Article 3."""

    def execute(self, account_data: Dict) -> Dict:
        agent_name = 'article_3_process_agent'
        dormancy_status = account_data.get('dormancy_status')
        contact_attempts = account_data.get('contact_attempts_made', 0)
        result = {'agent': agent_name, 'status': 'N/A', 'action': None, 'regulatory_citation': 'CBUAE Art. 3'}

        if dormancy_status == 'dormant' and contact_attempts < 3:
            result.update({
                'status': 'Process Pending',
                'action': "Initiate/Review CBUAE Article 3 Contact Process",
                'priority': Priority.MEDIUM.value,
                'risk_level': RiskLevel.MEDIUM.value
            })
            context = f"Dormant account with {contact_attempts} contact attempts. CBUAE Article 3 process for contact is needed."
            result['recommendation'] = self.get_llm_recommendation(context, "banking_compliance")
            result['ai_summary'] = self.generate_summary(
                {'account_id': account_data.get('account_id'), 'contact_attempts_made': contact_attempts})
        return result


# 8. High-Value Account Agent
class HighValueAccountAgent(DormantAgent):
    """Balance-based prioritization for high-value accounts."""

    def execute(self, account_data: Dict) -> Dict:
        agent_name = 'high_value_account_agent'
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
            result['recommendation'] = self.get_llm_recommendation(context, "risk_assessment")
            result['ai_summary'] = self.generate_summary(
                {'account_id': account_data.get('account_id'), 'balance': balance})
        return result


# 9. Transition Detection Agent
class TransitionDetectionAgent(DormantAgent):
    """Dormant-to-active identification."""

    def execute(self, account_data: Dict) -> Dict:
        agent_name = 'transition_detection_agent'
        previous_status = account_data.get('previous_dormancy_status')
        current_status = account_data.get('current_activity_status')
        result = {'agent': agent_name, 'status': 'No Transition', 'action': None, 'regulatory_citation': 'Internal'}

        if previous_status == 'dormant' and current_status == 'active':
            result.update({
                'status': 'Reactivated',
                'action': "Verify reactivation process and update internal records",
                'priority': Priority.LOW.value,
                'risk_level': RiskLevel.LOW.value
            })
            context = f"Account transitioned from dormant to active. Verify compliance with reactivation procedures."
            result['recommendation'] = self.get_llm_recommendation(context, "banking_compliance")
            result['ai_summary'] = self.generate_summary(
                {'account_id': account_data.get('account_id'), 'transition': 'Dormant to Active'})
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

    # --- END OF FILE DormantAgent.py ---