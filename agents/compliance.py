# --- COMPLETE 17 COMPLIANCE VERIFICATION AGENTS (As per PDF Documentation) ---

from enum import Enum
import os
import csv
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import logging
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Constants and Enums
class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL_COMPLIANT = "partial_compliant"
    PENDING_REVIEW = "pending_review"
    CRITICAL_VIOLATION = "critical_violation"


class ViolationType(Enum):
    ARTICLE_2_VIOLATION = "article_2_violation"
    ARTICLE_3_1_VIOLATION = "article_3_1_violation"
    ARTICLE_3_4_VIOLATION = "article_3_4_violation"
    CONTACT_VIOLATION = "contact_violation"
    TRANSFER_VIOLATION = "transfer_violation"
    DOCUMENTATION_VIOLATION = "documentation_violation"
    TIMELINE_VIOLATION = "timeline_violation"
    AMOUNT_VIOLATION = "amount_violation"
    REPORTING_VIOLATION = "reporting_violation"
    FX_VIOLATION = "fx_violation"
    CLAIM_VIOLATION = "claim_violation"
    AUDIT_VIOLATION = "audit_violation"


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


class CBUAEArticle(Enum):
    ARTICLE_2 = "article_2"
    ARTICLE_3_1 = "article_3_1"
    ARTICLE_3_4 = "article_3_4"
    ARTICLE_3_6 = "article_3_6"
    ARTICLE_3_7 = "article_3_7"
    ARTICLE_3_9 = "article_3_9"
    ARTICLE_3_10 = "article_3_10"
    ARTICLE_4 = "article_4"
    ARTICLE_5 = "article_5"
    ARTICLE_7_3 = "article_7_3"
    ARTICLE_8 = "article_8"
    ARTICLE_8_5 = "article_8_5"


class AccountType(Enum):
    SAFE_DEPOSIT = "safe_deposit"
    INVESTMENT = "investment"
    FIXED_DEPOSIT = "fixed_deposit"
    DEMAND_DEPOSIT = "demand_deposit"
    UNCLAIMED_INSTRUMENT = "unclaimed_instrument"
    SAVINGS = "savings"
    CURRENT = "current"


class ActivityStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DORMANT = "dormant"
    UNCLAIMED = "unclaimed"


# Base Compliance Agent Class
class ComplianceAgent:
    def __init__(self, llm_client: Any = None, config: Any = None):
        self.llm_client = llm_client
        self.config = config
        self.llm_available = bool(llm_client)

        if self.llm_available and config and hasattr(config, 'rag_system') and hasattr(config.rag_system,
                                                                                       'llm_for_generation'):
            self.llm_config_gen = config.rag_system.llm_for_generation
            self.llm_provider = self.llm_config_gen.provider.lower() if hasattr(self.llm_config_gen,
                                                                                'provider') else 'unknown'
            self.llm_model_name = self.llm_config_gen.model_name if hasattr(self.llm_config_gen,
                                                                            'model_name') else 'unknown'
        else:
            self.llm_provider = 'unknown'
            self.llm_model_name = 'unknown'

    def _call_generative_llm(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Synchronous helper to call the configured generative LLM."""
        if not self.llm_available:
            return "LLM client not available - using rule-based analysis."

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            if self.llm_provider in ["openai", "azure_openai", "groq"]:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=messages,
                    temperature=0.1
                )
                return response.choices[0].message.content
            elif self.llm_provider == "ollama":
                response = self.llm_client.generate(
                    model=self.llm_model_name,
                    prompt=f"{system_prompt}\n{user_prompt}" if system_prompt else user_prompt,
                    stream=False
                )
                return response.get('response', 'No response from Ollama')
            else:
                return f"Unsupported LLM provider: {self.llm_provider}"

        except Exception as e:
            logger.error(f"Error calling generative LLM: {e}")
            return f"Failed to generate LLM response: {str(e)}"

    def get_llm_recommendation(self, violation_context: str, article: str = None) -> str:
        """Get AI-powered compliance recommendations."""
        if not self.llm_available:
            return "LLM not available for recommendations."

        system_prompt = """You are a CBUAE banking compliance expert. Provide specific regulatory analysis, 
        immediate remediation steps, and compliance recommendations for the given violation context."""

        user_prompt = f"""
        Compliance Analysis Required:
        Article: {article or 'General Compliance'}
        Violation: {violation_context}

        Provide:
        1. Regulatory assessment and citation
        2. Immediate remediation actions
        3. Risk mitigation recommendations
        4. Compliance monitoring steps
        """
        return self._call_generative_llm(user_prompt, system_prompt)

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        raise NotImplementedError("Each compliance agent must implement the execute method")


# =============================================================================
# ALL 17 COMPLIANCE VERIFICATION AGENTS (As per PDF Documentation)
# =============================================================================

# Agent 1: Article 2 Compliance Agent - Detect Dormant Accounts
class Article2ComplianceAgent(ComplianceAgent):
    """
    Purpose: Detects accounts that meet dormancy criteria per CBUAE Article 2
    Functionality: Verifies account dormancy classification based on inactivity periods
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'article_2_compliance_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Art. 2 - Dormant Account Definition'
        }

        last_activity_date = account_data.get('last_activity_date') or account_data.get('last_transaction_date')
        if isinstance(last_activity_date, str):
            try:
                last_activity_date = datetime.fromisoformat(last_activity_date.replace('Z', '+00:00'))
            except ValueError:
                result['compliance_status'] = ComplianceStatus.PENDING_REVIEW.value
                result['violations'] = ['Invalid last activity date format']
                return result

        dormancy_status = account_data.get('dormancy_status', 'active')

        if isinstance(last_activity_date, datetime):
            days_inactive = (datetime.now() - last_activity_date).days
            is_dormant_by_logic = days_inactive >= 1095  # 3 years as per CBUAE
            is_flagged_as_dormant = dormancy_status == 'dormant'

            if is_dormant_by_logic and not is_flagged_as_dormant:
                violation = f"Account meets dormancy criteria (inactive {days_inactive} days) but not flagged as dormant"
                result.update({
                    'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                    'violations': [violation],
                    'priority': Priority.HIGH.value,
                    'risk_level': RiskLevel.HIGH.value,
                    'action': 'Flag account as dormant and initiate dormancy procedures'
                })
                result['recommendation'] = self.get_llm_recommendation(violation, "CBUAE Article 2")

        return result


# Agent 2: Article 3.1 Process Compliance Agent - Contact Requirements
class Article31ProcessComplianceAgent(ComplianceAgent):
    """
    Purpose: Verifies compliance with customer contact requirements per CBUAE Article 3.1
    Functionality: Checks contact attempts, methods, and documentation
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'article_3_1_process_compliance_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Art. 3.1 - Customer Contact Requirements'
        }

        if account_data.get('dormancy_status') == 'dormant':
            violations = []
            contact_attempts = account_data.get('contact_attempts_made', 0)
            customer_type = account_data.get('customer_type', 'individual')
            account_value = account_data.get('balance_current', 0)

            # Determine required contact attempts
            required_attempts = 3  # Default minimum
            if customer_type == 'individual' and account_value >= 25000:
                required_attempts = 5
            elif customer_type == 'corporate':
                required_attempts = 5 if account_value >= 100000 else 4

            if contact_attempts < required_attempts:
                violations.append(f"Insufficient contact attempts: {contact_attempts} of {required_attempts} required")

            if not account_data.get('contact_log_present', False):
                violations.append("Contact efforts not properly documented")

            contact_methods = account_data.get('contact_methods_used', [])
            if isinstance(contact_methods, str):
                contact_methods = contact_methods.split(',')
            if len(contact_methods) < 2:
                violations.append("Contact attempts lack multi-channel diversity")

            if violations:
                result.update({
                    'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                    'violations': violations,
                    'priority': Priority.HIGH.value,
                    'risk_level': RiskLevel.HIGH.value,
                    'action': f"Complete {required_attempts - contact_attempts} additional contact attempts"
                })
                result['recommendation'] = self.get_llm_recommendation("\n".join(violations), "CBUAE Article 3.1")

        return result


# Agent 3: Article 3.4 Transfer Compliance Agent - Transfer to Central Bank
class Article34TransferComplianceAgent(ComplianceAgent):
    """
    Purpose: Ensures compliance with Central Bank transfer requirements per CBUAE Article 3.4
    Functionality: Verifies transfer eligibility, documentation, and timeline compliance
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'article_3_4_transfer_compliance_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Art. 3.4 - Central Bank Transfer Requirements'
        }

        if account_data.get('is_eligible_for_cb_transfer', False):
            violations = []

            if not account_data.get('transfer_docs_prepared', False):
                violations.append("Required documentation for Central Bank transfer not prepared")

            balance = account_data.get('balance_current', 0) or account_data.get('balance', 0)
            if balance <= 0:
                violations.append("Account marked for transfer has zero or negative balance")

            # Check transfer timeline compliance
            transfer_eligibility_date = account_data.get('transfer_eligibility_date')
            if transfer_eligibility_date:
                if isinstance(transfer_eligibility_date, str):
                    transfer_eligibility_date = datetime.fromisoformat(transfer_eligibility_date.replace('Z', '+00:00'))

                if isinstance(transfer_eligibility_date, datetime):
                    days_since_eligible = (datetime.now() - transfer_eligibility_date).days
                    if days_since_eligible > 90 and not account_data.get('transferred_to_cb_date'):
                        violations.append(f"Transfer overdue by {days_since_eligible - 90} days")

            if violations:
                result.update({
                    'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                    'violations': violations,
                    'priority': Priority.CRITICAL.value,
                    'risk_level': RiskLevel.CRITICAL.value,
                    'action': 'Immediately process Central Bank transfer'
                })
                result['recommendation'] = self.get_llm_recommendation("\n".join(violations), "CBUAE Article 3.4")

        return result


# Agent 4: Contact Verification Agent - Customer Contact Validation
class ContactVerificationAgent(ComplianceAgent):
    """
    Purpose: Validates customer contact information and communication attempts
    Functionality: Verifies contact details accuracy and communication effectiveness
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'contact_verification_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Art. 3.1 - Contact Verification'
        }

        violations = []

        # Verify contact information completeness
        email = account_data.get('customer_email', '')
        phone = account_data.get('customer_phone', '')
        address = account_data.get('customer_address', '')

        if not email or not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            violations.append("Invalid or missing customer email address")

        if not phone or len(re.sub(r'[^\d]', '', phone)) < 7:
            violations.append("Invalid or missing customer phone number")

        if not address or len(address.strip()) < 10:
            violations.append("Incomplete customer address information")

        # Check contact attempt outcomes
        if account_data.get('dormancy_status') == 'dormant':
            contact_outcomes = account_data.get('contact_outcomes', [])
            if isinstance(contact_outcomes, str):
                contact_outcomes = contact_outcomes.split(',')

            successful_contacts = len([outcome for outcome in contact_outcomes if 'successful' in outcome.lower()])
            if successful_contacts == 0 and account_data.get('contact_attempts_made', 0) > 0:
                violations.append("No successful customer contact despite multiple attempts")

        if violations:
            result.update({
                'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                'violations': violations,
                'priority': Priority.MEDIUM.value,
                'risk_level': RiskLevel.MEDIUM.value,
                'action': 'Update customer contact information and retry contact'
            })
            result['recommendation'] = self.get_llm_recommendation("\n".join(violations), "Contact Verification")

        return result


# Agent 5: Transfer Eligibility Agent - Eligibility Assessment
class TransferEligibilityAgent(ComplianceAgent):
    """
    Purpose: Assesses account eligibility for Central Bank transfer
    Functionality: Evaluates dormancy period, contact completion, and transfer criteria
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'transfer_eligibility_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Art. 3.4 - Transfer Eligibility Criteria'
        }

        account_type = account_data.get('account_type', '')
        last_activity_date = account_data.get('last_activity_date')
        contact_attempts = account_data.get('contact_attempts_made', 0)

        if isinstance(last_activity_date, str):
            last_activity_date = datetime.fromisoformat(last_activity_date.replace('Z', '+00:00'))

        if isinstance(last_activity_date, datetime):
            days_inactive = (datetime.now() - last_activity_date).days

            # Determine eligibility based on account type and dormancy period
            is_eligible = False
            required_dormancy_days = 1825  # Default 5 years

            if account_type == AccountType.SAFE_DEPOSIT.value:
                required_dormancy_days = 1825  # 5 years for safe deposit
            elif account_type == AccountType.UNCLAIMED_INSTRUMENT.value:
                required_dormancy_days = 1095  # 3 years for unclaimed instruments

            is_eligible = (days_inactive >= required_dormancy_days and
                           contact_attempts >= 3 and
                           account_data.get('dormancy_status') == 'dormant')

            current_eligibility = account_data.get('is_eligible_for_cb_transfer', False)

            if is_eligible and not current_eligibility:
                violation = f"Account meets transfer eligibility criteria but not marked as eligible"
                result.update({
                    'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                    'violations': [violation],
                    'priority': Priority.HIGH.value,
                    'risk_level': RiskLevel.HIGH.value,
                    'action': 'Mark account as eligible for Central Bank transfer'
                })
                result['recommendation'] = self.get_llm_recommendation(violation, "Transfer Eligibility")

            elif not is_eligible and current_eligibility:
                violation = f"Account marked as transfer eligible but doesn't meet criteria"
                result.update({
                    'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                    'violations': [violation],
                    'priority': Priority.MEDIUM.value,
                    'risk_level': RiskLevel.MEDIUM.value,
                    'action': 'Review and correct transfer eligibility status'
                })

        return result


# Agent 6: FX Conversion Check Agent - Foreign Currency Handling
class FXConversionCheckAgent(ComplianceAgent):
    """
    Purpose: Ensures foreign currency accounts are converted to AED before Central Bank transfer
    Functionality: Verifies currency conversion compliance per CBUAE Article 8.5
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'fx_conversion_check_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Art. 8.5 - Foreign Currency Conversion'
        }

        currency = account_data.get('currency', 'AED')
        balance = account_data.get('balance_current', 0) or account_data.get('balance', 0)

        if (account_data.get('is_eligible_for_cb_transfer', False) and
                currency != 'AED' and balance > 0):

            if not account_data.get('fx_conversion_complete', False):
                violation = f"Foreign currency account ({currency}, {balance:,.2f}) requires conversion to AED"
                result.update({
                    'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                    'violations': [violation],
                    'priority': Priority.HIGH.value,
                    'risk_level': RiskLevel.MEDIUM.value,
                    'action': f'Convert {currency} balance to AED before transfer'
                })
                result['recommendation'] = self.get_llm_recommendation(violation, "CBUAE Article 8.5")

        return result


# Agent 7: Process Management Agent - Overall Process Oversight
class ProcessManagementAgent(ComplianceAgent):
    """
    Purpose: Provides overall process management and coordination oversight
    Functionality: Monitors process flows, timelines, and inter-agent dependencies
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'process_management_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'Internal Process Management'
        }

        violations = []

        # Check process stage consistency
        dormancy_status = account_data.get('dormancy_status', 'active')
        process_stage = account_data.get('process_stage', 'initial')

        # Validate process flow logic
        if dormancy_status == 'dormant' and process_stage == 'initial':
            violations.append("Dormant account still in initial process stage")

        if account_data.get('is_eligible_for_cb_transfer') and process_stage not in ['transfer_ready', 'transferred']:
            violations.append("Transfer-eligible account not in appropriate process stage")

        # Check for process bottlenecks
        last_process_update = account_data.get('last_process_update_date')
        if last_process_update:
            if isinstance(last_process_update, str):
                last_process_update = datetime.fromisoformat(last_process_update.replace('Z', '+00:00'))

            if isinstance(last_process_update, datetime):
                days_since_update = (datetime.now() - last_process_update).days
                if days_since_update > 30 and dormancy_status == 'dormant':
                    violations.append(f"Process stagnant for {days_since_update} days")

        if violations:
            result.update({
                'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                'violations': violations,
                'priority': Priority.MEDIUM.value,
                'risk_level': RiskLevel.MEDIUM.value,
                'action': 'Review and advance process stage appropriately'
            })
            result['recommendation'] = self.get_llm_recommendation("\n".join(violations), "Process Management")

        return result


# Agent 8: Documentation Review Agent - Document Completeness
class DocumentationReviewAgent(ComplianceAgent):
    """
    Purpose: Reviews and validates completeness of required documentation
    Functionality: Checks KYC, audit trails, contact logs, and regulatory documentation
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'documentation_review_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'Internal Documentation Standards'
        }

        violations = []

        # Check KYC documentation
        if not account_data.get('customer_kyc_complete', True):
            violations.append("Customer KYC documentation incomplete or expired")

        # Check audit trail completeness
        if not account_data.get('audit_log_complete', True):
            violations.append("Comprehensive audit trail missing or incomplete")

        # Check contact documentation for dormant accounts
        if account_data.get('dormancy_status') == 'dormant':
            if not account_data.get('contact_log_present', False):
                violations.append("Contact attempt documentation missing")

            if not account_data.get('dormancy_declaration_signed', False):
                violations.append("Dormancy declaration not properly documented")

        # Check transfer documentation
        if account_data.get('is_eligible_for_cb_transfer', False):
            if not account_data.get('transfer_docs_prepared', False):
                violations.append("Transfer documentation not prepared")

            if not account_data.get('legal_verification_complete', False):
                violations.append("Legal verification documentation incomplete")

        if violations:
            result.update({
                'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                'violations': violations,
                'priority': Priority.MEDIUM.value,
                'risk_level': RiskLevel.MEDIUM.value,
                'action': 'Complete missing documentation and update records'
            })
            result['recommendation'] = self.get_llm_recommendation("\n".join(violations), "Documentation Standards")

        return result


# Agent 9: Timeline Compliance Agent - Regulatory Timeline Management
class TimelineComplianceAgent(ComplianceAgent):
    """
    Purpose: Monitors compliance with all regulatory timelines and deadlines
    Functionality: Tracks dormancy periods, contact timelines, transfer deadlines, and claim processing
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'timeline_compliance_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'Multiple CBUAE Articles - Timeline Requirements'
        }

        violations = []

        # Check claim processing timeline
        if account_data.get('claim_status') == 'pending':
            submission_date = account_data.get('claim_submission_date')
            if submission_date:
                if isinstance(submission_date, str):
                    submission_date = datetime.fromisoformat(submission_date.replace('Z', '+00:00'))

                if isinstance(submission_date, datetime):
                    days_pending = (datetime.now() - submission_date).days
                    if days_pending > 30:
                        violations.append(f"Claim processing overdue by {days_pending - 30} days")

        # Check transfer timeline
        transfer_eligibility_date = account_data.get('transfer_eligibility_date')
        if transfer_eligibility_date and not account_data.get('transfer_initiated_date'):
            if isinstance(transfer_eligibility_date, str):
                transfer_eligibility_date = datetime.fromisoformat(transfer_eligibility_date.replace('Z', '+00:00'))

            if isinstance(transfer_eligibility_date, datetime):
                days_overdue = (datetime.now() - transfer_eligibility_date).days
                if days_overdue > 90:
                    violations.append(f"Transfer initiation overdue by {days_overdue - 90} days")

        # Check dormancy classification timeline
        dormancy_trigger_date = account_data.get('dormancy_trigger_date')
        if dormancy_trigger_date and account_data.get('dormancy_status') != 'dormant':
            if isinstance(dormancy_trigger_date, str):
                dormancy_trigger_date = datetime.fromisoformat(dormancy_trigger_date.replace('Z', '+00:00'))

            if isinstance(dormancy_trigger_date, datetime):
                days_delayed = (datetime.now() - dormancy_trigger_date).days
                if days_delayed > 30:
                    violations.append(f"Dormancy classification delayed by {days_delayed - 30} days")

        # Check contact attempt timeline
        if account_data.get('dormancy_status') == 'dormant':
            last_contact_date = account_data.get('last_contact_attempt_date')
            if last_contact_date:
                if isinstance(last_contact_date, str):
                    last_contact_date = datetime.fromisoformat(last_contact_date.replace('Z', '+00:00'))

                if isinstance(last_contact_date, datetime):
                    days_since_contact = (datetime.now() - last_contact_date).days
                    if days_since_contact > 180:  # 6 months since last contact
                        violations.append(f"No contact attempts for {days_since_contact} days")

        if violations:
            result.update({
                'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                'violations': violations,
                'priority': Priority.HIGH.value,
                'risk_level': RiskLevel.HIGH.value,
                'action': 'Address overdue timeline violations immediately'
            })
            result['recommendation'] = self.get_llm_recommendation("\n".join(violations), "Timeline Compliance")

        return result


# Agent 10: Amount Verification Agent - Financial Amount Validation
class AmountVerificationAgent(ComplianceAgent):
    """
    Purpose: Verifies accuracy of financial amounts and balance calculations
    Functionality: Validates account balances, transaction sums, and amount consistency
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'amount_verification_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'Internal Financial Data Accuracy'
        }

        violations = []

        # Check balance consistency
        current_balance = account_data.get('balance_current', 0)
        ledger_balance = account_data.get('ledger_balance', current_balance)

        if abs(current_balance - ledger_balance) > 0.01:  # Allow for minor rounding
            violations.append(f"Balance mismatch: Current {current_balance}, Ledger {ledger_balance}")

        # Check negative balance for dormant accounts
        if account_data.get('dormancy_status') == 'dormant' and current_balance < 0:
            violations.append("Dormant account has negative balance")

        # Verify minimum balance for transfer eligibility
        if account_data.get('is_eligible_for_cb_transfer') and current_balance <= 0:
            violations.append("Transfer-eligible account has zero or negative balance")

        # Check for suspicious amount patterns
        if current_balance > 1000000:  # Large amounts require additional verification
            if not account_data.get('large_amount_verified', False):
                violations.append("Large balance amount not independently verified")

        # Verify currency conversion amounts
        if account_data.get('currency') != 'AED' and account_data.get('fx_conversion_complete'):
            converted_amount = account_data.get('converted_aed_amount', 0)
            if converted_amount <= 0 and current_balance > 0:
                violations.append("Foreign currency conversion amount missing or invalid")

        if violations:
            result.update({
                'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                'violations': violations,
                'priority': Priority.MEDIUM.value,
                'risk_level': RiskLevel.MEDIUM.value,
                'action': 'Verify and reconcile financial amounts'
            })
            result['recommendation'] = self.get_llm_recommendation("\n".join(violations), "Amount Verification")

        return result


# Agent 11: Claims Detection Agent - Outstanding Claims Management
class ClaimsDetectionAgent(ComplianceAgent):
    """
    Purpose: Detects and manages outstanding customer claims per CBUAE Article 4
    Functionality: Identifies pending claims, validates claim processing, and ensures timely resolution
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'claims_detection_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Art. 4 - Claims Processing'
        }

        claim_status = account_data.get('claim_status')
        violations = []

        if claim_status == 'pending':
            claim_id = account_data.get('claim_id', 'Unknown')
            violations.append(f"Outstanding claim (ID: {claim_id}) pending resolution")

            # Check claim documentation
            if not account_data.get('claim_documentation_complete', False):
                violations.append("Claim documentation incomplete")

            # Check claim investigation status
            if not account_data.get('claim_investigation_started', False):
                violations.append("Claim investigation not initiated")

        elif claim_status == 'under_review':
            review_start_date = account_data.get('claim_review_start_date')
            if review_start_date:
                if isinstance(review_start_date, str):
                    review_start_date = datetime.fromisoformat(review_start_date.replace('Z', '+00:00'))

                if isinstance(review_start_date, datetime):
                    days_in_review = (datetime.now() - review_start_date).days
                    if days_in_review > 45:  # Extended review period
                        violations.append(f"Claim under review for {days_in_review} days")

        # Check for multiple claims on same account
        claim_count = account_data.get('total_claims_count', 0)
        if claim_count > 1:
            violations.append(f"Multiple claims ({claim_count}) on single account require coordination")

        if violations:
            result.update({
                'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                'violations': violations,
                'priority': Priority.HIGH.value,
                'risk_level': RiskLevel.MEDIUM.value,
                'action': 'Expedite claim processing and resolution'
            })
            result['recommendation'] = self.get_llm_recommendation("\n".join(violations), "CBUAE Article 4")

        return result


# Agent 12: Flag Instructions Agent - System Flag Management
class FlagInstructionsAgent(ComplianceAgent):
    """
    Purpose: Manages system flags and special instructions for account handling
    Functionality: Processes flagging instructions, special handling requirements, and alert management
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'flag_instructions_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'Internal Flag Management'
        }

        violations = []

        # Check for manual review flags
        if account_data.get('requires_manual_review_flag', False):
            violations.append("Account flagged for manual review - requires immediate attention")

        # Check for legal hold flags
        if account_data.get('legal_hold_flag', False):
            if not account_data.get('legal_hold_documentation', False):
                violations.append("Legal hold flag active but documentation missing")

        # Check for high-risk customer flags
        if account_data.get('high_risk_customer_flag', False):
            if not account_data.get('enhanced_due_diligence_complete', False):
                violations.append("High-risk customer requires enhanced due diligence")

        # Check for regulatory reporting flags
        if account_data.get('regulatory_reporting_flag', False):
            last_report_date = account_data.get('last_regulatory_report_date')
            if not last_report_date:
                violations.append("Regulatory reporting flag active but no reports generated")

        # Check for dormancy processing flags
        if account_data.get('dormancy_status') == 'dormant':
            if not account_data.get('dormancy_processing_flag', False):
                violations.append("Dormant account missing dormancy processing flag")

        # Process any pending flag instructions
        flag_instruction = account_data.get('pending_flag_instruction')
        if flag_instruction:
            logger.info(f"Processing flag instruction for account {account_data.get('account_id')}: {flag_instruction}")
            result['action'] = f"Process flag instruction: {flag_instruction}"

        if violations:
            result.update({
                'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                'violations': violations,
                'priority': Priority.HIGH.value,
                'risk_level': RiskLevel.MEDIUM.value,
                'action': 'Address flagged conditions and update flag status'
            })
            result['recommendation'] = self.get_llm_recommendation("\n".join(violations), "Flag Management")

        return result


# Agent 13: Risk Assessment Agent - Compliance Risk Evaluation
class ComplianceRiskAssessmentAgent(ComplianceAgent):
    """
    Purpose: Assesses overall compliance risk based on multiple factors
    Functionality: Evaluates financial, operational, and regulatory risks for comprehensive assessment
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'compliance_risk_assessment_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'Internal Risk Management Framework'
        }

        violations = []
        risk_factors = []
        risk_score = 0

        balance = account_data.get('balance_current', 0) or account_data.get('balance', 0)
        dormancy_status = account_data.get('dormancy_status')

        # High-value account risk
        if balance > 500000:
            risk_factors.append("High-value account")
            risk_score += 3

            if dormancy_status == 'dormant' and not account_data.get('is_eligible_for_cb_transfer'):
                violations.append(f"High-value dormant account (AED {balance:,.2f}) poses significant risk")
                risk_score += 5

        # Customer type risk
        customer_type = account_data.get('customer_type', 'individual')
        if customer_type == 'corporate' and balance > 100000:
            risk_factors.append("High-value corporate account")
            risk_score += 2

        # Geographic risk
        customer_country = account_data.get('customer_country', 'UAE')
        high_risk_countries = ['Unknown', 'Sanctioned', 'High-Risk']
        if customer_country in high_risk_countries:
            risk_factors.append("High-risk geographic location")
            risk_score += 4

        # Process delay risk
        if account_data.get('dormancy_status') == 'dormant':
            dormancy_trigger_date = account_data.get('dormancy_trigger_date')
            if dormancy_trigger_date:
                if isinstance(dormancy_trigger_date, str):
                    dormancy_trigger_date = datetime.fromisoformat(dormancy_trigger_date.replace('Z', '+00:00'))

                if isinstance(dormancy_trigger_date, datetime):
                    days_dormant = (datetime.now() - dormancy_trigger_date).days
                    if days_dormant > 730:  # Over 2 years dormant
                        risk_factors.append("Extended dormancy period")
                        risk_score += 3

        # Multiple violation risk
        if dormancy_results and isinstance(dormancy_results, dict):
            violation_count = 0
            for res in dormancy_results.values():
                if isinstance(res, dict) and res.get('violations'):
                    violation_count += 1
            if violation_count > 3:
                risk_factors.append("Multiple compliance violations")
                risk_score += violation_count

        # Determine risk level
        if risk_score >= 10:
            result['risk_level'] = RiskLevel.CRITICAL.value
            result['priority'] = Priority.CRITICAL.value
        elif risk_score >= 7:
            result['risk_level'] = RiskLevel.HIGH.value
            result['priority'] = Priority.HIGH.value
        elif risk_score >= 4:
            result['risk_level'] = RiskLevel.MEDIUM.value
            result['priority'] = Priority.MEDIUM.value

        if risk_score >= 7:
            violations.append(f"High compliance risk (score: {risk_score}) - {', '.join(risk_factors)}")
            result.update({
                'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                'violations': violations,
                'action': 'Implement enhanced monitoring and risk mitigation measures'
            })

            result['recommendation'] = self.get_llm_recommendation(
                f"Risk score: {risk_score}, Factors: {', '.join(risk_factors)}",
                "Compliance Risk Management"
            )

        result['risk_score'] = risk_score
        result['risk_factors'] = risk_factors

        return result


# Agent 14: Regulatory Reporting Agent - CBUAE Reporting Compliance
class RegulatoryReportingAgent(ComplianceAgent):
    """
    Purpose: Ensures compliance with CBUAE regulatory reporting requirements
    Functionality: Validates reporting schedules, content accuracy, and submission timelines
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'regulatory_reporting_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'CBUAE Art. 3.10 - Regulatory Reporting'
        }

        violations = []

        if account_data.get('dormancy_status') == 'dormant':
            # Check annual CBUAE report inclusion
            last_report_date = account_data.get('last_cbuae_report_inclusion_date')
            if last_report_date:
                if isinstance(last_report_date, str):
                    last_report_date = datetime.fromisoformat(last_report_date.replace('Z', '+00:00'))

                if isinstance(last_report_date, datetime):
                    days_since_report = (datetime.now() - last_report_date).days
                    if days_since_report > 365:
                        violations.append(f"Not included in CBUAE report for {days_since_report} days")
            else:
                violations.append("No record of CBUAE report inclusion")

            # Check quarterly reporting for high-value accounts
            balance = account_data.get('balance_current', 0)
            if balance > 100000:
                last_quarterly_report = account_data.get('last_quarterly_report_date')
                if last_quarterly_report:
                    if isinstance(last_quarterly_report, str):
                        last_quarterly_report = datetime.fromisoformat(last_quarterly_report.replace('Z', '+00:00'))

                    if isinstance(last_quarterly_report, datetime):
                        days_since_quarterly = (datetime.now() - last_quarterly_report).days
                        if days_since_quarterly > 95:  # Over a quarter
                            violations.append("High-value account missing quarterly report")

        # Check transfer reporting
        if account_data.get('transferred_to_cb_date'):
            if not account_data.get('transfer_reported_to_cbuae', False):
                violations.append("Central Bank transfer not reported to CBUAE")

        # Check claim reporting
        if account_data.get('claim_status') in ['pending', 'resolved']:
            if not account_data.get('claim_reported_to_cbuae', False):
                violations.append("Customer claim not reported to CBUAE")

        if violations:
            result.update({
                'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                'violations': violations,
                'priority': Priority.MEDIUM.value,
                'risk_level': RiskLevel.MEDIUM.value,
                'action': 'Submit required regulatory reports to CBUAE'
            })
            result['recommendation'] = self.get_llm_recommendation("\n".join(violations), "CBUAE Article 3.10")

        return result


# Agent 15: Audit Trail Agent - Audit Documentation
class AuditTrailAgent(ComplianceAgent):
    """
    Purpose: Ensures comprehensive audit trail maintenance for all account activities
    Functionality: Validates audit log completeness, data integrity, and retention compliance
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'audit_trail_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'Internal Audit Requirements'
        }

        violations = []

        # Check basic audit log presence
        if not account_data.get('audit_log_complete', True):
            violations.append("Comprehensive audit trail missing or incomplete")

        # Check audit log entries for key events
        if account_data.get('dormancy_status') == 'dormant':
            if not account_data.get('dormancy_classification_logged', False):
                violations.append("Dormancy classification event not logged")

            if account_data.get('contact_attempts_made', 0) > 0:
                if not account_data.get('contact_attempts_logged', False):
                    violations.append("Contact attempts not properly logged")

        # Check transfer audit trail
        if account_data.get('is_eligible_for_cb_transfer'):
            if not account_data.get('transfer_eligibility_logged', False):
                violations.append("Transfer eligibility determination not logged")

        if account_data.get('transferred_to_cb_date'):
            if not account_data.get('transfer_completion_logged', False):
                violations.append("Transfer completion not logged")

        # Check modification audit trail
        last_modified_date = account_data.get('last_modified_date')
        last_modified_by = account_data.get('last_modified_by')

        if last_modified_date and not last_modified_by:
            violations.append("Account modification not properly attributed")

        # Check retention compliance
        account_creation_date = account_data.get('created_date')
        if account_creation_date:
            if isinstance(account_creation_date, str):
                account_creation_date = datetime.fromisoformat(account_creation_date.replace('Z', '+00:00'))

            if isinstance(account_creation_date, datetime):
                account_age_years = (datetime.now() - account_creation_date).days / 365
                if account_age_years > 7 and not account_data.get('long_term_retention_approved', False):
                    violations.append(f"Account records {account_age_years:.1f} years old - retention review required")

        if violations:
            result.update({
                'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                'violations': violations,
                'priority': Priority.MEDIUM.value,
                'risk_level': RiskLevel.MEDIUM.value,
                'action': 'Complete audit trail documentation and address gaps'
            })
            result['recommendation'] = self.get_llm_recommendation("\n".join(violations), "Audit Trail Management")

        return result


# Agent 16: Action Generation Agent - Remediation Action Planning
class ActionGenerationAgent(ComplianceAgent):
    """
    Purpose: Generates specific remediation actions based on compliance violations
    Functionality: Creates actionable tasks, assigns priorities, and schedules follow-up activities
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'action_generation_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'Internal Action Management'
        }

        generated_actions = []
        priority_actions = []

        # Analyze dormancy results to generate actions
        if dormancy_results and isinstance(dormancy_results, dict):
            for agent_name, agent_result in dormancy_results.items():
                if isinstance(agent_result, dict) and agent_result.get('violations'):
                    agent_priority = agent_result.get('priority', Priority.LOW.value)
                    agent_action = agent_result.get('action')

                    if agent_action:
                        action_item = {
                            'action': agent_action,
                            'source_agent': agent_name,
                            'priority': agent_priority,
                            'due_date': self._calculate_due_date(agent_priority),
                            'assigned_to': self._determine_assignee(agent_name),
                            'violation_count': len(agent_result['violations'])
                        }
                        generated_actions.append(action_item)

                        if agent_priority in [Priority.HIGH.value, Priority.CRITICAL.value]:
                            priority_actions.append(action_item)

        # Generate consolidated action plan
        if generated_actions:
            result['generated_actions'] = generated_actions
            result['priority_actions'] = priority_actions

            total_actions = len(generated_actions)
            high_priority_count = len(priority_actions)

            if high_priority_count > 0:
                violation_msg = f"Generated {total_actions} remediation actions ({high_priority_count} high priority)"
                result.update({
                    'compliance_status': ComplianceStatus.NON_COMPLIANT.value,
                    'violations': [violation_msg],
                    'priority': Priority.HIGH.value,
                    'risk_level': RiskLevel.HIGH.value,
                    'action': f"Execute {high_priority_count} high-priority actions immediately"
                })
                action_summary = "; ".join([a['action'] for a in priority_actions[:3]])
                result['recommendation'] = self.get_llm_recommendation(
                    f"High-priority actions: {action_summary}",
                    "Action Planning"
                )

        return result

    def _calculate_due_date(self, priority: str) -> str:
        """Calculate due date based on priority level"""
        now = datetime.now()
        if priority == Priority.CRITICAL.value:
            due_date = now + timedelta(hours=24)
        elif priority == Priority.HIGH.value:
            due_date = now + timedelta(days=3)
        elif priority == Priority.MEDIUM.value:
            due_date = now + timedelta(days=7)
        else:
            due_date = now + timedelta(days=14)

        return due_date.isoformat()

    def _determine_assignee(self, agent_name: str) -> str:
        """Determine appropriate assignee based on agent type"""
        if 'contact' in agent_name:
            return 'Customer Relations Team'
        elif 'transfer' in agent_name:
            return 'Transfer Processing Team'
        elif 'documentation' in agent_name:
            return 'Documentation Team'
        elif 'reporting' in agent_name:
            return 'Regulatory Reporting Team'
        elif 'risk' in agent_name:
            return 'Risk Management Team'
        else:
            return 'Compliance Team'


# Agent 17: Final Verification Agent - Comprehensive Validation
class FinalVerificationAgent(ComplianceAgent):
    """
    Purpose: Performs final comprehensive validation of all compliance checks
    Functionality: Cross-validates results, identifies conflicts, and provides overall assessment
    """

    def execute(self, account_data: Dict, dormancy_results: Dict = None) -> Dict:
        result = {
            'agent': 'final_verification_agent',
            'compliance_status': ComplianceStatus.COMPLIANT.value,
            'violations': [],
            'action': None,
            'priority': Priority.LOW.value,
            'risk_level': RiskLevel.LOW.value,
            'recommendation': None,
            'regulatory_citation': 'Comprehensive Compliance Validation'
        }

        violations = []
        verification_summary = {}

        RISK_LEVEL_MAP = {level.value: i for i, level in enumerate(RiskLevel)}
        PRIORITY_MAP = {level.value: i for i, level in enumerate(Priority)}

        if dormancy_results and isinstance(dormancy_results, dict):
            # Analyze overall compliance status
            statuses = []
            all_violations = []
            risk_levels = []
            priorities = []

            for agent_name, agent_result in dormancy_results.items():
                if isinstance(agent_result, dict):
                    status = agent_result.get('compliance_status')
                    if status:
                        statuses.append(status)

                    agent_violations = agent_result.get('violations', [])
                    if agent_violations:
                        all_violations.extend(agent_violations)

                    risk_level = agent_result.get('risk_level')
                    if risk_level:
                        risk_levels.append(risk_level)

                    priority = agent_result.get('priority')
                    if priority:
                        priorities.append(priority)

            # Generate verification summary
            verification_summary = {
                'total_agents_executed': len([r for r in dormancy_results.values() if isinstance(r, dict)]),
                'compliant_agents': statuses.count(ComplianceStatus.COMPLIANT.value),
                'non_compliant_agents': statuses.count(ComplianceStatus.NON_COMPLIANT.value),
                'critical_violations': statuses.count(ComplianceStatus.CRITICAL_VIOLATION.value),
                'total_violations': len(all_violations),
                'highest_risk_level': max(risk_levels, key=lambda r: RISK_LEVEL_MAP.get(r,
                                                                                        -1)) if risk_levels else RiskLevel.LOW.value,
                'highest_priority': max(priorities,
                                        key=lambda p: PRIORITY_MAP.get(p, -1)) if priorities else Priority.LOW.value
            }

            # Check for conflicting results
            if ComplianceStatus.COMPLIANT.value in statuses and ComplianceStatus.CRITICAL_VIOLATION.value in statuses:
                violations.append("Conflicting compliance statuses detected across agents")

            # Validate data consistency
            if account_data.get('dormancy_status') == 'dormant':
                dormancy_agents = ['article_2_compliance_agent', 'article_3_1_process_compliance_agent']
                dormancy_violations = any(
                    dormancy_results.get(agent, {}).get('violations') for agent in dormancy_agents
                )
                if not dormancy_violations:
                    violations.append("Dormant account shows no violations in core dormancy agents")

            # Check transfer consistency
            if account_data.get('is_eligible_for_cb_transfer'):
                transfer_agents = ['article_3_4_transfer_compliance_agent', 'fx_conversion_check_agent']
                transfer_ready = all(
                    not dormancy_results.get(agent, {}).get('violations') for agent in transfer_agents
                )
                if not transfer_ready and not account_data.get('transferred_to_cb_date'):
                    violations.append("Transfer-eligible account has unresolved transfer compliance issues")

            # Overall compliance determination
            if verification_summary.get('critical_violations', 0) > 0:
                overall_status = ComplianceStatus.CRITICAL_VIOLATION.value
                overall_priority = Priority.CRITICAL.value
                overall_risk = RiskLevel.CRITICAL.value
            elif verification_summary.get('non_compliant_agents', 0) > verification_summary.get('compliant_agents', 0):
                overall_status = ComplianceStatus.NON_COMPLIANT.value
                overall_priority = Priority.HIGH.value
                overall_risk = RiskLevel.HIGH.value
            elif verification_summary.get('non_compliant_agents', 0) > 0:
                overall_status = ComplianceStatus.PARTIAL_COMPLIANT.value
                overall_priority = Priority.MEDIUM.value
                overall_risk = RiskLevel.MEDIUM.value
            else:
                overall_status = ComplianceStatus.COMPLIANT.value
                overall_priority = Priority.LOW.value
                overall_risk = RiskLevel.LOW.value

            if violations or overall_status != ComplianceStatus.COMPLIANT.value:
                if not violations:
                    violations = [f"Overall compliance status: {overall_status}"]

                result.update({
                    'compliance_status': overall_status,
                    'violations': violations,
                    'priority': overall_priority,
                    'risk_level': overall_risk,
                    'action': 'Address compliance issues identified in verification summary'
                })
                summary_text = json.dumps(verification_summary, indent=2)
                result['recommendation'] = self.get_llm_recommendation(
                    f"Verification Summary: {summary_text}",
                    "Final Verification"
                )

        result['verification_summary'] = verification_summary
        return result


# =============================================================================
# Master Orchestrator for All 17 Compliance Verification Agents
# =============================================================================

class ComplianceOrchestrator:
    """
    Master orchestrator managing all 17 compliance verification agents
    """

    # Complete mapping of all 17 agents
    AGENT_CLASS_MAP = {
        'article_2_compliance_agent': Article2ComplianceAgent,
        'article_3_1_process_compliance_agent': Article31ProcessComplianceAgent,
        'article_3_4_transfer_compliance_agent': Article34TransferComplianceAgent,
        'contact_verification_agent': ContactVerificationAgent,
        'transfer_eligibility_agent': TransferEligibilityAgent,
        'fx_conversion_check_agent': FXConversionCheckAgent,
        'process_management_agent': ProcessManagementAgent,
        'documentation_review_agent': DocumentationReviewAgent,
        'timeline_compliance_agent': TimelineComplianceAgent,
        'amount_verification_agent': AmountVerificationAgent,
        'claims_detection_agent': ClaimsDetectionAgent,
        'flag_instructions_agent': FlagInstructionsAgent,
        'compliance_risk_assessment_agent': ComplianceRiskAssessmentAgent,
        'regulatory_reporting_agent': RegulatoryReportingAgent,
        'audit_trail_agent': AuditTrailAgent,
        'action_generation_agent': ActionGenerationAgent,
        'final_verification_agent': FinalVerificationAgent,
    }

    def __init__(self, llm_client=None, config=None):
        """Initialize the orchestrator with LLM client and configuration."""
        self.llm_client = llm_client
        self.config = config
        self.agents = {}

        # Initialize all agents
        for agent_name, agent_class in self.AGENT_CLASS_MAP.items():
            try:
                self.agents[agent_name] = agent_class(llm_client, config)
            except Exception as e:
                logger.error(f"Failed to initialize {agent_name}: {e}")
                self.agents[agent_name] = None

    def process_account(self, account_data):
        """
        Process a single account through all compliance agents.

        Args:
            account_data (dict): Account data dictionary

        Returns:
            dict: Results from all agents for this account
        """
        account_results = {}

        for agent_name, agent in self.agents.items():
            if agent is None:
                account_results[agent_name] = {
                    'error': f'Agent {agent_name} not initialized',
                    'compliance_status': 'pending_review'
                }
                continue

            try:
                # Execute the agent on this account
                result = agent.execute(account_data)
                account_results[agent_name] = result

            except Exception as e:
                logger.error(
                    f"Error executing {agent_name} on account {account_data.get('account_id', 'Unknown')}: {e}")
                account_results[agent_name] = {
                    'error': str(e),
                    'compliance_status': 'error'
                }

        return account_results

    def process_multiple_accounts(self, account_list):
        """
        Process multiple accounts through all compliance agents.

        Args:
            account_list (list): List of account data dictionaries

        Returns:
            list: Results for each account
        """
        results = []

        for account_data in account_list:
            account_result = self.process_account(account_data)
            results.append(account_result)

        return results

    def run_all_agents(self, account_data, dormancy_results=None):
        """
        Run all agents on a single account (compatibility method).

        Args:
            account_data (dict): Account data
            dormancy_results (dict): Optional dormancy analysis results

        Returns:
            dict: Combined results from all agents
        """
        return self.process_account(account_data)

    def get_agent_summary(self, results):
        """
        Generate summary statistics from agent results.

        Args:
            results (list): List of account results

        Returns:
            dict: Summary statistics
        """
        total_accounts = len(results)
        compliant_count = 0
        violations = []

        for account_results in results:
            account_compliant = True

            for agent_name, agent_result in account_results.items():
                if isinstance(agent_result, dict):
                    status = agent_result.get('compliance_status', 'unknown')
                    if status in ['non_compliant', 'critical_violation']:
                        account_compliant = False

                    agent_violations = agent_result.get('violations', [])
                    violations.extend(agent_violations)

            if account_compliant:
                compliant_count += 1

        return {
            'total_accounts': total_accounts,
            'compliant_accounts': compliant_count,
            'non_compliant_accounts': total_accounts - compliant_count,
            'total_violations': len(violations),
            'compliance_rate': (compliant_count / total_accounts * 100) if total_accounts > 0 else 0
        }
