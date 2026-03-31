# Copyright 2024-2026 Agentics Transformation Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Domain-specific AI Agents for FLUID Forge

Specialized agents with domain expertise for creating data products
in specific industries and use cases.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from fluid_build.cli.console import cprint
from fluid_build.cli.console import error as console_error
from fluid_build.cli.forge_dialogs import build_choice, normalize_choice_value

YES_NO_CHOICES = [
    build_choice("Yes", "yes", aliases=["y", "yeah", "yep", "sure", "real time"]),
    build_choice("No", "no", aliases=["n", "nope", "batch", "not really"]),
]

FINANCE_PRODUCT_CHOICES = [
    build_choice("Risk Analytics", "risk_analytics", aliases=["risk", "market risk", "risk scoring"]),
    build_choice("Trading Platform", "trading_platform", aliases=["trading", "trades", "execution"]),
    build_choice("Fraud Detection", "fraud_detection", aliases=["fraud", "fraud analytics", "fraud monitoring"]),
    build_choice("Customer Analytics", "customer_analytics", aliases=["customer", "customer insights", "customer 360"]),
    build_choice(
        "Compliance Reporting",
        "compliance_reporting",
        aliases=["compliance", "regulatory reporting", "sox reporting"],
    ),
]

FINANCE_COMPLIANCE_CHOICES = [
    build_choice("SOX", "sox", aliases=["sarbanes oxley"]),
    build_choice("GDPR", "gdpr", aliases=["privacy"]),
    build_choice("PCI DSS", "pci_dss", aliases=["pci", "card compliance"]),
    build_choice("Basel III", "basel_iii", aliases=["basel"]),
    build_choice("MiFID II", "mifid_ii", aliases=["mifid"]),
    build_choice("None / not sure", "none", aliases=["none", "not sure", "no compliance"]),
]

HEALTHCARE_PRODUCT_CHOICES = [
    build_choice("Patient Analytics", "patient_analytics", aliases=["patient", "patient insights"]),
    build_choice("Clinical Research", "clinical_research", aliases=["research", "clinical studies"]),
    build_choice("Population Health", "population_health", aliases=["population", "public health"]),
    build_choice("EHR Integration", "ehr_integration", aliases=["ehr", "emr", "clinical integration"]),
    build_choice("Drug Discovery", "drug_discovery", aliases=["drug", "discovery", "r and d"]),
]

PHI_CHOICES = [
    build_choice("Yes", "yes", aliases=["phi", "full phi"]),
    build_choice("No", "no", aliases=["no phi"]),
    build_choice("De-identified only", "deidentified_only", aliases=["deidentified", "de identified", "masked"]),
]

RETAIL_PRODUCT_CHOICES = [
    build_choice("Customer 360", "customer_360", aliases=["customer 360", "cdp", "customer profile"]),
    build_choice(
        "Recommendation Engine",
        "recommendation_engine",
        aliases=["recommendations", "rec engine", "personalization"],
    ),
    build_choice(
        "Inventory Optimization",
        "inventory_optimization",
        aliases=["inventory", "stock optimization", "replenishment"],
    ),
    build_choice("Price Optimization", "price_optimization", aliases=["pricing", "price"]),
    build_choice("Demand Forecasting", "demand_forecasting", aliases=["forecasting", "demand"]),
]

RETAIL_SCALE_CHOICES = [
    build_choice("Small (<1M records)", "small (<1m records)", aliases=["small", "<1m", "under 1m"]),
    build_choice(
        "Medium (1M-100M)",
        "medium (1m-100m)",
        aliases=["medium", "mid", "1m to 100m", "10m"],
    ),
    build_choice("Large (>100M)", "large (>100m)", aliases=["large", "huge", "100m+", "enterprise scale"]),
]


def _raw_answer(context: Dict[str, Any], key: str) -> str:
    raw_answers = context.get("raw_answers") or {}
    return str(raw_answers.get(key) or context.get(key) or "").strip()


def _resolve_context_choice(
    context: Dict[str, Any],
    *,
    field_name: str,
    choices: List[Dict[str, Any]],
    default: Optional[str] = None,
) -> Optional[str]:
    return normalize_choice_value(
        context.get(field_name),
        field_name=field_name,
        choices=choices,
        default=None,
    ) or normalize_choice_value(
        _raw_answer(context, field_name),
        field_name=field_name,
        choices=choices,
        default=default,
    )


def _choice_label(choices: List[Dict[str, Any]], value: Any) -> str:
    current = str(value or "").strip()
    for choice in choices:
        if str(choice.get("value") or "").strip() == current:
            return str(choice.get("label") or current)
    return current.replace("_", " ").title() if current else "Not specified"


class AIAgentBase:
    """Base class for AI agents - minimal version for domain agents"""

    def __init__(self, name: str, description: str, domain: str):
        self.name = name
        self.description = description
        self.domain = domain
        try:
            from rich.console import Console

            self.console = Console()
        except ImportError:
            self.console = None

    def get_questions(self) -> List[Dict[str, Any]]:
        """Get questions for this agent"""
        raise NotImplementedError

    def analyze_requirements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze requirements and provide suggestions"""
        raise NotImplementedError

    def create_project(self, target_dir: Path, context: Dict[str, Any]) -> bool:
        """Create project using AI agent - delegates to ForgeEngine"""
        try:
            # Analyze requirements and generate suggestions
            suggestions = self.analyze_requirements(context)

            # Show AI analysis to user
            self._show_ai_analysis(context, suggestions)

            # Create project configuration for ForgeEngine
            project_config = self._create_forge_config(target_dir, context, suggestions)

            # Use ForgeEngine to create and validate the project properly
            success = self._create_with_forge_engine(project_config)

            if success:
                # Show next steps
                self._show_next_steps(target_dir, context, suggestions)
                return True
            else:
                if self.console:
                    self.console.print("[red]❌ Project creation failed validation[/red]")
                return False

        except Exception as e:
            if self.console:
                self.console.print(f"[red]❌ Failed to create project: {e}[/red]")
            else:
                console_error(f"Failed to create project: {e}")
            return False

    def _create_forge_config(
        self, target_dir: Path, context: Dict[str, Any], suggestions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create configuration for ForgeEngine"""
        goal = context.get("project_goal", "Data Product")

        # Create project name that will pass validation
        project_name = self._sanitize_project_name(goal)

        return {
            "name": project_name,
            "description": f"AI-generated {goal} ({self.domain} domain)",
            "template": suggestions["recommended_template"],
            "provider": suggestions["recommended_provider"],
            "target_dir": str(target_dir),
            "ai_context": context,
            "ai_suggestions": suggestions,
            "domain": self.domain,
        }

    def _sanitize_project_name(self, goal: str) -> str:
        """Create a valid project name from goal"""
        import re

        name = goal.lower()
        name = re.sub(r"[^a-z0-9\s\-_]", "", name)
        name = re.sub(r"\s+", "-", name)
        name = re.sub(r"-+", "-", name)
        name = name.strip("-")

        if name and not name[0].isalpha():
            name = f"project-{name}"

        if not name:
            name = f"{self.domain}-data-product"

        return name

    def _create_with_forge_engine(self, project_config: Dict[str, Any]) -> bool:
        """Use ForgeEngine to create and validate project"""
        try:
            from fluid_build.forge import ForgeEngine

            if self.console:
                with self.console.status(
                    f"[bold blue]🔧 Generating {self.domain} project...", spinner="dots"
                ):
                    engine = ForgeEngine()
                    success = engine.run_with_config(project_config, dry_run=False)
                return success
            else:
                cprint(f"🔧 Generating {self.domain} project...")
                engine = ForgeEngine()
                return engine.run_with_config(project_config, dry_run=False)

        except Exception as e:
            if self.console:
                self.console.print(f"[red]❌ ForgeEngine integration failed: {e}[/red]")
            return False

    def _show_ai_analysis(self, context: Dict[str, Any], suggestions: Dict[str, Any]):
        """Show AI analysis to user"""
        if not self.console:
            return

        from rich.panel import Panel

        goal = context.get("project_goal", "Data Product")
        data_sources = context.get("data_sources", "Not specified")
        try:
            product_choices = self.get_questions()[0]["choices"]
        except (NotImplementedError, IndexError, KeyError, TypeError):
            product_choices = []
        product_type = _choice_label(product_choices, context.get("product_type"))
        patterns = ", ".join(suggestions.get("recommended_patterns", []) or ["Standard scaffolding"])

        analysis = f"""
🎯 **Project Goal:** {goal}
📊 **Data Sources:** {data_sources}
🏷️ **Domain Focus:** {product_type}

🤖 **Recommendations:**
• Template: {suggestions.get('recommended_template')}
• Provider: {suggestions.get('recommended_provider')}
• Patterns: {patterns}

[dim]Optimized for {self.domain} workflows and guardrails.[/dim]
        """

        self.console.print(Panel(analysis.strip(), title="🧠 AI Analysis", border_style="blue"))

    def _show_next_steps(
        self, target_dir: Path, context: Dict[str, Any], suggestions: Dict[str, Any]
    ):
        """Show intelligent next steps"""
        if not self.console:
            return

        from rich.panel import Panel

        next_steps = f"""
🎯 **Immediate Next Steps:**
1. Review and customize contract.fluid.yaml
2. Run `fluid validate contract.fluid.yaml` to check your setup
3. Configure your {suggestions['recommended_provider']} provider credentials

🚀 **Recommended Workflow:**
1. `fluid validate contract.fluid.yaml` - Validate your contract
2. `fluid plan contract.fluid.yaml --out runtime/plan.json` - Generate execution plan
3. `fluid apply runtime/plan.json` - Deploy your data product

💡 **{self.domain.title()}-Specific Tips:**
"""

        # Add domain-specific tips
        if self.domain == "finance" and suggestions.get("security_requirements"):
            next_steps += "• Review security and compliance requirements\n"
            next_steps += "• Set up audit logging and access controls\n"
        elif self.domain == "healthcare":
            next_steps += "• Ensure HIPAA compliance measures are in place\n"
            next_steps += "• Review PHI handling procedures\n"
        elif self.domain == "retail":
            next_steps += "• Configure personalization engine settings\n"
            next_steps += "• Set up A/B testing framework\n"

        next_steps += "• Run `fluid auth status` to confirm provider access\n"
        next_steps += "• Use `fluid doctor` if anything looks off\n"
        next_steps += "\n[dim]Generated by FLUID AI Agent - Domain: " + self.domain + "[/dim]"

        self.console.print(Panel(next_steps.strip(), title="🚀 What's Next?", border_style="green"))


class FinanceAgent(AIAgentBase):
    """Finance and banking domain expert"""

    def __init__(self):
        super().__init__(
            name="finance",
            description="Expert in financial data products, regulatory compliance, and risk analytics",
            domain="finance",
        )

    def get_questions(self) -> List[Dict[str, Any]]:
        """Get finance-specific questions"""
        return [
            {
                "key": "product_type",
                "question": "What type of financial product are you building?",
                "type": "choice",
                "choices": FINANCE_PRODUCT_CHOICES,
                "required": True,
            },
            {
                "key": "data_sources",
                "question": "What data sources will you use?",
                "type": "text",
                "required": True,
                "default": "Transaction data, customer profiles, market data",
            },
            {
                "key": "compliance_requirements",
                "question": "Any specific compliance requirements?",
                "type": "choice",
                "choices": FINANCE_COMPLIANCE_CHOICES,
                "required": False,
            },
            {
                "key": "real_time",
                "question": "Do you need real-time processing?",
                "type": "choice",
                "choices": YES_NO_CHOICES,
                "default": "no",
            },
        ]

    def analyze_requirements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze finance-specific requirements"""
        product_type = _resolve_context_choice(
            context,
            field_name="product_type",
            choices=FINANCE_PRODUCT_CHOICES,
            default="customer_analytics",
        )
        compliance = _resolve_context_choice(
            context,
            field_name="compliance_requirements",
            choices=FINANCE_COMPLIANCE_CHOICES,
            default="none",
        )
        real_time = (
            _resolve_context_choice(
                context,
                field_name="real_time",
                choices=YES_NO_CHOICES,
                default="no",
            )
            == "yes"
        )

        suggestions = {
            "recommended_template": "finance-analytics",
            "recommended_provider": "gcp",
            "recommended_patterns": [],
            "architecture_suggestions": [],
            "best_practices": [],
            "technology_stack": [],
            "security_requirements": [],
        }

        # Template selection
        if product_type == "risk_analytics":
            suggestions["recommended_template"] = "risk-analytics-template"
            suggestions["recommended_patterns"].extend(["monte_carlo", "stress_testing"])
            suggestions["technology_stack"].extend(["Python", "pandas", "numpy", "scikit-learn"])
        elif product_type == "trading_platform":
            suggestions["recommended_template"] = "trading-platform-template"
            suggestions["recommended_patterns"].extend(["event_sourcing", "cqrs"])
            suggestions["technology_stack"].extend(["Python", "Redis", "Kafka"])
        elif product_type == "fraud_detection":
            suggestions["recommended_template"] = "ml-pipeline-template"
            suggestions["recommended_patterns"].extend(["anomaly_detection", "real_time_scoring"])
            suggestions["technology_stack"].extend(["Python", "TensorFlow", "Kafka"])
        else:
            suggestions["recommended_template"] = "analytics-template"

        # Real-time processing
        if real_time:
            suggestions["recommended_patterns"].append("streaming_pipeline")
            suggestions["technology_stack"].extend(["Apache Kafka", "Apache Flink"])
            suggestions["architecture_suggestions"].append("Implement stream processing with Kafka")

        # Compliance requirements
        if compliance != "none":
            suggestions["security_requirements"].extend(
                [
                    "Implement data encryption at rest and in transit",
                    "Set up audit logging for all data access",
                    "Implement role-based access control (RBAC)",
                    "Enable data retention policies",
                    "Set up compliance reporting dashboards",
                ]
            )

            if compliance == "sox":
                suggestions["security_requirements"].append("Implement SOX-compliant audit trails")
            elif compliance == "gdpr":
                suggestions["security_requirements"].extend(
                    [
                        "Implement right-to-be-forgotten workflows",
                        "Enable data portability",
                        "Set up consent management",
                    ]
                )
            elif compliance == "pci_dss":
                suggestions["security_requirements"].extend(
                    [
                        "Tokenize credit card data",
                        "Implement network segmentation",
                        "Regular security scans",
                    ]
                )

        # Best practices
        suggestions["best_practices"].extend(
            [
                "Implement data lineage tracking",
                "Use version control for all models and code",
                "Set up automated testing for data quality",
                "Implement disaster recovery procedures",
                "Document data governance policies",
            ]
        )

        # Architecture
        suggestions["architecture_suggestions"].extend(
            [
                "Use medallion architecture (bronze/silver/gold)",
                "Implement data validation at ingestion",
                "Set up monitoring and alerting",
                "Consider data mesh for multiple teams",
            ]
        )

        return suggestions


class HealthcareAgent(AIAgentBase):
    """Healthcare and life sciences domain expert"""

    def __init__(self):
        super().__init__(
            name="healthcare",
            description="Expert in healthcare data products, HIPAA compliance, and clinical analytics",
            domain="healthcare",
        )

    def get_questions(self) -> List[Dict[str, Any]]:
        """Get healthcare-specific questions"""
        return [
            {
                "key": "product_type",
                "question": "What type of healthcare product are you building?",
                "type": "choice",
                "choices": HEALTHCARE_PRODUCT_CHOICES,
                "required": True,
            },
            {
                "key": "data_sources",
                "question": "What data sources will you use?",
                "type": "text",
                "required": True,
                "default": "EHR data, claims data, lab results",
            },
            {
                "key": "hipaa_required",
                "question": "Is HIPAA compliance required?",
                "type": "choice",
                "choices": YES_NO_CHOICES,
                "default": "yes",
            },
            {
                "key": "phi_handling",
                "question": "Will you handle PHI (Protected Health Information)?",
                "type": "choice",
                "choices": PHI_CHOICES,
                "default": "yes",
            },
        ]

    def analyze_requirements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze healthcare-specific requirements"""
        product_type = _resolve_context_choice(
            context,
            field_name="product_type",
            choices=HEALTHCARE_PRODUCT_CHOICES,
            default="patient_analytics",
        )
        hipaa = (
            _resolve_context_choice(
                context,
                field_name="hipaa_required",
                choices=YES_NO_CHOICES,
                default="yes",
            )
            == "yes"
        )
        phi = _resolve_context_choice(
            context,
            field_name="phi_handling",
            choices=PHI_CHOICES,
            default="yes",
        )

        suggestions = {
            "recommended_template": "healthcare-analytics",
            "recommended_provider": "gcp",
            "recommended_patterns": [],
            "architecture_suggestions": [],
            "best_practices": [],
            "technology_stack": [],
            "security_requirements": [],
        }

        # Template selection
        if product_type == "clinical_research":
            suggestions["recommended_template"] = "clinical-research-template"
            suggestions["technology_stack"].extend(["Python", "R", "Jupyter"])
        elif product_type == "population_health":
            suggestions["recommended_template"] = "population-health-template"
            suggestions["recommended_patterns"].append("cohort_analysis")
        else:
            suggestions["recommended_template"] = "analytics-template"

        # HIPAA compliance
        if hipaa:
            suggestions["security_requirements"].extend(
                [
                    "Implement HIPAA-compliant data encryption",
                    "Set up BAA (Business Associate Agreement) tracking",
                    "Enable comprehensive audit logging",
                    "Implement access controls with MFA",
                    "Set up breach notification procedures",
                    "Regular HIPAA compliance audits",
                ]
            )

        # PHI handling
        if phi == "yes":
            suggestions["security_requirements"].extend(
                [
                    "Implement PHI data masking",
                    "Use secure enclaves for PHI processing",
                    "Enable data lineage for PHI",
                    "Set up automatic PHI detection",
                ]
            )
            suggestions["architecture_suggestions"].append("Separate PHI data into secure zones")
        elif phi == "deidentified_only":
            suggestions["architecture_suggestions"].append(
                "Implement de-identification pipeline at ingestion"
            )
            suggestions["best_practices"].append(
                "Validate de-identification meets Safe Harbor or Expert Determination"
            )

        # Best practices
        suggestions["best_practices"].extend(
            [
                "Use standardized medical codes (ICD-10, SNOMED)",
                "Implement data quality checks for clinical data",
                "Document all data transformations",
                "Set up monitoring for data anomalies",
                "Implement version control for algorithms",
            ]
        )

        # Architecture
        suggestions["architecture_suggestions"].extend(
            [
                "Use FHIR standards for interoperability",
                "Implement HL7 message processing if needed",
                "Consider federated learning for privacy",
                "Set up secure data sharing mechanisms",
            ]
        )

        return suggestions


class RetailAgent(AIAgentBase):
    """Retail and e-commerce domain expert"""

    def __init__(self):
        super().__init__(
            name="retail",
            description="Expert in retail analytics, customer personalization, and inventory optimization",
            domain="retail",
        )

    def get_questions(self) -> List[Dict[str, Any]]:
        """Get retail-specific questions"""
        return [
            {
                "key": "product_type",
                "question": "What type of retail product are you building?",
                "type": "choice",
                "choices": RETAIL_PRODUCT_CHOICES,
                "required": True,
            },
            {
                "key": "data_sources",
                "question": "What data sources will you use?",
                "type": "text",
                "required": True,
                "default": "Transaction data, customer profiles, inventory data",
            },
            {
                "key": "real_time_personalization",
                "question": "Do you need real-time personalization?",
                "type": "choice",
                "choices": YES_NO_CHOICES,
                "default": "yes",
            },
            {
                "key": "scale",
                "question": "Expected data scale?",
                "type": "choice",
                "choices": RETAIL_SCALE_CHOICES,
                "required": True,
            },
        ]

    def analyze_requirements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze retail-specific requirements"""
        product_type = _resolve_context_choice(
            context,
            field_name="product_type",
            choices=RETAIL_PRODUCT_CHOICES,
            default="customer_360",
        )
        real_time = (
            _resolve_context_choice(
                context,
                field_name="real_time_personalization",
                choices=YES_NO_CHOICES,
                default="yes",
            )
            == "yes"
        )
        scale = _resolve_context_choice(
            context,
            field_name="scale",
            choices=RETAIL_SCALE_CHOICES,
            default="medium (1m-100m)",
        )

        suggestions = {
            "recommended_template": "retail-analytics",
            "recommended_provider": "gcp",
            "recommended_patterns": [],
            "architecture_suggestions": [],
            "best_practices": [],
            "technology_stack": [],
            "performance_optimization": [],
        }

        # Template selection
        if product_type == "recommendation_engine":
            suggestions["recommended_template"] = "ml-pipeline-template"
            suggestions["recommended_patterns"].extend(["collaborative_filtering", "content_based"])
            suggestions["technology_stack"].extend(["Python", "TensorFlow", "Redis"])
        elif product_type == "inventory_optimization":
            suggestions["recommended_template"] = "optimization-template"
            suggestions["recommended_patterns"].append("demand_forecasting")
            suggestions["technology_stack"].extend(["Python", "OR-Tools", "Prophet"])
        elif product_type == "customer_360":
            suggestions["recommended_template"] = "customer360-template"
            suggestions["recommended_patterns"].append("customer_segmentation")
        else:
            suggestions["recommended_template"] = "analytics-template"

        # Real-time requirements
        if real_time:
            suggestions["recommended_patterns"].append("streaming_pipeline")
            suggestions["technology_stack"].extend(["Redis", "Kafka"])
            suggestions["architecture_suggestions"].extend(
                [
                    "Implement feature store for real-time serving",
                    "Use caching layer for frequently accessed data",
                    "Set up A/B testing framework",
                ]
            )

        # Scale considerations
        if "large" in scale:
            suggestions["performance_optimization"].extend(
                [
                    "Use partitioning for large tables",
                    "Implement incremental processing",
                    "Consider distributed processing (Spark)",
                    "Set up data lake architecture",
                    "Use columnar storage (Parquet)",
                ]
            )
            suggestions["recommended_provider"] = "gcp"  # BigQuery for scale
        elif "small" in scale:
            suggestions["recommended_provider"] = "local"  # Can start locally
            suggestions["performance_optimization"].append("PostgreSQL sufficient for this scale")

        # Best practices
        suggestions["best_practices"].extend(
            [
                "Implement customer privacy controls",
                "Set up experimentation framework",
                "Track customer journey metrics",
                "Implement data quality monitoring",
                "Set up business metrics dashboards",
            ]
        )

        # Architecture
        suggestions["architecture_suggestions"].extend(
            [
                "Use event-driven architecture for transactions",
                "Implement CDC (Change Data Capture)",
                "Set up data marts for business teams",
                "Consider customer data platform (CDP) patterns",
            ]
        )

        return suggestions


# Registry of domain agents
DOMAIN_AGENTS = {
    "finance": FinanceAgent,
    "healthcare": HealthcareAgent,
    "retail": RetailAgent,
}


def get_agent(agent_name: str) -> AIAgentBase:
    """Get a domain agent by name

    Args:
        agent_name: Name of the agent

    Returns:
        AIAgentBase instance

    Raises:
        ValueError: If agent not found
    """
    if agent_name not in DOMAIN_AGENTS:
        raise ValueError(
            f"Agent '{agent_name}' not found. Available: {', '.join(DOMAIN_AGENTS.keys())}"
        )

    agent_class = DOMAIN_AGENTS[agent_name]
    return agent_class()


def list_agents() -> List[Dict[str, str]]:
    """List all available domain agents

    Returns:
        List of agent info dictionaries
    """
    agents = []
    for name, agent_class in DOMAIN_AGENTS.items():
        agent = agent_class()
        agents.append(
            {"name": agent.name, "domain": agent.domain, "description": agent.description}
        )
    return agents
