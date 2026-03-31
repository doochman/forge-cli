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

"""Compatibility surface for Forge domain agents."""

from __future__ import annotations

from typing import Dict, List

from .forge_domain_agent_base import (
    AIAgentBase,
    DeclarativeDomainAgent,
    _choice_label,
    _raw_answer,
    _resolve_context_choice,
)


class _SpecBackedDomainAgent(DeclarativeDomainAgent):
    """Shared declarative domain-agent binding for named compatibility classes."""

    spec_name = ""

    def __init__(self) -> None:
        super().__init__(self.spec_name)


class FinanceAgent(_SpecBackedDomainAgent):
    """Finance and banking domain expert."""

    spec_name = "finance"


class HealthcareAgent(_SpecBackedDomainAgent):
    """Healthcare and life sciences domain expert."""

    spec_name = "healthcare"


class RetailAgent(_SpecBackedDomainAgent):
    """Retail and e-commerce domain expert."""

    spec_name = "retail"


class TelcoAgent(_SpecBackedDomainAgent):
    """TM Forum SID-aligned telecom domain agent."""

    spec_name = "telco"


DOMAIN_AGENTS = {
    "finance": FinanceAgent,
    "healthcare": HealthcareAgent,
    "retail": RetailAgent,
    "telco": TelcoAgent,
}


def get_agent(agent_name: str) -> AIAgentBase:
    """Get a domain agent by name."""
    if agent_name not in DOMAIN_AGENTS:
        raise ValueError(
            f"Agent '{agent_name}' not found. Available: {', '.join(DOMAIN_AGENTS.keys())}"
        )
    return DOMAIN_AGENTS[agent_name]()


def list_agents() -> List[Dict[str, str]]:
    """List all available domain agents."""
    agents = []
    for name, agent_class in DOMAIN_AGENTS.items():
        agent = agent_class()
        agents.append(
            {"name": agent.name, "domain": agent.domain, "description": agent.description}
        )
    return agents


__all__ = [
    "AIAgentBase",
    "FinanceAgent",
    "HealthcareAgent",
    "RetailAgent",
    "TelcoAgent",
    "DOMAIN_AGENTS",
    "get_agent",
    "list_agents",
    "_raw_answer",
    "_resolve_context_choice",
    "_choice_label",
]
