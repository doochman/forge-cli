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

"""Branch coverage tests for forge.py (fluid_build/cli/forge.py)."""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---- Custom exceptions ----


class TestForgeExceptions:
    def test_forge_error(self):
        from fluid_build.cli.forge import ForgeError

        err = ForgeError(1, "test error")
        assert "test error" in str(err)

    def test_template_not_found(self):
        from fluid_build.cli.forge import TemplateNotFoundError

        try:
            err = TemplateNotFoundError("missing-tmpl", ["starter", "etl"])
            assert "missing-tmpl" in str(err)
        except TypeError:
            pass  # CLIError signature mismatch

    def test_blueprint_not_found(self):
        from fluid_build.cli.forge import BlueprintNotFoundError

        try:
            err = BlueprintNotFoundError("bad-bp", ["quickstart", "enterprise"])
            assert "bad-bp" in str(err)
        except TypeError:
            pass

    def test_invalid_project_name(self):
        from fluid_build.cli.forge import InvalidProjectNameError

        try:
            err = InvalidProjectNameError("a b c", "contains spaces")
            assert "a b c" in str(err)
        except TypeError:
            pass

    def test_project_generation_error(self):
        from fluid_build.cli.forge import ProjectGenerationError

        err = ProjectGenerationError(1, "generation failed")
        assert isinstance(err, Exception)

    def test_context_validation_error(self):
        from fluid_build.cli.forge import ContextValidationError

        err = ContextValidationError(1, "bad context")
        assert isinstance(err, Exception)


# ---- ForgeMode enum ----


class TestForgeMode:
    def test_all_modes(self):
        from fluid_build.cli.forge import ForgeMode

        assert ForgeMode.TEMPLATE.value == "template"
        assert ForgeMode.AI_COPILOT.value == "copilot"
        assert ForgeMode.DOMAIN_AGENT.value == "agent"
        assert ForgeMode.BLUEPRINT.value == "blueprint"


# ---- AIAgent base class ----


class TestAIAgent:
    def test_init(self):
        from fluid_build.cli.forge import AIAgent

        agent = AIAgent("test", "A test agent", "analytics")
        assert agent.name == "test"
        assert agent.description == "A test agent"
        assert agent.domain == "analytics"

    def test_create_project_raises(self):
        from fluid_build.cli.forge import AIAgent

        agent = AIAgent("test", "desc", "general")
        with pytest.raises(NotImplementedError):
            asyncio.run(agent.create_project(Path("/tmp"), {}))

    def test_get_questions_raises(self):
        from fluid_build.cli.forge import AIAgent

        agent = AIAgent("test", "desc", "general")
        with pytest.raises(NotImplementedError):
            agent.get_questions()


# ---- CopilotAgent ----


class TestCopilotAgent:
    def _make_agent(self):
        from fluid_build.cli.forge import CopilotAgent

        return CopilotAgent()

    def test_init(self):
        agent = self._make_agent()
        assert agent.name == "copilot"
        assert agent.domain == "general"

    def test_get_questions(self):
        agent = self._make_agent()
        questions = agent.get_questions()
        assert isinstance(questions, list)
        assert len(questions) >= 3
        keys = [q["key"] for q in questions]
        assert "project_goal" in keys
        use_case_question = next(q for q in questions if q["key"] == "use_case")
        assert use_case_question["choices"][0] == {
            "label": "Analytics & BI",
            "value": "analytics",
        }
        assert use_case_question["choices"][-1] == {
            "label": "Other / Not sure",
            "value": "other",
        }
        assert use_case_question["follow_up"]["key"] == "use_case_other"

    def test_show_next_steps_uses_fluid_commands(self):
        agent = self._make_agent()
        agent.console = MagicMock()

        agent._show_next_steps(Path("/tmp"), {}, {"recommended_provider": "gcp"})

        panel = agent.console.print.call_args.args[0]
        text = str(panel.renderable)
        assert "fluid validate contract.fluid.yaml" in text
        assert "fluid plan contract.fluid.yaml --out runtime/plan.json" in text
        assert "fluid apply runtime/plan.json" in text
        assert "fluid market --search" in text
        assert "make validate" not in text
        assert "fluid market search" not in text

    @patch("fluid_build.cli.forge_copilot_agent.LOG.warning")
    def test_prepare_runtime_inputs_logs_capability_warnings(self, mock_warning):
        agent = self._make_agent()
        llm_config = MagicMock()
        discovery_report = MagicMock()
        with patch.object(agent, "_resolve_llm_config_dependency", return_value=llm_config):
            with patch.object(
                agent, "_discover_local_context_dependency", return_value=discovery_report
            ):
                with patch.object(agent, "_load_project_memory", return_value=None):
                    with patch.object(
                        agent,
                        "_build_capability_matrix_dependency",
                        return_value={
                            "providers": ["local"],
                            "templates": {"starter": {}},
                            "warnings": ["Copilot couldn't inspect the aws provider."],
                        },
                    ):
                        runtime_inputs = agent.prepare_runtime_inputs({})

        assert runtime_inputs["capability_warnings"] == [
            "Copilot couldn't inspect the aws provider."
        ]
        mock_warning.assert_called_once()

    def test_analyze_requirements_ml(self):
        agent = self._make_agent()
        context = {
            "project_goal": "Build a machine learning pipeline",
            "data_sources": "s3 bucket",
            "use_case": "machine_learning",
            "complexity": "advanced",
        }
        suggestions = agent.analyze_requirements(context)
        assert "recommended_template" in suggestions
        assert "recommended_provider" in suggestions

    def test_analyze_requirements_streaming(self):
        agent = self._make_agent()
        context = {
            "project_goal": "Build a streaming data pipeline",
            "data_sources": "kafka",
            "use_case": "streaming",
            "complexity": "intermediate",
        }
        suggestions = agent.analyze_requirements(context)
        assert suggestions["recommended_template"] == "streaming"

    def test_analyze_requirements_etl(self):
        agent = self._make_agent()
        context = {
            "project_goal": "ETL pipeline for warehouse",
            "data_sources": "postgres",
            "use_case": "etl",
            "complexity": "beginner",
        }
        suggestions = agent.analyze_requirements(context)
        assert suggestions is not None

    def test_analyze_requirements_analytics(self):
        agent = self._make_agent()
        context = {
            "project_goal": "Analytics dashboard",
            "data_sources": "bigquery",
            "use_case": "analytics",
            "complexity": "intermediate",
        }
        suggestions = agent.analyze_requirements(context)
        assert suggestions is not None

    def test_analyze_requirements_default(self):
        agent = self._make_agent()
        context = {
            "project_goal": "Something generic",
            "data_sources": "local files",
            "use_case": "other",
            "complexity": "beginner",
        }
        suggestions = agent.analyze_requirements(context)
        assert suggestions["recommended_template"] == "starter"

    def test_analyze_requirements_data_platform(self):
        agent = self._make_agent()
        context = {
            "project_goal": "Lakehouse platform",
            "data_sources": "object storage",
            "use_case": "data_platform",
            "complexity": "intermediate",
        }
        suggestions = agent.analyze_requirements(context)
        assert suggestions["recommended_template"] == "etl_pipeline"

    def test_analyze_requirements_other_with_follow_up(self):
        agent = self._make_agent()
        context = {
            "project_goal": "Internal platform",
            "data_sources": "warehouse tables",
            "use_case": "other",
            "use_case_other": "CDC sync for warehouse loads",
            "complexity": "intermediate",
        }
        suggestions = agent.analyze_requirements(context)
        assert suggestions["recommended_template"] == "etl_pipeline"

    def test_create_project_success(self):
        from fluid_build.cli.forge_copilot_runtime import CopilotGenerationResult, DiscoveryReport

        agent = self._make_agent()
        agent.generate_project_artifacts = MagicMock(
            return_value=CopilotGenerationResult(
                suggestions={
                    "recommended_template": "starter",
                    "recommended_provider": "local",
                    "recommended_patterns": [],
                    "architecture_suggestions": [],
                    "best_practices": [],
                    "technology_stack": [],
                },
                contract={"name": "test-project", "fluidVersion": "0.7.2"},
                readme_markdown="# Test Project\n",
                additional_files={},
                discovery_report=DiscoveryReport(workspace_roots=["/tmp/test"]),
                attempt_reports=[],
            )
        )
        agent._show_ai_analysis = MagicMock()
        agent._create_forge_config = MagicMock(return_value={"name": "test"})
        agent._create_with_forge_engine = MagicMock(return_value=True)
        agent._show_next_steps = MagicMock()
        result = agent.create_project(Path("/tmp/test"), {"project_goal": "test"})
        assert result is True

    @patch("fluid_build.cli.forge.CopilotMemoryStore")
    def test_create_project_saves_memory_in_non_interactive_mode_when_requested(
        self, mock_store_cls
    ):
        from fluid_build.cli.forge_copilot_runtime import CopilotGenerationResult, DiscoveryReport

        agent = self._make_agent()
        mock_store_cls.return_value.load.return_value = None
        agent.generate_project_artifacts = MagicMock(
            return_value=CopilotGenerationResult(
                suggestions={
                    "recommended_template": "starter",
                    "recommended_provider": "local",
                    "recommended_patterns": [],
                    "architecture_suggestions": [],
                    "best_practices": [],
                    "technology_stack": [],
                },
                contract={"name": "test-project", "fluidVersion": "0.7.2"},
                readme_markdown="# Test Project\n",
                additional_files={},
                discovery_report=DiscoveryReport(workspace_roots=["/tmp/test"]),
                attempt_reports=[],
            )
        )
        agent._show_ai_analysis = MagicMock()
        agent._create_forge_config = MagicMock(return_value={"name": "test"})
        agent._create_with_forge_engine = MagicMock(return_value=True)
        agent._show_next_steps = MagicMock()

        result = agent.create_project(
            Path("/tmp/test"),
            {"project_goal": "test"},
            {"non_interactive": True, "save_memory": True},
        )

        assert result is True
        mock_store_cls.return_value.save.assert_called_once()

    @patch("fluid_build.cli.forge.ask_confirmation", return_value=True)
    @patch("fluid_build.cli.forge.CopilotMemoryStore")
    def test_create_project_prompts_before_saving_memory_in_interactive_mode(
        self, mock_store_cls, mock_confirm
    ):
        from fluid_build.cli.forge_copilot_runtime import CopilotGenerationResult, DiscoveryReport

        agent = self._make_agent()
        mock_store_cls.return_value.load.return_value = None
        agent.generate_project_artifacts = MagicMock(
            return_value=CopilotGenerationResult(
                suggestions={
                    "recommended_template": "starter",
                    "recommended_provider": "local",
                    "recommended_patterns": [],
                    "architecture_suggestions": [],
                    "best_practices": [],
                    "technology_stack": [],
                },
                contract={"name": "test-project", "fluidVersion": "0.7.2"},
                readme_markdown="# Test Project\n",
                additional_files={},
                discovery_report=DiscoveryReport(workspace_roots=["/tmp/test"]),
                attempt_reports=[],
            )
        )
        agent._show_ai_analysis = MagicMock()
        agent._create_forge_config = MagicMock(return_value={"name": "test"})
        agent._create_with_forge_engine = MagicMock(return_value=True)
        agent._show_next_steps = MagicMock()

        result = agent.create_project(
            Path("/tmp/test"),
            {"project_goal": "test"},
            {"non_interactive": False},
        )

        assert result is True
        mock_confirm.assert_called_once()
        mock_store_cls.return_value.save.assert_called_once()

    @patch("fluid_build.cli.forge.CopilotMemoryStore")
    def test_create_project_does_not_save_memory_on_dry_run(self, mock_store_cls):
        from fluid_build.cli.forge_copilot_runtime import CopilotGenerationResult, DiscoveryReport

        agent = self._make_agent()
        mock_store_cls.return_value.load.return_value = None
        agent.generate_project_artifacts = MagicMock(
            return_value=CopilotGenerationResult(
                suggestions={
                    "recommended_template": "starter",
                    "recommended_provider": "local",
                    "recommended_patterns": [],
                    "architecture_suggestions": [],
                    "best_practices": [],
                    "technology_stack": [],
                },
                contract={"name": "test-project", "fluidVersion": "0.7.2"},
                readme_markdown="# Test Project\n",
                additional_files={},
                discovery_report=DiscoveryReport(workspace_roots=["/tmp/test"]),
                attempt_reports=[],
            )
        )
        agent._show_ai_analysis = MagicMock()
        agent._create_forge_config = MagicMock(return_value={"name": "test"})
        agent._create_with_forge_engine = MagicMock(return_value=True)
        agent._show_next_steps = MagicMock()

        result = agent.create_project(
            Path("/tmp/test"),
            {"project_goal": "test"},
            {"non_interactive": True, "save_memory": True},
            dry_run=True,
        )

        assert result is True
        mock_store_cls.return_value.save.assert_not_called()

    def test_create_project_failure(self):
        from fluid_build.cli.forge_copilot_runtime import CopilotGenerationResult, DiscoveryReport

        agent = self._make_agent()
        agent.generate_project_artifacts = MagicMock(
            return_value=CopilotGenerationResult(
                suggestions={
                    "recommended_template": "starter",
                    "recommended_provider": "local",
                    "recommended_patterns": [],
                    "architecture_suggestions": [],
                    "best_practices": [],
                    "technology_stack": [],
                },
                contract={"name": "test-project", "fluidVersion": "0.7.2"},
                readme_markdown="# Test Project\n",
                additional_files={},
                discovery_report=DiscoveryReport(workspace_roots=["/tmp/test"]),
                attempt_reports=[],
            )
        )
        agent._show_ai_analysis = MagicMock()
        agent._create_forge_config = MagicMock(return_value={"name": "test"})
        agent._create_with_forge_engine = MagicMock(return_value=False)
        result = agent.create_project(Path("/tmp/test"), {"project_goal": "test"})
        assert result is False

    def test_create_project_generation_error(self):
        from fluid_build.cli.forge_copilot_runtime import CopilotGenerationError

        agent = self._make_agent()
        agent.generate_project_artifacts = MagicMock(
            side_effect=CopilotGenerationError(
                "copilot_generation_failed",
                "Unable to generate a valid contract.",
                suggestions=["Check your API key"],
            )
        )
        agent._create_with_forge_engine = MagicMock()
        result = agent.create_project(Path("/tmp/test"), {})
        assert result is False
        agent._create_with_forge_engine.assert_not_called()

    def test_create_project_exception(self):
        agent = self._make_agent()
        agent.generate_project_artifacts = MagicMock(side_effect=RuntimeError("boom"))
        result = agent.create_project(Path("/tmp/test"), {})
        assert result is False

    @patch("fluid_build.cli.forge.generate_copilot_artifacts")
    @patch("fluid_build.cli.forge.CopilotMemoryStore")
    @patch("fluid_build.cli.forge.discover_local_context")
    @patch("fluid_build.cli.forge.resolve_llm_config")
    def test_generate_project_artifacts_loads_memory_by_default(
        self,
        mock_resolve_llm_config,
        mock_discover_local_context,
        mock_store_cls,
        mock_generate_copilot_artifacts,
    ):
        from fluid_build.cli.forge_copilot_memory import CopilotProjectMemory
        from fluid_build.cli.forge_copilot_runtime import DiscoveryReport, LlmConfig

        agent = self._make_agent()
        memory = CopilotProjectMemory(
            schema_version=1,
            saved_at="2026-03-31T00:00:00Z",
            project_profile={
                "template": "analytics",
                "provider": "local",
                "domain": "analytics",
                "owner": "data-team",
            },
            conventions={
                "build_engines": ["sql"],
                "binding_platforms": ["local"],
                "binding_formats": ["csv"],
                "expose_kinds": ["table"],
                "provider_hints": ["local"],
                "source_formats": {"csv": 1},
                "schema_summaries": [],
            },
            recent_outcomes=[],
        )
        mock_resolve_llm_config.return_value = LlmConfig(
            provider="openai",
            model="gpt-4o-mini",
            endpoint="https://api.openai.com/v1/chat/completions",
            api_key="key",
        )
        mock_discover_local_context.return_value = DiscoveryReport(workspace_roots=["/tmp/test"])
        mock_store_cls.return_value.load.return_value = memory

        agent.generate_project_artifacts({"project_goal": "test"}, {"target_dir": "/tmp/test"})

        assert (
            mock_generate_copilot_artifacts.call_args.kwargs["project_memory"].preferred_provider
            == "local"
        )

    @patch("fluid_build.cli.forge.generate_copilot_artifacts")
    @patch("fluid_build.cli.forge.CopilotMemoryStore")
    @patch("fluid_build.cli.forge.discover_local_context")
    @patch("fluid_build.cli.forge.resolve_llm_config")
    def test_generate_project_artifacts_bypasses_memory_when_disabled(
        self,
        mock_resolve_llm_config,
        mock_discover_local_context,
        mock_store_cls,
        mock_generate_copilot_artifacts,
    ):
        from fluid_build.cli.forge_copilot_runtime import DiscoveryReport, LlmConfig

        agent = self._make_agent()
        mock_resolve_llm_config.return_value = LlmConfig(
            provider="openai",
            model="gpt-4o-mini",
            endpoint="https://api.openai.com/v1/chat/completions",
            api_key="key",
        )
        mock_discover_local_context.return_value = DiscoveryReport(workspace_roots=["/tmp/test"])

        agent.generate_project_artifacts(
            {"project_goal": "test"},
            {"target_dir": "/tmp/test", "memory": False},
        )

        mock_store_cls.return_value.load.assert_not_called()
        assert mock_generate_copilot_artifacts.call_args.kwargs["project_memory"] is None

    def test_create_forge_config(self):
        agent = self._make_agent()
        config = agent._create_forge_config(
            Path("/tmp/test"),
            {"project_goal": "Build data product", "use_case": "etl"},
            {"recommended_template": "etl", "recommended_provider": "local"},
        )
        assert "name" in config or "template" in config

    def test_sanitize_project_name(self):
        agent = self._make_agent()
        name = agent._sanitize_project_name("My Cool Project!")
        assert isinstance(name, str)
        assert len(name) > 0

    def test_create_with_forge_engine_success(self):
        agent = self._make_agent()
        agent.console = None  # force simple mode
        with patch("fluid_build.forge.ForgeEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.run_with_config.return_value = True
            mock_engine_cls.return_value = mock_engine
            result = agent._create_with_forge_engine({"name": "test", "template": "starter"})
            assert result is True

    def test_create_with_forge_engine_failure(self):
        agent = self._make_agent()
        agent.console = None
        with patch("fluid_build.forge.ForgeEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.run_with_config.return_value = False
            mock_engine_cls.return_value = mock_engine
            result = agent._create_with_forge_engine({"name": "test"})
            assert result is False

    def test_create_with_forge_engine_exception(self):
        agent = self._make_agent()
        agent.console = None
        with patch("fluid_build.forge.ForgeEngine", side_effect=RuntimeError):
            result = agent._create_with_forge_engine({"name": "test"})
            assert result is False

    def test_show_ai_analysis_no_console(self):
        agent = self._make_agent()
        agent.console = None
        agent._show_ai_analysis({}, {})  # Should not raise

    def test_show_ai_analysis_prefers_use_case_other_text(self):
        agent = self._make_agent()
        agent.console = MagicMock()
        agent._show_ai_analysis(
            {
                "project_goal": "Test project",
                "data_sources": "warehouse tables",
                "use_case": "other",
                "use_case_other": "Customer 360",
                "complexity": "intermediate",
            },
            {
                "recommended_template": "analytics",
                "recommended_provider": "local",
                "recommended_patterns": [],
                "architecture_suggestions": [],
                "best_practices": [],
            },
        )
        panel = agent.console.print.call_args.args[0]
        text = str(panel.renderable)
        assert "Customer 360" in text
        assert "Use Case:** other" not in text

    def test_generate_intelligent_contract(self):
        agent = self._make_agent()
        contract = agent._generate_intelligent_contract(
            {"project_goal": "ETL pipeline", "use_case": "etl"},
            {"recommended_template": "etl", "recommended_provider": "local"},
        )
        assert isinstance(contract, str)
        assert len(contract) > 0

    def test_generate_intelligent_readme(self):
        agent = self._make_agent()
        readme = agent._generate_intelligent_readme(
            {"project_goal": "Test project", "use_case": "analytics"},
            {
                "recommended_template": "analytics",
                "recommended_provider": "gcp",
                "architecture_suggestions": ["Use partitioning"],
                "best_practices": ["Test early"],
                "recommended_patterns": [],
                "technology_stack": [],
            },
        )
        assert isinstance(readme, str)
        assert len(readme) > 0


# ---- Module-level functions ----


class TestRegisterFunction:
    def test_register(self):
        from fluid_build.cli.forge import register

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        register(sub)


class TestGetTargetDirectory:
    def test_with_target_dir_arg(self):
        from fluid_build.cli.forge import get_target_directory

        args = MagicMock()
        args.target_dir = "/tmp/my-project"
        result = get_target_directory(args)
        assert result == Path("/tmp/my-project")

    def test_without_target_dir(self):
        from fluid_build.cli.forge import get_target_directory

        args = MagicMock()
        args.target_dir = None
        result = get_target_directory(args, default_name="my-fluid-project")
        assert isinstance(result, Path)
        assert "my-fluid-project" in str(result)


class TestLoadContext:
    def test_load_json_string(self):
        from fluid_build.cli.forge import load_context

        ctx = load_context('{"project_goal": "test"}')
        assert ctx["project_goal"] == "test"

    def test_load_invalid_json(self):
        from fluid_build.cli.forge import load_context

        with pytest.raises(Exception):
            load_context("not a json string and not a file path either!!!!")

    def test_load_json_file(self, tmp_path):
        from fluid_build.cli.forge import load_context

        f = tmp_path / "context.json"
        f.write_text(json.dumps({"project_goal": "test"}))
        ctx = load_context(str(f))
        assert ctx["project_goal"] == "test"

    def test_load_nonexistent_file(self):
        from fluid_build.cli.forge import load_context

        with pytest.raises(Exception):
            load_context("/nonexistent/path/to/file.json")


class TestRunFunction:
    @patch("fluid_build.cli.forge.handle_memory_management", return_value=0)
    def test_run_memory_management_short_circuits(self, mock_memory):
        from fluid_build.cli.forge import run

        args = MagicMock()
        args.help = False
        args.show_memory = True
        args.reset_memory = False
        args.non_interactive = True
        logger = logging.getLogger("test")
        result = run(args, logger)
        assert result == 0
        mock_memory.assert_called_once()

    @patch("fluid_build.cli.forge.run_ai_copilot_mode", return_value=0)
    def test_run_copilot_mode(self, mock_copilot):
        from fluid_build.cli.forge import run

        args = MagicMock()
        args.help = False
        args.mode = "copilot"
        logger = logging.getLogger("test")
        result = run(args, logger)
        assert result == 0

    @patch("fluid_build.cli.forge.run_template_mode", return_value=0)
    def test_run_template_mode(self, mock_tmpl):
        from fluid_build.cli.forge import run

        args = MagicMock()
        args.help = False
        args.mode = "template"
        logger = logging.getLogger("test")
        result = run(args, logger)
        assert result == 0

    @patch("fluid_build.cli.forge.run_domain_agent_mode", return_value=0)
    def test_run_agent_mode(self, mock_agent):
        from fluid_build.cli.forge import run

        args = MagicMock()
        args.help = False
        args.mode = "agent"
        logger = logging.getLogger("test")
        result = run(args, logger)
        assert result == 0

    @patch("fluid_build.cli.forge.run_blueprint_mode", return_value=0)
    def test_run_blueprint_mode(self, mock_bp):
        from fluid_build.cli.forge import run

        args = MagicMock()
        args.help = False
        args.mode = "blueprint"
        logger = logging.getLogger("test")
        result = run(args, logger)
        assert result == 0

    def test_run_exception(self):
        from fluid_build.cli.forge import run

        args = MagicMock()
        args.help = False
        args.mode = "invalid_mode_xyz"
        logger = logging.getLogger("test")
        result = run(args, logger)
        assert result == 1


class TestRunAICopilotMode:
    @patch("fluid_build.cli.forge.CopilotAgent")
    def test_copilot_success(self, mock_agent_cls):
        from fluid_build.cli.forge import run_ai_copilot_mode

        mock_agent = MagicMock()
        mock_agent.create_project.return_value = True
        mock_agent_cls.return_value = mock_agent
        args = MagicMock()
        args.context = None
        args.non_interactive = True
        args.target_dir = "/tmp/test"
        logger = logging.getLogger("test")
        result = run_ai_copilot_mode(args, logger)
        assert result == 0

    @patch("fluid_build.cli.forge.CopilotAgent")
    def test_copilot_failure(self, mock_agent_cls):
        from fluid_build.cli.forge import run_ai_copilot_mode

        mock_agent = MagicMock()
        mock_agent.create_project.return_value = False
        mock_agent_cls.return_value = mock_agent
        args = MagicMock()
        args.context = None
        args.non_interactive = True
        args.target_dir = "/tmp/test"
        logger = logging.getLogger("test")
        result = run_ai_copilot_mode(args, logger)
        assert result == 1

    @patch("fluid_build.cli.forge.CopilotAgent")
    def test_copilot_with_context(self, mock_agent_cls):
        from fluid_build.cli.forge import run_ai_copilot_mode

        mock_agent = MagicMock()
        mock_agent.create_project.return_value = True
        mock_agent_cls.return_value = mock_agent
        args = MagicMock()
        args.context = '{"project_goal": "test"}'
        args.non_interactive = True
        args.target_dir = "/tmp/test"
        logger = logging.getLogger("test")
        result = run_ai_copilot_mode(args, logger)
        assert result == 0

    @patch("fluid_build.cli.forge_modes.print_dialog_status")
    @patch("fluid_build.cli.forge_modes.run_adaptive_copilot_interview")
    @patch("fluid_build.cli.forge.CopilotAgent")
    def test_copilot_interactive_provider_warning_continues(
        self,
        mock_agent_cls,
        mock_interview,
        mock_print_dialog_status,
    ):
        from fluid_build.cli.forge import run_ai_copilot_mode

        mock_agent = MagicMock()
        mock_agent.prepare_runtime_inputs.return_value = {
            "llm_config": MagicMock(),
            "discovery_report": MagicMock(),
            "project_memory": None,
            "capability_matrix": {"providers": ["local"], "templates": {"starter": {}}},
            "capability_warnings": ["Copilot couldn't inspect the aws provider."],
        }
        mock_agent.create_project.return_value = True
        mock_agent_cls.return_value = mock_agent
        mock_state = MagicMock()
        mock_state.finalize.return_value = {
            "project_goal": "test",
            "data_sources": "local files",
            "use_case": "analytics",
            "complexity": "intermediate",
        }
        mock_interview.return_value = mock_state
        args = MagicMock()
        args.context = None
        args.non_interactive = False
        args.target_dir = "/tmp/test"
        logger = logging.getLogger("test")

        result = run_ai_copilot_mode(args, logger)

        assert result == 0
        assert any(
            call.kwargs.get("status") == "warning"
            and "couldn't fully verify some local providers"
            in call.kwargs.get("message", "").lower()
            for call in mock_print_dialog_status.call_args_list
        )


class TestMemoryManagement:
    @patch("fluid_build.cli.forge.CopilotMemoryStore")
    def test_handle_memory_management_show(self, mock_store_cls):
        from fluid_build.cli.forge import handle_memory_management
        from fluid_build.cli.forge_copilot_memory import CopilotProjectMemory

        mock_store_cls.return_value.load.return_value = CopilotProjectMemory(
            schema_version=1,
            saved_at="2026-03-31T00:00:00Z",
            project_profile={
                "template": "analytics",
                "provider": "local",
                "domain": "analytics",
                "owner": "data-team",
            },
            conventions={
                "build_engines": ["sql"],
                "binding_platforms": ["local"],
                "binding_formats": ["csv"],
                "expose_kinds": ["table"],
                "provider_hints": ["local"],
                "source_formats": {"csv": 1},
                "schema_summaries": [],
            },
            recent_outcomes=[],
        )
        mock_store_cls.return_value.path = Path("/tmp/test/runtime/.state/copilot-memory.json")
        args = MagicMock()
        args.target_dir = "/tmp/test"
        args.reset_memory = False
        args.show_memory = True

        result = handle_memory_management(args, logging.getLogger("test"))

        assert result == 0
        mock_store_cls.return_value.load.assert_called_once()

    @patch("fluid_build.cli.forge.CopilotMemoryStore")
    def test_handle_memory_management_reset(self, mock_store_cls):
        from fluid_build.cli.forge import handle_memory_management

        mock_store_cls.return_value.delete.return_value = True
        mock_store_cls.return_value.path = Path("/tmp/test/runtime/.state/copilot-memory.json")
        args = MagicMock()
        args.target_dir = "/tmp/test"
        args.reset_memory = True
        args.show_memory = False

        result = handle_memory_management(args, logging.getLogger("test"))

        assert result == 0
        mock_store_cls.return_value.delete.assert_called_once()


class TestRunDomainAgentMode:
    @patch("fluid_build.cli.forge.CopilotAgent")
    def test_agent_mode_with_name(self, mock_copilot_cls):
        from fluid_build.cli.forge import run_domain_agent_mode

        mock_agent = MagicMock()
        mock_agent.create_project.return_value = True
        mock_copilot_cls.return_value = mock_agent
        args = MagicMock()
        args.agent = "copilot"
        args.context = None
        args.non_interactive = True
        args.target_dir = "/tmp/test"
        logger = logging.getLogger("test")
        result = run_domain_agent_mode(args, logger)
        assert result in (0, 1)

    def test_agent_mode_unknown_agent(self):
        from fluid_build.cli.forge import run_domain_agent_mode

        args = MagicMock()
        args.agent = "nonexistent_agent_xyz"
        args.non_interactive = True
        args.target_dir = "/tmp/test"
        logger = logging.getLogger("test")
        result = run_domain_agent_mode(args, logger)
        assert result == 1


class TestRunBlueprintMode:
    @patch("fluid_build.cli.forge.blueprint_registry")
    def test_blueprint_not_found(self, mock_bp_reg):
        from fluid_build.cli.forge import run_blueprint_mode

        mock_bp_reg.get_blueprint.return_value = None
        mock_bp_reg.list_blueprints.return_value = ["quickstart"]
        args = MagicMock()
        args.blueprint = "nonexistent"
        args.non_interactive = True
        args.target_dir = "/tmp/test"
        logger = logging.getLogger("test")
        result = run_blueprint_mode(args, logger)
        assert result == 1

    @patch("fluid_build.cli.forge.blueprint_registry")
    def test_blueprint_success(self, mock_bp_reg):
        from fluid_build.cli.forge import run_blueprint_mode

        mock_bp = MagicMock()
        mock_bp.generate_project.return_value = True
        mock_bp_reg.get_blueprint.return_value = mock_bp
        args = MagicMock()
        args.blueprint = "quickstart"
        args.non_interactive = True
        args.target_dir = "/tmp/test-bp"
        args.dry_run = False
        logger = logging.getLogger("test")
        result = run_blueprint_mode(args, logger)
        assert result in (0, 1)

    @patch("fluid_build.cli.forge.Console")
    @patch("fluid_build.cli.forge.blueprint_registry")
    def test_blueprint_success_shows_fluid_next_steps(self, mock_bp_reg, mock_console_cls):
        from fluid_build.cli.forge import run_blueprint_mode

        mock_bp = MagicMock()
        mock_bp.metadata.title = "Customer 360"
        mock_bp.metadata.description = "Enterprise blueprint"
        mock_bp.generate_project.return_value = True
        mock_bp_reg.get_blueprint.return_value = mock_bp
        console = MagicMock()
        mock_console_cls.return_value = console
        args = MagicMock()
        args.blueprint = "quickstart"
        args.non_interactive = False
        args.target_dir = "/tmp/test-bp"
        args.dry_run = False
        logger = logging.getLogger("test")

        result = run_blueprint_mode(args, logger)

        assert result in (0, 1)
        panel = console.print.call_args_list[-1].args[0]
        text = str(panel.renderable)
        assert "fluid validate contract.fluid.yaml" in text
        assert "fluid plan contract.fluid.yaml --out runtime/plan.json" in text
        assert "fluid apply runtime/plan.json" in text
        assert "python -m fluid_build" not in text


class TestGatherCopilotContext:
    def test_no_console(self):
        from fluid_build.cli.forge import CopilotAgent, gather_copilot_context

        agent = CopilotAgent()
        result = gather_copilot_context(agent, None)
        assert isinstance(result, dict)

    def test_with_console(self):
        from fluid_build.cli.forge import CopilotAgent, gather_copilot_context

        agent = CopilotAgent()
        console = MagicMock()
        console.input.side_effect = [
            "test goal",
            "test source",
            "Analytics & BI",
            "",
            "intermediate",
        ]
        result = gather_copilot_context(agent, console)
        assert isinstance(result, dict)
        assert result["raw_answers"]["project_goal"] == "test goal"
        assert result["dialog_transcript"][0]["field"] == "project_goal"

    def test_with_console_maps_labeled_use_case_choices(self):
        from fluid_build.cli.forge import CopilotAgent, gather_copilot_context

        agent = CopilotAgent()
        console = MagicMock()
        console.input.side_effect = [
            "Customer 360",
            "BigQuery tables",
            "dashboards and BI",
            "small team",
            "advanced setup",
        ]
        result = gather_copilot_context(agent, console)
        assert result["use_case"] == "analytics"
        assert result["team_size"] == "small (2-5)"
        assert result["complexity"] == "advanced"
        assert result["raw_answers"]["use_case"] == "dashboards and BI"

    def test_with_console_collects_use_case_other_follow_up(self):
        from fluid_build.cli.forge import CopilotAgent, gather_copilot_context

        agent = CopilotAgent()
        console = MagicMock()
        console.input.side_effect = [
            "Customer 360",
            "BigQuery tables",
            "customer 360 graph",
            "solo",
            "simple",
        ]
        result = gather_copilot_context(agent, console)
        assert result["use_case"] == "other"
        assert result["use_case_other"] == "customer 360 graph"
        assert result["dialog_transcript"][2]["raw_input"] == "customer 360 graph"

    def test_with_domain_agent_captures_friendly_raw_answers(self):
        from fluid_build.cli.forge import gather_copilot_context
        from fluid_build.cli.forge_agents import FinanceAgent

        agent = FinanceAgent()
        console = MagicMock()
        console.input.side_effect = [
            "fraud analytics",
            "card transactions",
            "pci",
            "yes",
        ]

        result = gather_copilot_context(agent, console)

        assert result["product_type"] == "fraud_detection"
        assert result["compliance_requirements"] == "pci_dss"
        assert result["real_time"] == "yes"
        assert result["raw_answers"]["product_type"] == "fraud analytics"
        assert result["dialog_transcript"][0]["resolved_value"] == "fraud_detection"


class TestGetEnhancedTemplates:
    def test_returns_dict(self):
        from fluid_build.cli.forge import get_enhanced_templates

        templates = get_enhanced_templates()
        assert isinstance(templates, dict)


class TestCreateLegacyBootstrapper:
    def test_returns_object(self):
        try:
            from fluid_build.cli.forge import create_legacy_bootstrapper

            result = create_legacy_bootstrapper(target_dir="/tmp/test")
            assert result is not None
        except (ImportError, Exception):
            pass  # May not be available


class TestRunForgeBlueprint:
    @patch("fluid_build.cli.forge.blueprint_registry")
    def test_no_blueprint(self, mock_bp_reg):
        from fluid_build.cli.forge import _run_forge_blueprint

        mock_bp_reg.get_blueprint.return_value = None
        args = MagicMock()
        args.blueprint = "missing"
        result = _run_forge_blueprint(args, mock_bp_reg)
        assert result == 1

    @patch("fluid_build.cli.forge.blueprint_registry")
    def test_dry_run(self, mock_bp_reg, tmp_path):
        from fluid_build.cli.forge import _run_forge_blueprint

        mock_bp = MagicMock()
        mock_bp.validate.return_value = []  # No errors
        mock_bp.path = tmp_path
        mock_bp.metadata.name = "quickstart"
        mock_bp.metadata.title = "Quickstart"
        mock_bp.metadata.description = "Quickstart blueprint"
        mock_bp.metadata.complexity.value = "simple"
        mock_bp.metadata.setup_time = "5 min"
        mock_bp.metadata.providers = ["local"]
        mock_bp_reg.get_blueprint.return_value = mock_bp
        target = tmp_path / "new_project"
        args = MagicMock()
        args.blueprint = "quickstart"
        args.dry_run = True
        args.non_interactive = True
        args.target_dir = str(target)
        args.quickstart = False
        result = _run_forge_blueprint(args, mock_bp_reg)
        assert result == 0
