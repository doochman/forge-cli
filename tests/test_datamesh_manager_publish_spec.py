from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests

from fluid_build.cli.datamesh_manager import _cmd_publish, _publish_exit_code
from fluid_build.providers.base import ProviderError
from fluid_build.providers.datamesh_manager import DataMeshManagerProvider


def _sample_contract():
    return {
        "id": "sales-product",
        "metadata": {
            "name": "Sales Product",
            "description": "demo",
            "status": "active",
            "owner": {"team": "analytics"},
        },
        "owner": {"team": "analytics"},
        "exposes": [],
        "expects": [],
    }


def _sample_contract_with_exposes():
    return {
        "id": "sales-product",
        "metadata": {
            "name": "Sales Product",
            "description": "demo",
            "status": "active",
            "owner": {"team": "analytics"},
        },
        "owner": {"team": "analytics"},
        "exposes": [
            {"id": "orders", "provider": "gcp", "contract": {"schema": []}},
            {"id": "customers", "provider": "gcp", "contract": {"schema": []}},
        ],
        "expects": [],
    }


def test_apply_dry_run_defaults_to_dps_spec():
    provider = DataMeshManagerProvider(api_key="dummy", api_url="https://api.entropy-data.com")

    result = provider.apply(_sample_contract(), dry_run=True)

    assert result["payload"]["dataProductSpecification"] == "0.0.1"


def test_apply_dry_run_uses_odps_spec_when_provider_hint_is_odps():
    provider = DataMeshManagerProvider(api_key="dummy", api_url="https://api.entropy-data.com")

    result = provider.apply(_sample_contract(), dry_run=True, provider_hint="odps")

    payload = result["payload"]
    assert payload["kind"] == "DataProduct"
    assert "apiVersion" in payload
    assert "info" not in payload


def test_apply_dry_run_allows_explicit_spec_override():
    provider = DataMeshManagerProvider(api_key="dummy", api_url="https://api.entropy-data.com")

    result = provider.apply(
        _sample_contract(),
        dry_run=True,
        provider_hint="odps",
        data_product_specification="4.1.0",
    )

    assert result["payload"]["dataProductSpecification"] == "4.1.0"


def test_apply_dry_run_keeps_per_expose_data_contract_ids_for_dps():
    provider = DataMeshManagerProvider(api_key="dummy", api_url="https://api.entropy-data.com")

    result = provider.apply(_sample_contract_with_exposes(), dry_run=True, publish_contract=True)

    output_ports = result["payload"].get("outputPorts", [])
    data_contract_ids = [port.get("dataContractId") for port in output_ports]
    assert data_contract_ids == ["sales-product.orders", "sales-product.customers"]


def test_apply_dry_run_sets_per_expose_contract_ids_for_odps():
    provider = DataMeshManagerProvider(api_key="dummy", api_url="https://api.entropy-data.com")

    result = provider.apply(
        _sample_contract_with_exposes(),
        dry_run=True,
        publish_contract=True,
        provider_hint="odps",
    )

    output_ports = result["payload"].get("outputPorts", [])
    contract_ids = [port.get("contractId") for port in output_ports]
    assert contract_ids == ["sales-product.orders", "sales-product.customers"]


def test_cmd_publish_passes_provider_hint_to_apply():
    args = SimpleNamespace(
        contract="contract.fluid.yaml",
        overlay=None,
        dry_run=True,
        team_id=None,
        no_create_team=False,
        with_contract=False,
        contract_format="odcs",
        data_product_spec=None,
        validate_generated_contracts=True,
        validation_mode="strict",
        fail_on_contract_error=False,
        provider="odps",
    )

    mock_provider = MagicMock()
    mock_provider.apply.return_value = {
        "dry_run": True,
        "method": "PUT",
        "url": "https://api.entropy-data.com/api/dataproducts/sales-product",
        "payload": {"id": "sales-product", "kind": "DataProduct", "apiVersion": "v1.0.0"},
    }

    with patch("fluid_build.cli.datamesh_manager.load_contract_with_overlay", return_value=_sample_contract()):
        with patch("fluid_build.cli.datamesh_manager._make_provider", return_value=mock_provider):
            with patch("fluid_build.cli.datamesh_manager._print_dry_run"):
                code = _cmd_publish(args)

    assert code == 0
    _, kwargs = mock_provider.apply.call_args
    assert kwargs["provider_hint"] == "odps"
    assert kwargs["data_product_specification"] is None
    assert kwargs["validate_generated_contracts"] is True
    assert kwargs["validation_mode"] == "strict"


def test_cmd_publish_fail_on_contract_error_returns_non_zero():
    args = SimpleNamespace(
        contract="contract.fluid.yaml",
        overlay=None,
        dry_run=False,
        team_id=None,
        no_create_team=False,
        with_contract=True,
        contract_format="odcs",
        data_product_spec=None,
        validate_generated_contracts=False,
        validation_mode="warn",
        fail_on_contract_error=True,
        provider="odps",
    )

    mock_provider = MagicMock()
    mock_provider.apply.return_value = {
        "success": True,
        "product_id": "sales-product",
        "odcs_contracts": [
            {"contract_id": "sales-product.a", "success": True},
            {"contract_id": "sales-product.b", "success": False, "error": "boom"},
        ],
    }

    with patch("fluid_build.cli.datamesh_manager.load_contract_with_overlay", return_value=_sample_contract()):
        with patch("fluid_build.cli.datamesh_manager._make_provider", return_value=mock_provider):
            with patch("fluid_build.cli.datamesh_manager._print_publish_result"):
                code = _cmd_publish(args)

    assert code == 1


def test_request_wraps_retry_error_as_provider_error():
    provider = DataMeshManagerProvider(api_key="dummy", api_url="https://api.entropy-data.com")
    session = MagicMock()
    session.request.side_effect = requests.exceptions.RetryError("too many 500 responses")
    provider._session_instance = session

    with pytest.raises(ProviderError) as excinfo:
        provider._request("PUT", "/api/datacontracts/x.y", json_body={"id": "x.y"})

    assert "HTTP request failed" in str(excinfo.value)


def test_publish_exit_code_strict_mode_on_invalid_contract():
    args = SimpleNamespace(validation_mode="strict", fail_on_contract_error=False)
    result = {
        "odcs_contracts": [
            {"contract_id": "a", "success": True, "valid": True},
            {"contract_id": "b", "success": True, "valid": False},
        ]
    }

    assert _publish_exit_code(result, args) == 1


def test_publish_exit_code_fail_on_contract_error():
    args = SimpleNamespace(validation_mode="warn", fail_on_contract_error=True)
    result = {
        "odcs_contracts": [
            {"contract_id": "a", "success": True},
            {"contract_id": "b", "success": False, "error": "boom"},
        ]
    }

    assert _publish_exit_code(result, args) == 1


def test_publish_odcs_strict_validation_skips_put_on_invalid(monkeypatch):
    provider = DataMeshManagerProvider(api_key="dummy", api_url="https://api.entropy-data.com")
    contract = {
        "id": "sales-product",
        "metadata": {"name": "Sales Product"},
        "owner": {"team": "analytics"},
        "exposes": [{"id": "port_a"}],
        "expects": [],
    }

    class _FakeOdcsProvider:
        def render(self, fluid, expose_id=None):
            return {"id": f"sales-product.{expose_id}", "kind": "DataContract"}

    monkeypatch.setattr(
        "fluid_build.providers.odcs.OdcsProvider",
        _FakeOdcsProvider,
        raising=True,
    )
    monkeypatch.setattr(
        provider,
        "_validate_generated_odcs_contract",
        lambda _odcs_provider, _odcs_body: (False, "ODCS validation failed"),
    )

    request_calls = []
    monkeypatch.setattr(
        provider,
        "_request",
        lambda *args, **kwargs: request_calls.append((args, kwargs)),
    )

    results = provider._publish_odcs_per_expose(
        contract,
        "sales-product",
        validate_generated_contracts=True,
        validation_mode="strict",
    )

    assert request_calls == []
    assert results[0]["success"] is False
    assert results[0]["valid"] is False
    assert results[0]["error_type"] == "VALIDATION_FAILED"


def test_publish_odcs_warn_validation_still_puts(monkeypatch):
    provider = DataMeshManagerProvider(api_key="dummy", api_url="https://api.entropy-data.com")
    contract = {
        "id": "sales-product",
        "metadata": {"name": "Sales Product"},
        "owner": {"team": "analytics"},
        "exposes": [{"id": "port_a"}],
        "expects": [],
    }

    class _FakeOdcsProvider:
        def render(self, fluid, expose_id=None):
            return {"id": f"sales-product.{expose_id}", "kind": "DataContract"}

    monkeypatch.setattr(
        "fluid_build.providers.odcs.OdcsProvider",
        _FakeOdcsProvider,
        raising=True,
    )
    monkeypatch.setattr(
        provider,
        "_validate_generated_odcs_contract",
        lambda _odcs_provider, _odcs_body: (False, "ODCS validation failed"),
    )

    class _Resp:
        status_code = 200

    request_calls = []
    monkeypatch.setattr(
        provider,
        "_request",
        lambda *args, **kwargs: (request_calls.append((args, kwargs)) or _Resp()),
    )

    results = provider._publish_odcs_per_expose(
        contract,
        "sales-product",
        validate_generated_contracts=True,
        validation_mode="warn",
    )

    assert len(request_calls) == 1
    assert results[0]["success"] is True
    assert results[0]["valid"] is False
    assert "validation_error" in results[0]
    assert "schema_objects" in results[0]
    assert "schema_properties" in results[0]
