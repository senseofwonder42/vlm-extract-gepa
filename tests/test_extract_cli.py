"""End-to-end extraction with a mocked model, run directories and the CLI."""

import json

import pytest
from typer.testing import CliRunner

from vex import cli
from vex.config import config_hash
from vex.extract import extract_document
from vex.prompt import candidate_hash, seed_candidate
from vex.runs import create_run_dir
from tests.conftest import DEFAULT_YAML, VALID_VALUES, chat_response


async def test_extract_document_sends_pages_and_parses(make_client, fake_openrouter, pdf_path):
    fake_openrouter.queue.append(chat_response(json.dumps(VALID_VALUES), cost=0.002))
    candidate = seed_candidate()
    async with make_client() as client:
        result = await extract_document(client, pdf_path, candidate)
    assert result.values == VALID_VALUES
    assert result.candidate_hash == candidate_hash(candidate)
    assert result.usd == pytest.approx(0.002)
    user_content = fake_openrouter.chat_requests[0]["messages"][1]["content"]
    assert [part["type"] for part in user_content] == ["text", "image_url"]
    assert fake_openrouter.chat_requests[0]["temperature"] == 0.0  # extractor_eval by default


async def test_extract_document_failure_is_not_an_exception(make_client, fake_openrouter, pdf_path):
    fake_openrouter.queue += [chat_response("garbage"), chat_response("garbage again")]
    async with make_client() as client:
        result = await extract_document(client, pdf_path, seed_candidate())
    assert result.failed and result.error


def test_run_dir_contains_resolved_config_and_hash(config):
    run_dir = create_run_dir(config, "extract")
    assert (run_dir / "config.sha256").read_text().strip() == config_hash(config)
    assert "qwen/qwen3.8-27b" in (run_dir / "config.resolved.yaml").read_text()
    assert run_dir.name.endswith(f"-extract-{config_hash(config)[:8]}")


runner = CliRunner()


def _base_args(tmp_path) -> list[str]:
    return ["--config", str(DEFAULT_YAML), "--set", f"paths.runs={tmp_path / 'runs'}",
            "--set", f"paths.prompts={tmp_path / 'prompts'}", "--set", f"cache.dir={tmp_path / 'cache'}"]


def test_cli_config_show_applies_overrides(tmp_path):
    result = runner.invoke(cli.app, [*_base_args(tmp_path), "--set", "budget.max_usd=1.5", "config", "show"])
    assert result.exit_code == 0
    assert "max_usd: 1.5" in result.output and "config sha256" in result.output


def test_cli_rejects_bad_override(tmp_path):
    result = runner.invoke(cli.app, [*_base_args(tmp_path), "--set", "budget.max_usdd=1", "config", "show"])
    assert result.exit_code != 0


def test_cli_prompt_seed_writes_versioned_file(tmp_path):
    result = runner.invoke(cli.app, [*_base_args(tmp_path), "prompt", "seed"])
    assert result.exit_code == 0
    [saved] = (tmp_path / "prompts").glob("seed.*.json")
    assert json.loads(saved.read_text())["hash"] == candidate_hash(seed_candidate())


def test_cli_extract_writes_result_and_cost(tmp_path, monkeypatch, fake_openrouter, pdf_path):
    fake_openrouter.queue.append(chat_response(json.dumps(VALID_VALUES), cost=0.002))
    original = cli.OpenRouterClient

    def client_with_fake_transport(*args, **kwargs):
        return original(*args, **kwargs, transport=fake_openrouter.transport())

    monkeypatch.setattr(cli, "OpenRouterClient", client_with_fake_transport)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-0000000000000000")
    result = runner.invoke(cli.app, [*_base_args(tmp_path), "extract", str(pdf_path)])
    assert result.exit_code == 0, result.output
    assert '"immatriculation": "AB-123-CD"' in result.output
    [run_dir] = (tmp_path / "runs").iterdir()
    assert json.loads((run_dir / "cost.json").read_text())["total_usd"] == pytest.approx(0.002)
    assert json.loads((run_dir / "result.json").read_text())["values"] == VALID_VALUES
