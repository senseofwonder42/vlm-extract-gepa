"""Config loading, `--set` overrides, role resolution, hashing, secrets."""

import pydantic
import pytest

from vex.config import Settings, apply_override, config_hash, load_config
from tests.conftest import DEFAULT_YAML


def test_default_yaml_validates():
    config = load_config(DEFAULT_YAML)
    assert config.models.extractor.model == "qwen/qwen3.8-27b"
    assert set(config.metric.weights) == {
        "assureur", "numero_contrat", "nom_souscripteur", "adresse_souscripteur", "immatriculation",
        "marque_vehicule", "date_debut_validite", "date_fin_validite", "code_assureur", "garanties",
    }


def test_override_parses_yaml_scalars_lists_and_dicts():
    config = load_config(DEFAULT_YAML, [
        "models.extractor.params.temperature=0.2",
        "metric.fuzzy_fields=[assureur]",
        "models.extractor.extra_body.provider={order: [deepinfra]}",
        "cache.enabled=false",
    ])
    assert config.models.extractor.params["temperature"] == 0.2
    assert config.metric.fuzzy_fields == ["assureur"]
    assert config.models.extractor.extra_body["provider"] == {"order": ["deepinfra"]}
    assert config.cache.enabled is False


def test_override_can_add_new_passthrough_keys():
    config = load_config(DEFAULT_YAML, ["models.extractor.params.frequency_penalty=0.1"])
    assert config.models.extractor.params["frequency_penalty"] == 0.1


def test_override_typo_in_typed_section_is_rejected():
    with pytest.raises(pydantic.ValidationError):
        load_config(DEFAULT_YAML, ["budget.max_usdd=3"])


@pytest.mark.parametrize("bad", ["no_equals_sign", "=1", "budget.max_usd.x=1"])
def test_malformed_override_raises(bad):
    data = {"budget": {"max_usd": 5}}
    with pytest.raises(ValueError):
        apply_override(data, bad)


def test_extractor_eval_is_deep_merged_over_extractor():
    config = load_config(DEFAULT_YAML)
    role = config.role("extractor_eval")
    assert role.params["temperature"] == 0.0
    assert role.params["presence_penalty"] == 0.0
    assert role.params["top_k"] == 20  # inherited
    assert role.extra_body["reasoning"] == {"effort": "none"}  # inherited
    assert config.models.extractor.params["temperature"] == 0.7  # base untouched


def test_reasoning_settings_per_role():
    config = load_config(DEFAULT_YAML)
    assert config.role("extractor").extra_body["reasoning"] == {"effort": "none"}
    for role in ("preannotator", "diagnostic", "reflection"):
        assert config.role(role).extra_body["reasoning"] == {"effort": "medium"}


def test_hash_is_stable_and_sensitive_to_overrides():
    assert config_hash(load_config(DEFAULT_YAML)) == config_hash(load_config(DEFAULT_YAML))
    assert config_hash(load_config(DEFAULT_YAML)) != config_hash(
        load_config(DEFAULT_YAML, ["models.extractor.params.temperature=0.2"])
    )


def test_secrets_stay_out_of_config(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret-value-123456")
    settings = Settings(_env_file=None)
    assert settings.openrouter_api_key.get_secret_value() == "sk-or-secret-value-123456"
    assert "sk-or-secret" not in repr(settings)
    dumped = str(load_config(DEFAULT_YAML).model_dump())
    assert "sk-or" not in dumped
