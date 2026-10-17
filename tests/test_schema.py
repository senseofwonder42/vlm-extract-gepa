"""`AttestationAuto` validation and the strict JSON schema."""

import pydantic
import pytest

from vex.schema import FIELD_NAMES, AttestationAuto, field_descriptions, strict_json_schema
from tests.conftest import VALID_VALUES


def test_field_names_match_model():
    assert tuple(AttestationAuto.model_fields) == FIELD_NAMES


def test_valid_values_round_trip():
    assert AttestationAuto.model_validate(VALID_VALUES).model_dump() == VALID_VALUES


def test_empty_strings_and_lists_become_null():
    model = AttestationAuto.model_validate({"assureur": "  ", "garanties": [], "numero_contrat": " X1 "})
    assert model.assureur is None
    assert model.garanties is None
    assert model.numero_contrat == "X1"


def test_blank_list_items_are_dropped():
    assert AttestationAuto.model_validate({"garanties": ["Vol", " ", ""]}).garanties == ["Vol"]
    assert AttestationAuto.model_validate({"garanties": [" "]}).garanties is None


@pytest.mark.parametrize("bad", ["01/02/2025", "2025-2-1", "1er février 2025", "2025-13-01"])
def test_non_iso_dates_are_rejected(bad):
    with pytest.raises(pydantic.ValidationError):
        AttestationAuto.model_validate({"date_debut_validite": bad})


def test_unknown_fields_are_rejected():
    with pytest.raises(pydantic.ValidationError):
        AttestationAuto.model_validate({"vin": "VF1"})


def test_every_description_documents_the_null_rule():
    for name, description in field_descriptions().items():
        assert "null" in description, name


def test_multi_vehicle_rule_is_in_vehicle_fields():
    descriptions = field_descriptions()
    assert "PREMIER" in descriptions["immatriculation"]
    assert "PREMIER" in descriptions["marque_vehicule"]


def test_strict_schema_shape():
    schema = strict_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["required"] == list(FIELD_NAMES)
    for prop in schema["properties"].values():
        assert "default" not in prop and "title" not in prop
        assert {"type": "null"} in prop["anyOf"]


def test_strict_schema_uses_given_descriptions():
    custom = {name: f"desc {name}" for name in FIELD_NAMES}
    schema = strict_json_schema(custom)
    assert schema["properties"]["assureur"]["description"] == "desc assureur"
