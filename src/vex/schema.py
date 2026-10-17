"""Extraction schema for French motor insurance certificates.

`AttestationAuto` covers the four document families met in practice:
the free-form "attestation d'assurance", the "Mémo Véhicule Assuré" (mandatory
since 1 April 2024), the white "carte internationale d'assurance automobile",
and the legacy green card ("carte verte").

Field descriptions are the seed text of the `field:<name>` prompt components:
GEPA rewrites them, so the class docstrings stay in English while the
descriptions, which the model and the annotators read, are in French.

Several vehicles on one certificate: only the FIRST listed vehicle is
extracted. This rule is stated in the `immatriculation` and
`marque_vehicule` descriptions so the model and the annotators apply it alike.
"""

import copy
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator

FIELD_NAMES: tuple[str, ...] = (
    "assureur",
    "numero_contrat",
    "nom_souscripteur",
    "adresse_souscripteur",
    "immatriculation",
    "marque_vehicule",
    "date_debut_validite",
    "date_fin_validite",
    "code_assureur",
    "garanties",
)

_ABSENT = "Si le champ est absent ou illisible → null (jamais de chaîne vide)."


class AttestationAuto(BaseModel):
    """Fields extracted from one motor insurance certificate. Absent → `None`."""

    model_config = ConfigDict(extra="forbid")

    assureur: str | None = Field(
        default=None,
        description=(
            "Nom de la compagnie d'assurance qui porte le risque, tel qu'imprimé : en-tête, "
            "logo écrit en toutes lettres, case « Assureur » / « Société d'assurance » de la "
            "carte internationale, ou bloc de signature. Ne pas prendre un courtier ou "
            "intermédiaire (« Votre conseiller », « Intermédiaire ») sauf s'il est aussi "
            "l'assureur. Recopier le nom sans forme juridique ni mentions légales "
            f"(« S.A. au capital de… », RCS). {_ABSENT}"
        ),
    )
    numero_contrat: str | None = Field(
        default=None,
        description=(
            "Numéro du contrat, de police ou d'adhésion (« N° de contrat », « Police n° », "
            "« Contrat n° », case 4 de la carte internationale après les codes pays et "
            "assureur). Recopier exactement lettres, chiffres, tirets et barres obliques, "
            "sans le libellé. Ne pas confondre avec le numéro de client ou de sociétaire, "
            f"le SIREN, ni le code assureur. {_ABSENT}"
        ),
    )
    nom_souscripteur: str | None = Field(
        default=None,
        description=(
            "Nom du souscripteur, c'est-à-dire le titulaire du contrat (« Souscripteur », "
            "« Assuré », « Nom et adresse de l'assuré », case 8 de la carte internationale) : "
            "prénom et nom d'une personne (sans civilité M., Mme) ou raison sociale d'une "
            "entreprise. Si un conducteur désigné est aussi mentionné, prendre le "
            f"souscripteur. {_ABSENT}"
        ),
    )
    adresse_souscripteur: str | None = Field(
        default=None,
        description=(
            "Adresse postale complète du souscripteur sur une seule ligne : numéro et voie, "
            "complément, code postal et commune, séparés par des virgules (ex. « 12 rue des "
            "Lilas, Bât. B, 69003 Lyon »). Jamais l'adresse de l'assureur ou de l'agence. "
            f"{_ABSENT}"
        ),
    )
    immatriculation: str | None = Field(
        default=None,
        description=(
            "Numéro d'immatriculation du véhicule assuré (« Immatriculation », « N° "
            "d'immatriculation », case 5 de la carte internationale). Format SIV (depuis 2009) : "
            "AA-123-AA ; ancien format FNI : 1234 AB 56 ou 123 ABC 75 (chiffres, lettres, "
            "département). Recopier en majuscules, avec les tirets pour le SIV. Ne pas "
            "confondre avec le numéro VIN (17 caractères) ni le numéro de contrat. Si plusieurs "
            "véhicules sont listés, prendre uniquement le PREMIER. Si le véhicule est « en cours "
            f"d'immatriculation » → null. {_ABSENT}"
        ),
    )
    marque_vehicule: str | None = Field(
        default=None,
        description=(
            "Marque (constructeur) du véhicule assuré, sans modèle ni version (ex. « Peugeot » "
            "pour « Peugeot 208 Allure » ; case 6 « Catégorie et marque du véhicule » de la "
            "carte internationale). Si plusieurs véhicules sont listés, celle du PREMIER, le "
            f"même que pour l'immatriculation. {_ABSENT}"
        ),
    )
    date_debut_validite: str | None = Field(
        default=None,
        description=(
            "Date de début de validité de l'attestation ou de la carte (« Valable du », "
            "« Période de validité du », « Date d'effet », case 3 de la carte internationale), "
            "au format ISO AAAA-MM-JJ. Les dates françaises s'écrivent jour d'abord : "
            "« 01/02/2025 » et « 1er février 2025 » → 2025-02-01. Ne pas prendre la date "
            f"d'émission ou d'édition du document. {_ABSENT}"
        ),
    )
    date_fin_validite: str | None = Field(
        default=None,
        description=(
            "Date de fin de validité de l'attestation ou de la carte (« au », « Valable "
            "jusqu'au », « Date d'échéance », case 3 de la carte internationale), au format "
            "ISO AAAA-MM-JJ, jour d'abord dans les dates françaises (« 31/01/2026 » → "
            f"2026-01-31). {_ABSENT}"
        ),
    )
    code_assureur: str | None = Field(
        default=None,
        description=(
            "Code d'identification de l'assureur (« Code assureur », « Code compagnie »), "
            "surtout présent sur la carte internationale (ex-carte verte, case 4, entre le "
            "code pays « F » et le numéro de contrat) et sur le Mémo Véhicule Assuré. "
            "Recopier uniquement le code de l'assureur, sans le code pays ni le numéro de "
            f"contrat. {_ABSENT}"
        ),
    )
    garanties: list[str] | None = Field(
        default=None,
        description=(
            "Liste des garanties souscrites telles qu'énumérées sur le document (ex. "
            "« Responsabilité civile », « Vol », « Incendie », « Bris de glace », « Dommages "
            "tous accidents », « Assistance »), un élément par garantie, dans l'ordre du "
            "document, libellés recopiés sans montants ni franchises. Ne lister que les "
            "garanties marquées comme souscrites (cochées, « oui »). Si le document ne liste "
            "aucune garantie → null, jamais une liste vide."
        ),
    )

    @field_validator("*", mode="before")
    @classmethod
    def _empty_to_none(cls, value: object) -> object:
        """Turn empty strings, blank list items and empty lists into `None`."""
        if isinstance(value, str):
            value = value.strip()
            return value or None
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            items = [item.strip() for item in value if item.strip()]
            return items or None
        return value

    @field_validator("date_debut_validite", "date_fin_validite")
    @classmethod
    def _iso_date(cls, value: str | None) -> str | None:
        """Require ISO `YYYY-MM-DD`; a French date here triggers the repair retry."""
        if value is not None:
            try:
                date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"expected ISO date YYYY-MM-DD, got {value!r}") from exc
            if len(value) != 10:
                raise ValueError(f"expected ISO date YYYY-MM-DD, got {value!r}")
        return value


def strict_json_schema(descriptions: dict[str, str] | None = None) -> dict:
    """Derive the strict JSON schema sent as `response_format`.

    Strict structured output requires every property to be listed in `required`,
    `additionalProperties: false`, and no `default`/`title` keywords. Nullable
    fields stay nullable through `anyOf [..., null]`.

    Args:
        descriptions: Field descriptions to use instead of the class ones (this is
            how a GEPA candidate's `field:<name>` texts reach the model).

    Returns:
        A JSON schema dict.
    """
    schema = copy.deepcopy(AttestationAuto.model_json_schema())
    schema.pop("title", None)
    for name, prop in schema["properties"].items():
        prop.pop("title", None)
        prop.pop("default", None)
        if descriptions is not None:
            prop["description"] = descriptions[name]
    schema["required"] = list(FIELD_NAMES)
    schema["additionalProperties"] = False
    return schema


def field_descriptions() -> dict[str, str]:
    """Return the seed description of every field, in schema order.

    Returns:
        Mapping field name → description.
    """
    return {name: AttestationAuto.model_fields[name].description or "" for name in FIELD_NAMES}
