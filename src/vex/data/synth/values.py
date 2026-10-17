"""Sample the content of one synthetic certificate.

Two things come out of `sample_document`:

- `display`: everything the template prints, including distractors
  (member number, VIN, issue date, legal mentions, broker) and the
  formatting variants (dates as "1er février 2025", plates without dashes...);
- `meta`: the values an annotator should extract, following the field
  descriptions. A field the layout does not print is `None` in `meta`.

`meta` is saved as `synthetic_meta`. It is NEVER used as a label for training
or evaluation (see docs/PLAN.md §6); only tests and sanity checks read it.
All insurers are fictional (see `FICTIONAL_INSURERS` and its test).
"""

import random
from dataclasses import dataclass
from datetime import date, timedelta

from faker import Faker

LAYOUTS: tuple[str, ...] = (
    "memo",
    "carte_internationale",
    "carte_verte",
    "attestation_lettre",
    "attestation_tableau",
)


@dataclass(frozen=True)
class Insurer:
    """A fictional insurer.

    Attributes:
        name: Display name.
        code: Insurer code printed on cards and mémos.
        city: Head-office city.
        color: Brand colour (hex).
        shape: Logo shape drawn by `logos.draw_logo`.
    """

    name: str
    code: str
    city: str
    color: str
    shape: str


FICTIONAL_INSURERS: tuple[Insurer, ...] = (
    Insurer("Mutuelle Imaginaire de l'Ouest", "0457", "Nantes", "#2a6f97", "circle"),
    Insurer("Assurances Fictives du Val", "812", "Lyon", "#1b5e20", "shield"),
    Insurer("Groupe Assurfictif", "3391", "Marseille", "#8e2c48", "square"),
    Insurer("Caisse Mutuelle des Trois Rivières", "0271", "Orléans", "#b5651d", "triangle"),
    Insurer("La Prévoyance Chimérique", "5520", "Lille", "#4a3b8f", "diamond"),
    Insurer("Compagnie Utopia Assurances", "118", "Bordeaux", "#00796b", "circle"),
    Insurer("Mutuelle du Pays d'Ailleurs", "0906", "Rennes", "#c62828", "shield"),
    Insurer("Assur'Mirage", "7742", "Toulouse", "#37474f", "square"),
    Insurer("Société d'Assurances Nébuleuse", "2604", "Strasbourg", "#5d4037", "triangle"),
    Insurer("Hypothétique Assurances", "649", "Nice", "#283593", "diamond"),
    Insurer("Mutualité Fantaisie", "1835", "Dijon", "#6a1b9a", "circle"),
    Insurer("Atlantide Assurances Auto", "0388", "Brest", "#0277bd", "shield"),
)

BRANDS: dict[str, tuple[str, ...]] = {
    "Peugeot": ("208", "308", "2008", "3008", "Partner"),
    "Renault": ("Clio V", "Mégane", "Captur", "Kangoo", "Zoé"),
    "Citroën": ("C3", "C4", "Berlingo", "Jumpy"),
    "Dacia": ("Sandero", "Duster", "Spring"),
    "Toyota": ("Yaris", "Corolla", "C-HR"),
    "Volkswagen": ("Polo", "Golf", "T-Roc"),
    "Ford": ("Fiesta", "Puma", "Transit"),
    "Fiat": ("500", "Panda", "Tipo"),
    "Opel": ("Corsa", "Astra"),
    "Kia": ("Picanto", "Niro"),
    "Tesla": ("Model 3", "Model Y"),
    "Mercedes-Benz": ("Classe A", "Vito"),
    "Škoda": ("Fabia", "Octavia"),
}

GUARANTEES: tuple[str, ...] = (
    "Responsabilité civile",
    "Défense pénale et recours",
    "Vol",
    "Incendie",
    "Bris de glace",
    "Dommages tous accidents",
    "Assistance 0 km",
    "Catastrophes naturelles",
    "Garantie du conducteur",
    "Événements climatiques",
)

MONTHS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
          "septembre", "octobre", "novembre", "décembre")
_SIV_LETTERS = "ABCDEFGHJKLMNPQRSTVWXYZ"  # SIV excludes I, O and U


def _letters(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(_SIV_LETTERS) for _ in range(n))


def _siv_plate(rng: random.Random) -> str:
    return f"{_letters(rng, 2)}-{rng.randint(1, 999):03d}-{_letters(rng, 2)}"


def _fni_plate(rng: random.Random) -> str:
    return f"{rng.randint(1, 9999)} {_letters(rng, rng.choice((2, 3)))} {rng.randint(1, 95):02d}"


def format_date_fr(value: date, style: str) -> str:
    """Format a date the way French certificates print it.

    Args:
        value: Date to format.
        style: `slash` (01/02/2025), `dot` (01.02.2025), `long` (1er février 2025)
            or `iso` (2025-02-01).

    Returns:
        The formatted date.
    """
    if style == "slash":
        return value.strftime("%d/%m/%Y")
    if style == "dot":
        return value.strftime("%d.%m.%Y")
    if style == "long":
        day = "1er" if value.day == 1 else str(value.day)
        return f"{day} {MONTHS[value.month - 1]} {value.year}"
    return value.isoformat()


@dataclass(frozen=True)
class SampledDocument:
    """One synthetic document's content.

    Attributes:
        layout: Template name.
        display: Template variables.
        meta: Expected extraction (`AttestationAuto` fields, `None` if not printed).
        pages: Number of pages (1 or 2).
    """

    layout: str
    display: dict
    meta: dict
    pages: int


def sample_document(index: int, seed: int) -> SampledDocument:
    """Sample the content of document `index` for a generator `seed`.

    The layout cycles over `LAYOUTS` so every layout gets the same share.

    Args:
        index: Document index (0-based).
        seed: Generator seed.

    Returns:
        The sampled document.
    """
    rng = random.Random(f"{seed}:{index}")
    fake = Faker("fr_FR")
    fake.seed_instance(rng.randint(0, 2**31))
    layout = LAYOUTS[index % len(LAYOUTS)]
    insurer = rng.choice(FICTIONAL_INSURERS)

    is_company = rng.random() < 0.15
    holder = fake.company() if is_company else fake.name()
    civility = "" if is_company else rng.choice(("M.", "Mme", ""))
    street, postcode, city = fake.street_address(), fake.postcode(), fake.city()
    address_line = f"{street}, {postcode} {city}"

    legacy = layout == "carte_verte"
    start = date(rng.randint(2018, 2022) if legacy else rng.randint(2024, 2026), rng.randint(1, 12), rng.randint(1, 28))
    duration = rng.choices((365, 182, 30), weights=(8, 1, 1))[0]
    end = start + timedelta(days=duration - 1)
    date_style = "slash" if layout in ("carte_internationale", "carte_verte") else rng.choice(("slash", "long", "dot", "slash"))

    n_vehicles = 2 if layout == "attestation_tableau" and rng.random() < 0.3 else 1
    vehicles = []
    for _ in range(n_vehicles):
        brand = rng.choice(list(BRANDS))
        plate = _fni_plate(rng) if (legacy or rng.random() < 0.12) else _siv_plate(rng)
        vehicles.append({"brand": brand, "model": rng.choice(BRANDS[brand]), "plate": plate,
                         "plate_display": plate.replace("-", " ") if rng.random() < 0.15 else plate,
                         "vin": "VF" + "".join(rng.choice("0123456789ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(15))})
    pending_registration = layout == "attestation_lettre" and rng.random() < 0.05

    guarantees = rng.sample(GUARANTEES[1:], k=rng.randint(1, 5))
    guarantees = ["Responsabilité civile", *sorted(guarantees, key=GUARANTEES.index)]
    unchecked = [g for g in GUARANTEES if g not in guarantees][: rng.randint(0, 3)]

    contract = rng.choice((
        f"{insurer.name[:2].upper()}-{start.year}-{rng.randint(1, 999999):06d}",
        f"{rng.randint(10, 99)} {rng.randint(100, 999)} {rng.randint(100, 999)} {rng.randint(10, 99)}",
        f"AUTO/{start.year}/{rng.randint(1000, 99999)}",
        f"{rng.choice('ABCDEFGH')}{rng.randint(1000000, 9999999)}-{rng.randint(1, 9)}",
    ))

    shows = {
        "address": layout != "memo" or rng.random() < 0.6,
        "code": layout in ("carte_internationale", "carte_verte") or (layout == "memo" and rng.random() < 0.5)
        or (layout == "attestation_tableau" and rng.random() < 0.4),
        "guarantees": layout == "attestation_tableau" or (layout == "attestation_lettre" and rng.random() < 0.7)
        or (layout == "memo" and rng.random() < 0.3),
        "end_date": not (layout == "attestation_lettre" and rng.random() < 0.15),
    }
    pages = 2 if layout in ("attestation_lettre", "attestation_tableau") and rng.random() < 0.2 else 1

    display = {
        "insurer": insurer,
        "holder": holder,
        "civility": civility,
        "street": street,
        "postcode": postcode,
        "city": city,
        "show_address": shows["address"],
        "contract": contract,
        "member_number": str(rng.randint(1000000, 9999999)),
        "vehicles": vehicles,
        "pending_registration": pending_registration,
        "start": format_date_fr(start, date_style),
        "end": format_date_fr(end, date_style) if shows["end_date"] else None,
        "start_parts": (f"{start.day:02d}", f"{start.month:02d}", f"{start.year}"),
        "end_parts": (f"{end.day:02d}", f"{end.month:02d}", f"{end.year}"),
        "issued": format_date_fr(start - timedelta(days=rng.randint(1, 20)), "slash"),
        "show_code": shows["code"],
        "show_guarantees": shows["guarantees"],
        "guarantees": guarantees,
        "unchecked": unchecked,
        "broker": f"Cabinet {fake.last_name()}" if rng.random() < 0.3 else None,
        "rcs": f"RCS {insurer.city} {rng.randint(100, 999)} {rng.randint(100, 999)} {rng.randint(100, 999)}",
        "capital": f"{rng.choice((1, 2, 5, 10, 50))} 000 000",
        "phone": f"0{rng.randint(1, 9)} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)}",
        "two_pages": pages == 2,
    }
    first = vehicles[0]
    meta = {
        "assureur": insurer.name,
        "numero_contrat": contract,
        "nom_souscripteur": holder,
        "adresse_souscripteur": address_line if shows["address"] else None,
        "immatriculation": None if pending_registration else first["plate"],
        "marque_vehicule": first["brand"],
        "date_debut_validite": start.isoformat(),
        "date_fin_validite": end.isoformat() if shows["end_date"] else None,
        "code_assureur": insurer.code if shows["code"] else None,
        "garanties": guarantees if shows["guarantees"] else None,
    }
    return SampledDocument(layout=layout, display=display, meta=meta, pages=pages)
