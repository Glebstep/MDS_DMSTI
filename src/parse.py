"""
Parsing & normalization.

Takes the messy raw text from eLABa and produces:
  * structured publication records (authors, DOI, year, title)
  * deduplicated publication set (the same paper reached by N different
    co-author queries is one paper, not N)
  * mapping from authors-in-publications to our 9-group researcher roster

Author matching uses a normalized surname + first-name initial — robust to
diacritics, transliteration, and Lithuanian inflected forms.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Optional

from unidecode import unidecode

logger = logging.getLogger(__name__)



# A single author fragment:  "Surname, Firstname [Secondname] [(role)]"
# - surname: everything up to the comma (allow internal spaces & hyphens)
# - firstname: capitalized tokens after the comma
# - optional role in parens, e.g. "(tyrėjas)" — eLABa tags dataset authors this way
_AUTHOR_RE = re.compile(
    r"""
    ^\s*
    (?P<surname>[^,]+?)\s*,\s*
    (?P<firstname>[^.;()]+?)
    (?:\s*\([^)]+\))?
    \s*$
    """,
    re.VERBOSE,
)

# DOI usually appears as "DOI: 10.XXXX/..." — sometimes wrapped in markdown brackets.
# We accept any 10.XXXX form and stop at whitespace or closing brackets.
_DOI_RE = re.compile(r"DOI:\s*\[?\s*(10\.\d+/[^\s\]\)]+)", re.IGNORECASE)

# Publication year — 19XX or 20XX as a standalone token.
# We pick the LAST match (the citation often lists conference dates + pub year;
# pub year is conventionally at the end).
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def parse_publication(raw: str) -> Optional[dict]:
    """
    Convert one raw eLABa citation string into a structured record.

    Returns None if no authors could be identified (truly broken record).

    Output shape:
        {
            "authors": [(surname, firstname), ...],
            "title":   str | None,
            "doi":     str | None,
            "year":    int | None,
            "raw":     str,
        }
    """
    # The substring before the first '//' contains "Authors. Title".
    # Datasets (no journal) skip the //; we still parse what's there.
    head = raw.split("//", 1)[0]

    # Authors are separated by ';'. The LAST fragment carries the title:
    # "Surname, Firstname. Title with periods possibly".
    fragments = [f.strip() for f in head.split(";") if f.strip()]
    if not fragments:
        return None

    authors: list[tuple[str, str]] = []
    title: Optional[str] = None

    for i, frag in enumerate(fragments):
        is_last = i == len(fragments) - 1

        if is_last:
            # Find the first ". " — that splits the last author from the title.
            # We use a non-greedy author chunk so we don't gobble title text.
            m = re.match(r"^(.*?)\.\s+(.+)$", frag, re.DOTALL)
            if m:
                author_part, title = m.group(1), m.group(2).strip()
            else:
                # No title separator — entire fragment is an author (rare,
                # e.g. dataset records that end with the citation block).
                author_part = frag.rstrip(".")
        else:
            author_part = frag

        am = _AUTHOR_RE.match(author_part)
        if am:
            authors.append((am.group("surname").strip(), am.group("firstname").strip()))
        else:
            logger.debug("Skipping unparseable author fragment: %r", author_part)

    if not authors:
        return None

    doi_m = _DOI_RE.search(raw)
    doi = doi_m.group(1).rstrip(".") if doi_m else None

    # Strip identifier strings (DOI/ISBN/ISSN values) before year extraction.
    # Without this the year of "v52i1.1988" or ISBN "97819..." can win.
    cleaned_for_year = re.sub(
        r"(?:DOI|ISBN|eISBN|ISSN|eISSN):\s*\S+",
        " ",
        raw,
        flags=re.IGNORECASE,
    )
    years = _YEAR_RE.findall(cleaned_for_year)
    year = int(years[-1]) if years else None

    return {
        "authors": authors,
        "title": title,
        "doi": doi,
        "year": year,
        "raw": raw,
    }



def _norm(s: str) -> str:
    """Lowercase + transliterate diacritics. The canonical comparison key."""
    return unidecode(s.lower().strip())


def _normalize_title(t: Optional[str]) -> str:
    """Aggressively normalize a title for duplicate detection."""
    if not t:
        return ""
    t = unidecode(t.lower())
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def publication_key(pub: dict) -> str:
    """
    A stable identifier for one paper.
      * DOI when present — that's the strongest signal.
      * Otherwise: normalized-title + year. Two records with the same title
        and year are almost certainly the same paper.
    """
    if pub.get("doi"):
        return f"doi:{pub['doi'].lower()}"
    return f"title:{_normalize_title(pub.get('title'))}|year:{pub.get('year')}"


def build_researcher_index(researchers: list[dict]) -> dict[str, list[int]]:
    """
    Build {normalized_surname: [roster_idx, ...]}.
    Surnames may collide (e.g. Žilinskas Antanas vs Žilinskas Julius),
    hence the list. Disambiguation happens in match_author().
    """
    idx: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(researchers):
        idx[_norm(r["surname"])].append(i)
    return dict(idx)


def match_author(
    surname: str,
    firstname: str,
    researchers: list[dict],
    by_surname: dict[str, list[int]],
) -> Optional[int]:
    """
    Given an author appearing in some publication, find their index in the
    DMSTI roster. Returns None when there's no plausible match.

    Matching strategy:
      1) compare normalized surnames (handles Č -> C, ė -> e, casing, etc.)
      2) on collision, disambiguate by first letter of given name
    """
    candidates = by_surname.get(_norm(surname), [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    fn_initial = _norm(firstname)[:1]
    for ci in candidates:
        if _norm(researchers[ci]["firstname"])[:1] == fn_initial:
            return ci
    return None



def resolve_publications(
    publications_by_author: dict[str, list[str]],
    researchers: list[dict],
) -> dict[str, set[int]]:
    """
    Take the raw cache from scrape.fetch_publications() and:
      1) parse every raw string into a publication record
      2) deduplicate by publication_key (the same paper appears under N
         authors' rosters; we keep one copy)
      3) resolve every author mention against the researcher roster
      4) drop publications that involve fewer than two roster members
         (irrelevant for the co-author matrix — they add no edges)

    Returns: { publication_key: {roster_idx, roster_idx, ...} }
    """
    by_surname = build_researcher_index(researchers)
    seen: dict[str, set[int]] = {}

    n_parsed = n_unique = n_internal = 0

    for surname_key, raw_list in publications_by_author.items():
        for raw in raw_list:
            pub = parse_publication(raw)
            if pub is None:
                continue
            n_parsed += 1

            key = publication_key(pub)

            # Resolve every author against the roster
            roster_ids: set[int] = set()
            for sn, fn in pub["authors"]:
                ri = match_author(sn, fn, researchers, by_surname)
                if ri is not None:
                    roster_ids.add(ri)

            if not roster_ids:
                continue  # no DMSTI members involved (shouldn't really happen)

            if key in seen:
                # Same paper, second perspective — merge author sets defensively.
                seen[key] |= roster_ids
            else:
                seen[key] = roster_ids
                n_unique += 1
                if len(roster_ids) >= 2:
                    n_internal += 1

    logger.info(
        "Parsed %d raw rows → %d unique publications, of which %d link "
        "at least two DMSTI researchers (the ones that contribute edges).",
        n_parsed, n_unique, n_internal,
    )
    return seen



if __name__ == "__main__":
    import json
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    fixture = Path(__file__).resolve().parent.parent / "data" / "fixture_sabaliauskas.json"
    with fixture.open(encoding="utf-8") as f:
        raws = json.load(f)

    print(f"Parsing {len(raws)} fixture publications...\n")
    for raw in raws:
        pub = parse_publication(raw)
        if pub is None:
            print("FAILED:", raw[:80])
            continue
        authors_str = "; ".join(f"{s}, {n}" for s, n in pub["authors"])
        print(f"[{pub['year']}] {authors_str}")
        print(f"   title: {(pub['title'] or '')[:90]}")
        print(f"   doi:   {pub['doi']}")
        print()
