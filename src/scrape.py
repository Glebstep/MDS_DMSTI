"""
Scraping module: pulls researcher rosters from VU DMSTI and publications from eLABa.

Data sources:
  * fetch_researchers(): research group staff from /staff-2/by-departmentt
  * fetch_additional_staff(): lecturers + project employees from the same page
  * fetch_publications(): eLABa for each person -> list of raw citation strings

All network calls cached as JSON under data/.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

MII_BY_DEPARTMENT_URL = "https://www.mii.lt/en/structure/staff-2/by-departmentt"
ELABA_URL = "https://elaba.mb.vu.lt/dmsti/"

RESEARCH_GROUPS = {
    "Artificial Intelligence Laboratory",
    "Blockchain and Quantum Technologies Group",
    "Cognitive Computing Group",
    "Cyber-Social Systems Engineering Group",
    "Education Systems Group",
    "Global Optimization Group",
    "Image and Signal Analysis Group",
    "Intelligent Technologies Research Group",
    "Interdisciplinary Statistical Research Group",
}

# Sections on by-departmentt that contain potential "hidden" researchers.
ADDITIONAL_SECTIONS = {"Lecturers", "Employees in Projects", "Employees"}

NAME_OVERRIDES: dict[str, tuple[str, str]] = {
    "Moura de lima Glauco endrigo": ("Glauco Endrigo", "Moura de Lima"),
}

REQUEST_DELAY = 0.5
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DMSTI-Scientometrics-Bot/1.0; academic task)"
}


def split_display_name(display_name: str) -> tuple[str, str]:
    if display_name in NAME_OVERRIDES:
        return NAME_OVERRIDES[display_name]
    parts = display_name.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Cannot split: {display_name!r}")
    surname, *given = parts
    return " ".join(given), surname


def _dedup_key(firstname: str, surname: str) -> str:
    return f"{surname.lower()}_{firstname[:1].lower()}"


def _parse_by_departmentt(
    soup: BeautifulSoup,
    target_sections: set[str],
    seen: set[str],
    group_label: str | None = None,
) -> list[dict]:
    """Parse specific sections from the by-departmentt page."""
    results: list[dict] = []
    for heading in soup.find_all(["h3", "h2"]):
        section_name = " ".join(heading.get_text(strip=True).split())
        if section_name not in target_sections:
            continue
        member_list = heading.find_next("ul")
        if member_list is None:
            continue
        for a in member_list.find_all("a"):
            display = a.get_text(strip=True)
            if not display:
                continue
            try:
                firstname, surname = split_display_name(display)
            except ValueError:
                continue
            key = _dedup_key(firstname, surname)
            if key in seen:
                logger.info("Dedup: %s already in roster", display)
                continue
            seen.add(key)
            results.append({
                "display_name": display,
                "firstname": firstname,
                "surname": surname,
                "group": group_label,
            })
    return results


def fetch_researchers(
    cache_path: Path = Path("data/researchers.json"),
    refresh: bool = False,
) -> list[dict]:
    """Fetch research-group staff. Each entry has a real group name."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        with cache_path.open(encoding="utf-8") as f:
            return json.load(f)

    logger.info("Fetching researcher roster from %s", MII_BY_DEPARTMENT_URL)
    resp = requests.get(MII_BY_DEPARTMENT_URL, headers=DEFAULT_HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    seen: set[str] = set()
    researchers: list[dict] = []

    for heading in soup.find_all(["h3", "h2"]):
        group_name = " ".join(heading.get_text(strip=True).split())
        if group_name not in RESEARCH_GROUPS:
            continue
        member_list = heading.find_next("ul")
        if member_list is None:
            continue
        for a in member_list.find_all("a"):
            display = a.get_text(strip=True)
            if not display:
                continue
            try:
                firstname, surname = split_display_name(display)
            except ValueError:
                continue
            key = _dedup_key(firstname, surname)
            if key in seen:
                logger.info("Dedup: %s already in roster, skipping", display)
                continue
            seen.add(key)
            researchers.append({
                "display_name": display,
                "firstname": firstname,
                "surname": surname,
                "group": group_name,
            })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(researchers, f, ensure_ascii=False, indent=2)
    logger.info("Cached %d researchers to %s", len(researchers), cache_path)
    return researchers


def fetch_additional_staff(
    existing_keys: set[str],
    cache_path: Path = Path("data/additional_staff.json"),
    refresh: bool = False,
) -> list[dict]:
    """
    Fetch lecturers and project employees from by-departmentt.
    These are candidates — group is set to None. main.py will assign
    groups based on co-authorship or discard if isolated.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        with cache_path.open(encoding="utf-8") as f:
            return json.load(f)

    logger.info("Fetching additional staff (lecturers, project employees)...")
    resp = requests.get(MII_BY_DEPARTMENT_URL, headers=DEFAULT_HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    results = _parse_by_departmentt(soup, ADDITIONAL_SECTIONS, existing_keys, group_label=None)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Cached %d additional staff", len(results))
    return results

def fetch_phd_students(
    existing_keys: set[str],
    cache_path: Path = Path("data/phd_students.json"),
    refresh: bool = False,
) -> list[dict]:
    """Scrape PhD students, return those not already in roster. Group=None."""
    PHD_URL = "https://www.mii.lt/en/doctoral-studies/phd-students-list/alphabetical-list"
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        with cache_path.open(encoding="utf-8") as f:
            return json.load(f)

    logger.info("Fetching PhD students from %s", PHD_URL)
    resp = requests.get(PHD_URL, headers=DEFAULT_HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    students: list[dict] = []
    for a in soup.select("a[href*='/structure/staff/']"):
        display = " ".join(a.get_text(strip=True).split())
        if not display or len(display) < 3:
            continue
        try:
            firstname, surname = split_display_name(display)
        except ValueError:
            continue
        key = _dedup_key(firstname, surname)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        students.append({
            "display_name": display,
            "firstname": firstname,
            "surname": surname,
            "group": None,
        })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(students, f, ensure_ascii=False, indent=2)
    logger.info("Cached %d new PhD students", len(students))
    return students


def _fetch_one_author(firstname: str, surname: str) -> list[str]:
    query = f"{firstname} {surname}"
    resp = requests.get(ELABA_URL, params={"aut": query}, headers=DEFAULT_HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    raw: list[str] = []
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) != 2:
            continue
        if cells[0].get_text(strip=True) == "Eil. Nr.":
            continue
        text = cells[1].get_text(separator=" ", strip=True)
        if text:
            raw.append(text)
    return raw


def fetch_publications(
    researchers: list[dict],
    cache_path: Path = Path("data/publications.json"),
    refresh: bool = False,
) -> dict[str, list[str]]:
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        with cache_path.open(encoding="utf-8") as f:
            return json.load(f)

    publications: dict[str, list[str]] = {}
    n = len(researchers)
    for i, r in enumerate(researchers, 1):
        logger.info("[%d/%d] Fetching publications for %s %s", i, n, r["firstname"], r["surname"])
        try:
            raw = _fetch_one_author(r["firstname"], r["surname"])
        except requests.RequestException as e:
            logger.error("Failed for %s: %s", r["display_name"], e)
            raw = []
        publications[r["display_name"]] = raw
        time.sleep(REQUEST_DELAY)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(publications, f, ensure_ascii=False, indent=2)
    logger.info("Cached publications for %d authors (total %d records)",
                len(publications), sum(len(v) for v in publications.values()))
    return publications


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    rs = fetch_researchers()
    print(f"\n{len(rs)} researchers across {len({r['group'] for r in rs})} groups")
    for g in sorted({r["group"] for r in rs}):
        print(f"  {g}: {sum(1 for r in rs if r['group'] == g)}")



