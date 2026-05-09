"""Scrape UFCStats.com for historical fight data, fighter stats, and events.

UFCStats.com structure (verified):
  - Events: /statistics/events/completed?page=all
    → table.b-statistics__table-events → rows with event links
  - Event detail: /event-details/{id}
    → table.b-fight-details__table_type_event-details → rows with fight links
    → Each row has: W/L, Fighter names, Kd, Str, Td, Sub, Weight, Method, Round, Time
  - Fight detail: /fight-details/{id}
    → Round-by-round stats tables (Totals + Significant Strikes)
    → Tale-of-the-tape (height, reach, stance, DOB)
  - Fighter detail: /fighter-details/{id}
    → Career stats, physical attributes
"""

import re
import time
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import requests
from bs4 import BeautifulSoup

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
DELAY = 0.5  # Be polite to the server


def _fet(url: str) -> BeautifulSoup:
    """Fetch URL, return BeautifulSoup."""
    time.sleep(DELAY)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.content, "lxml")


# ═══ SCRAPE EVENTS ═══════════════════════════════════════════════════

def scrape_events() -> pd.DataFrame:
    """Scrape all completed UFC events."""
    log.info("Scraping completed events list...")
    soup = _fet(config.UFOSTATS_EVENTS)

    rows = []
    for tr in soup.select("table.b-statistics__table-events tbody tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue

        # First td has event name (link) + date (span)
        link = tds[0].find("a")
        if not link:
            continue

        href = link.get("href", "")
        event_id = href.rstrip("/").split("/")[-1] if href else ""
        event_name = link.text.strip()

        # Date is in a <span class="b-statistics__date">
        date_span = tds[0].find("span", class_="b-statistics__date")
        event_date = date_span.text.strip() if date_span else ""

        location = tds[1].text.strip() if len(tds) > 1 else ""

        if event_id:
            rows.append({
                "event_id": event_id,
                "event_name": event_name,
                "event_date": event_date,
                "location": location,
                "event_url": href,
            })

    df = pd.DataFrame(rows)
    df.to_csv(config.RAW_EVENTS, index=False)
    log.info("Saved %d events", len(df))
    return df


# ═══ SCRAPE FIGHTS FROM AN EVENT ═════════════════════════════════════

def _scrape_event_fights(event_url: str, event_id: str, event_date: str) -> list[dict]:
    """Scrape all fights from a single event detail page."""
    try:
        soup = _fet(event_url)
    except Exception as e:
        log.warning("Failed to fetch event %s: %s", event_id, e)
        return []

    fights = []
    for row in soup.select("tr.b-fight-details__table-row__hover"):
        tds = row.find_all("td")
        if len(tds) < 10:
            continue

        fight_link = row.get("data-link", "")
        fight_id = fight_link.rstrip("/").split("/")[-1] if fight_link else ""

        # Get fighter names from the links inside the second td
        fighter_links = tds[1].find_all("a")
        if len(fighter_links) >= 2:
            fighter_a = fighter_links[0].text.strip()
            fighter_b = fighter_links[1].text.strip()
        else:
            continue

        # W/L flag
        wl_text = tds[0].get_text(strip=True).lower()
        # Winner is the first-named fighter if "win" appears
        winner = fighter_a if "win" in wl_text else (fighter_b if wl_text else "")

        # Kd, Str, Td, Sub
        kd_a, kd_b = _split_pair(tds[2].get_text(strip=True))
        str_a, str_b = _split_pair(tds[3].get_text(strip=True))
        td_a, td_b = _split_pair(tds[4].get_text(strip=True))
        sub_a, sub_b = _split_pair(tds[5].get_text(strip=True))
        weight_class = tds[6].get_text(strip=True)
        method = tds[7].get_text(strip=True)
        fight_round = tds[8].get_text(strip=True)
        fight_time = tds[9].get_text(strip=True)

        fights.append({
            "fight_id": fight_id,
            "fight_url": fight_link,
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "winner": winner,
            "kd_a": kd_a, "kd_b": kd_b,
            "str_a": str_a, "str_b": str_b,
            "td_a": td_a, "td_b": td_b,
            "sub_a": sub_a, "sub_b": sub_b,
            "weight_class": weight_class,
            "method": method,
            "round": fight_round,
            "time": fight_time,
            "event_id": event_id,
            "event_date": event_date,
        })

    return fights


def _split_pair(text: str) -> tuple[str, str]:
    """Split a concatenated pair like '12' or '1 of 20 of 0' into (a_val, b_val).

    Handles formats like:
      - '12' (two digits for A and B)
      - '1 of 20 of 0' (attempted/landed format)
    """
    text = text.strip()
    if not text:
        return "", ""

    # Try splitting digits: e.g. '12' → '1', '2'; '010' → '0', '10'
    # Match patterns like: digits, optionally 'of digits', etc.
    parts = re.findall(r'\d+\s*(?:of\s*\d+)?', text)
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    if len(parts) == 1:
        return parts[0].strip(), "0"

    return text[:len(text)//2], text[len(text)//2:]


# ═══ SCRAPE FIGHTER DETAILS ══════════════════════════════════════════

def _scrape_fighter_detail(fighter_name: str, fighter_url: str) -> dict:
    """Scrape fighter physical and career stats from detail page."""
    try:
        soup = _fet(fighter_url)
    except Exception as e:
        log.warning("Failed to fetch fighter %s: %s", fighter_name, e)
        return {}

    attrs = {"fighter_name": fighter_name}

    # Career stats from the stat blocks
    stat_blocks = soup.select("ul.b-list__box-list li")
    for li in stat_blocks:
        text = li.get_text(" ", strip=True)
        text_clean = text.replace("\n", " ").replace("  ", " ")

        for label, key in [
            ("Height:", "height"), ("Weight:", "weight"), ("Reach:", "reach"),
            ("STANCE:", "stance"), ("DOB:", "dob"),
            ("SLpM:", "slpm"), ("Str. Acc.:", "str_acc"),
            ("SApM:", "sapm"), ("Str. Def:", "str_def"),
            ("TD Avg.:", "td_avg"), ("TD Acc.:", "td_acc"),
            ("TD Def.:", "td_def"), ("Sub. Avg.:", "sub_avg"),
        ]:
            if label in text_clean:
                val = text_clean.replace(label, "").strip()
                if val:
                    attrs[key] = val
                break

    # Also try the record header
    record_el = soup.select_one("span.b-content__title-record")
    if record_el:
        attrs["record"] = record_el.get_text(strip=True)

    return attrs


def _collect_fighter_urls(fighter_names: set[str]) -> dict[str, str]:
    """Build fighter name → URL mapping by searching the fighter index."""
    fighter_urls = {}
    log.info("Searching fighter URLs for %d fighters...", len(fighter_names))

    for letter in "abcdefghijklmnopqrstuvwxyz":
        url = f"{config.UFOSTATS_FIGHTERS}?char={letter}&page=all"
        try:
            soup = _fet(url)
        except Exception:
            continue

        for row in soup.select("table.b-statistics__table tbody tr"):
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            first_name = cols[0].get_text(strip=True)
            last_name = cols[1].get_text(strip=True)
            full = f"{first_name} {last_name}"

            if full in fighter_names:
                link = cols[0].find("a")
                href = link.get("href", "") if link else ""
                fighter_urls[full] = href

    log.info("Found URLs for %d/%d fighters", len(fighter_urls), len(fighter_names))
    return fighter_urls


def _scrape_all_fighter_details(fights_df: pd.DataFrame) -> pd.DataFrame:
    """Scrape physical attributes for all fighters referenced in fight data."""
    fighter_names = set(fights_df["fighter_a"].unique()) | set(fights_df["fighter_b"].unique())
    fighter_urls = _collect_fighter_urls(fighter_names)

    rows = []
    for name, url in fighter_urls.items():
        attrs = _scrape_fighter_detail(name, url)
        if attrs:
            rows.append(attrs)

    df = pd.DataFrame(rows)
    df.to_csv(config.RAW_FIGHTERS, index=False)
    log.info("Saved %d fighter profiles", len(df))
    return df


# ═══ SCRAPE FIGHT DETAILS (ROUND STATS) ══════════════════════════════

def _scrape_fight_detail(fight_id: str, fight_url: str) -> dict | None:
    """Scrape detailed per-round fight stats from a fight details page.

    Parses the round-by-round tables and aggregates totals across all rounds.
    """
    try:
        soup = _fet(fight_url)
    except Exception as e:
        log.warning("Failed to fetch fight %s: %s", fight_id, e)
        return None

    # Find round-by-round stats tables
    tables = soup.select("table.b-fight-details__table")
    if len(tables) < 1:
        return None

    # Aggregate round stats
    totals = defaultdict(float)

    for table in tables:
        # Each row in tbody is one round (or a section header)
        for row in table.select("tbody tr"):
            tds = row.find_all("td")
            if len(tds) < 3:
                continue

            # Column 0 = stat label, Column 1 = fighter A value, Column 2 = fighter B value
            # But the exact structure varies. Let's look for pattern.
            texts = [td.get_text(strip=True) for td in tds]

            if len(texts) >= 3 and texts[0]:
                stat_name = texts[0].lower().replace(" ", "_")
                val_a = _safe_numeric(texts[1])
                val_b = _safe_numeric(texts[2])
                if val_a is not None:
                    totals[f"{stat_name}_a"] += val_a
                if val_b is not None:
                    totals[f"{stat_name}_b"] += val_b

    if totals:
        totals["fight_id"] = fight_id
        return dict(totals)

    return None


def _safe_numeric(text: str) -> float | None:
    """Extract the first numeric value from text like '15 of 30' → 15."""
    if not text:
        return None
    m = re.search(r'(\d+(?:\.\d+)?)', text)
    if m:
        return float(m.group(1))
    return None


# ═══ MAIN ════════════════════════════════════════════════════════════

def run():
    """Main scraping pipeline."""
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1. Scrape events
    events = scrape_events()

    # 2. Scrape fights from each event
    all_fights = []
    for _, ev in events.iterrows():
        log.info("Event: %s (%s)", ev["event_name"], ev["event_date"])
        fights = _scrape_event_fights(
            ev.get("event_url", ""),
            ev["event_id"],
            ev["event_date"],
        )
        all_fights.extend(fights)

    fights_df = pd.DataFrame(all_fights)
    fights_df.to_csv(config.RAW_FIGHTS, index=False)
    log.info("Saved %d fights", len(fights_df))

    if fights_df.empty:
        log.warning("No fights scraped. Check URL/selectors.")
        return

    # 3. Scrape fighter details
    _scrape_all_fighter_details(fights_df)

    # 4. Scrape detailed fight stats (optional, slower)
    log.info("Scraping detailed fight stats (this takes a while)...")
    detail_rows = []
    for _, fight in fights_df.iterrows():
        detail = _scrape_fight_detail(fight["fight_id"], fight["fight_url"])
        if detail:
            detail_rows.append(detail)

    if detail_rows:
        details_df = pd.DataFrame(detail_rows)
        fights_df = fights_df.merge(
            details_df, on="fight_id", how="left", suffixes=("", "_detail")
        )

    fights_df.to_csv(config.RAW_FIGHTS, index=False)
    log.info("*** Scraping complete ***")


if __name__ == "__main__":
    run()
