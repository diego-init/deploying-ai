# Create an improved, AI-friendly and time zone–aware version of the script.
# It reorganizes the code, adds data classes, robust parsing, and converts
# UTC timestamps to Boston's America/New_York timezone.

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone, timedelta
import zoneinfo

# -----------------------------
# Constants & Config
# -----------------------------
ALERTS_URL = "https://cdn.mbta.com/realtime/Alerts_enhanced.json"
BOSTON_TZ = zoneinfo.ZoneInfo("America/New_York")
GTFS_DIR = Path("./gtfs")

# -----------------------------
# HTTP Session with Retries
# -----------------------------
def build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=False  # for compatibility across urllib3 versions
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "MBTA-AI-Agent/2.0"})
    return session

SESSION = build_session()

def fetch_json(url: str) -> Dict[str, Any]:
    resp = SESSION.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()

# -----------------------------
# GTFS Loaders
# -----------------------------
@dataclass
class GTFSData:
    routes: pd.DataFrame
    stops: pd.DataFrame
    shapes: pd.DataFrame
    facilities: pd.DataFrame
    trips: pd.DataFrame

def load_gtfs(gtfs_dir: Path) -> GTFSData:
    def read_csv(name: str) -> pd.DataFrame:
        p = gtfs_dir / name
        if not p.exists():
            # Return empty DF with no columns—lookups will handle empties gracefully.
            return pd.DataFrame()
        return pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[])

    return GTFSData(
        routes=read_csv("routes.txt"),
        stops=read_csv("stops.txt"),
        shapes=read_csv("shapes.txt"),
        facilities=read_csv("facilities.txt"),
        trips=read_csv("trips.txt"),
    )

GTFS = load_gtfs(GTFS_DIR)

# -----------------------------
# Time Utilities
# -----------------------------
def utc_epoch_to_boston_iso(ts: Optional[int]) -> Optional[str]:
    """Convert a UTC epoch (seconds) to ISO 8601 in America/New_York."""
    if ts is None:
        return None
    try:
        dt_utc = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        dt_bos = dt_utc.astimezone(BOSTON_TZ)
        # Use timespec='seconds' for concise ISO format
        return dt_bos.isoformat(timespec="seconds")
    except Exception:
        return None

def humanize_timedelta(seconds: Optional[int]) -> Optional[str]:
    if seconds is None:
        return None
    try:
        td = timedelta(seconds=int(seconds))
        # Simple humanization: D days, HH:MM:SS
        days = td.days
        rem = td - timedelta(days=days)
        hours, rem = divmod(rem.seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        if days:
            return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except Exception:
        return None

# -----------------------------
# Data Models
# -----------------------------
@dataclass
class AlertRecord:
    alert_id: str
    effect: Optional[str]
    effect_detail: Optional[str]
    cause: Optional[str]
    cause_detail: Optional[str]
    lifecycle: Optional[str]

    created_at_boston: Optional[str]
    last_modified_boston: Optional[str]
    age_seconds: Optional[int]
    age_human: Optional[str]

    header_texts: List[str]
    routes_affected: List[str]
    route_ids_affected: List[str]
    route_desc_affected: List[str]
    stops_affected: List[str]
    stops_latlon_affected: List[Tuple[str, str]]
    facilities_affected: List[str]
    shapes_affected: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# -----------------------------
# Lookups
# -----------------------------
def route_name(route_id: str, routes_df: pd.DataFrame) -> Optional[str]:
    if routes_df.empty or not route_id:
        return None
    m = routes_df[routes_df["route_id"] == route_id]
    if m.empty:
        return None
    # Prefer long name, fallback to short if needed
    val = m["route_long_name"].values[0] if "route_long_name" in m else None
    if not val or str(val).strip() == "":
        val = m["route_short_name"].values[0] if "route_short_name" in m else None
    return val

def route_desc(route_id: str, routes_df: pd.DataFrame) -> Optional[str]:
    if routes_df.empty or not route_id or "route_desc" not in routes_df.columns:
        return None
    m = routes_df[routes_df["route_id"] == route_id]
    if m.empty:
        return None
    return m["route_desc"].values[0]

def shape_id_from_route(route_id: str, trips_df: pd.DataFrame) -> Optional[str]:
    if trips_df.empty or not route_id or "route_id" not in trips_df.columns or "shape_id" not in trips_df.columns:
        return None
    m = trips_df[trips_df["route_id"] == route_id]
    if m.empty:
        return None
    return m["shape_id"].values[0]

def stop_name_and_latlon(stop_id: str, stops_df: pd.DataFrame) -> Tuple[Optional[str], Optional[Tuple[str, str]]]:
    if stops_df.empty or not stop_id:
        return None, None
    m = stops_df[stops_df["stop_id"] == stop_id]
    if m.empty:
        return None, None
    name = m["stop_name"].values[0] if "stop_name" in m else None
    lat = m["stop_lat"].values[0] if "stop_lat" in m else None
    lon = m["stop_lon"].values[0] if "stop_lon" in m else None
    if lat is not None and lon is not None:
        return name, (str(lat), str(lon))
    return name, None

# -----------------------------
# Parser
# -----------------------------
def parse_alerts(payload: Dict[str, Any], gtfs: GTFSData) -> List[AlertRecord]:
    entities = payload.get("entity", []) or []
    alerts: List[AlertRecord] = []

    for ent in entities:
        try:
            alert = ent.get("alert", {})
            effect = alert.get("effect")
            effect_detail = alert.get("effect_detail")
            cause = alert.get("cause")
            cause_detail = alert.get("cause_detail")
            lifecycle = alert.get("alert_lifecycle")

            created_ts = alert.get("created_timestamp")
            last_ts = alert.get("last_modified_timestamp")

            created_iso = utc_epoch_to_boston_iso(created_ts)
            last_iso = utc_epoch_to_boston_iso(last_ts)

            age_sec = None
            if isinstance(created_ts, int) and isinstance(last_ts, int):
                age_sec = int(last_ts) - int(created_ts)
            elif created_ts is not None and last_ts is not None:
                # attempt coercion
                try:
                    age_sec = int(last_ts) - int(created_ts)
                except Exception:
                    age_sec = None

            age_h = humanize_timedelta(age_sec)

            # Header texts
            header_texts: List[str] = []
            header = alert.get("header_text", {})
            trans = header.get("translation", []) or []
            for t in trans:
                txt = t.get("text")
                if txt:
                    header_texts.append(str(txt))

            # Informed entities
            informed = alert.get("informed_entity", []) or []
            route_ids: List[str] = []
            route_names: List[str] = []
            route_descs: List[str] = []
            shapes: List[str] = []
            stops_names: List[str] = []
            stops_latlon: List[Tuple[str, str]] = []
            facilities: List[str] = []

            for info in informed:
                rid = info.get("route_id")
                sid = info.get("stop_id")
                fid = info.get("facility_id")

                # Routes
                if rid:
                    route_ids.append(str(rid))
                    rname = route_name(str(rid), gtfs.routes)
                    rdesc = route_desc(str(rid), gtfs.routes)
                    shid = shape_id_from_route(str(rid), gtfs.trips)
                    if rname:
                        route_names.append(rname)
                    if rdesc:
                        route_descs.append(rdesc)
                    if shid:
                        shapes.append(shid)

                # Stops
                if sid:
                    sname, latlon = stop_name_and_latlon(str(sid), gtfs.stops)
                    if sname:
                        stops_names.append(sname)
                    if latlon:
                        stops_latlon.append(latlon)

                # Facilities
                if fid:
                    facilities.append(str(fid))

            # Provide defaults if lists ended empty, to keep structure explicit.
            if not route_ids:
                route_ids = []
            if not route_names:
                route_names = []
            if not route_descs:
                route_descs = []
            if not shapes:
                shapes = []
            if not stops_names:
                stops_names = []
            if not stops_latlon:
                stops_latlon = []
            if not facilities:
                facilities = []

            alerts.append(
                AlertRecord(
                    alert_id=str(ent.get("id", "")),
                    effect=effect,
                    effect_detail=effect_detail,
                    cause=cause,
                    cause_detail=cause_detail,
                    lifecycle=lifecycle,
                    created_at_boston=created_iso,
                    last_modified_boston=last_iso,
                    age_seconds=age_sec,
                    age_human=age_h,
                    header_texts=header_texts,
                    routes_affected=route_names,
                    route_ids_affected=route_ids,
                    route_desc_affected=route_descs,
                    stops_affected=stops_names,
                    stops_latlon_affected=stops_latlon,
                    facilities_affected=facilities,
                    shapes_affected=shapes,
                )
            )
        except Exception:
            # Skip malformed entity but continue
            continue

    return alerts

# -----------------------------
# Public API for an AI Agent
# -----------------------------
def get_alerts_for_ai() -> List[Dict[str, Any]]:
    """
    Fetch the latest MBTA alerts, enrich with GTFS context, and return
    AI-friendly dictionaries with Boston-local timestamps and compact fields.
    """
    payload = fetch_json(ALERTS_URL)
    alerts = parse_alerts(payload, GTFS)
    return [a.to_dict() for a in alerts]

def get_alerts_json(indent: Optional[int] = 2) -> str:
    """Return the same information as a JSON string (UTF-8)."""
    return json.dumps(get_alerts_for_ai(), ensure_ascii=False, indent=indent)
