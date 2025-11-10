# main.py
# Orchestrates generation of mbta_chat_conductor_ai.json for agent consumption.

from pathlib import Path
import json
import mbta_chat_conductor

OUT_PATH = Path("mbta_chat_conductor_ai.json")

def generate_json(out_path: Path = OUT_PATH) -> int:
    """Generate alerts JSON file. Returns number of alerts written."""
    data = mbta_chat_conductor.get_alerts_for_ai()
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(data)
