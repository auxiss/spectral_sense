"""
demod.py

The "modulations" folder holds, side by side:

    WFM.json, NFM.json, AM.json ...   one small file per modulation (the labels)
    wfm.py, nfm.py, am.py ...         the demodulators (code)

A JSON file points at its demodulator by name:

    {"name": "WFM", "demodulator": "wfm", "digital": false, "protocols": []}

"wfm" means the file modulations/wfm.py, which must define

    DEMODULATOR = SomeClass        # a Demodulator subclass

A modulation without a "demodulator" is label-only: it is just a name for
the dataset and its Demod button stays disabled. Dropping a new .py file in
the folder and naming it in a JSON file is all it takes to add a demodulator.

Older JSON files that say "analog_voice" still work: the demodulator is then
looked up by the modulation's own name (WFM -> wfm.py).
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import time
from pathlib import Path


AUDIO_RATE = 48000          # every demodulator outputs audio at this rate


DEFAULT_ENTRIES = [
    {"name": "unknown", "demodulator": None,  "digital": False, "protocols": []},
    {"name": "WFM",     "demodulator": "wfm", "digital": False, "protocols": []},
    {"name": "NFM",     "demodulator": "nfm", "digital": False, "protocols": []},
    {"name": "AM",      "demodulator": "am",  "digital": False, "protocols": []},
    {"name": "BPSK",    "demodulator": None,  "digital": True,  "protocols": []},
    {"name": "QPSK",    "demodulator": None,  "digital": True,  "protocols": []},
    {"name": "FSK",     "demodulator": None,  "digital": True,  "protocols": []},
    {"name": "OOK",     "demodulator": None,  "digital": True,  "protocols": []},
]


# ----------------------------------------------------------------------
# Demodulator base class
# ----------------------------------------------------------------------

class Demodulator:
    """
    process() gets isolated channel IQ (any sample rate, in chunks) and
    returns float32 audio at AUDIO_RATE, or None if there is nothing yet.
    Every demodulator keeps its own state between chunks, so chunk
    boundaries are invisible in the audio.
    """

    def process(self, iq, sample_rate):
        raise NotImplementedError

    def reset(self):
        """Forget all state (called when a new transmission starts)."""


# ----------------------------------------------------------------------
# Registry (the "modulations" folder)
# ----------------------------------------------------------------------

class ModulationRegistry:
    def __init__(self, folder="modulations"):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

        # the demodulators are imported as <folder name>.<file name>
        parent = str(self.folder.resolve().parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)

        self._entries: dict[str, dict] = {}
        self._load()

        if not self._entries:
            for entry in DEFAULT_ENTRIES:
                self._entries[entry["name"]] = dict(entry)
                self._save(entry["name"])

    # ------------------------------------------------------------ storage
    def _load(self):
        for path in sorted(self.folder.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._entries[data["name"]] = {
                    "name": data["name"],
                    "demodulator": data.get("demodulator"),
                    "digital": bool(data.get("digital", False)),
                    "protocols": list(data.get("protocols", [])),
                }
            except (OSError, ValueError, KeyError):
                continue                  # ignore unreadable files

    def _save(self, name):
        safe = re.sub(r"[^\w\-]+", "_", name)
        path = self.folder / f"{safe}.json"
        path.write_text(json.dumps(self._entries[name], indent=2), encoding="utf-8")

    def _find(self, name):
        """Case-insensitive lookup. Returns the stored name or None."""
        wanted = (name or "").strip().lower()
        for stored in self._entries:
            if stored.lower() == wanted:
                return stored
        return None

    def _module_key(self, entry):
        """File name (without .py) of the entry's demodulator, or None."""

        key = entry.get("demodulator")
        if not key:
            return None

        if (self.folder / f"{key}.py").exists():
            return key

        legacy = entry["name"].lower()    # old files say "analog_voice"
        if (self.folder / f"{legacy}.py").exists():
            return legacy

        return None

    # --------------------------------------------------------------- query
    def names(self):
        others = sorted(n for n in self._entries if n != "unknown")
        return (["unknown"] if "unknown" in self._entries else []) + others

    def protocols(self, modulation):
        key = self._find(modulation)
        return list(self._entries[key]["protocols"]) if key else []

    def can_decode(self, modulation, protocol=None):
        """
        True if the Demod button may be enabled.
        Analog: the modulation's demodulator file must exist.
        Digital: would also need a protocol decoder (none exist yet).
        """
        key = self._find(modulation)
        if key is None:
            return False

        entry = self._entries[key]

        if entry["digital"]:
            return False                  # TODO: protocol decoders

        return self._module_key(entry) is not None

    def create_demodulator(self, modulation):
        key = self._find(modulation)
        if key is None:
            return None

        module_key = self._module_key(self._entries[key])
        if module_key is None:
            return None

        try:
            importlib.invalidate_caches()
            module = importlib.import_module(f"{self.folder.name}.{module_key}")
            return module.DEMODULATOR()
        except Exception as exc:
            print(f"Could not load demodulator '{module_key}': {exc}")
            return None

    # ------------------------------------------------------------- editing
    def add_modulation(self, name):
        """Create a label-only modulation. Returns False if it exists or is blank."""
        name = (name or "").strip()

        if not name or self._find(name):
            return False

        self._entries[name] = {
            "name": name, "demodulator": None, "digital": False, "protocols": [],
        }
        self._save(name)
        return True

    def add_protocol(self, modulation, protocol):
        """Add a label-only protocol under a modulation."""
        protocol = (protocol or "").strip()
        key = self._find(modulation)

        if key is None or not protocol or protocol.lower() == "unknown":
            return False

        if protocol in self._entries[key]["protocols"]:
            return False

        self._entries[key]["protocols"].append(protocol)
        self._save(key)
        return True


# ----------------------------------------------------------------------
# What the observer worker calls
# ----------------------------------------------------------------------

class DemodRouter:
    """
    Picks the demodulator of the channel's modulation, feeds it the isolated
    channel samples and hands the audio to the player.
    """

    GAP_RESET_S = 0.25      # silence this long = a new transmission: reset the demodulator

    def __init__(self, registry: ModulationRegistry, player=None):
        self.registry = registry
        self.player = player
        self._instances = {}          # (channel id, modulation) -> [demodulator, last call time]

    def process(self, iq, sample_rate, channel):
        key = (channel.id, channel.modulation)
        entry = self._instances.get(key)

        if entry is None:
            entry = [self.registry.create_demodulator(channel.modulation), 0.0]
            self._instances[key] = entry

        demodulator = entry[0]

        if demodulator is None:
            return None

        now = time.monotonic()

        if now - entry[1] > self.GAP_RESET_S:
            demodulator.reset()

        entry[1] = now

        audio = demodulator.process(iq, sample_rate)

        if audio is not None and len(audio) and self.player is not None:
            self.player.push(audio)

        return audio

    def close(self):
        if self.player is not None:
            self.player.stop()
