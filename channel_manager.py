import csv
import shutil
from dataclasses import dataclass, asdict, fields
from pathlib import Path

from datetime import datetime, timezone
import numpy as np
from scipy.io import wavfile


UNKNOWN = "unknown"
CAPTURE_INFO_HEADER = ["capture_file", "gain", "time", "num_samples", "bandwidth"]


@dataclass
class RFRange:
    id: int
    name: str
    start_freq: float
    end_freq: float
    modulation: str
    protocol: str
    samples_dir: str
    info: str


# folder structure:
# channels/
# |-- channels.csv                 (all channels info)
# |-- samples/
#     |-- <channel_id>/
#     |     |-- captures.csv       (capture_file, gain, time, num_samples, bandwidth)
#     |     |-- capture_0.wav
#     |     |-- capture_1.wav
#     |-- <channel_id>/
#           ...


class ChannelManager:
    def __init__(self, path="channels"):
        self.path = Path(path)
        self.csv_path = self.path / "channels.csv"
        self.samples_root = self.path / "samples"
        self.path.mkdir(parents=True, exist_ok=True)
        self.samples_root.mkdir(exist_ok=True)

        self.channels: dict[int, RFRange] = {}
        self._load()

    # ---------------------------------------------------------- persistence
    def _load(self):
        self.channels.clear()
        if not self.csv_path.exists():
            self._save()
            return
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ch = RFRange(
                    id=int(row["id"]),
                    name=row["name"],
                    start_freq=float(row["start_freq"]),
                    end_freq=float(row["end_freq"]),
                    modulation=row["modulation"],
                    protocol=row["protocol"],
                    samples_dir=row["samples_dir"],
                    info=row["info"],
                )
                self.channels[ch.id] = ch

    def _save(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[x.name for x in fields(RFRange)])
            writer.writeheader()
            for ch in sorted(self.channels.values(), key=lambda c: c.id):
                writer.writerow(asdict(ch))

    def _next_id(self):
        return max(self.channels, default=-1) + 1

    def _create_samples_dir(self, channel_id):
        rel = Path("samples") / str(channel_id)
        full = self.path / rel
        full.mkdir(parents=True, exist_ok=True)
        info_csv = full / "captures.csv"
        if not info_csv.exists():
            with open(info_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CAPTURE_INFO_HEADER)
        return str(rel)

    @staticmethod
    def _given(value):
        """True if the caller actually supplied a value (not empty / 'unknown')."""
        return value not in ("", None, UNKNOWN)

    # ------------------------------------------------------------------ API
    def get_channels(self):
        return sorted(self.channels.values(), key=lambda c: c.start_freq)

    def add_update_channel(self, start_freq, end_freq, name="",
                           modulation=UNKNOWN, protocol=UNKNOWN, info=""):
        ch, change_type = self._add_update(start_freq, end_freq, name,
                                           modulation, protocol, info)
        if change_type is not None:
            self._save()
        return ch

    def _add_update(self, start_freq, end_freq, name="",
                    modulation=UNKNOWN, protocol=UNKNOWN, info=""):
        """Return the channel and whether it was new, updated, or unchanged."""
        if start_freq > end_freq:
            start_freq, end_freq = end_freq, start_freq

        contained = []   # existing channels inside the reported range
        containing = []  # existing channels that contain the reported range
        for ch in self.channels.values():
            if ch.start_freq >= start_freq and ch.end_freq <= end_freq:
                contained.append(ch)
            elif ch.start_freq <= start_freq and ch.end_freq >= end_freq:
                containing.append(ch)

        # Reported range is inside an existing channel: never shrink it.
        if containing and not contained:
            target = min(containing, key=lambda c: c.id)
            before = asdict(target)
            self._apply_info(target, name, modulation, protocol, info)
            change_type = "updated" if asdict(target) != before else None
            return target, change_type

        # Reported range covers existing channel(s): widen to the new range.
        if contained:
            contained.sort(key=lambda c: c.id)
            target = contained[0]
            before = asdict(target)
            target.start_freq = start_freq
            target.end_freq = end_freq
            self._apply_info(target, name, modulation, protocol, info)
            for other in contained[1:]:
                self._fill_missing(target, other)
                del self.channels[other.id]
            changed = asdict(target) != before or len(contained) > 1
            return target, "updated" if changed else None

        # Brand-new channel.
        if name == "":
            center = start_freq + (end_freq - start_freq) / 2
            name = f"{center}-{end_freq - start_freq}-{modulation}"
        new_id = self._next_id()
        ch = RFRange(
            id=new_id,
            name=name,
            start_freq=start_freq,
            end_freq=end_freq,
            modulation=modulation or UNKNOWN,
            protocol=protocol or UNKNOWN,
            samples_dir=self._create_samples_dir(new_id),
            info=info,
        )
        self.channels[new_id] = ch
        return ch, "new"

    # ---------------------------------------------------------- scanner input
    @staticmethod
    def beacon_to_range(beacon):
        """Scanner beacon {'frequency', 'bandwidth'} -> (start_freq, end_freq)."""
        center = beacon.get("frequency")
        bandwidth = beacon.get("bandwidth")
        if center is None or bandwidth is None:
            return None
        half = float(bandwidth) / 2
        return float(center) - half, float(center) + half

    def report_beacons(self, beacons):
        """
        Feed the scanner's beacon list straight in.         Saves once at the end, and only if something changed. Returns counts
        keyed by "new" and "updated".
        """
        changes = {"new": 0, "updated": 0}
        for beacon in beacons:
            rng = self.beacon_to_range(beacon)
            if rng is None:
                continue
            _, change_type = self._add_update(*rng)
            if change_type is not None:
                changes[change_type] += 1
        if any(changes.values()):
            self._save()
        return changes

    def _apply_info(self, ch, name, modulation, protocol, info):
        """Overwrite only the fields the caller actually provided."""
        if self._given(name):
            ch.name = name
        if self._given(modulation):
            ch.modulation = modulation
        if self._given(protocol):
            ch.protocol = protocol
        if self._given(info):
            ch.info = info

    def _fill_missing(self, target, other):
        """Carry over info from a merged-away channel if target lacks it."""
        if not self._given(target.modulation):
            target.modulation = other.modulation
        if not self._given(target.protocol):
            target.protocol = other.protocol
        if not self._given(target.info):
            target.info = other.info

    def delete_channel(self, id, remove_samples=False):
        ch = self.channels.pop(id, None)
        if ch is None:
            return False
        if remove_samples:
            shutil.rmtree(self.path / ch.samples_dir, ignore_errors=True)
        self._save()
        return True

    def edit_channel(self, id, start_freq=None, end_freq=None, name=None,
                     modulation=None, protocol=None, info=None):
        """Direct edit by id; None means 'leave unchanged' (unlike add_update_channel)."""
        ch = self.channels.get(id)
        if ch is None:
            raise KeyError(f"No channel with id {id}")
        new_start = ch.start_freq if start_freq is None else start_freq
        new_end = ch.end_freq if end_freq is None else end_freq
        if new_start > new_end:
            raise ValueError("start_freq must be <= end_freq")
        ch.start_freq, ch.end_freq = new_start, new_end
        if name is not None:
            ch.name = name
        if modulation is not None:
            ch.modulation = modulation
        if protocol is not None:
            ch.protocol = protocol
        if info is not None:
            ch.info = info
        self._save()
        return ch

    def refine_channel(self, id, start_freq, end_freq, merge_overlap=0.5):
        """
        Trusted update from the observer: overwrite the channel's range with the
        measured one, even if it is narrower than before.

        If the new range overlaps another channel by at least `merge_overlap`
        of the narrower channel's width, both are treated as the same signal
        and merged. The lower id survives and takes the refined range.
        Sample folders of merged-away channels stay on disk.

        Returns the surviving channel.
        """
        ch = self.channels.get(id)
        if ch is None:
            raise KeyError(f"No channel with id {id}")
        if start_freq > end_freq:
            start_freq, end_freq = end_freq, start_freq
        if end_freq - start_freq <= 0:
            raise ValueError("refined range has zero width")

        ch.start_freq, ch.end_freq = start_freq, end_freq

        for other in list(self.channels.values()):
            if other.id == ch.id or other.id not in self.channels:
                continue
            if self._overlap_fraction(ch, other) >= merge_overlap:
                survivor, loser = (ch, other) if ch.id < other.id else (other, ch)
                survivor.start_freq, survivor.end_freq = start_freq, end_freq
                self._fill_missing(survivor, loser)
                del self.channels[loser.id]
                ch = survivor

        self._save()
        return ch

    def refine_channel_by_center(self, id, center_freq, bandwidth, merge_overlap=0.5):
        """Same as refine_channel, but takes what the observer measures."""
        half = float(bandwidth) / 2
        return self.refine_channel(id, center_freq - half, center_freq + half,
                                   merge_overlap)

    @staticmethod
    def _overlap_fraction(a, b):
        """Overlap width divided by the width of the narrower channel (0 to 1)."""
        overlap = min(a.end_freq, b.end_freq) - max(a.start_freq, b.start_freq)
        if overlap <= 0:
            return 0.0
        narrower = min(a.end_freq - a.start_freq, b.end_freq - b.start_freq)
        return overlap / narrower if narrower > 0 else 0.0



    # ------------------------------------------------------------- captures
    def _captures_csv(self, channel_id):
        ch = self.channels.get(channel_id)
        if ch is None:
            raise KeyError(f"No channel with id {channel_id}")
        folder = self.path / ch.samples_dir
        folder.mkdir(parents=True, exist_ok=True)
        csv_path = folder / "captures.csv"
        if not csv_path.exists():
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CAPTURE_INFO_HEADER)
        return folder, csv_path

    def append_capture(self, channel_id, samples, capture_bandwidth,
                       gain=None, timestamp=None):
        """
        Save one continuous burst as a new .wav file for the channel.

        samples:            complex IQ array (saved as 2-channel float32: I, Q)
                            or a real array (saved as mono float32)
        capture_bandwidth:  sample rate / bandwidth of THIS capture in Hz
                            (not the channel bandwidth)
        gain:               receiver gain used for the capture
        timestamp:          datetime of capture start (defaults to now, UTC)
        Returns the path of the new .wav file.
        """
        samples = np.asarray(samples)
        if samples.size == 0:
            raise ValueError("samples is empty")

        folder, csv_path = self._captures_csv(channel_id)

        # next index, based on the rows already in the csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            index = sum(1 for _ in csv.DictReader(f))
        wav_name = f"capture_{index}.wav"
        while (folder / wav_name).exists():  # never overwrite
            index += 1
            wav_name = f"capture_{index}.wav"

        if np.iscomplexobj(samples):
            data = np.column_stack((samples.real, samples.imag)).astype(np.float32)
        else:
            data = samples.astype(np.float32)

        # wav header sample rate must be an int
        wavfile.write(folder / wav_name, int(round(capture_bandwidth)), data)

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                wav_name,
                "" if gain is None else gain,
                timestamp.isoformat(),
                len(samples),          # number of samples in this burst
                capture_bandwidth,
            ])
        return folder / wav_name

    def get_captures(self, channel_id):
        """List of dicts, one per capture, with 'path' added."""
        folder, csv_path = self._captures_csv(channel_id)
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["path"] = folder / row["capture_file"]
        return rows