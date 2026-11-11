import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class rf_range:
    id: int
    start_freq: float
    end_freq: float
    info: str
    valid_golbe_region: str
    hex_color: str


#itu: internasion telecomunicasion unit
# * itu regions
# region 1 erope, africa
# region 2 amicas
# region 3 asion, osiania





class ranges_manager:
    _CSV_FIELDS = (
        "id",
        "start_freq",
        "end_freq",
        "info",
        "valid_golbe_region",
        "hex_color",
    )

    def __init__(self, file_name="known_ranges.csv"):
        self.file_name = Path(file_name)
        self.ranges = []



        if self.file_name.exists():
            self.ranges = self._load()
        else:
            self._save(self.ranges)

    def _load(self):
        with self.file_name.open("r", newline="", encoding="utf-8") as ranges_file:
            reader = csv.DictReader(ranges_file)
            if reader.fieldnames != list(self._CSV_FIELDS):
                raise ValueError(
                    f"{self.file_name} must have CSV columns "
                    f"{', '.join(self._CSV_FIELDS)}"
                )

            return [
                rf_range(
                    id=int(row["id"]),
                    start_freq=float(row["start_freq"]),
                    end_freq=float(row["end_freq"]),
                    info=row["info"],
                    valid_golbe_region=row["valid_golbe_region"],
                    hex_color=row["hex_color"],
                )
                for row in reader
            ]

    def _save(self, ranges):
        with self.file_name.open("w", newline="", encoding="utf-8") as ranges_file:
            writer = csv.DictWriter(ranges_file, fieldnames=self._CSV_FIELDS)
            writer.writeheader()
            writer.writerows(
                {
                    "id": item.id,
                    "start_freq": item.start_freq,
                    "end_freq": item.end_freq,
                    "info": item.info,
                    "valid_golbe_region": item.valid_golbe_region,
                    "hex_color": item.hex_color,
                }
                for item in ranges
            )

    def get_ranges(self):
        return self.ranges

    def add(self, range: rf_range):
        if any(item.id == range.id for item in self.ranges):
            raise ValueError(f"A frequency range with id {range.id} already exists")

        updated_ranges = [*self.ranges, range]
        self._save(updated_ranges)
        self.ranges = updated_ranges

    def delete(self, id):
        updated_ranges = [item for item in self.ranges if item.id != id]
        if len(updated_ranges) == len(self.ranges):
            raise KeyError(f"No frequency range with id {id} exists")

        self._save(updated_ranges)
        self.ranges = updated_ranges

    def replace(self, range: rf_range):
        for index, item in enumerate(self.ranges):
            if item.id == range.id:
                updated_ranges = self.ranges.copy()
                updated_ranges[index] = range
                self._save(updated_ranges)
                self.ranges = updated_ranges
                return

        raise KeyError(f"No frequency range with id {range.id} exists")

    