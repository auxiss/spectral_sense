from dataclasses import dataclass


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





class ranges_manager():
    def __init__(self, file_name):

        self.ranges = []

        #load form file

        self.ranges.append(rf_range(id=0, start_freq=87.5e6, end_freq=108e6, info="WB-FM brodcast", valid_golbe_region="itu:1", hex_color="#53CEFF"))
        self.ranges.append(rf_range(id=1, start_freq=108e6, end_freq=137e6, info="aviation", valid_golbe_region="itu:1", hex_color="#538FFF"))
        self.ranges.append(rf_range(id=2, start_freq=144e6, end_freq=146e6, info="2m_band.", valid_golbe_region="itu:1", hex_color="#FF5370"))
        self.ranges.append(rf_range(id=3, start_freq=694e6, end_freq=862e6, info="mobile broadband", valid_golbe_region="itu:1", hex_color="#FCFF53"))
        self.ranges.append(rf_range(id=4, start_freq=335e6, end_freq=470e6, info="defense/emergency/pub_safety, TETRA/PMR", valid_golbe_region="itu:1", hex_color="#59FF53"))
        self.ranges.append(rf_range(id=5, start_freq=156e6, end_freq=170e6, info="maritimeVHF/distress/safety", valid_golbe_region="itu:1", hex_color="#3854A1"))
        self.ranges.append(rf_range(id=6, start_freq=2.412e9, end_freq=2.484e9, info="WiFi", valid_golbe_region="itu:1", hex_color="#FCFF53"))

        print(type(self.ranges))


    def get_ranges(self):
        return self.ranges

    def add(self, range: rf_range):
        pass

    def delete(self, id):
        pass

    