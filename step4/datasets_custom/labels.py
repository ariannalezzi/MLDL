from abc import ABCMeta
from dataclasses import dataclass
from typing import Tuple


class BaseGTALabels(metaclass=ABCMeta):
    pass


@dataclass
class GTA5Label:
    name: str
    ID: int
    color: Tuple[int, int, int]


class GTA5Labels_TaskCV2017(BaseGTALabels):
    road = GTA5Label(name="road", ID=0, color=(128, 64, 128))
    sidewalk = GTA5Label(name="sidewalk", ID=1, color=(244, 35, 232))
    building = GTA5Label(name="building", ID=2, color=(70, 70, 70))
    wall = GTA5Label(name="wall", ID=3, color=(102, 102, 156))
    fence = GTA5Label(name="fence", ID=4, color=(190, 153, 153))
    pole = GTA5Label(name="pole", ID=5, color=(153, 153, 153))
    light = GTA5Label(name="light", ID=6, color=(250, 170, 30))
    sign = GTA5Label(name="sign", ID=7, color=(220, 220, 0))
    vegetation = GTA5Label(name="vegetation", ID=8, color=(107, 142, 35))
    terrain = GTA5Label(name="terrain", ID=9, color=(152, 251, 152))
    sky = GTA5Label(name="sky", ID=10, color=(70, 130, 180))
    person = GTA5Label(name="person", ID=11, color=(220, 20, 60))
    rider = GTA5Label(name="rider", ID=12, color=(255, 0, 0))
    car = GTA5Label(name="car", ID=13, color=(0, 0, 142))
    truck = GTA5Label(name="truck", ID=14, color=(0, 0, 70))
    bus = GTA5Label(name="bus", ID=15, color=(0, 60, 100))
    train = GTA5Label(name="train", ID=16, color=(0, 80, 100))
    motocycle = GTA5Label(name="motocycle", ID=17, color=(0, 0, 230))
    bicycle = GTA5Label(name="bicycle", ID=18, color=(119, 11, 32))


    list_ = [
        road,
        sidewalk,
        building,
        wall,
        fence,
        pole,
        light,
        sign,
        vegetation,
        terrain,
        sky,
        person,
        rider,
        car,
        truck,
        bus,
        train,
        motocycle,
        bicycle,
    ]

    @property
    def support_id_list(self):
        ret = [label.ID for label in self.list_]
        return ret
