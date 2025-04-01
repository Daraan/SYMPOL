"""
Object Types
Each object type in the MiniGrid environment is assigned a specific integer value. Here are the common object types:

[0,1,2,8,9]



unseen	    0 |  0
empty	    1 |  1
wall	    2 |  2
floor	    3 |  3
door	    4 |  4
key	        5 |  5
ball	    6 |  6
box	        7 |  7
goal	    8 |  8
lava	    9 |  9
agent      10 | 10
-----------------------


Colors
Colors are also encoded as integer values. Here are the typical colors used in the MiniGrid environment:

Color	Value
Red  	0 | 11
Green	1 | 12
Blue	2 | 13
Purple	3 | 14
Yellow	4 | 15
Grey	5 | 16
-----------------------
States
The state value provides additional context about the object. For some objects, this might indicate whether they are open or closed, picked up, etc. Here are some common state values:

State	Value
Open	0 | 17
Closed	1 | 18
Locked	2 | 19


| **Num** | **Name**   |
|:-------:|:----------:|
| **0**   | **left**   |
| **1**   | **right**  |
| **2**   | **forward**|
| **3**   | **pickup** |
| **4**   | **drop**   |
| **5**   | **toggle** |
| **6**   | **done**   |

"""

from gymnasium.envs.registration import register
from minigrid.core.world_object import Goal
from minigrid.envs.distshift import DistShiftEnv


class RandomGoalDistShiftEnv2(DistShiftEnv):
    def __init__(self, strip2_row=5, **kwargs):
        super().__init__(strip2_row=strip2_row, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Remove the old goal
        self.grid.set(width - 2, 1, None)

        # Randomly place the goal somewhere in the grid
        while True:
            goal_x = self._rand_int(0, width)
            goal_y = self._rand_int(0, height)
            if self.grid.get(goal_x, goal_y) is None:
                self.grid.set(goal_x, goal_y, Goal())
                break

        self.mission = "Get to the green goal square"


class RandomGoalDistShiftEnv(DistShiftEnv):
    def __init__(self, strip2_row=2, **kwargs):
        super().__init__(strip2_row=strip2_row, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Remove the old goal
        self.grid.set(width - 2, 1, None)

        # Randomly place the goal somewhere in the grid
        while True:
            goal_x = self._rand_int(0, width)
            goal_y = self._rand_int(0, height)
            if self.grid.get(goal_x, goal_y) is None:
                self.grid.set(goal_x, goal_y, Goal())
                break

        self.mission = "Get to the green goal square"


# Register the environment with Gymnasium
register(
    id="MiniGrid-DistShift3-v0",
    entry_point="utils:RandomGoalDistShiftEnv",
)

register(
    id="MiniGrid-DistShift4-v0",
    entry_point="utils:RandomGoalDistShiftEnv2",
)
