# 文件名: custom_actions.py
import habitat
from habitat.core.registry import registry
from habitat.core.embodied_task import SimulatorTaskAction

@registry.register_task_action
class NoOpAction(SimulatorTaskAction):
    def step(self, *args, **kwargs):
        return self._sim.get_sensor_observations()