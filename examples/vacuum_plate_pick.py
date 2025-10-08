import math
import time
from pathlib import Path

import numpy as np

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = PROJECT_ROOT / "stretch_mujoco" / "models" / "plate_pick_scene.xml"

PLATE_CENTER = np.array([0.0, -0.7, 0.52])
PLATE_SURFACE_Z = PLATE_CENTER[2] + 0.01
SAFE_HEIGHT = PLATE_SURFACE_Z + 0.20
CONTACT_HEIGHT = PLATE_SURFACE_Z + 0.01
LIFTED_HEIGHT = PLATE_SURFACE_Z + 0.30

VACUUM_TIP_OFFSET = np.array([0.0, 0.0, -0.09])
BASE_APPROACH_DISTANCE = 0.35

WRIST_PITCH_DOWN = 0
GRIPPER_CLOSED = -0.25


def get_tip_position(sim: StretchMujocoSimulator) -> np.ndarray:
    grasp_pose = sim.get_link_pose("link_grasp_center")
    tip_local = np.append(VACUUM_TIP_OFFSET, 1.0)
    tip_world = grasp_pose @ tip_local
    return tip_world[:3]


def estimate_axis_sign(
    sim: StretchMujocoSimulator, actuator: Actuators, measure_fn, step: float
) -> float:
    baseline = measure_fn()

    sim.move_by(actuator, step)
    sim.wait_while_is_moving(actuator)
    moved = measure_fn()

    sim.move_by(actuator, -step)
    sim.wait_while_is_moving(actuator)

    if math.isclose(moved, baseline, abs_tol=1e-4):
        return 1.0
    return 1.0 if moved > baseline else -1.0


def raise_tip_to_height(
    sim: StretchMujocoSimulator,
    target_z: float,
    lift_sign: float,
    gain_lift: float = 2.0,
    tolerance: float = 0.005,
    max_iterations: int = 30,
) -> None:
    for _ in range(max_iterations):
        current_z = get_tip_position(sim)[2]
        error = target_z - current_z
        if abs(error) < tolerance:
            break
        delta = np.clip(error * gain_lift * lift_sign, -0.03, 0.03)
        if abs(delta) < 1e-4:
            break
        sim.move_by(Actuators.lift, float(delta))
        sim.wait_while_is_moving(Actuators.lift)


def align_tip(
    sim: StretchMujocoSimulator,
    target: np.ndarray,
    gain_arm: float,
    gain_yaw: float,
    gain_lift: float,
    arm_sign: float,
    yaw_sign: float,
    lift_sign: float,
    tolerance: float = 0.01,
    max_iterations: int = 30,
    allow_vertical: bool = True,
) -> None:
    for _ in range(max_iterations):
        current = get_tip_position(sim)
        error = target - current

        if np.linalg.norm(error) < tolerance:
            break

        if allow_vertical and abs(error[2]) > tolerance:
            delta = np.clip(error[2] * gain_lift * lift_sign, -0.02, 0.02)
            if abs(delta) > 1e-4:
                sim.move_by(Actuators.lift, float(delta))
                sim.wait_while_is_moving(Actuators.lift)
            continue

        if abs(error[0]) > tolerance:
            delta = np.clip(error[0] * gain_arm * arm_sign, -0.04, 0.04)
            if abs(delta) > 1e-4:
                sim.move_by(Actuators.arm, float(delta))
                sim.wait_while_is_moving(Actuators.arm)

        if abs(error[1]) > tolerance:
            delta = np.clip(error[1] * gain_yaw * yaw_sign, -0.2, 0.2)
            if abs(delta) > 1e-4:
                sim.move_by(Actuators.wrist_yaw, float(delta))
                sim.wait_until_at_setpoint(Actuators.wrist_yaw)


def main() -> None:
    sim = StretchMujocoSimulator(scene_xml_path=str(SCENE_PATH))
    sim.start(headless=False)

    try:
        sim.home()

        sim.move_to(Actuators.gripper, GRIPPER_CLOSED)
        sim.wait_until_at_setpoint(Actuators.gripper)
        
        sim.move_to(Actuators.wrist_roll, 0.0)
        sim.move_to(Actuators.wrist_yaw, 0.0)
        sim.move_to(Actuators.wrist_pitch, WRIST_PITCH_DOWN)
        sim.move_to(Actuators.arm, 0.0)
        sim.move_to(Actuators.lift, 0.55)

        sim.wait_until_at_setpoint(Actuators.wrist_roll)
        sim.wait_until_at_setpoint(Actuators.wrist_yaw)
        sim.wait_until_at_setpoint(Actuators.wrist_pitch)
        sim.wait_until_at_setpoint(Actuators.arm)
        sim.wait_until_at_setpoint(Actuators.lift)

        sim.move_by(Actuators.base_translate, BASE_APPROACH_DISTANCE)
        sim.wait_while_is_moving(Actuators.base_translate)


        # # arm_sign = estimate_axis_sign(
        # #     sim, Actuators.arm, lambda: get_tip_position(sim)[0], 0.02
        # # )
        # # yaw_sign = estimate_axis_sign(
        # #     sim, Actuators.wrist_yaw, lambda: get_tip_position(sim)[1], 0.05
        # # )
        lift_sign = estimate_axis_sign(
            sim, Actuators.lift, lambda: get_tip_position(sim)[2], 0.02
        )

        # Base alignment
        desired_base_x = 0.02  # align with table center
        while True:
            base_x, base_y, _ = sim.get_base_pose()
            print(base_x)
            print(" ")
            error = desired_base_x - base_x
            if abs(error) < 0.01:
                break
            sim.move_by(Actuators.base_translate, error * 0.8)
            sim.wait_while_is_moving(Actuators.base_translate)

        # Hover over plate
        sim.move_to(Actuators.arm, 0.3)
        sim.wait_until_at_setpoint(Actuators.arm)

        # Move down to make contact
        raise_tip_to_height(sim, CONTACT_HEIGHT, lift_sign)
        
        time.sleep(2.0)


        sim.set_equality_active("plate_to_pen", True)

        # Lift plate
        raise_tip_to_height(sim, LIFTED_HEIGHT, lift_sign)

        time.sleep(1.0)

        sim.move_by(Actuators.base_translate, 0.3)
        sim.wait_while_is_moving(Actuators.base_translate)

        raise_tip_to_height(sim, CONTACT_HEIGHT, lift_sign)
        sim.set_equality_active("plate_to_pen", False)

        raise_tip_to_height(sim, SAFE_HEIGHT, lift_sign)


    except KeyboardInterrupt:
        pass
    finally:
        time.sleep(0.5)
        sim.stop()


if __name__ == "__main__":
    main()
