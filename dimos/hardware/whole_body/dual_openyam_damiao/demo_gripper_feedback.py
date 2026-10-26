# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Query each OpenYAM gripper's CAN feedback without enabling its motor."""

import argparse

import can_motor_control
from can_motor_control import damiao


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-can-port", required=True)
    parser.add_argument("--right-can-port", required=True)
    args = parser.parse_args()

    buses = {
        side: can_motor_control.GsUsbBus(
            vendor_id=0x1D50,
            product_id=0x606F,
            serial_number=serial,
        )
        for side, serial in (
            ("left", args.left_can_port),
            ("right", args.right_can_port),
        )
    }
    builder = can_motor_control.Robot.builder()
    for side, bus in buses.items():
        builder.add_bus(side, bus, damiao.DamiaoCodec())
        builder.add_arm(
            f"{side}_arm",
            bus=side,
            motors=[
                can_motor_control.MotorSpec(
                    f"{side}_joint{index}",
                    damiao.MotorType.DM4340 if index <= 3 else damiao.MotorType.DM4310,
                    index,
                    index + 0x10,
                )
                for index in range(1, 7)
            ],
        )
        builder.add_gripper(
            f"{side}_gripper",
            bus=side,
            motor=can_motor_control.MotorSpec(
                f"{side}_gripper", damiao.MotorType.DM4310, 0x08, 0x18
            ),
        )
    robot = builder.build()
    robot.connect()
    try:
        arms: dict[str, can_motor_control.Arm] = {}
        grippers: dict[str, can_motor_control.Gripper] = {}
        for side in buses:
            arm = robot[f"{side}_arm"]
            gripper = robot[f"{side}_gripper"]
            if not isinstance(arm, can_motor_control.Arm):
                raise TypeError(f"{side}_arm is not an Arm")
            if not isinstance(gripper, can_motor_control.Gripper):
                raise TypeError(f"{side}_gripper is not a Gripper")
            arms[side] = arm
            grippers[side] = gripper
        for _ in range(25):
            for side in buses:
                arms[side].refresh()
                grippers[side].refresh()
            robot.tick(10_000)
        for side, bus in buses.items():
            arm = arms[side]
            motor = grippers[side].motor
            print(
                f"{side}: serial={getattr(args, f'{side}_can_port')} "
                f"rx={bus.rx_received} tx={bus.tx_completed} dropped={bus.rx_dropped} "
                f"arm_positions={arm.positions().round(3).tolist()} "
                f"arm_temperatures={[arm[f'{side}_joint{i}'].temperature_mos for i in range(1, 7)]} "
                f"gripper_recv_id=0x{motor.recv_id:x} position={motor.position:.4f} "
                f"temperature={motor.temperature_mos}/{motor.temperature_rotor} "
                f"fault={motor.fault}"
            )
    finally:
        robot.disable()


if __name__ == "__main__":
    main()
