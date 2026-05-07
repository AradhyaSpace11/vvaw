import argparse
import os
import sys
import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import ROBOT_URDF

RENDER_W = 640
RENDER_H = 640


def get_camera_view(renderer):
    view = p.computeViewMatrixFromYawPitchRoll([0.2, 0, 0.1], 1.2, 45, -40, 0, 2)
    proj = p.computeProjectionMatrixFOV(50, 1.0, 0.1, 4.0)
    _, _, rgb, _, _ = p.getCameraImage(
        RENDER_W,
        RENDER_H,
        view,
        proj,
        shadow=1,
        renderer=renderer,
    )
    frame = np.reshape(rgb, (RENDER_H, RENDER_W, 4))[:, :, :3].astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def build_args():
    parser = argparse.ArgumentParser(description="Run the VVAW PyBullet smoke simulation.")
    parser.add_argument("--direct", action="store_true", help="Run without opening the PyBullet GUI.")
    parser.add_argument("--steps", type=int, default=0, help="Exit after N simulation steps. 0 runs until quit.")
    parser.add_argument("--no-camera-window", action="store_true", help="Do not open the OpenCV camera window.")
    parser.add_argument(
        "--renderer",
        choices=("auto", "opengl", "tiny"),
        default="auto",
        help="Camera renderer. auto uses TinyRenderer in --direct mode and OpenGL in GUI mode.",
    )
    return parser.parse_args()


def main():
    args = build_args()

    if not ROBOT_URDF.exists():
        print(f"Robot URDF not found: {ROBOT_URDF}")
        return

    connect_mode = p.DIRECT if args.direct else p.GUI
    physics_client = p.connect(connect_mode)
    if physics_client < 0:
        print("Failed to connect to PyBullet.")
        return

    if args.renderer == "tiny" or (args.renderer == "auto" and args.direct):
        renderer = p.ER_TINY_RENDERER
    else:
        renderer = p.ER_BULLET_HARDWARE_OPENGL

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    if not args.direct:
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=45,
            cameraPitch=-35,
            cameraTargetPosition=[0.2, 0.0, 0.1],
        )

    floor_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[5, 5, 0.01])
    floor_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[5, 5, 0.01],
        rgbaColor=[0.3, 0.3, 0.3, 1],
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=floor_collision,
        baseVisualShapeIndex=floor_visual,
        basePosition=[0, 0, -0.01],
    )

    robot_id = p.loadURDF(str(ROBOT_URDF), basePosition=[0, 0, 0], useFixedBase=True)
    cube_id = p.loadURDF("cube.urdf", basePosition=[0.4, 0.0, 0.05], globalScaling=0.05)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])

    for joint_id in range(min(6, p.getNumJoints(robot_id))):
        p.resetJointState(robot_id, joint_id, 0)

    show_camera = not args.no_camera_window and not args.direct
    if show_camera:
        cv2.namedWindow("Robot Camera View", cv2.WINDOW_NORMAL)

    print("PyBullet test simulation running. Press q in camera window or Ctrl+C to exit.")
    try:
        step_count = 0
        while p.isConnected():
            p.stepSimulation()
            frame = get_camera_view(renderer)
            if show_camera:
                cv2.imshow("Robot Camera View", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            step_count += 1
            if args.steps and step_count >= args.steps:
                print(f"Completed {step_count} simulation steps. Camera frame: {frame.shape}")
                break
            time.sleep(1 / 240)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
