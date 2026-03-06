import json
import time
import queue
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
from ultralytics import YOLO
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================
# ROBOT CONTROL
# =========================

@dataclass
class CommandResult:
    ok: bool
    text: str = ""
    error: str = ""
    timed_out: bool = False


class RoArmM2S:
    def __init__(self, ip: str = "192.168.4.1"):
        self.base_url = f"http://{ip}/js"
        self._session = requests.Session()

        retry = Retry(
            total=2,
            connect=2,
            read=0,
            backoff_factor=0.15,
            status_forcelist=[429, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )

        self._session.mount(
            "http://",
            HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1),
        )

        self._q = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

        self.min_gap_s = 0.05
        self._last_send_ts = 0.0

    def _http_send(self, cmd, connect_timeout, read_timeout):
        now = time.time()
        gap = self.min_gap_s - (now - self._last_send_ts)
        if gap > 0:
            time.sleep(gap)

        try:
            r = self._session.get(
                self.base_url,
                params={"json": json.dumps(cmd)},
                timeout=(connect_timeout, read_timeout),
            )
            self._last_send_ts = time.time()
            return CommandResult(ok=True, text=r.text)

        except requests.exceptions.RequestException as e:
            return CommandResult(ok=False, error=str(e))

    def _run(self):
        while True:
            cmd, ct, rt, retq = self._q.get()
            res = self._http_send(cmd, ct, rt)
            retq.put(res)

    def send(self, cmd, connect_timeout=0.6, read_timeout=0.6):
        retq = queue.Queue(maxsize=1)
        self._q.put((cmd, connect_timeout, read_timeout, retq))
        return retq.get()

    def feedback(self):
        res = self.send({"T": 105})
        try:
            return json.loads(res.text.strip())
        except:
            return None

    def move_xyzt(self, x, y, z, t):
        return self.send({"T": 1041, "x": x, "y": y, "z": z, "t": t})

    def gripper_open(self):
        return self.send({"T": 106, "cmd": 1.08})

    def gripper_close(self):
        return self.send({"T": 106, "cmd": 3.14})


# =========================
# CONFIGURATION
# =========================

ROARM_IP = "192.168.4.1"
IP_URL = "http://192.168.0.3:8080/video" #change IP address here

# NEW MODEL PATH
MODEL_PATH = r"C:\Users\Jashwanth\Downloads\WasteYOLO_FRESH\training_run\weights\best.pt"

PLASTIC_CLASS = "plastic"

T_TOOL = 3.14

Z_HOVER = 250
Z_PICK = 160
Z_PLACE = 165

PICK_XY = (220, 40)
RIGHT_BIN_XY = (0, -240)

MOVE_SETTLE = 0.40
DOWN_SETTLE = 0.40
GRIP_SETTLE = 0.40


def go(arm, x, y, z, settle):
    arm.move_xyzt(x, y, z, T_TOOL)
    time.sleep(settle)


def pick_and_place(arm):

    arm.gripper_open()
    time.sleep(GRIP_SETTLE)

    go(arm, PICK_XY[0], PICK_XY[1], Z_HOVER, MOVE_SETTLE)
    go(arm, PICK_XY[0], PICK_XY[1], Z_PICK, DOWN_SETTLE)

    arm.gripper_close()
    time.sleep(GRIP_SETTLE)

    go(arm, PICK_XY[0], PICK_XY[1], Z_HOVER, MOVE_SETTLE)

    go(arm, RIGHT_BIN_XY[0], RIGHT_BIN_XY[1], Z_HOVER, MOVE_SETTLE)
    go(arm, RIGHT_BIN_XY[0], RIGHT_BIN_XY[1], Z_PLACE, DOWN_SETTLE)

    arm.gripper_open()
    time.sleep(GRIP_SETTLE)

    go(arm, RIGHT_BIN_XY[0], RIGHT_BIN_XY[1], Z_HOVER, MOVE_SETTLE)


# =========================
# MAIN LOOP
# =========================

def run():

    arm = RoArmM2S(ROARM_IP)
    print("Robot feedback:", arm.feedback())

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(IP_URL)

    CONF = 0.55
    COOLDOWN = 1.2
    last_action = 0

    print("Model classes:", model.names)
    print("Robot will pick ONLY plastic")

    while True:

        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (1080, 720))

        results = model(frame, conf=CONF)

        best_label = None
        best_conf = 0

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                label = model.names[cls].lower()

                if conf > best_conf:
                    best_conf = conf
                    best_label = label

        if best_label == PLASTIC_CLASS and time.time() - last_action > COOLDOWN:

            print(f"Plastic detected (conf {best_conf:.2f})")

            pick_and_place(arm)

            last_action = time.time()

        cv2.imshow("Waste Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()