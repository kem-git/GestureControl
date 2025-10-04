import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import comtypes
import screen_brightness_control as sbc
import time
import pyautogui

# --------- Setup MediaPipe ---------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --------- Setup Audio ---------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
volume_ctrl = comtypes.cast(interface, comtypes.POINTER(IAudioEndpointVolume))
vol_min, vol_max = volume_ctrl.GetVolumeRange()[:2]

# --------- Webcam ---------
cap = cv2.VideoCapture(0)

# --------- Lock & smoothing variables ---------
lock_time = 2.0  # seconds to hold to lock/unlock
last_thumb_index_dist = 0
lock_start = None
locked = False
current_vol = None
current_brightness = None

# --------- Hand movement tracking for desktop switching ---------
last_x_pos = None
desktop_trigger_threshold = 0.15  # relative x movement
desktop_cooldown = 1.0  # seconds between triggers
last_desktop_time = 0

# --------- Helper: distance ---------
def distance(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

# --------- Main Loop ---------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    tick_on_screen = False

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        thumb = hand.landmark[4]
        index = hand.landmark[8]

        h, w, _ = frame.shape
        cv2.circle(frame, (int(thumb.x*w), int(thumb.y*h)), 6, (255,255,255), -1)
        cv2.circle(frame, (int(index.x*w), int(index.y*h)), 6, (255,255,255), -1)
        cv2.line(frame, (int(thumb.x*w), int(thumb.y*h)), (int(index.x*w), int(index.y*h)), (200,200,200), 2)

        # --- Finger distances ---
        x_dist = abs(thumb.x - index.x)
        y_dist = abs(thumb.y - index.y)
        x_dist = max(x_dist, 0.03)
        y_dist = max(y_dist, 0.03)

        # --- Check for pinch (unlock gesture) ---
        pinch_threshold = 0.04
        if locked and x_dist < pinch_threshold:
            if lock_start is None:
                lock_start = time.time()
            elif (time.time() - lock_start) > lock_time:
                locked = False
                lock_start = None
        # --- Lock check ---
        elif not locked:
            dist_change = abs(last_thumb_index_dist - x_dist)
            if dist_change < 0.005:  # holding steady
                if lock_start is None:
                    lock_start = time.time()
                elif (time.time() - lock_start) > lock_time:
                    locked = True
                    lock_start = None
            else:
                lock_start = None

        last_thumb_index_dist = x_dist

        # --- Update volume and brightness if not locked ---
        if not locked:
            vol = np.interp(y_dist, [0.03, 0.4], [vol_min, vol_max])
            volume_ctrl.SetMasterVolumeLevel(vol, None)
            current_vol = vol

            brightness = np.interp(x_dist, [0.03, 0.6], [0, 100])
            try:
                sbc.set_brightness(int(brightness))
                current_brightness = brightness
            except:
                pass
        else:
            tick_on_screen = True
            if current_vol is not None:
                volume_ctrl.SetMasterVolumeLevel(current_vol, None)
            if current_brightness is not None:
                try:
                    sbc.set_brightness(int(current_brightness))
                except:
                    pass

        # --- Desktop switching ---
        if last_x_pos is not None:
            dx = thumb.x - last_x_pos
            now = time.time()
            if abs(dx) > desktop_trigger_threshold and (now - last_desktop_time) > desktop_cooldown:
                if dx > 0:
                    pyautogui.hotkey('ctrl', 'win', 'right')
                else:
                    pyautogui.hotkey('ctrl', 'win', 'left')
                last_desktop_time = now
        last_x_pos = thumb.x

    # --- Draw lock tick ---
    if tick_on_screen:
        cv2.putText(frame, "X", (frame.shape[1]-50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3, cv2.LINE_AA)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
