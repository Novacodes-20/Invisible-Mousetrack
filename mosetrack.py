import cv2
import mediapipe as mp
import pyautogui
from pynput.mouse import Controller

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=0
)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False
mouse = Controller()

prev_x, prev_y = 0, 0
smoothening = 5
click_cooldown = 0
prev_thumb_relative_y = None
scroll_direction_count = 0  # positive = down, negative = up
DIRECTION_THRESHOLD = 3  # needs 3 consistent frames to scroll

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if click_cooldown > 0:
        click_cooldown -= 1

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            lm = hand.landmark

            index_tip_y = int(lm[8].y * h)
            index_mid_y = int(lm[6].y * h)

            thumb_tip_y = int(lm[4].y * h)
            wrist_y = int(lm[0].y * h)
            thumb_relative_y = thumb_tip_y - wrist_y

            # --- CURSOR ---
            screen_x = int(lm[8].x * screen_w)
            screen_y = int(lm[8].y * screen_h)
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y, duration=0)
            prev_x, prev_y = curr_x, curr_y

            # --- CLICK ---
            if index_tip_y > index_mid_y + 10:
                if click_cooldown == 0:
                    pyautogui.click()
                    click_cooldown = 20
                    cv2.putText(frame, "CLICK", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # --- SCROLL (debounced) ---
            if prev_thumb_relative_y is not None:
                diff = thumb_relative_y - prev_thumb_relative_y

                if abs(diff) > 3:
                    # build up direction count
                    if diff > 0:
                        scroll_direction_count = min(scroll_direction_count + 1, 10)
                    else:
                        scroll_direction_count = max(scroll_direction_count - 1, -10)

                    # only scroll after 3 consistent frames
                    if abs(scroll_direction_count) >= DIRECTION_THRESHOLD:
                        scroll_amount = max(1, int(abs(diff) / 3))
                        mouse.scroll(0, -scroll_amount if scroll_direction_count > 0 else scroll_amount)
                        cv2.putText(frame, "SCROLL DOWN" if scroll_direction_count > 0 else "SCROLL UP",
                                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                else:
                    # gradually reset when thumb is still
                    if scroll_direction_count > 0:
                        scroll_direction_count -= 1
                    elif scroll_direction_count < 0:
                        scroll_direction_count += 1

            prev_thumb_relative_y = thumb_relative_y

    cv2.imshow("Invisible Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()