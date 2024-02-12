import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      print("Ignoring empty camera frame.")
      continue

    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    counter = 0

    if results.multi_hand_landmarks:

      for hand_landmarks in results.multi_hand_landmarks:
        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
        handLabel = results.multi_handedness[handIndex].classification[0].label

        Hand_Landmarkers = []

        for landmarks in hand_landmarks.landmark:
          Hand_Landmarkers.append([landmarks.x, landmarks.y])

      
        if handLabel == "Left" and Hand_Landmarkers[4][0] > Hand_Landmarkers[3][0]:
          counter = counter+1
        elif handLabel == "Right" and Hand_Landmarkers[4][0] < Hand_Landmarkers[3][0]:
          counter = counter+1

 
        if Hand_Landmarkers[8][1] < Hand_Landmarkers[6][1]:      
          counter = counter+1
        if Hand_Landmarkers[12][1] < Hand_Landmarkers[10][1]:     
          counter = counter+1
        if Hand_Landmarkers[16][1] < Hand_Landmarkers[14][1]:     
          counter = counter+1
        if Hand_Landmarkers[20][1] < Hand_Landmarkers[18][1]:     
          counter = counter+1

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    cv2.putText(frame, str(counter), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    cv2.imshow('Main Window', frame)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()