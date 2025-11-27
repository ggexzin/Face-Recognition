import cv2, mediapipe as mp
import pickle

IMG_SIZE = (200, 200)
MAX_CONFIDENCE = 70  # limite para reconhecer como válido
MAX_HISTORY = 5      # suavização de frames

MODEL_FILE = 'lbph_model.yml'
LABELS_FILE = 'labels.pkl'

def preprocess_face(face_img):
    return cv2.equalizeHist(cv2.resize(face_img, IMG_SIZE))

def recognise(cam_index=0):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    with open(LABELS_FILE, 'rb') as f:
        labels = pickle.load(f)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Your camera is off or your device doesn't have one!")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    previous_name = None
    previous_frames = 0

    print("Recognition started. Press 'escape' to leave")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            overlay = frame.copy()
            for lm in results.multi_face_landmarks:
                h, w, _ = frame.shape
                points = [(int(p.x*w), int(p.y*h)) for p in lm.landmark]

                x_min, x_max = min(p[0] for p in points), max(p[0] for p in points)
                y_min, y_max = min(p[1] for p in points), max(p[1] for p in points)

                # recortar face e reconhecer
                face_img = preprocess_face(cv2.cvtColor(frame[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY))
                label, confidence = recognizer.predict(face_img)

                if confidence > MAX_CONFIDENCE:
                    name = "Unknown"
                else:
                    name = labels.get(label, "Unknown")

                # suavização
                if previous_name == name and previous_frames < MAX_HISTORY:
                    name = previous_name
                    previous_frames += 1
                else:
                    previous_name = name
                    previous_frames = 0

                # desenhar linhas da mesh no overlay (com transparência)
                [cv2.line(overlay, points[i], points[i+1], (255,255,255), 1) for i in range(len(points)-1)]

                # desenhar o retângulo sólido diretamente no frame, sem transparência
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,0,0), 2)

                cv2.putText(
                    frame,
                    name,                      # texto
                    (x_min, y_min - 10),       # posição acima do retângulo
                    cv2.FONT_HERSHEY_SIMPLEX,  # fonte
                    0.9,                        # escala
                    (255, 255, 255),               # cor (verde)
                    2,                         # espessura
                    cv2.LINE_AA
                )

            # aplicar overlay transparente apenas para as linhas
            frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)


        cv2.imshow("Face Mesh Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27: # de acordo com a documentação do cv2 a tecla 27 corresponde ao "escape"
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognise()
