import os, cv2, mediapipe as mp

IMG_SIZE = (200, 200)
MAX_IMAGES = 300

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def preprocess_face(face_img):
    return cv2.equalizeHist(cv2.resize(face_img, IMG_SIZE))

def capture_images(name, cam_index=0):
    folder = os.path.join('dataset', name)
    ensure_dir(folder)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened(): 
        raise RuntimeError("Não foi possível abrir a câmera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1,
        refine_landmarks=True, 
        min_detection_confidence=0.5
    )

    total_saved = 0
    print("Olhe para a câmera. Captura iniciada...")

    while total_saved < MAX_IMAGES:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            overlay = frame.copy()  # cópia para desenhar linhas
            for lm in results.multi_face_landmarks:
                h, w, _ = frame.shape
                points = [(int(p.x*w), int(p.y*h)) for p in lm.landmark]
                x_min, x_max = min(p[0] for p in points), max(p[0] for p in points)
                y_min, y_max = min(p[1] for p in points), max(p[1] for p in points)
                [cv2.line(overlay, points[i], points[i+1], (255,255,255), 1) for i in range(len(points)-1)]
                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0,0,0), 2)
                
                # salvar imagem processada
                face_img = preprocess_face(cv2.cvtColor(frame[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY))
                cv2.imwrite(os.path.join(folder, f"{name}_{total_saved:03d}.jpg"), face_img)
                total_saved += 1

                # se atingir o limite, parar imediatamente
                if total_saved >= MAX_IMAGES:
                    break

            # overlay transparente
            frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

        cv2.imshow("Face Mesh Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captura concluída. {total_saved} fotos salvas em: {folder}")


if __name__ == "__main__":
    person_name = input("Nome da pessoa: ").strip()
    while not person_name:
        person_name = input("Por favor insira um nome válido: ").strip()
    capture_images(person_name)
