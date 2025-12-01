import cv2
import numpy as np
import pickle
from pathlib import Path

MODEL_FILE = 'lbph_model.yml'
LABELS_FILE = 'labels.pkl'
IMG_SIZE = (200, 200)  # tamanho uniforme para todos os rostos

def preprocess_face(img):
    # Redimensiona e aplica equalização de histograma.
    img_resized = cv2.resize(img, IMG_SIZE)
    img_equalized = cv2.equalizeHist(img_resized)
    return img_equalized

def train(dataset_dir: str = 'dataset'):
    dataset = Path(dataset_dir)
    faces = []
    labels = []
    label_dict = {}
    current_label = 0

    # Percorrer todas as pastas de pessoas
    for person_dir in sorted(dataset.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        label_dict[current_label] = name

        # Percorrer imagens
        for img_file in person_dir.glob('*.jpg'):
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Invalid image ignored: {img_file}")
                continue
            img = preprocess_face(img)
            faces.append(img)
            labels.append(current_label)

        current_label += 1

    if not faces:
        raise RuntimeError('No valid images were found for training.')

    # --- Criar LBPH recognizer ---
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8 #quantas a imagem terá para criar histogramas
        )
    except AttributeError: #no caso do OpenCV não ter o módulo “face”
        raise RuntimeError(
            "LBPHFaceRecognizer not found. "
            "Install opencv-contrib-python: pip install opencv-contrib-python"
        )

    recognizer.train(faces, np.array(labels)) # Quais são os padrões faciais de cada pessoa | A que label cada face corresponde
    recognizer.write(MODEL_FILE) #cria lbph_model.yml

    # salvar mapping de labels para nomes
    with open(LABELS_FILE, 'wb') as f:
        pickle.dump(label_dict, f)

    print(f'Training complete. Model saved at {MODEL_FILE}')
    print(f'Labels saved to {LABELS_FILE}: {label_dict}')

if __name__ == '__main__':
    train()
