# Reconhecimento Facial com OpenCV, Mediapipe e LBPH

Este projeto contém três módulos principais:

1.  **Captura de imagens** usando FaceMesh (Mediapipe)
2.  **Treinamento de modelo LBPH** com OpenCV
3.  **Reconhecimento facial em tempo real**

Todos os scripts usam OpenCV, Mediapipe, NumPy e outras dependências
listadas no ficheiro requirements.txt

------------------------------------------------------------------------

## 1. Requisitos do Sistema

### ✔ Windows, Linux ou macOS

### ✔ Python 3.8 -- 3.12

(Obrigatório para compatibilidade com OpenCV e Mediapipe)

### ✔ Webcam

------------------------------------------------------------------------

## 🛠 2. Instalação das Bibliotecas

Antes de tudo, recomenda-se criar um ambiente virtual:

``` bash
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

### Instale todas as dependências:

```bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 3. Estrutura de Diretórios Recomendada

    📂 face-recognition/
     ┣ 📂 dataset/
     ┃ ┗ 📂 nome_da_pessoa/
     ┣ capture.py
     ┣ train.py
     ┣ recognise.py
     ┗ README.md

Os scripts criam as pastas automaticamente.

------------------------------------------------------------------------

## 4. Captura de Imagens (FaceMesh)

Execute:

``` bash
python3 capture.py
```

Será solicitado um nome para criar a pasta:

    Person name: Joao

As imagens serão salvas em:

    dataset/Joao/

------------------------------------------------------------------------

## 5. Treinar o Modelo LBPH

``` bash
python3 train.py
```

Isso irá gerar:

-   `lbph_model.yml` → modelo treinado\
-   `labels.pkl` → dicionário de labels

------------------------------------------------------------------------

## 6. Reconhecer Faces em Tempo Real

``` bash
python3 recognise.py
```

Pressione **ESC** para sair da aplicação.

------------------------------------------------------------------------

## Licença

Uso livre para estudo e desenvolvimento.
