# ⚽ Soccer Vision & Tracking

Sistema de visão computacional e inteligência artificial para deteção, rastreio e análise tática em vídeos de futebol. Este projeto combina modelos de deep learning em PyTorch com anotações e marcadores visuais dinâmicos.

---

## 📌 Funcionalidades

- **Deteção Automática:** Identificação de jogadores, equipas de arbitragem e bola.
- **Rastreio Inteligente:** Acompanhamento contínuo dos atletas e da trajetória da bola no relvado.
- **Marcadores Personalizados (`custom_markers.py`):** Visualizações gráficas como halos nos pés, caixas delimitadoras personalizadas, indicadores de posse de bola e mapas táticos.
- **Integração com Pesos Personalizados (`custom_model.pt`):** Modelo pré-treinado/afinado pronto para inferência rápida com aceleração por GPU.

---

## 📂 Estrutura do Projeto

```text
soccer/
├── custom_markers.py    # Módulo de anotações e renderização visual
├── custom_model.pt      # Pesos treinados do modelo (PyTorch / YOLO)
├── .gitignore           # Configurações de ficheiros ignorados pelo Git
└── README.md            # Documentação principal
```

---

## 🚀 Instalação e Configuração

### Pré-requisitos
- Python 3.10+ instalado
- Placa de vídeo compatível com CUDA (recomendado para inferência em tempo real)

### 1. Clonar o repositório
```bash
git clone https://github.com/teu-usuario/soccer.git
cd soccer
```

### 2. Configurar o ambiente virtual
```bash
# Criar o ambiente
python -m venv venv

# Ativar no Linux/macOS:
source venv/bin/activate

# Ativar no Windows:
venv\Scripts\activate
```

### 3. Instalar dependências
```bash
pip install torch torchvision opencv-python ultralytics numpy
```

---

## 💻 Como Utilizar

### Exemplo Básico de Inferência e Desenho

```python
import cv2
import torch
from custom_markers import draw_markers

# 1. Carregar o modelo treinado
model = torch.load("custom_model.pt")
model.eval()

# 2. Ler o vídeo ou frame de entrada
cap = cv2.VideoCapture("input_match.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Processar inferência
    # results = model(frame)

    # 4. Aplicar os marcadores visuais personalizados
    # annotated_frame = draw_markers(frame, results)

    # 5. Exibir saída
    # cv2.imshow("Soccer Vision", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🎨 Personalização dos Marcadores

O ficheiro `custom_markers.py` permite parametrizar:
- **Cores dos times:** Separação cromática para jogadores da casa, visitantes e árbitros.
- **Triângulo de Posse:** Indicador flutuante sobre a cabeça do jogador em posse da bola.
- **Trajetória da Bola:** Efeito de rastro na movimentação da bola.

---

## 📄 Licença

Este projeto está sob a licença [MIT](LICENSE).