import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import av

# Configuração inicial da página Web
st.set_page_config(page_title="SafeSignal AI", page_icon="🚨", layout="centered")
st.title("SafeSignal AI 🚨")
st.subheader("Deteção Automatizada de Sinais de Socorro")
st.write("Demonstração Mobile - Rode o protótipo no seu telemóvel/celular.")

# Classe que processa o vídeo vindo da câmara
class HandProcessor(VideoProcessorBase):
    def __init__(self):
        # Inicializa o MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        # Transforma o frame de vídeo recebido num array do OpenCV (BGR)
        img = frame.to_ndarray(format="bgr24")
        
        # O MediaPipe precisa da imagem em RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resultados = self.hands.process(img_rgb)
        
        # Se detetar uma mão...
        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                # Desenha as articulações da mão na tela
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # --- AQUI IRÁ A LOGICA DO SEU ALGORITMO ANTI-FALSO POSITIVO ---
                # Exemplo visual apenas para a demonstração:
                cv2.putText(img, "Mao em analise...", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        # Devolve o frame processado de volta para o ecrã do telemóvel
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Cria o botão de Start/Stop da câmara na página web
webrtc_streamer(
    key="safesignal-detection", 
    video_processor_factory=HandProcessor,
    media_stream_constraints={"video": True, "audio": False} # Desliga o áudio para evitar microfonia
)

st.info("Nota: Ao clicar em 'Start', dê permissão para o navegador aceder à sua câmara.")