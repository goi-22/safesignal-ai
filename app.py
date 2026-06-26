import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import av

# Configuração inicial da página Web
st.set_page_config(page_title="SafeSignal AI", page_icon="🚨", layout="centered")
st.title("SafeSignal AI 🚨")
st.subheader("Detecção Automatizada de Sinais de Socorro")
st.write("Demonstração Mobile - Rode o protótipo no seu telemóvel/celular.")

# Configuração de servidores STUN públicos para garantir que o vídeo conecte em redes escolares/4G
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

class HandProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resultados = self.hands.process(img_rgb)
        
        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                cv2.putText(img, "Mao em analise...", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Criamos um container fixo para o WebRTC não brigar com o resto da página
with st.container():
    webrtc_streamer(
        key="safesignal-detection-v2", # Mudamos a chave para forçar o Streamlit a recriar o componente limpo
        video_processor_factory=HandProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=RTC_CONFIGURATION, # Adicionado para estabilizar a conexão de vídeo
        async_processing=True # Força o processamento assíncrono para não travar a interface gráfica
    )

st.info("Nota: Se o erro persistir, verifique se o Tradutor do Google Chrome não está ativo nesta página.")