import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from PIL import Image
import gdown
import os
import asyncio
import tempfile

# Custom CSS for card styling
st.markdown("""
    <style>
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        text-align: center;
        height: 200px;
    }
    .card-title {
        font-size: 1.5em;
        margin-bottom: 10px;
        color: black;
    }
    .card-image {
        width: 70%;
        height: 90px;
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Function to create a card
def create_card(title, image_url):
    card_html = f"""
    <div class="card">
        <img class="card-image" src="{image_url}" alt="{title}">
        <div class="card-title">{title}</div>
    </div>
    """
    return card_html

# Function to download the model from Google Drive
def download_model_from_gdrive(gdrive_url, output_path):
    gdown.download(gdrive_url, output_path, quiet=False, fuzzy=True)

# Cache the model loading
@st.cache_resource
def load_model():
    model_path = 'lays_model.pt'
    gdrive_url = 'https://drive.google.com/uc?id=1zyckUO6xBQjrZko-RDzme-oxLfg3f_cH'
    if not os.path.exists(model_path):
        download_model_from_gdrive(gdrive_url, model_path)
    model = YOLO(model_path)
    return model

model = load_model()

classes = [
    'Bar.B.Q', 'Barbecue', 'Cheddar-Sour Cream', 'Cheddar Jalapeno', 'Classic', 'Dill Pickle',
    'Flamin Hot', 'French Cheese', 'Honey Barbecue', 'Lays', 'Masala', 'PAPRIKA', 'Poppables',
    'Salt-Vinegar', 'Salted', 'Sour Cream-Onion', 'Sweet Southern Heat Barbecue', 'Wavy', 'Yogurt-Herb'
]

detected_classes = set()

def get_class_html(cls, detected_classes):
    detected_style = 'background-color:#FF4B4B;padding:4px 4px;border-radius:5px;margin:2px; display:inline-block; color:white;'
    default_style = 'padding:4px 4px;border-radius:5px;margin:2px; display:inline-block; background-color:white; color:black;'
    style = detected_style if cls in detected_classes else default_style
    return f'<span style="{style}">{cls}</span>'

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.confidence = 0.25

    def set_params(self, model, confidence):
        self.model = model
        self.confidence = confidence

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.model:
            results = self.model(img_rgb, conf=self.confidence)
            if results:
                annotated_frame = results[0].plot()
                return av.VideoFrame.from_ndarray(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")

async def process_image(image, model, confidence):
    img = Image.open(image)
    results = await asyncio.to_thread(model, img, conf=confidence)
    return results

def main():
    st.title("Detección de Objetos")
    activities = ["Principal", "Usar cámara", "Subir imagen", "Subir vídeo"]
    choice = st.sidebar.selectbox("Selecciona actividad", activities)
    st.sidebar.markdown('---')

    if choice == "Principal":
        st.markdown("<h4 style='color:white;'>Aplicación web de detección de Papas Lays usando Yolov9, Google Colab, Roboflow, Streamlit y lenguaje de programación Python.</h4>", unsafe_allow_html=True)
        html_classesp = [get_class_html(cls, detected_classes) for cls in classes]
        st.markdown(f"<div style='padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;'><h4 style='color:#FF4B4B;text-align:center;'>19 Clases</h4><p style='color:white;text-align:center;'>{' '.join(html_classesp)}</p></div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.markdown(create_card("Usar cámara", "https://st2.depositphotos.com/1915171/5331/v/450/depositphotos_53312473-stock-illustration-webcam-sign-icon-web-video.jpg"), unsafe_allow_html=True)
        col2.markdown(create_card("Subir imagen", "https://i.pinimg.com/736x/e1/91/5c/e1915cea845d5e31e1ec113a34b45fd8.jpg"), unsafe_allow_html=True)
        col3.markdown(create_card("Subir vídeo", "https://static.vecteezy.com/system/resources/previews/005/919/290/original/video-play-film-player-movie-solid-icon-illustration-logo-template-suitable-for-many-purposes-free-vector.jpg"), unsafe_allow_html=True)

    elif choice == "Usar cámara":
        st.header("Utiliza tu cámara")
        if model:
            confidence_slider = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
            start_detection = st.checkbox("Iniciar detección de objetos")
            video_transformer = VideoTransformer()
            if start_detection:
                video_transformer.set_params(model, confidence_slider)
            webrtc_streamer(key="example", video_transformer_factory=lambda: video_transformer, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    elif choice == "Subir imagen":
        confidence_slider = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
        html_classes = [get_class_html(cls, detected_classes) for cls in classes]
        text_placeholder = st.empty()
        text_placeholder.markdown(f"<div style='padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;'><h4 style='color:#FF4B4B;text-align:center;'>19 Clases</h4><p style='color:white;text-align:center;'>{' '.join(html_classes)}</p></div>", unsafe_allow_html=True)
        change_text = st.checkbox("Objetos Detectados")
        image = st.file_uploader('Sube imagen', type=['png', 'jpg', 'jpeg', 'gif'])

        if image:
            col1, col2, col3 = st.columns([1, 1, 1])
            col1.image(image, caption='Imagen original')
            if model:
                with col2:
                    with st.spinner('Procesando imagen...'):
                        results = asyncio.run(process_image(image, model, confidence_slider))
                        if results:
                            annotated_frame = results[0].plot()
                            col2.image(annotated_frame, caption='Imagen anotada')
                            for result in results[0].boxes:
                                idx = int(result.cls.cpu().numpy()[0])
                                confidence = result.conf.cpu().numpy()[0]
                                detected_class = classes[idx]
                                detected_classes.add(detected_class)
                                col3.markdown(f"<div style='background-color:#f0f0f0;padding:5px;border-radius:5px;margin:5px 0; color:black;'><b>Clase:</b> <span style='color:black'>{detected_class}</span><br><b>Confianza:</b> {confidence:.2f}<br></div>", unsafe_allow_html=True)
                        else:
                            col3.write("No se detectaron objetos.")
            else:
                st.error("Model is not loaded. Please check the logs for errors.")
        if change_text:
            html_classes = [get_class_html(cls, detected_classes) for cls in classes]
            text_placeholder.markdown(f"<div style='padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;'><h4 style='color:#FF4B4B;text-align:center;'>19 Clases</h4><p style='color:white;text-align:center;'>{' '.join(html_classes)}</p></div>", unsafe_allow_html=True)

    elif choice == "Subir vídeo":
        confidence_slider = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
        html_classes = [get_class_html(cls, detected_classes) for cls in classes]
        text_placeholder = st.empty()
        text_placeholder.markdown(f"<div style='padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;'><h4 style='color:#FF4B4B;text-align:center;'>19 Clases</h4><p style='color:white;text-align:center;'>{' '.join(html_classes)}</p></div>", unsafe_allow_html=True)
        change_text = st.checkbox("Objetos Detectados")
        video_file_buffer = st.sidebar.file_uploader("Sube un vídeo", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
        tfflie = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        if video_file_buffer:
            tfflie.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tfflie.name)
            st.sidebar.video(video_file_buffer)
            stframe = st.empty()
            while vid.isOpened():
                success, frame = vid.read()
                if success:
                    if model:
                        results = model(frame, conf=confidence_slider)
                        for result in results[0].boxes:
                            idx = int(result.cls.cpu().numpy()[0])
                            confidence = result.conf.cpu().numpy()[0]
                            detected_class = classes[idx]
                            detected_classes.add(detected_class)
                        if change_text:
                            html_classes = [get_class_html(cls, detected_classes) for cls in classes]
                            text_placeholder.markdown(f"<div style='padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;'><h4 style='color:#FF4B4B;text-align:center;'>19 Clases</h4><p style='color:white;text-align:center;'>{' '.join(html_classes)}</p></div>", unsafe_allow_html=True)
                        annotated_frame = results[0].plot()
                        stframe.image(annotated_frame, channels='BGR', use_column_width=True, caption="Vídeo Anotado")
            vid.release()

if __name__ == "__main__":
    main()
