import streamlit as st
try:
    import numpy as np
    import cv2
    from ultralytics import YOLO
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    import av
    from PIL import Image
    import gdown
    import os
    import tempfile

except ImportError as e:
    st.error(f"Error importing modules: {e}")

st.markdown("""
    <style>
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        text-align: center;
        height: 220px; /* Ensure all cards have the same height */
    }
    .card-title {
        font-size: 1.5em;
        margin-bottom: 10px;
        color: black;
    }
    .card-image {
        width: 80%;
        height: 120px; /* Set a fixed height for the images */
        object-fit: cover; /* Ensure images cover the specified dimensions */
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
    gdown.download(gdrive_url, output_path, quiet=False)

# Cache the model loading
@st.cache_resource
def load_model():
    model_path = 'lays_model.pt'
    gdrive_url = 'https://drive.google.com/uc?export=download&id=1zyckUO6xBQjrZko-RDzme-oxLfg3f_cH'

    if not os.path.exists(model_path):
        try:
            download_model_from_gdrive(gdrive_url, model_path)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return None

    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

classes = [
    'Bar.B.Q', 'Barbecue', 'Cheddar-Sour Cream', 'Cheddar Jalapeno', 'Classic', 'Dill Pickle',
    'Flamin Hot', 'French Cheese', 'Honey Barbecue', 'Lays', 'Masala', 'PAPRIKA', 'Poppables',
    'Salt-Vinegar', 'Salted', 'Sour Cream-Onion', 'Sweet Southern Heat Barbecue', 'Wavy', 'Yogurt-Herb'
]

detected_classes = set()
detected_classes_set = set()

def get_class_html(cls, detected_classes):
    detected_style = 'background-color:#FF4B4B;padding:4px 4px;border-radius:5px;margin:2px; display:inline-block; color:white;'
    default_style = 'padding:4px 4px;border-radius:5px;margin:2px; display:inline-block; background-color:white; color:black;'
    
    style = detected_style if cls in detected_classes else default_style
    return f'<span style="{style}">{cls}</span>'


class VideoTransformer(VideoTransformerBase):
    def __init__(self, model=None, confidence=0.25):
        self.model = model
        self.confidence = confidence

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.model:
            results = self.model(img_rgb, conf=self.confidence)

            if results:
                annotated_frame = results[0].plot()  # Annotate frame
                return av.VideoFrame.from_ndarray(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), format="bgr24")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():

    st.title("Detección de Objetos")
    activiteis = ["Principal", "Usar cámara", "Subir imagen", "Subir vídeo"]
    choice = st.sidebar.selectbox("Selecciona actividad", activiteis)
    st.sidebar.markdown('---')

    if choice == "Principal":
        html_temp_home1 = """<div style="padding:10px">
                                            <h4 style="color:white;text-align:left;">
                                            Aplicación web de detección de Papas Lays usando Yolov9, Google Colab, Roboflow, Streamlit y lenguaje de programación Python.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)

        html_classesp = [get_class_html(cls, detected_classes) for cls in classes]

        html_tempp = f"""
                <div style="padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;">
                    <h4 style="color:#FF4B4B;text-align:center;">19 Clases</h4>
                    <p style="color:white;text-align:center;">{" ".join(html_classesp)}</p>
                </div>
                <br>
                """
        st.markdown(html_tempp, unsafe_allow_html=True)
    
        # Example cards with images
        card1 = create_card("Usar cámara", "https://st2.depositphotos.com/1915171/5331/v/450/depositphotos_53312473-stock-illustration-webcam-sign-icon-web-video.jpg")
        card2 = create_card("Subir imagen", "https://i.pinimg.com/736x/e1/91/5c/e1915cea845d5e31e1ec113a34b45fd8.jpg")
        card3 = create_card("Subir vídeo", "https://static.vecteezy.com/system/resources/previews/005/919/290/original/video-play-film-player-movie-solid-icon-illustration-logo-template-suitable-for-many-purposes-free-vector.jpg")

        # Display cards in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(card1, unsafe_allow_html=True)
        with col2:
            st.markdown(card2, unsafe_allow_html=True)
        with col3:
            st.markdown(card3, unsafe_allow_html=True)

    elif choice == "Usar cámara":
        st.header("Utiliza tu cámara")
        st.write("Selecciona el disposito que quieres utilizar y empieza a detectar objetos")

        if model:
            confidence_slider = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)

            # Add button to start detection
            start_detection = st.checkbox("Iniciar detección de objetos")

            if start_detection:
                st.write("Detección de objetos activada")
                webrtc_streamer(
                    key="example",
                    video_processor_factory=lambda: VideoTransformer(model, confidence_slider)
                )
            else:
                st.write("Esperando para iniciar la detección...")
                webrtc_streamer(
                    key="example",
                    video_transformer_factory=lambda: VideoTransformer()
                )
        else:
            st.error("El Modelo no se ha cargado correctamente")

    elif choice == "Subir imagen":
        
        detected_class = None

        confidence_slider = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
        
        html_classes = [get_class_html(cls, detected_classes) for cls in classes]

        html_temp = f"""
                <div style="padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;">
                    <h4 style="color:#FF4B4B;text-align:center;">19 Clases</h4>
                    <p style="color:white;text-align:center;">{" ".join(html_classes)}</p>
                </div>
                """

        # Create a placeholder for the text
        text_placeholder = st.empty()

        # Display the original text
        text_placeholder.markdown(html_temp, unsafe_allow_html=True)
        # Checkbox to trigger text change
        change_text = st.checkbox("Objetos Detectados")

        image = st.file_uploader('Sube imagen', type=['png', 'jpg', 'jpeg', 'gif'])
        if image is not None:
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.image(image, caption='Imagen original')

            with col2:
                with st.spinner('Procesando imagen...'):
                    img = Image.open(image)

                if model:
                    results = model(img, conf = confidence_slider)
                    if results:
                        annotated_frame = results[0].plot()
                        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                        st.image(annotated_frame, caption='Imagen anotada')
                        with col3:
                            st.write("**Detalles de detección:**")
                            for result in results[0].boxes:
                                idx = int(result.cls.cpu().numpy()[0])
                                confidence = result.conf.cpu().numpy()[0]
                                detected_class = classes[idx]
                                detected_classes.add(detected_class)

                                    
                                st.markdown(f"""
                                                    <div style="background-color:#f0f0f0;padding:5px;border-radius:5px;margin:5px 0; color:black;">
                                                        <b>Clase:</b> <span style="color:black">{detected_class}</span><br>
                                                        <b>Confianza:</b> {confidence:.2f}
                                                        <br>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                    
                    else:
                        st.write("No se detectaron objetos.")
                else:
                    st.error("Model is not loaded. Please check the logs for errors.")
        
        if change_text:
            html_classes = [get_class_html(cls, detected_classes) for cls in classes]
            html_temp2 = f"""
                <div style="padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;">
                    <h4 style="color:#FF4B4B;text-align:center;">19 Clases</h4>
                    <p style="color:white;text-align:center;">{" ".join(html_classes)}</p>
                </div>
            """
            text_placeholder.markdown(html_temp2, unsafe_allow_html=True)
     

    elif choice == "Subir vídeo":
        
        confidence_slider = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)

        html_classes = [get_class_html(cls, detected_classes) for cls in classes]

        html_temp = f"""
                <div style="padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;">
                    <h4 style="color:#FF4B4B;text-align:center;">19 Clases</h4>
                    <p style="color:white;text-align:center;">{" ".join(html_classes)}</p>
                </div>
                """

        # Create a placeholder for the text
        text_placeholder = st.empty()

        # Display the original text
        text_placeholder.markdown(html_temp, unsafe_allow_html=True)
        # Checkbox to trigger text change
        change_text = st.checkbox("Objetos Detectados")

        video_file_buffer = st.sidebar.file_uploader("Sube un vídeo", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
        tfflie = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        vid = None
        if video_file_buffer is not None:
            tfflie.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tfflie.name)
            
            st.sidebar.text('Reproduciendo Video')
            st.sidebar.video(video_file_buffer)
        
        
        st.sidebar.text('Reproduciendo Video')

        stframe = st.empty()

        if vid is not None:
            while vid.isOpened():
                success, frame = vid.read()
                if success:
                    if model:
                        # Run YOLOv8 inference on the frame
                        results = model(frame, conf = confidence_slider)
                        for result in results[0].boxes:
                            idx = int(result.cls.cpu().numpy()[0])
                            confidence = result.conf.cpu().numpy()[0]
                            detected_class = classes[idx]
                            detected_classes_set.add(detected_class)
                            print(detected_classes_set)

                        if change_text:
                            html_classes = [get_class_html(cls, detected_classes_set) for cls in classes]
                            html_temp2 = f"""
                                <div style="padding:4px; border: 2px solid #FF4B4B; border-radius: 10px;">
                                    <h4 style="color:#FF4B4B;text-align:center;">19 Clases</h4>
                                    <p style="color:white;text-align:center;">{" ".join(html_classes)}</p>
                                </div>
                            """
                            text_placeholder.markdown(html_temp2, unsafe_allow_html=True)
                        # Visualize the results on the frame
                        annotated_frame = results[0].plot()
                        stframe.image(annotated_frame, channels='BGR', use_column_width=True, caption="Vídeo Anotado")
            vid.release()                   


if __name__ == "__main__":
    main()
