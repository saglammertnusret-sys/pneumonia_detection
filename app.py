from pathlib import Path
import torch
import streamlit as st
from PIL import Image, UnidentifiedImageError
from pneumonia.pipeline import load_checkpoint, make_transform

st.set_page_config(page_title='Chest X-ray research demo', layout='centered')
st.title('Chest X-ray classification')
st.caption('Research prototype. Model scores are not a medical diagnosis or calibrated clinical probabilities.')

@st.cache_resource
def load_model(path, modified_ns):
    return load_checkpoint(path, torch.device('cpu'))

checkpoint_path = st.text_input('Trained checkpoint path', 'runs/resnet18/best.pt')
path = Path(checkpoint_path)
if not path.is_file():
    st.info('Train a model first, then enter the path to its best.pt checkpoint.')
    st.stop()
try:
    model, checkpoint = load_model(str(path.resolve()), path.stat().st_mtime_ns)
except Exception as error:
    st.error(f'Cannot load this checkpoint: {error}')
    st.stop()

config = checkpoint['config']
st.caption(f"Architecture: {config['model']} · CLAHE: {'enabled' if config['clahe'] else 'disabled'}")
uploaded = st.file_uploader('Upload a JPG or PNG chest X-ray', type=['jpg', 'jpeg', 'png'])
if uploaded:
    try:
        image = Image.open(uploaded)
        image.load()
        st.image(image, caption='Uploaded image', width=450)
        tensor = make_transform(config['image_size'], config['clahe'])(image).unsqueeze(0)
        with torch.inference_mode():
            score = model(tensor).flatten().sigmoid().item()
        st.metric('Pneumonia model score', f'{score:.3f}')
        st.write(f"Model label at threshold 0.5: **{'PNEUMONIA' if score >= .5 else 'NORMAL'}**")
        st.caption('This binary classifier does not verify that the input is a chest X-ray.')
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f'Cannot read this image: {error}')
