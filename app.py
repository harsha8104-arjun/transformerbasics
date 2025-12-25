
import streamlit as st
from transformers import pipeline, CLIPProcessor, CLIPModel
from PIL import Image
import torch

@st.cache_resource
def load_pipelines():
    return {
        "sentiment": pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"),
        "generator": pipeline("text-generation", model="gpt2"),
        "image_classifier": pipeline("image-classification", model="google/vit-base-patch16-224"),
        "asr": pipeline("automatic-speech-recognition", model="openai/whisper-small")
    }

@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

pipelines = load_pipelines()
clip_model, clip_processor = load_clip()


st.title("🤖 AI Playground with Transformers")

task = st.sidebar.selectbox("Choose a task:",
                            ["Sentiment Analysis", "Text Generation", "Image Classification", "Text-Image Matching", "Speech-to-Text"])


if task == "Sentiment Analysis":
    text = st.text_area("Enter text:", "I love using transformers for AI projects!")
    if st.button("Analyze"):
        result = pipelines["sentiment"](text)[0]
        st.write(f"**Label:** {result['label']}, **Confidence:** {result['score']:.2f}")


elif task == "Text Generation":
    prompt = st.text_input("Enter a prompt:", "Once upon a time")
    if st.button("Generate"):
        result = pipelines["generator"](prompt, max_length=50, num_return_sequences=1)
        st.write(result[0]["generated_text"])


elif task == "Image Classification":
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Classify"):
            results = pipelines["image_classifier"](image)
            for r in results:
                st.write(f"**{r['label']}** ({r['score']:.2f})")


elif task == "Text-Image Matching":
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png"])
    text = st.text_input("Enter a description:", "a photo of a cat")
    if uploaded_file and text:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Compare"):
            inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).item()
            st.write(f"Similarity score with '{text}': **{probs:.2f}**")


elif task == "Speech-to-Text":
    uploaded_file = st.file_uploader("Upload an audio file:", type=["wav", "mp3", "flac"])
    if uploaded_file:
        if st.button("Transcribe"):
            result = pipelines["asr"](uploaded_file)
            st.write("**Transcription:**")
            st.write(result["text"])