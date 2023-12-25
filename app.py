import os

import scipy
import streamlit as st
import torch
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from transformers import AutoProcessor, BarkModel, pipeline


# create image-to-text pipeline
@st.cache_resource
def create_image_to_text_pipeline():
    """create image to text pipeline"""

    task = "image-to-text"
    model = "Salesforce/blip-image-captioning-base"
    img_to_text_pipeline = pipeline(task, model=model)
    return img_to_text_pipeline


# generate information about the image
def image_to_text(url):
    """image to text"""

    generate_kwargs = {
        "do_sample": True,
        "temperature": 0.7,
        "max_new_tokens": 256,
    }

    pipe = create_image_to_text_pipeline()
    txt = pipe(url, generate_kwargs=generate_kwargs)[0]["generated_text"]
    return txt


# load language models
@st.cache_resource
def load_llm_model(openai_key):
    """load llm model"""

    model = ChatOpenAI(
        model_name="gpt-3.5-turbo", openai_api_key=openai_key, temperature=0
    )
    return model


# generate audio script
def generate_audio_script(openai_key, scenario):
    """generate audio script"""

    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a story teller. "
                    "You can generate a story based on a simple narrative, "
                    "the story be no more than 40 words."
                )
            ),
            HumanMessagePromptTemplate.from_template("{scenario}"),
        ]
    )

    llm_model = load_llm_model(openai_key)
    ai_response = llm_model(chat_template.format_messages(scenario=scenario))
    script = ai_response.content
    return script


# load audio pipeline
@st.cache_resource
def load_audio_pipeline():
    """load audio pipeline"""

    synthesiser = BarkModel.from_pretrained("suno/bark-small")
    audio_processor = AutoProcessor.from_pretrained("suno/bark")
    return synthesiser, audio_processor


def generate_audio(script):
    """generate audio"""

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    print("Script: ", script)
    model, processor = load_audio_pipeline()

    inputs = processor(script)
    model = model.to(device)

    speech_output = model.generate(**inputs.to(device))
    sampling_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(
        "audio/bark_output.wav", rate=sampling_rate, data=speech_output[0].cpu().numpy()
    )


def main():
    """main"""

    st.set_page_config(
        page_title="Image to Speech",
        page_icon="📢",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.header("The Image Reader 📢", divider="rainbow")

    st.subheader(
        "This application :red[analyzes] the uploaded image, generates an :green[imaginative phrase], and then converts it into :blue[audio] :sunglasses:"
    )

    st.markdown("[check out the repository](https://github.com/ThivaV/image_to_audio)")

    openai_key = st.text_input("Enter your OpenAI key 👇", type="password")

    progress_bar_message = "Operation in progress. Please wait."

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
    if uploaded_image is not None:
        progress_bar = st.progress(0, text=progress_bar_message)

        # rename all the uploaded images to "uploaded_image"
        image_ext = os.path.splitext(uploaded_image.name)[1]
        new_image_name = "uploaded_image" + image_ext
        image_save_path = "img/" + new_image_name

        byte_data = uploaded_image.getvalue()
        with open(image_save_path, "wb") as file:
            file.write(byte_data)

        # 10% completed
        progress_bar.progress(10, text=progress_bar_message)

        col_1, col_2 = st.columns([6, 4])

        with col_1:
            st.image(uploaded_image, caption="Uploaded image.", use_column_width=True)

        # 20% completed
        progress_bar.progress(20, text=progress_bar_message)

        scenario = image_to_text(image_save_path)

        # 40% completed
        progress_bar.progress(40, text=progress_bar_message)

        script = generate_audio_script(openai_key, scenario)

        # 60% completed
        progress_bar.progress(60, text=progress_bar_message)

        generate_audio(script)

        # 90% completed
        progress_bar.progress(90, text=progress_bar_message)

        with col_2:
            with st.expander("About the image"):
                st.write(scenario)

            with st.expander("Script"):
                st.write(script)

            st.audio("audio/bark_output.wav")

        # 100% completed
        progress_bar.progress(
            100, text="Operation completed. Thank you for your patients."
        )


if __name__ == "__main__":
    main()
