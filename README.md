---
title: Image To Audio
emoji: 📢
colorFrom: gray
colorTo: yellow
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
---

# The Image Reader 📢

[The Image Reader 📢 - Playground](www.google.com)

This application analyzes the uploaded image, generates an imaginative phrase, and then converts it into audio.

- For **image_to_audio** following technologies were used:
    - **Image Reader:** 
        - HuggingFace ```image-to-text``` task used with ```Salesforce/blip-image-captioning-base``` pretrained model. Which produces a small description about the image.
        - [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
            - BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation
    - **Generate an imaginative phrase:**
        - OpenAI ```GPT-3.5-Turbo``` used to produce an imaginative narrative from the description generated earlier.
        - The phrase generated with more than 40 words.
        - [GPT-3.5 Turbo](https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates)
    - **text-to-audio:**
        - ```suno/bark-small``` used to generate the audio version of the imaginative narrative earlier.
        - [suno/bark-small](https://huggingface.co/suno/bark-small)
            - **BARK**: Bark is a transformer-based text-to-audio model created by [Suno](https://www.suno.ai/). Bark can generate highly realistic, multilingual speech as well as other audio - including music, background noise and simple sound effects. The model can also produce nonverbal communications like laughing, sighing and crying.
