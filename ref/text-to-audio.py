# text-to-audio using suno/bark
import scipy
import torch
from transformers import AutoProcessor
from transformers import BarkModel

model = BarkModel.from_pretrained("suno/bark-small")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device: ", device)

processor = AutoProcessor.from_pretrained("suno/bark")

# prepare the inputs
text_prompt = "You are a story teller. You can generate a story based on a simple narrative, the story be no more than 20 words."
inputs = processor(text_prompt)

# generate speech
model = model.to(device)
speech_output = model.generate(**inputs.to(device))

sampling_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=speech_output[0].cpu().numpy())