import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Replace the model version with your required version if needed
pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32
)

# Running the inference on CPU
pipeline = pipeline.to('cpu')  # Changed from 'cuda' to 'cpu'

prompt = "Your prompt here"

image = pipeline(prompt=prompt).images[0]

image.show()
from IPython.display import display
display(image)
