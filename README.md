# Stable Diffusion API Wrapper
## AlexScotland

# Description
Fast API Endpoint that supports LoRA integration with Image Generation!


# Dependencies
- NVIDIA GPU with at least 12GB of VRAM (with a batch size of 1)
- NVIDIA drivers installed + NVIDIA Container Toolkit installed

# How to use
## Starting the Software
### Docker
As of right now, I do not have this working dockerized.  That may come in the future.
### Bare Metal
#### Setup
First thing we need to do is assign the following ENV variables:
- `BASE_DIRECTORY`: This is the base directory of the repository.
  - This is used when finding LoRAs and PipelineFiles
- `SELECTED_MODEL`: This is used to select the current Image Model.
  - In the future, we will be assigning this via POST request.
- `SELECTED_VIDEO_MODEL`: This *WILL* be used for IMG2Video.  The code is here and **working** - I just found it was not a very good experience for my usecase', but can be used with the code in the deprectated [API V1 file](routes/api.py)

#### Adding Models
The purpose of this repository was to learn how I can create image generation pipelines, without the need for external network connectivity.  
I really didn't want to spend money and wanted to do it all on my own`¯\_( ͡° ͜ʖ ͡°)_/¯`

To add a model, we do one of the following:
##### Via API (Easiest)
We can make a `PUT` request to `/download/` containing the name of the Model.
```json
{
"model_name": "stabilityai/stable-video-diffusion-img2vid"
}
```
This will download the model into the `MODEL_DIRECTORY` folder.  
A successful message looks like:
```json
{
  "Status": "Downloaded"
}
```
Now we can set the `SELECTED_MODEL` to the name of the folder we just downloaded.
##### Manually
1. Navigate to our `models/full_models/` directory.
2. Download the local `.safetensors` file for the ImageGenerationPipeline, 
   1. We can use the Diffusers Pipeline to pull an image from `HuggingFace` using `pipeline = DiffusionPipeline.from_pretrained(model)`
3. Open (or use the existing Pipeline above) and run `pipeline.save_pretrained(f"{MODEL_DIRECTORY}/{model.split('/')[-1]}")`.  This is done in the deprecated api.
4. Once we have our model downloded, we can set the `SELECTED_MODEL` to a path to that local file.

#### Starting API
To start the API, we use `fastapi run` or `fastapi dev`.  
I found `fastapi run` works better, as `dev` seems to freeze sometimes.

# FAQ
## What is `base_lora` and `contextual_lora`?
These are fields used to help generate images using multiple LoRAs.  
An Example of a `base_lora` may be a LoRA used as a *style*.

i.e. A LoRA used for creating images in specific art styles.

The `contextual_lora` would be something that would modify something in the picture.  
An example of this could be a LoRA for fixing a finger count, or better image-quality.

## Why is negative prompt prefilled?
This is a canned negative prompt to help better generate pictures, feel free to remove and change for your use cases!

# Happy Hacking!
![If you are reading this, then you got slow internet or have bad sight. If it is the latter, I am sorry to expose you](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExcTNybmhtNHBtbzd1ODlkeHYya2R3eDVxbmFwb3VqN3U0dXVxZTlibiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/o0vwzuFwCGAFO/giphy.webp)