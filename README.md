---
title: Video Model Studio
emoji: 🎥
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 5.32.1
app_file: app.py
pinned: true
license: apache-2.0
short_description: All-in-one tool for AI video training
---

# 🎥 Video Model Studio (VMS)

![example](https://media.githubusercontent.com/media/jbilcke-hf/VideoModelStudio/main/docs/screenshots/importing-a-video-dataset.jpg)

## Presentation

### What is this project?

VMS is a Gradio app that wraps around Finetrainers, to provide a simple UI to train AI video models on Hugging Face.

You can deploy it to a private space, and start long-running training jobs in the background.

## Funding

VideoModelStudio is 100% open-source project, I develop and maintain it during both my pro and personal time. If you like it, you can tip! If not, have a good day 🫶

<a href="https://www.buymeacoffee.com/flngr" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

## News
- 🔥 **2025-06-04**: Upgrade to Cuda 12.8, Pytorch 2.6, Gradio 5.32
- 🔥 **2025-03-12**: VMS now officially supports Wan!
- 🔥 **2025-03-11**: I have added a tab to preview a model!
- 🔥 **2025-03-10**: Various small fixes and improvements
- 🔥 **2025-03-09**: I have added a basic CPU/RAM monitor (no GPU yet)
- 🔥 **2025-03-02**: Made some fixes to improve Finetrainer reliability when working with big datasets
- 🔥 **2025-02-18**: I am working to add better recovery in case of a failed run (this is still in beta)
- 🔥 **2025-02-18**: I have added persistence of UI settings. So if you reload Gradio, you won't lose your settings!

## TODO
- Add `Aya-Vision-8B` for frame analysis (currently we use `Qwen2-VL-7B`)

### See also

#### Internally used project: Finetrainers

VMS uses Finetrainers under the hood: https://github.com/a-r-r-o-w/finetrainers

#### Similar project: diffusion-pipe-ui

I wasn't aware of its existence when I started my project, but there is also this open-source initiative (which is similar in terms of dataset management etc): https://github.com/alisson-anjos/diffusion-pipe-ui

## Features

### Run Finetrainers in the background

The main feature of VMS is the ability to run a Finetrainers training session in the background.

You can start your job, close the web browser tab, and come back the next morning to see the result.

### Automatic scene splitting

VMS uses PySceneDetect to split scenes.

### Automatic clip captioning

VMS uses `LLaVA-Video-7B-Qwen2` for captioning. You can customize the system prompt if you want to.

### Download your dataset

Not interested in using VMS for training? That's perfectly fine!

You can use VMS for video splitting and captioning, and export the data for training on another platform eg. on Replicate or Fal.

## Supported models

VMS uses `Finetrainers` under the hood. In theory any model supported by Finetrainers should work in VMS.

In practice, a PR (pull request) will be necessary to adapt the UI a bit to accomodate for each model specificities.


### Wan

I am currently testing Wan LoRA training!

### LTX-Video

I have tested training a LTX-Video LoRA model using videos (not images), on a single A100 instance.
It requires about 18/19 Gb of VRAM, depending on your settings.

### HunyuanVideo

I have tested training a HunyuanVideo LoRA model using videos (not images),, on a single A100 instance.

It requires about 47~49 Gb of VRAM, depending on your settings.

### CogVideoX

Do you want support for this one? Let me know in the comments!

## Limitations

### No AV1 on A100

If your dataset contains videos encoded using the AV1 codec, you might not be able to decode them (eg. during scene splitting) if your machine doesn't support hardware decoding.

Nvidia A100 don't support hardware AV1 decoding for instance.

It might be possible to convert them on server-side or use software decoding directly from Python, but I haven't looked into that yet (you can submit a PR if you have an idea).

My recommendation is to make sure your data comes in h264.

You can use FFmpeg to do this, eg:

```bash
ffmpeg -i input_video_in_av1.mp4 -vcodec libx264 -acodec aac output_video_in_h264.mp4
```

### One-user-per-space design

Currently CMS can only support one training job at a time, anybody with access to your Gradio app will be able to upload or delete everything etc.

This means you have to run VMS in a *PRIVATE* HF Space, or locally if you require full privacy.

## Deployment

VMS is built on top of Finetrainers and Gradio, and designed to run as a Hugging Face Space (but you can deploy it anywhere that has a NVIDIA GPU and supports Docker).

### Full installation at Hugging Face

Easy peasy: create a Space (make sure to use the `Gradio` type/template), and push the repo. No Docker needed!

That said, please see the "RUN" section for info about environement variables.

### Dev mode on Hugging Face

I recommend to not use the dev mode for a production usage (ie not use dev mode when training a real model), unless you know what you are doing.

That's because the dev mode can be unstable and cause space restarts.

If you still want to open the dev mode in the space, then open VSCode in local or remote and run:

```
pip install -r requirements.txt
```

As this is not automatic, then click on "Restart" in the space dev mode UI widget.

Important: if you see errors like "API not found" etc, it might indicate an issue with the dev mode and Gradio, not an issue with VMS itself.

### Full installation somewhere else

I haven't tested it, but you can try to provided Dockerfile

### Prerequisites

About Python:

I haven't tested Python 3.11 or 3.12, but I noticed some incompatibilities with Python 3.13 dependencies failing to install.

So I recommend you to install [pyenv](https://github.com/pyenv/pyenv) to switch between versions of Python.

If you are on macOS, you might already have some versions of Python installed, you can see them by typing:

```bash
% python3.10 --version
Python 3.10.16
% python3.11 --version
Python 3.11.11
% python3.12 --version
Python 3.12.9
% python3.13 --version
Python 3.13.2
```

Once pyenv is installed you can type:

```bash
pyenv install 3.10.16
```

### Full installation in local

the full installation requires:
- Linux
- CUDA 12
- Python 3.10

This is because of flash attention, which is defined in the `requirements.txt` using an URL to download a prebuilt wheel expecting this exact configuration (python bindings for a native library)

```bash
./setup.sh
```

### macOS is NOT supported

Currently macOS is NOT supported due to two separate dependency issues:

- issue with flash attention
- issue with decord

However, the paradox is that I actually develop and maintain VMS using macOS!

So don't be surprised by the presence of "degraded_setup.sh" and all, it's just a workaround for me to be able to at least test the web ui on macOS.

## Run

### Running the Gradio app

Note: please make sure you properly define the environment variables for `STORAGE_PATH` (eg. `/data/`) and `HF_HOME` (eg. `/data/huggingface/`)

```bash
python3.10 app.py
```

### Running locally

See above remarks about the environment variable.

By default `run.sh` will store stuff in `.data/` (located inside the current working directory):

```bash
./run.sh
```

### Environment Variables

- `STORAGE_PATH`: Specifies the base storage path (default: '.data')
- `HF_API_TOKEN`: Your Hugging Face API token for accessing models and publishing
- `USE_LARGE_DATASET`: Set to "true" or "1" to enable large dataset mode, which:
  - Hides the caption list in the caption tab
  - Disables preview and editing of individual captions
  - Disables the dataset download button
  - Use this when working with large datasets that would be too slow to display in the UI
- `PRELOAD_CAPTIONING_MODEL`: Preloads the captioning model at startup
- `ASK_USER_TO_DUPLICATE_SPACE`: Prompts users to duplicate the space
