---
title: Video Model Studio
emoji: 🎥
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 5.15.0
app_file: app.py
pinned: true
license: apache-2.0
short_description: All-in-one tool for AI video training
---

# 🎥 Video Model Studio (VMS)

## Presentation

### What is this project?

VMS is a Gradio app that wraps around Finetrainers, to provide a simple UI to train AI video models on Hugging Face.

You can deploy it to a private space, and start long-running training jobs in the background.

### Similar projects

I wasn't aware of its existence when I started my project, but there is also this open-source initiative: https://github.com/alisson-anjos/diffusion-pipe-ui

## News
- 🔥 **2025-03-02**: Made some fixes to improve Finetrainer reliability when working with big datasets
- 🔥 **2025-02-18**: I am working to add better recovery in case of a failed run (this is still in beta)
- 🔥 **2025-02-18**: I have added persistence of UI settings. So if you reload Gradio, you won't lose your settings!


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

### LTX-Video

I have tested training a LTX-Video LoRA model using videos (not images), on a single A100 instance.
It requires about 18/19 Gb of VRAM, depending on your settings.

### HunyuanVideo

I have tested training a HunyuanVideo LoRA model using videos (not images),, on a single A100 instance.

It requires about 47~49 Gb of VRAM, depending on your settings.

### CogVideoX

Do you want support for this one? Let me know in the comments!

## Limitations

### One-user-per-space design

Currently CMS can only support one training job at a time, anybody with access to your Gradio app will be able to upload or delete everything etc.

This means you have to run VMS in a *PRIVATE* HF Space, or locally if you require full privacy.

## Deployment

VMS is built on top of Finetrainers and Gradio, and designed to run as a Hugging Face Space (but you can deploy it anywhere that has a NVIDIA GPU and supports Docker).

### Full installation at Hugging Face

Easy peasy: create a Space (make sure to use the `Gradio` type/template), and push the repo. No Docker needed!

That said, please see the "RUN" section for info about environement variables.

### Dev mode on Hugging Face

Enable dev mode in the space, then open VSCode in local or remote and run:

```
pip install -r requirements.txt
```

As this is not automatic, then click on "Restart" in the space dev mode UI widget.

### Full installation somewhere else

I haven't tested it, but you can try to provided Dockerfile

### Full installation in local

the full installation requires:
- Linux
- CUDA 12
- Python 3.10

This is because of flash attention, which is defined in the `requirements.txt` using an URL to download a prebuilt wheel (python bindings for a native library)

```bash
./setup.sh
```

### Degraded installation in local

If you cannot meet the requirements, you can:

- solution 1: fix requirements.txt to use another prebuilt wheel
- solution 2: manually build/install flash attention
- solution 3: don't use clip captioning

Here is how to do solution 3:
```bash
./setup_no_captions.sh
```

## Run

### Running the Gradio app

Note: please make sure you properly define the environment variables for `STORAGE_PATH` (eg. `/data/`) and `HF_HOME` (eg. `/data/huggingface/`)

```bash
python app.py
```

### Running locally

See above remarks about the environment variable.

By default `run.sh` will store stuff in `.data/` (located inside the current working directory):

```bash
./run.sh
```