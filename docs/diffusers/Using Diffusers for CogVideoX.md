[](#cogvideox)CogVideoX
=======================

![LoRA](https://img.shields.io/badge/LoRA-d8b4fe?style=flat)

[CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) from Tsinghua University & ZhipuAI, by Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, Jie Tang.

The abstract from the paper is:

_We introduce CogVideoX, a large-scale diffusion transformer model designed for generating videos based on text prompts. To efficently model video data, we propose to levearge a 3D Variational Autoencoder (VAE) to compresses videos along both spatial and temporal dimensions. To improve the text-video alignment, we propose an expert transformer with the expert adaptive LayerNorm to facilitate the deep fusion between the two modalities. By employing a progressive training technique, CogVideoX is adept at producing coherent, long-duration videos characterized by significant motion. In addition, we develop an effectively text-video data processing pipeline that includes various data preprocessing strategies and a video captioning method. It significantly helps enhance the performance of CogVideoX, improving both generation quality and semantic alignment. Results show that CogVideoX demonstrates state-of-the-art performance across both multiple machine metrics and human evaluations. The model weight of CogVideoX-2B is publicly available at [https://github.com/THUDM/CogVideo](https://github.com/THUDM/CogVideo)._

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

This pipeline was contributed by [zRzRzRzRzRzRzR](https://github.com/zRzRzRzRzRzRzR). The original codebase can be found [here](https://huggingface.co/THUDM). The original weights can be found under [hf.co/THUDM](https://huggingface.co/THUDM).

There are three official CogVideoX checkpoints for text-to-video and video-to-video.

checkpoints

recommended inference dtype

[`THUDM/CogVideoX-2b`](https://huggingface.co/THUDM/CogVideoX-2b)

torch.float16

[`THUDM/CogVideoX-5b`](https://huggingface.co/THUDM/CogVideoX-5b)

torch.bfloat16

[`THUDM/CogVideoX1.5-5b`](https://huggingface.co/THUDM/CogVideoX1.5-5b)

torch.bfloat16

There are two official CogVideoX checkpoints available for image-to-video.

checkpoints

recommended inference dtype

[`THUDM/CogVideoX-5b-I2V`](https://huggingface.co/THUDM/CogVideoX-5b-I2V)

torch.bfloat16

[`THUDM/CogVideoX-1.5-5b-I2V`](https://huggingface.co/THUDM/CogVideoX-1.5-5b-I2V)

torch.bfloat16

For the CogVideoX 1.5 series:

*   Text-to-video (T2V) works best at a resolution of 1360x768 because it was trained with that specific resolution.
*   Image-to-video (I2V) works for multiple resolutions. The width can vary from 768 to 1360, but the height must be 768. The height/width must be divisible by 16.
*   Both T2V and I2V models support generation with 81 and 161 frames and work best at this value. Exporting videos at 16 FPS is recommended.

There are two official CogVideoX checkpoints that support pose controllable generation (by the [Alibaba-PAI](https://huggingface.co/alibaba-pai) team).

checkpoints

recommended inference dtype

[`alibaba-pai/CogVideoX-Fun-V1.1-2b-Pose`](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-Pose)

torch.bfloat16

[`alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose`](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose)

torch.bfloat16

[](#inference)Inference
-----------------------

Use [`torch.compile`](https://huggingface.co/docs/diffusers/main/en/tutorials/fast_diffusion#torchcompile) to reduce the inference latency.

First, load the pipeline:

Copied

import torch
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
from diffusers.utils import export\_to\_video,load\_image
pipe = CogVideoXPipeline.from\_pretrained("THUDM/CogVideoX-5b").to("cuda") \# or "THUDM/CogVideoX-2b" 

If you are using the image-to-video pipeline, load it as follows:

Copied

pipe = CogVideoXImageToVideoPipeline.from\_pretrained("THUDM/CogVideoX-5b-I2V").to("cuda")

Then change the memory layout of the pipelines `transformer` component to `torch.channels_last`:

Copied

pipe.transformer.to(memory\_format=torch.channels\_last)

Compile the components and run inference:

Copied

pipe.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)

\# CogVideoX works well with long and well-described prompts
prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
video = pipe(prompt=prompt, guidance\_scale=6, num\_inference\_steps=50).frames\[0\]

The [T2V benchmark](https://gist.github.com/a-r-r-o-w/5183d75e452a368fd17448fcc810bd3f) results on an 80GB A100 machine are:

Copied

Without torch.compile(): Average inference time: 96.89 seconds.
With torch.compile(): Average inference time: 76.27 seconds.

### [](#memory-optimization)Memory optimization

CogVideoX-2b requires about 19 GB of GPU memory to decode 49 frames (6 seconds of video at 8 FPS) with output resolution 720x480 (W x H), which makes it not possible to run on consumer GPUs or free-tier T4 Colab. The following memory optimizations could be used to reduce the memory footprint. For replication, you can refer to [this](https://gist.github.com/a-r-r-o-w/3959a03f15be5c9bd1fe545b09dfcc93) script.

*   `pipe.enable_model_cpu_offload()`:
    *   Without enabling cpu offloading, memory usage is `33 GB`
    *   With enabling cpu offloading, memory usage is `19 GB`
*   `pipe.enable_sequential_cpu_offload()`:
    *   Similar to `enable_model_cpu_offload` but can significantly reduce memory usage at the cost of slow inference
    *   When enabled, memory usage is under `4 GB`
*   `pipe.vae.enable_tiling()`:
    *   With enabling cpu offloading and tiling, memory usage is `11 GB`
*   `pipe.vae.enable_slicing()`

[](#quantization)Quantization
-----------------------------

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [CogVideoXPipeline](/docs/diffusers/main/en/api/pipelines/cogvideox#diffusers.CogVideoXPipeline) for inference with bitsandbytes.

Copied

import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, CogVideoXTransformer3DModel, CogVideoXPipeline
from diffusers.utils import export\_to\_video
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel

quant\_config = BitsAndBytesConfig(load\_in\_8bit=True)
text\_encoder\_8bit = T5EncoderModel.from\_pretrained(
    "THUDM/CogVideoX-2b",
    subfolder="text\_encoder",
    quantization\_config=quant\_config,
    torch\_dtype=torch.float16,
)

quant\_config = DiffusersBitsAndBytesConfig(load\_in\_8bit=True)
transformer\_8bit = CogVideoXTransformer3DModel.from\_pretrained(
    "THUDM/CogVideoX-2b",
    subfolder="transformer",
    quantization\_config=quant\_config,
    torch\_dtype=torch.float16,
)

pipeline = CogVideoXPipeline.from\_pretrained(
    "THUDM/CogVideoX-2b",
    text\_encoder=text\_encoder\_8bit,
    transformer=transformer\_8bit,
    torch\_dtype=torch.float16,
    device\_map="balanced",
)

prompt = "A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting."
video = pipeline(prompt=prompt, guidance\_scale=6, num\_inference\_steps=50).frames\[0\]
export\_to\_video(video, "ship.mp4", fps=8)

[](#diffusers.CogVideoXPipeline)CogVideoXPipeline
-------------------------------------------------

### class diffusers.CogVideoXPipeline

[](#diffusers.CogVideoXPipeline)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py#L147)

( tokenizer: T5Tokenizertext\_encoder: T5EncoderModelvae: AutoencoderKLCogVideoXtransformer: CogVideoXTransformer3DModelscheduler: typing.Union\[diffusers.schedulers.scheduling\_ddim\_cogvideox.CogVideoXDDIMScheduler, diffusers.schedulers.scheduling\_dpm\_cogvideox.CogVideoXDPMScheduler\] )

Parameters

*   [](#diffusers.CogVideoXPipeline.vae)**vae** ([AutoencoderKL](/docs/diffusers/main/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) — Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
*   [](#diffusers.CogVideoXPipeline.text_encoder)**text\_encoder** (`T5EncoderModel`) — Frozen text-encoder. CogVideoX uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the [t5-v1\_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
*   [](#diffusers.CogVideoXPipeline.tokenizer)**tokenizer** (`T5Tokenizer`) — Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
*   [](#diffusers.CogVideoXPipeline.transformer)**transformer** ([CogVideoXTransformer3DModel](/docs/diffusers/main/en/api/models/cogvideox_transformer3d#diffusers.CogVideoXTransformer3DModel)) — A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
*   [](#diffusers.CogVideoXPipeline.scheduler)**scheduler** ([SchedulerMixin](/docs/diffusers/main/en/api/schedulers/overview#diffusers.SchedulerMixin)) — A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

Pipeline for text-to-video generation using CogVideoX.

This model inherits from [DiffusionPipeline](/docs/diffusers/main/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### \_\_call\_\_

[](#diffusers.CogVideoXPipeline.__call__)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py#L505)

( prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Nonenegative\_prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Noneheight: typing.Optional\[int\] = Nonewidth: typing.Optional\[int\] = Nonenum\_frames: typing.Optional\[int\] = Nonenum\_inference\_steps: int = 50timesteps: typing.Optional\[typing.List\[int\]\] = Noneguidance\_scale: float = 6use\_dynamic\_cfg: bool = Falsenum\_videos\_per\_prompt: int = 1eta: float = 0.0generator: typing.Union\[torch.\_C.Generator, typing.List\[torch.\_C.Generator\], NoneType\] = Nonelatents: typing.Optional\[torch.FloatTensor\] = Noneprompt\_embeds: typing.Optional\[torch.FloatTensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.FloatTensor\] = Noneoutput\_type: str = 'pil'return\_dict: bool = Trueattention\_kwargs: typing.Optional\[typing.Dict\[str, typing.Any\]\] = Nonecallback\_on\_step\_end: typing.Union\[typing.Callable\[\[int, int, typing.Dict\], NoneType\], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType\] = Nonecallback\_on\_step\_end\_tensor\_inputs: typing.List\[str\] = \['latents'\]max\_sequence\_length: int = 226 ) → export const metadata = 'undefined';[CogVideoXPipelineOutput](/docs/diffusers/main/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) or `tuple`

Expand 19 parameters

Parameters

*   [](#diffusers.CogVideoXPipeline.__call__.prompt)**prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.
*   [](#diffusers.CogVideoXPipeline.__call__.negative_prompt)**negative\_prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
*   [](#diffusers.CogVideoXPipeline.__call__.height)**height** (`int`, _optional_, defaults to self.transformer.config.sample\_height \* self.vae\_scale\_factor\_spatial) — The height in pixels of the generated image. This is set to 480 by default for the best results.
*   [](#diffusers.CogVideoXPipeline.__call__.width)**width** (`int`, _optional_, defaults to self.transformer.config.sample\_height \* self.vae\_scale\_factor\_spatial) — The width in pixels of the generated image. This is set to 720 by default for the best results.
*   [](#diffusers.CogVideoXPipeline.__call__.num_frames)**num\_frames** (`int`, defaults to `48`) — Number of frames to generate. Must be divisible by self.vae\_scale\_factor\_temporal. Generated video will contain 1 extra frame because CogVideoX is conditioned with (num\_seconds \* fps + 1) frames where num\_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that needs to be satisfied is that of divisibility mentioned above.
*   [](#diffusers.CogVideoXPipeline.__call__.num_inference_steps)**num\_inference\_steps** (`int`, _optional_, defaults to 50) — The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
*   [](#diffusers.CogVideoXPipeline.__call__.timesteps)**timesteps** (`List[int]`, _optional_) — Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.
*   [](#diffusers.CogVideoXPipeline.__call__.guidance_scale)**guidance\_scale** (`float`, _optional_, defaults to 7.0) — Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
*   [](#diffusers.CogVideoXPipeline.__call__.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — The number of videos to generate per prompt.
*   [](#diffusers.CogVideoXPipeline.__call__.generator)**generator** (`torch.Generator` or `List[torch.Generator]`, _optional_) — One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.
*   [](#diffusers.CogVideoXPipeline.__call__.latents)**latents** (`torch.FloatTensor`, _optional_) — Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will ge generated by sampling using the supplied random `generator`.
*   [](#diffusers.CogVideoXPipeline.__call__.prompt_embeds)**prompt\_embeds** (`torch.FloatTensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.
*   [](#diffusers.CogVideoXPipeline.__call__.negative_prompt_embeds)**negative\_prompt\_embeds** (`torch.FloatTensor`, _optional_) — Pre-generated negative text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, negative\_prompt\_embeds will be generated from `negative_prompt` input argument.
*   [](#diffusers.CogVideoXPipeline.__call__.output_type)**output\_type** (`str`, _optional_, defaults to `"pil"`) — The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
*   [](#diffusers.CogVideoXPipeline.__call__.return_dict)**return\_dict** (`bool`, _optional_, defaults to `True`) — Whether or not to return a `~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` instead of a plain tuple.
*   [](#diffusers.CogVideoXPipeline.__call__.attention_kwargs)**attention\_kwargs** (`dict`, _optional_) — A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
*   [](#diffusers.CogVideoXPipeline.__call__.callback_on_step_end)**callback\_on\_step\_end** (`Callable`, _optional_) — A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
*   [](#diffusers.CogVideoXPipeline.__call__.callback_on_step_end_tensor_inputs)**callback\_on\_step\_end\_tensor\_inputs** (`List`, _optional_) — The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.
*   [](#diffusers.CogVideoXPipeline.__call__.max_sequence_length)**max\_sequence\_length** (`int`, defaults to `226`) — Maximum sequence length in encoded prompt. Must be consistent with `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

Returns

export const metadata = 'undefined';

[CogVideoXPipelineOutput](/docs/diffusers/main/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) or `tuple`

export const metadata = 'undefined';

[CogVideoXPipelineOutput](/docs/diffusers/main/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) if `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

[](#diffusers.CogVideoXPipeline.__call__.example)

Examples:

Copied

\>>> import torch
\>>> from diffusers import CogVideoXPipeline
\>>> from diffusers.utils import export\_to\_video

\>>> \# Models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
\>>> pipe = CogVideoXPipeline.from\_pretrained("THUDM/CogVideoX-2b", torch\_dtype=torch.float16).to("cuda")
\>>> prompt = (
...     "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
...     "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
...     "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
...     "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
...     "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
...     "atmosphere of this unique musical performance."
... )
\>>> video = pipe(prompt=prompt, guidance\_scale=6, num\_inference\_steps=50).frames\[0\]
\>>> export\_to\_video(video, "output.mp4", fps=8)

#### encode\_prompt

[](#diffusers.CogVideoXPipeline.encode_prompt)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py#L244)

( prompt: typing.Union\[str, typing.List\[str\]\]negative\_prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Nonedo\_classifier\_free\_guidance: bool = Truenum\_videos\_per\_prompt: int = 1prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonemax\_sequence\_length: int = 226device: typing.Optional\[torch.device\] = Nonedtype: typing.Optional\[torch.dtype\] = None )

Parameters

*   [](#diffusers.CogVideoXPipeline.encode_prompt.prompt)**prompt** (`str` or `List[str]`, _optional_) — prompt to be encoded
*   [](#diffusers.CogVideoXPipeline.encode_prompt.negative_prompt)**negative\_prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
*   [](#diffusers.CogVideoXPipeline.encode_prompt.do_classifier_free_guidance)**do\_classifier\_free\_guidance** (`bool`, _optional_, defaults to `True`) — Whether to use classifier free guidance or not.
*   [](#diffusers.CogVideoXPipeline.encode_prompt.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
*   [](#diffusers.CogVideoXPipeline.encode_prompt.prompt_embeds)**prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.
*   [](#diffusers.CogVideoXPipeline.encode_prompt.negative_prompt_embeds)**negative\_prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated negative text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, negative\_prompt\_embeds will be generated from `negative_prompt` input argument.
*   [](#diffusers.CogVideoXPipeline.encode_prompt.device)**device** — (`torch.device`, _optional_): torch device
*   [](#diffusers.CogVideoXPipeline.encode_prompt.dtype)**dtype** — (`torch.dtype`, _optional_): torch dtype

Encodes the prompt into text encoder hidden states.

#### fuse\_qkv\_projections

[](#diffusers.CogVideoXPipeline.fuse_qkv_projections)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py#L428)

( )

Enables fused QKV projections.

#### unfuse\_qkv\_projections

[](#diffusers.CogVideoXPipeline.unfuse_qkv_projections)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py#L433)

( )

Disable QKV projection fusion if enabled.

[](#diffusers.CogVideoXImageToVideoPipeline)CogVideoXImageToVideoPipeline
-------------------------------------------------------------------------

### class diffusers.CogVideoXImageToVideoPipeline

[](#diffusers.CogVideoXImageToVideoPipeline)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py#L164)

( tokenizer: T5Tokenizertext\_encoder: T5EncoderModelvae: AutoencoderKLCogVideoXtransformer: CogVideoXTransformer3DModelscheduler: typing.Union\[diffusers.schedulers.scheduling\_ddim\_cogvideox.CogVideoXDDIMScheduler, diffusers.schedulers.scheduling\_dpm\_cogvideox.CogVideoXDPMScheduler\] )

Parameters

*   [](#diffusers.CogVideoXImageToVideoPipeline.vae)**vae** ([AutoencoderKL](/docs/diffusers/main/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) — Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
*   [](#diffusers.CogVideoXImageToVideoPipeline.text_encoder)**text\_encoder** (`T5EncoderModel`) — Frozen text-encoder. CogVideoX uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the [t5-v1\_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
*   [](#diffusers.CogVideoXImageToVideoPipeline.tokenizer)**tokenizer** (`T5Tokenizer`) — Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
*   [](#diffusers.CogVideoXImageToVideoPipeline.transformer)**transformer** ([CogVideoXTransformer3DModel](/docs/diffusers/main/en/api/models/cogvideox_transformer3d#diffusers.CogVideoXTransformer3DModel)) — A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
*   [](#diffusers.CogVideoXImageToVideoPipeline.scheduler)**scheduler** ([SchedulerMixin](/docs/diffusers/main/en/api/schedulers/overview#diffusers.SchedulerMixin)) — A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

Pipeline for image-to-video generation using CogVideoX.

This model inherits from [DiffusionPipeline](/docs/diffusers/main/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### \_\_call\_\_

[](#diffusers.CogVideoXImageToVideoPipeline.__call__)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py#L602)

( image: typing.Union\[PIL.Image.Image, numpy.ndarray, torch.Tensor, typing.List\[PIL.Image.Image\], typing.List\[numpy.ndarray\], typing.List\[torch.Tensor\]\]prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Nonenegative\_prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Noneheight: typing.Optional\[int\] = Nonewidth: typing.Optional\[int\] = Nonenum\_frames: int = 49num\_inference\_steps: int = 50timesteps: typing.Optional\[typing.List\[int\]\] = Noneguidance\_scale: float = 6use\_dynamic\_cfg: bool = Falsenum\_videos\_per\_prompt: int = 1eta: float = 0.0generator: typing.Union\[torch.\_C.Generator, typing.List\[torch.\_C.Generator\], NoneType\] = Nonelatents: typing.Optional\[torch.FloatTensor\] = Noneprompt\_embeds: typing.Optional\[torch.FloatTensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.FloatTensor\] = Noneoutput\_type: str = 'pil'return\_dict: bool = Trueattention\_kwargs: typing.Optional\[typing.Dict\[str, typing.Any\]\] = Nonecallback\_on\_step\_end: typing.Union\[typing.Callable\[\[int, int, typing.Dict\], NoneType\], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType\] = Nonecallback\_on\_step\_end\_tensor\_inputs: typing.List\[str\] = \['latents'\]max\_sequence\_length: int = 226 ) → export const metadata = 'undefined';[CogVideoXPipelineOutput](/docs/diffusers/main/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) or `tuple`

Expand 20 parameters

Parameters

*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.image)**image** (`PipelineImageInput`) — The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.prompt)**prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.negative_prompt)**negative\_prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.height)**height** (`int`, _optional_, defaults to self.transformer.config.sample\_height \* self.vae\_scale\_factor\_spatial) — The height in pixels of the generated image. This is set to 480 by default for the best results.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.width)**width** (`int`, _optional_, defaults to self.transformer.config.sample\_height \* self.vae\_scale\_factor\_spatial) — The width in pixels of the generated image. This is set to 720 by default for the best results.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.num_frames)**num\_frames** (`int`, defaults to `48`) — Number of frames to generate. Must be divisible by self.vae\_scale\_factor\_temporal. Generated video will contain 1 extra frame because CogVideoX is conditioned with (num\_seconds \* fps + 1) frames where num\_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that needs to be satisfied is that of divisibility mentioned above.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.num_inference_steps)**num\_inference\_steps** (`int`, _optional_, defaults to 50) — The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.timesteps)**timesteps** (`List[int]`, _optional_) — Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.guidance_scale)**guidance\_scale** (`float`, _optional_, defaults to 7.0) — Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — The number of videos to generate per prompt.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.generator)**generator** (`torch.Generator` or `List[torch.Generator]`, _optional_) — One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.latents)**latents** (`torch.FloatTensor`, _optional_) — Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will ge generated by sampling using the supplied random `generator`.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.prompt_embeds)**prompt\_embeds** (`torch.FloatTensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.negative_prompt_embeds)**negative\_prompt\_embeds** (`torch.FloatTensor`, _optional_) — Pre-generated negative text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, negative\_prompt\_embeds will be generated from `negative_prompt` input argument.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.output_type)**output\_type** (`str`, _optional_, defaults to `"pil"`) — The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.return_dict)**return\_dict** (`bool`, _optional_, defaults to `True`) — Whether or not to return a `~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` instead of a plain tuple.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.attention_kwargs)**attention\_kwargs** (`dict`, _optional_) — A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.callback_on_step_end)**callback\_on\_step\_end** (`Callable`, _optional_) — A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.callback_on_step_end_tensor_inputs)**callback\_on\_step\_end\_tensor\_inputs** (`List`, _optional_) — The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.
*   [](#diffusers.CogVideoXImageToVideoPipeline.__call__.max_sequence_length)**max\_sequence\_length** (`int`, defaults to `226`) — Maximum sequence length in encoded prompt. Must be consistent with `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

Returns

export const metadata = 'undefined';

[CogVideoXPipelineOutput](/docs/diffusers/main/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) or `tuple`

export const metadata = 'undefined';

[CogVideoXPipelineOutput](/docs/diffusers/main/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) if `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

[](#diffusers.CogVideoXImageToVideoPipeline.__call__.example)

Examples:

Copied

\>>> import torch
\>>> from diffusers import CogVideoXImageToVideoPipeline
\>>> from diffusers.utils import export\_to\_video, load\_image

\>>> pipe = CogVideoXImageToVideoPipeline.from\_pretrained("THUDM/CogVideoX-5b-I2V", torch\_dtype=torch.bfloat16)
\>>> pipe.to("cuda")

\>>> prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
\>>> image = load\_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
... )
\>>> video = pipe(image, prompt, use\_dynamic\_cfg=True)
\>>> export\_to\_video(video.frames\[0\], "output.mp4", fps=8)

#### encode\_prompt

[](#diffusers.CogVideoXImageToVideoPipeline.encode_prompt)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py#L267)

( prompt: typing.Union\[str, typing.List\[str\]\]negative\_prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Nonedo\_classifier\_free\_guidance: bool = Truenum\_videos\_per\_prompt: int = 1prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonemax\_sequence\_length: int = 226device: typing.Optional\[torch.device\] = Nonedtype: typing.Optional\[torch.dtype\] = None )

Parameters

*   [](#diffusers.CogVideoXImageToVideoPipeline.encode_prompt.prompt)**prompt** (`str` or `List[str]`, _optional_) — prompt to be encoded
*   [](#diffusers.CogVideoXImageToVideoPipeline.encode_prompt.negative_prompt)**negative\_prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
*   [](#diffusers.CogVideoXImageToVideoPipeline.encode_prompt.do_classifier_free_guidance)**do\_classifier\_free\_guidance** (`bool`, _optional_, defaults to `True`) — Whether to use classifier free guidance or not.
*   [](#diffusers.CogVideoXImageToVideoPipeline.encode_prompt.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
*   [](#diffusers.CogVideoXImageToVideoPipeline.encode_prompt.prompt_embeds)**prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.
*   [](#diffusers.CogVideoXImageToVideoPipeline.encode_prompt.negative_prompt_embeds)**negative\_prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated negative text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, negative\_prompt\_embeds will be generated from `negative_prompt` input argument.
*   [](#diffusers.CogVideoXImageToVideoPipeline.encode_prompt.device)**device** — (`torch.device`, _optional_): torch device
*   [](#diffusers.CogVideoXImageToVideoPipeline.encode_prompt.dtype)**dtype** — (`torch.dtype`, _optional_): torch dtype

Encodes the prompt into text encoder hidden states.

#### fuse\_qkv\_projections

[](#diffusers.CogVideoXImageToVideoPipeline.fuse_qkv_projections)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py#L523)

( )

Enables fused QKV projections.

#### unfuse\_qkv\_projections

[](#diffusers.CogVideoXImageToVideoPipeline.unfuse_qkv_projections)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py#L529)

( )

Disable QKV projection fusion if enabled.

[](#diffusers.CogVideoXVideoToVideoPipeline)CogVideoXVideoToVideoPipeline
-------------------------------------------------------------------------

### class diffusers.CogVideoXVideoToVideoPipeline

[](#diffusers.CogVideoXVideoToVideoPipeline)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_video2video.py#L169)

( tokenizer: T5Tokenizertext\_encoder: T5EncoderModelvae: AutoencoderKLCogVideoXtransformer: CogVideoXTransformer3DModelscheduler: typing.Union\[diffusers.schedulers.scheduling\_ddim\_cogvideox.CogVideoXDDIMScheduler, diffusers.schedulers.scheduling\_dpm\_cogvideox.CogVideoXDPMScheduler\] )

Parameters

*   [](#diffusers.CogVideoXVideoToVideoPipeline.vae)**vae** ([AutoencoderKL](/docs/diffusers/main/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) — Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.text_encoder)**text\_encoder** (`T5EncoderModel`) — Frozen text-encoder. CogVideoX uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the [t5-v1\_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.tokenizer)**tokenizer** (`T5Tokenizer`) — Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
*   [](#diffusers.CogVideoXVideoToVideoPipeline.transformer)**transformer** ([CogVideoXTransformer3DModel](/docs/diffusers/main/en/api/models/cogvideox_transformer3d#diffusers.CogVideoXTransformer3DModel)) — A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.scheduler)**scheduler** ([SchedulerMixin](/docs/diffusers/main/en/api/schedulers/overview#diffusers.SchedulerMixin)) — A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

Pipeline for video-to-video generation using CogVideoX.

This model inherits from [DiffusionPipeline](/docs/diffusers/main/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### \_\_call\_\_

[](#diffusers.CogVideoXVideoToVideoPipeline.__call__)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_video2video.py#L575)

( video: typing.List\[PIL.Image.Image\] = Noneprompt: typing.Union\[str, typing.List\[str\], NoneType\] = Nonenegative\_prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Noneheight: typing.Optional\[int\] = Nonewidth: typing.Optional\[int\] = Nonenum\_inference\_steps: int = 50timesteps: typing.Optional\[typing.List\[int\]\] = Nonestrength: float = 0.8guidance\_scale: float = 6use\_dynamic\_cfg: bool = Falsenum\_videos\_per\_prompt: int = 1eta: float = 0.0generator: typing.Union\[torch.\_C.Generator, typing.List\[torch.\_C.Generator\], NoneType\] = Nonelatents: typing.Optional\[torch.FloatTensor\] = Noneprompt\_embeds: typing.Optional\[torch.FloatTensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.FloatTensor\] = Noneoutput\_type: str = 'pil'return\_dict: bool = Trueattention\_kwargs: typing.Optional\[typing.Dict\[str, typing.Any\]\] = Nonecallback\_on\_step\_end: typing.Union\[typing.Callable\[\[int, int, typing.Dict\], NoneType\], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType\] = Nonecallback\_on\_step\_end\_tensor\_inputs: typing.List\[str\] = \['latents'\]max\_sequence\_length: int = 226 ) → export const metadata = 'undefined';[CogVideoXPipelineOutput](/docs/diffusers/main/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) or `tuple`

Expand 20 parameters

Parameters

*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.video)**video** (`List[PIL.Image.Image]`) — The input video to condition the generation on. Must be a list of images/frames of the video.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.prompt)**prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.negative_prompt)**negative\_prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.height)**height** (`int`, _optional_, defaults to self.transformer.config.sample\_height \* self.vae\_scale\_factor\_spatial) — The height in pixels of the generated image. This is set to 480 by default for the best results.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.width)**width** (`int`, _optional_, defaults to self.transformer.config.sample\_height \* self.vae\_scale\_factor\_spatial) — The width in pixels of the generated image. This is set to 720 by default for the best results.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.num_inference_steps)**num\_inference\_steps** (`int`, _optional_, defaults to 50) — The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.timesteps)**timesteps** (`List[int]`, _optional_) — Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.strength)**strength** (`float`, _optional_, defaults to 0.8) — Higher strength leads to more differences between original video and generated video.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.guidance_scale)**guidance\_scale** (`float`, _optional_, defaults to 7.0) — Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — The number of videos to generate per prompt.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.generator)**generator** (`torch.Generator` or `List[torch.Generator]`, _optional_) — One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.latents)**latents** (`torch.FloatTensor`, _optional_) — Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will ge generated by sampling using the supplied random `generator`.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.prompt_embeds)**prompt\_embeds** (`torch.FloatTensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.negative_prompt_embeds)**negative\_prompt\_embeds** (`torch.FloatTensor`, _optional_) — Pre-generated negative text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, negative\_prompt\_embeds will be generated from `negative_prompt` input argument.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.output_type)**output\_type** (`str`, _optional_, defaults to `"pil"`) — The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.return_dict)**return\_dict** (`bool`, _optional_, defaults to `True`) — Whether or not to return a `~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` instead of a plain tuple.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.attention_kwargs)**attention\_kwargs** (`dict`, _optional_) — A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.callback_on_step_end)**callback\_on\_step\_end** (`Callable`, _optional_) — A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.callback_on_step_end_tensor_inputs)**callback\_on\_step\_end\_tensor\_inputs** (`List`, _optional_) — The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.__call__.max_sequence_length)**max\_sequence\_length** (`int`, defaults to `226`) — Maximum sequence length in encoded prompt. Must be consistent with `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

Returns

export const metadata = 'undefined';

[CogVideoXPipelineOutput](/docs/diffusers/main/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) or `tuple`

export const metadata = 'undefined';

[CogVideoXPipelineOutput](/docs/diffusers/main/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) if `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

[](#diffusers.CogVideoXVideoToVideoPipeline.__call__.example)

Examples:

Copied

\>>> import torch
\>>> from diffusers import CogVideoXDPMScheduler, CogVideoXVideoToVideoPipeline
\>>> from diffusers.utils import export\_to\_video, load\_video

\>>> \# Models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
\>>> pipe = CogVideoXVideoToVideoPipeline.from\_pretrained("THUDM/CogVideoX-5b", torch\_dtype=torch.bfloat16)
\>>> pipe.to("cuda")
\>>> pipe.scheduler = CogVideoXDPMScheduler.from\_config(pipe.scheduler.config)

\>>> input\_video = load\_video(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
... )
\>>> prompt = (
...     "An astronaut stands triumphantly at the peak of a towering mountain. Panorama of rugged peaks and "
...     "valleys. Very futuristic vibe and animated aesthetic. Highlights of purple and golden colors in "
...     "the scene. The sky is looks like an animated/cartoonish dream of galaxies, nebulae, stars, planets, "
...     "moons, but the remainder of the scene is mostly realistic."
... )

\>>> video = pipe(
...     video=input\_video, prompt=prompt, strength=0.8, guidance\_scale=6, num\_inference\_steps=50
... ).frames\[0\]
\>>> export\_to\_video(video, "output.mp4", fps=8)

#### encode\_prompt

[](#diffusers.CogVideoXVideoToVideoPipeline.encode_prompt)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_video2video.py#L269)

( prompt: typing.Union\[str, typing.List\[str\]\]negative\_prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Nonedo\_classifier\_free\_guidance: bool = Truenum\_videos\_per\_prompt: int = 1prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonemax\_sequence\_length: int = 226device: typing.Optional\[torch.device\] = Nonedtype: typing.Optional\[torch.dtype\] = None )

Parameters

*   [](#diffusers.CogVideoXVideoToVideoPipeline.encode_prompt.prompt)**prompt** (`str` or `List[str]`, _optional_) — prompt to be encoded
*   [](#diffusers.CogVideoXVideoToVideoPipeline.encode_prompt.negative_prompt)**negative\_prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
*   [](#diffusers.CogVideoXVideoToVideoPipeline.encode_prompt.do_classifier_free_guidance)**do\_classifier\_free\_guidance** (`bool`, _optional_, defaults to `True`) — Whether to use classifier free guidance or not.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.encode_prompt.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
*   [](#diffusers.CogVideoXVideoToVideoPipeline.encode_prompt.prompt_embeds)**prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.encode_prompt.negative_prompt_embeds)**negative\_prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated negative text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, negative\_prompt\_embeds will be generated from `negative_prompt` input argument.
*   [](#diffusers.CogVideoXVideoToVideoPipeline.encode_prompt.device)**device** — (`torch.device`, _optional_): torch device
*   [](#diffusers.CogVideoXVideoToVideoPipeline.encode_prompt.dtype)**dtype** — (`torch.dtype`, _optional_): torch dtype

Encodes the prompt into text encoder hidden states.

#### fuse\_qkv\_projections

[](#diffusers.CogVideoXVideoToVideoPipeline.fuse_qkv_projections)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_video2video.py#L496)

( )

Enables fused QKV projections.

#### unfuse\_qkv\_projections

[](#diffusers.CogVideoXVideoToVideoPipeline.unfuse_qkv_projections)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_video2video.py#L502)

( )

Disable QKV projection fusion if enabled.

[](#diffusers.CogVideoXFunControlPipeline)CogVideoXFunControlPipeline
---------------------------------------------------------------------

### class diffusers.CogVideoXFunControlPipeline

[](#diffusers.CogVideoXFunControlPipeline)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py#L154)

( tokenizer: T5Tokenizertext\_encoder: T5EncoderModelvae: AutoencoderKLCogVideoXtransformer: CogVideoXTransformer3DModelscheduler: KarrasDiffusionSchedulers )

Parameters

*   [](#diffusers.CogVideoXFunControlPipeline.vae)**vae** ([AutoencoderKL](/docs/diffusers/main/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) — Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
*   [](#diffusers.CogVideoXFunControlPipeline.text_encoder)**text\_encoder** (`T5EncoderModel`) — Frozen text-encoder. CogVideoX uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the [t5-v1\_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
*   [](#diffusers.CogVideoXFunControlPipeline.tokenizer)**tokenizer** (`T5Tokenizer`) — Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
*   [](#diffusers.CogVideoXFunControlPipeline.transformer)**transformer** ([CogVideoXTransformer3DModel](/docs/diffusers/main/en/api/models/cogvideox_transformer3d#diffusers.CogVideoXTransformer3DModel)) — A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
*   [](#diffusers.CogVideoXFunControlPipeline.scheduler)**scheduler** ([SchedulerMixin](/docs/diffusers/main/en/api/schedulers/overview#diffusers.SchedulerMixin)) — A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

Pipeline for controlled text-to-video generation using CogVideoX Fun.

This model inherits from [DiffusionPipeline](/docs/diffusers/main/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### \_\_call\_\_

[](#diffusers.CogVideoXFunControlPipeline.__call__)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py#L551)

( prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Nonenegative\_prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Nonecontrol\_video: typing.Optional\[typing.List\[PIL.Image.Image\]\] = Noneheight: typing.Optional\[int\] = Nonewidth: typing.Optional\[int\] = Nonenum\_inference\_steps: int = 50timesteps: typing.Optional\[typing.List\[int\]\] = Noneguidance\_scale: float = 6use\_dynamic\_cfg: bool = Falsenum\_videos\_per\_prompt: int = 1eta: float = 0.0generator: typing.Union\[torch.\_C.Generator, typing.List\[torch.\_C.Generator\], NoneType\] = Nonelatents: typing.Optional\[torch.Tensor\] = Nonecontrol\_video\_latents: typing.Optional\[torch.Tensor\] = Noneprompt\_embeds: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.Tensor\] = Noneoutput\_type: str = 'pil'return\_dict: bool = Trueattention\_kwargs: typing.Optional\[typing.Dict\[str, typing.Any\]\] = Nonecallback\_on\_step\_end: typing.Union\[typing.Callable\[\[int, int, typing.Dict\], NoneType\], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType\] = Nonecallback\_on\_step\_end\_tensor\_inputs: typing.List\[str\] = \['latents'\]max\_sequence\_length: int = 226 ) → export const metadata = 'undefined';[CogVideoXPipelineOutput](/docs/diffusers/main/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) or `tuple`

Expand 20 parameters

Parameters

*   [](#diffusers.CogVideoXFunControlPipeline.__call__.prompt)**prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.negative_prompt)**negative\_prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.control_video)**control\_video** (`List[PIL.Image.Image]`) — The control video to condition the generation on. Must be a list of images/frames of the video. If not provided, `control_video_latents` must be provided.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.height)**height** (`int`, _optional_, defaults to self.transformer.config.sample\_height \* self.vae\_scale\_factor\_spatial) — The height in pixels of the generated image. This is set to 480 by default for the best results.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.width)**width** (`int`, _optional_, defaults to self.transformer.config.sample\_height \* self.vae\_scale\_factor\_spatial) — The width in pixels of the generated image. This is set to 720 by default for the best results.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.num_inference_steps)**num\_inference\_steps** (`int`, _optional_, defaults to 50) — The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.timesteps)**timesteps** (`List[int]`, _optional_) — Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.guidance_scale)**guidance\_scale** (`float`, _optional_, defaults to 6.0) — Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — The number of videos to generate per prompt.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.generator)**generator** (`torch.Generator` or `List[torch.Generator]`, _optional_) — One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.latents)**latents** (`torch.Tensor`, _optional_) — Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will ge generated by sampling using the supplied random `generator`.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.control_video_latents)**control\_video\_latents** (`torch.Tensor`, _optional_) — Pre-generated control latents, sampled from a Gaussian distribution, to be used as inputs for controlled video generation. If not provided, `control_video` must be provided.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.prompt_embeds)**prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.negative_prompt_embeds)**negative\_prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated negative text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, negative\_prompt\_embeds will be generated from `negative_prompt` input argument.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.output_type)**output\_type** (`str`, _optional_, defaults to `"pil"`) — The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.return_dict)**return\_dict** (`bool`, _optional_, defaults to `True`) — Whether or not to return a `~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` instead of a plain tuple.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.attention_kwargs)**attention\_kwargs** (`dict`, _optional_) — A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.callback_on_step_end)**callback\_on\_step\_end** (`Callable`, _optional_) — A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.callback_on_step_end_tensor_inputs)**callback\_on\_step\_end\_tensor\_inputs** (`List`, _optional_) — The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.
*   [](#diffusers.CogVideoXFunControlPipeline.__call__.max_sequence_length)**max\_sequence\_length** (`int`, defaults to `226`) — Maximum sequence length in encoded prompt. Must be consistent with `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

Returns

export const metadata = 'undefined';

[CogVideoXPipelineOutput](/docs/diffusers/main/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) or `tuple`

export const metadata = 'undefined';

[CogVideoXPipelineOutput](/docs/diffusers/main/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) if `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

[](#diffusers.CogVideoXFunControlPipeline.__call__.example)

Examples:

Copied

\>>> import torch
\>>> from diffusers import CogVideoXFunControlPipeline, DDIMScheduler
\>>> from diffusers.utils import export\_to\_video, load\_video

\>>> pipe = CogVideoXFunControlPipeline.from\_pretrained(
...     "alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose", torch\_dtype=torch.bfloat16
... )
\>>> pipe.scheduler = DDIMScheduler.from\_config(pipe.scheduler.config)
\>>> pipe.to("cuda")

\>>> control\_video = load\_video(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
... )
\>>> prompt = (
...     "An astronaut stands triumphantly at the peak of a towering mountain. Panorama of rugged peaks and "
...     "valleys. Very futuristic vibe and animated aesthetic. Highlights of purple and golden colors in "
...     "the scene. The sky is looks like an animated/cartoonish dream of galaxies, nebulae, stars, planets, "
...     "moons, but the remainder of the scene is mostly realistic."
... )

\>>> video = pipe(prompt=prompt, control\_video=control\_video).frames\[0\]
\>>> export\_to\_video(video, "output.mp4", fps=8)

#### encode\_prompt

[](#diffusers.CogVideoXFunControlPipeline.encode_prompt)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py#L253)

( prompt: typing.Union\[str, typing.List\[str\]\]negative\_prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Nonedo\_classifier\_free\_guidance: bool = Truenum\_videos\_per\_prompt: int = 1prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonemax\_sequence\_length: int = 226device: typing.Optional\[torch.device\] = Nonedtype: typing.Optional\[torch.dtype\] = None )

Parameters

*   [](#diffusers.CogVideoXFunControlPipeline.encode_prompt.prompt)**prompt** (`str` or `List[str]`, _optional_) — prompt to be encoded
*   [](#diffusers.CogVideoXFunControlPipeline.encode_prompt.negative_prompt)**negative\_prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
*   [](#diffusers.CogVideoXFunControlPipeline.encode_prompt.do_classifier_free_guidance)**do\_classifier\_free\_guidance** (`bool`, _optional_, defaults to `True`) — Whether to use classifier free guidance or not.
*   [](#diffusers.CogVideoXFunControlPipeline.encode_prompt.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
*   [](#diffusers.CogVideoXFunControlPipeline.encode_prompt.prompt_embeds)**prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.
*   [](#diffusers.CogVideoXFunControlPipeline.encode_prompt.negative_prompt_embeds)**negative\_prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated negative text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, negative\_prompt\_embeds will be generated from `negative_prompt` input argument.
*   [](#diffusers.CogVideoXFunControlPipeline.encode_prompt.device)**device** — (`torch.device`, _optional_): torch device
*   [](#diffusers.CogVideoXFunControlPipeline.encode_prompt.dtype)**dtype** — (`torch.dtype`, _optional_): torch dtype

Encodes the prompt into text encoder hidden states.

#### fuse\_qkv\_projections

[](#diffusers.CogVideoXFunControlPipeline.fuse_qkv_projections)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py#L473)

( )

Enables fused QKV projections.

#### unfuse\_qkv\_projections

[](#diffusers.CogVideoXFunControlPipeline.unfuse_qkv_projections)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py#L478)

( )

Disable QKV projection fusion if enabled.

[](#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput)CogVideoXPipelineOutput
------------------------------------------------------------------------------------------------

### class diffusers.pipelines.cogvideo.pipeline\_output.CogVideoXPipelineOutput

[](#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_output.py#L8)

( frames: Tensor )

Parameters

*   [](#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput.frames)**frames** (`torch.Tensor`, `np.ndarray`, or List\[List\[PIL.Image.Image\]\]) — List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for CogVideo pipelines.

[< \> Update on GitHub](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/cogvideox.md)

CogVideoX

[←BLIP-Diffusion](/docs/diffusers/main/en/api/pipelines/blip_diffusion) [CogView3→](/docs/diffusers/main/en/api/pipelines/cogview3)