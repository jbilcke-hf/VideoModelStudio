[](#ltx-video)LTX Video
=======================

![LoRA](https://img.shields.io/badge/LoRA-d8b4fe?style=flat)

[LTX Video](https://huggingface.co/Lightricks/LTX-Video) is the first DiT-based video generation model capable of generating high-quality videos in real-time. It produces 24 FPS videos at a 768x512 resolution faster than they can be watched. Trained on a large-scale dataset of diverse videos, the model generates high-resolution videos with realistic and varied content. We provide a model for both text-to-video as well as image + text-to-video usecases.

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

Available models:

Model name

Recommended dtype

[`LTX Video 0.9.0`](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.safetensors)

`torch.bfloat16`

[`LTX Video 0.9.1`](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors)

`torch.bfloat16`

Note: The recommended dtype is for the transformer component. The VAE and text encoders can be either `torch.float32`, `torch.bfloat16` or `torch.float16` but the recommended dtype is `torch.bfloat16` as used in the original repository.

[](#loading-single-files)Loading Single Files
---------------------------------------------

Loading the original LTX Video checkpoints is also possible with `~ModelMixin.from_single_file`. We recommend using `from_single_file` for the Lightricks series of models, as they plan to release multiple models in the future in the single file format.

Copied

import torch
from diffusers import AutoencoderKLLTXVideo, LTXImageToVideoPipeline, LTXVideoTransformer3DModel

\# \`single\_file\_url\` could also be https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.1.safetensors
single\_file\_url = "https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.safetensors"
transformer = LTXVideoTransformer3DModel.from\_single\_file(
  single\_file\_url, torch\_dtype=torch.bfloat16
)
vae = AutoencoderKLLTXVideo.from\_single\_file(single\_file\_url, torch\_dtype=torch.bfloat16)
pipe = LTXImageToVideoPipeline.from\_pretrained(
  "Lightricks/LTX-Video", transformer=transformer, vae=vae, torch\_dtype=torch.bfloat16
)

\# ... inference code ...

Alternatively, the pipeline can be used to load the weights with `~FromSingleFileMixin.from_single_file`.

Copied

import torch
from diffusers import LTXImageToVideoPipeline
from transformers import T5EncoderModel, T5Tokenizer

single\_file\_url = "https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.safetensors"
text\_encoder = T5EncoderModel.from\_pretrained(
  "Lightricks/LTX-Video", subfolder="text\_encoder", torch\_dtype=torch.bfloat16
)
tokenizer = T5Tokenizer.from\_pretrained(
  "Lightricks/LTX-Video", subfolder="tokenizer", torch\_dtype=torch.bfloat16
)
pipe = LTXImageToVideoPipeline.from\_single\_file(
  single\_file\_url, text\_encoder=text\_encoder, tokenizer=tokenizer, torch\_dtype=torch.bfloat16
)

Loading [LTX GGUF checkpoints](https://huggingface.co/city96/LTX-Video-gguf) are also supported:

Copied

import torch
from diffusers.utils import export\_to\_video
from diffusers import LTXPipeline, LTXVideoTransformer3DModel, GGUFQuantizationConfig

ckpt\_path = (
    "https://huggingface.co/city96/LTX-Video-gguf/blob/main/ltx-video-2b-v0.9-Q3\_K\_S.gguf"
)
transformer = LTXVideoTransformer3DModel.from\_single\_file(
    ckpt\_path,
    quantization\_config=GGUFQuantizationConfig(compute\_dtype=torch.bfloat16),
    torch\_dtype=torch.bfloat16,
)
pipe = LTXPipeline.from\_pretrained(
    "Lightricks/LTX-Video",
    transformer=transformer,
    torch\_dtype=torch.bfloat16,
)
pipe.enable\_model\_cpu\_offload()

prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
negative\_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    prompt=prompt,
    negative\_prompt=negative\_prompt,
    width=704,
    height=480,
    num\_frames=161,
    num\_inference\_steps=50,
).frames\[0\]
export\_to\_video(video, "output\_gguf\_ltx.mp4", fps=24)

Make sure to read the [documentation on GGUF](../../quantization/gguf) to learn more about our GGUF support.

Loading and running inference with [LTX Video 0.9.1](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors) weights.

Copied

import torch
from diffusers import LTXPipeline
from diffusers.utils import export\_to\_video

pipe = LTXPipeline.from\_pretrained("a-r-r-o-w/LTX-Video-0.9.1-diffusers", torch\_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
negative\_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    prompt=prompt,
    negative\_prompt=negative\_prompt,
    width=768,
    height=512,
    num\_frames=161,
    decode\_timestep=0.03,
    decode\_noise\_scale=0.025,
    num\_inference\_steps=50,
).frames\[0\]
export\_to\_video(video, "output.mp4", fps=24)

Refer to [this section](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox#memory-optimization) to learn more about optimizing memory consumption.

[](#quantization)Quantization
-----------------------------

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [LTXPipeline](/docs/diffusers/main/en/api/pipelines/ltx_video#diffusers.LTXPipeline) for inference with bitsandbytes.

Copied

import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, LTXVideoTransformer3DModel, LTXPipeline
from diffusers.utils import export\_to\_video
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel

quant\_config = BitsAndBytesConfig(load\_in\_8bit=True)
text\_encoder\_8bit = T5EncoderModel.from\_pretrained(
    "Lightricks/LTX-Video",
    subfolder="text\_encoder",
    quantization\_config=quant\_config,
    torch\_dtype=torch.float16,
)

quant\_config = DiffusersBitsAndBytesConfig(load\_in\_8bit=True)
transformer\_8bit = LTXVideoTransformer3DModel.from\_pretrained(
    "Lightricks/LTX-Video",
    subfolder="transformer",
    quantization\_config=quant\_config,
    torch\_dtype=torch.float16,
)

pipeline = LTXPipeline.from\_pretrained(
    "Lightricks/LTX-Video",
    text\_encoder=text\_encoder\_8bit,
    transformer=transformer\_8bit,
    torch\_dtype=torch.float16,
    device\_map="balanced",
)

prompt = "A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting."
video = pipeline(prompt=prompt, num\_frames=161, num\_inference\_steps=50).frames\[0\]
export\_to\_video(video, "ship.mp4", fps=24)

[](#diffusers.LTXPipeline)LTXPipeline
-------------------------------------

### class diffusers.LTXPipeline

[](#diffusers.LTXPipeline)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx/pipeline_ltx.py#L143)

( scheduler: FlowMatchEulerDiscreteSchedulervae: AutoencoderKLLTXVideotext\_encoder: T5EncoderModeltokenizer: T5TokenizerFasttransformer: LTXVideoTransformer3DModel )

Parameters

*   [](#diffusers.LTXPipeline.transformer)**transformer** ([LTXVideoTransformer3DModel](/docs/diffusers/main/en/api/models/ltx_video_transformer3d#diffusers.LTXVideoTransformer3DModel)) — Conditional Transformer architecture to denoise the encoded video latents.
*   [](#diffusers.LTXPipeline.scheduler)**scheduler** ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/main/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) — A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
*   [](#diffusers.LTXPipeline.vae)**vae** ([AutoencoderKLLTXVideo](/docs/diffusers/main/en/api/models/autoencoderkl_ltx_video#diffusers.AutoencoderKLLTXVideo)) — Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
*   [](#diffusers.LTXPipeline.text_encoder)**text\_encoder** (`T5EncoderModel`) — [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/t5-v1\_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
*   [](#diffusers.LTXPipeline.tokenizer)**tokenizer** (`CLIPTokenizer`) — Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
*   [](#diffusers.LTXPipeline.tokenizer)**tokenizer** (`T5TokenizerFast`) — Second Tokenizer of class [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).

Pipeline for text-to-video generation.

Reference: [https://github.com/Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video)

#### \_\_call\_\_

[](#diffusers.LTXPipeline.__call__)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx/pipeline_ltx.py#L500)

( prompt: typing.Union\[str, typing.List\[str\]\] = Nonenegative\_prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Noneheight: int = 512width: int = 704num\_frames: int = 161frame\_rate: int = 25num\_inference\_steps: int = 50timesteps: typing.List\[int\] = Noneguidance\_scale: float = 3num\_videos\_per\_prompt: typing.Optional\[int\] = 1generator: typing.Union\[torch.\_C.Generator, typing.List\[torch.\_C.Generator\], NoneType\] = Nonelatents: typing.Optional\[torch.Tensor\] = Noneprompt\_embeds: typing.Optional\[torch.Tensor\] = Noneprompt\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonedecode\_timestep: typing.Union\[float, typing.List\[float\]\] = 0.0decode\_noise\_scale: typing.Union\[float, typing.List\[float\], NoneType\] = Noneoutput\_type: typing.Optional\[str\] = 'pil'return\_dict: bool = Trueattention\_kwargs: typing.Optional\[typing.Dict\[str, typing.Any\]\] = Nonecallback\_on\_step\_end: typing.Optional\[typing.Callable\[\[int, int, typing.Dict\], NoneType\]\] = Nonecallback\_on\_step\_end\_tensor\_inputs: typing.List\[str\] = \['latents'\]max\_sequence\_length: int = 128 ) → export const metadata = 'undefined';`~pipelines.ltx.LTXPipelineOutput` or `tuple`

Expand 22 parameters

Parameters

*   [](#diffusers.LTXPipeline.__call__.prompt)**prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.
*   [](#diffusers.LTXPipeline.__call__.height)**height** (`int`, defaults to `512`) — The height in pixels of the generated image. This is set to 480 by default for the best results.
*   [](#diffusers.LTXPipeline.__call__.width)**width** (`int`, defaults to `704`) — The width in pixels of the generated image. This is set to 848 by default for the best results.
*   [](#diffusers.LTXPipeline.__call__.num_frames)**num\_frames** (`int`, defaults to `161`) — The number of video frames to generate
*   [](#diffusers.LTXPipeline.__call__.num_inference_steps)**num\_inference\_steps** (`int`, _optional_, defaults to 50) — The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
*   [](#diffusers.LTXPipeline.__call__.timesteps)**timesteps** (`List[int]`, _optional_) — Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.
*   [](#diffusers.LTXPipeline.__call__.guidance_scale)**guidance\_scale** (`float`, defaults to `3` ) — Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
*   [](#diffusers.LTXPipeline.__call__.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — The number of videos to generate per prompt.
*   [](#diffusers.LTXPipeline.__call__.generator)**generator** (`torch.Generator` or `List[torch.Generator]`, _optional_) — One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.
*   [](#diffusers.LTXPipeline.__call__.latents)**latents** (`torch.Tensor`, _optional_) — Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will ge generated by sampling using the supplied random `generator`.
*   [](#diffusers.LTXPipeline.__call__.prompt_embeds)**prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.
*   [](#diffusers.LTXPipeline.__call__.prompt_attention_mask)**prompt\_attention\_mask** (`torch.Tensor`, _optional_) — Pre-generated attention mask for text embeddings.
*   [](#diffusers.LTXPipeline.__call__.negative_prompt_embeds)**negative\_prompt\_embeds** (`torch.FloatTensor`, _optional_) — Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not provided, negative\_prompt\_embeds will be generated from `negative_prompt` input argument.
*   [](#diffusers.LTXPipeline.__call__.negative_prompt_attention_mask)**negative\_prompt\_attention\_mask** (`torch.FloatTensor`, _optional_) — Pre-generated attention mask for negative text embeddings.
*   [](#diffusers.LTXPipeline.__call__.decode_timestep)**decode\_timestep** (`float`, defaults to `0.0`) — The timestep at which generated video is decoded.
*   [](#diffusers.LTXPipeline.__call__.decode_noise_scale)**decode\_noise\_scale** (`float`, defaults to `None`) — The interpolation factor between random noise and denoised latents at the decode timestep.
*   [](#diffusers.LTXPipeline.__call__.output_type)**output\_type** (`str`, _optional_, defaults to `"pil"`) — The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
*   [](#diffusers.LTXPipeline.__call__.return_dict)**return\_dict** (`bool`, _optional_, defaults to `True`) — Whether or not to return a `~pipelines.ltx.LTXPipelineOutput` instead of a plain tuple.
*   [](#diffusers.LTXPipeline.__call__.attention_kwargs)**attention\_kwargs** (`dict`, _optional_) — A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
*   [](#diffusers.LTXPipeline.__call__.callback_on_step_end)**callback\_on\_step\_end** (`Callable`, _optional_) — A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
*   [](#diffusers.LTXPipeline.__call__.callback_on_step_end_tensor_inputs)**callback\_on\_step\_end\_tensor\_inputs** (`List`, _optional_) — The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.
*   [](#diffusers.LTXPipeline.__call__.max_sequence_length)**max\_sequence\_length** (`int` defaults to `128` ) — Maximum sequence length to use with the `prompt`.

Returns

export const metadata = 'undefined';

`~pipelines.ltx.LTXPipelineOutput` or `tuple`

export const metadata = 'undefined';

If `return_dict` is `True`, `~pipelines.ltx.LTXPipelineOutput` is returned, otherwise a `tuple` is returned where the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

[](#diffusers.LTXPipeline.__call__.example)

Examples:

Copied

\>>> import torch
\>>> from diffusers import LTXPipeline
\>>> from diffusers.utils import export\_to\_video

\>>> pipe = LTXPipeline.from\_pretrained("Lightricks/LTX-Video", torch\_dtype=torch.bfloat16)
\>>> pipe.to("cuda")

\>>> prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
\>>> negative\_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

\>>> video = pipe(
...     prompt=prompt,
...     negative\_prompt=negative\_prompt,
...     width=704,
...     height=480,
...     num\_frames=161,
...     num\_inference\_steps=50,
... ).frames\[0\]
\>>> export\_to\_video(video, "output.mp4", fps=24)

#### encode\_prompt

[](#diffusers.LTXPipeline.encode_prompt)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx/pipeline_ltx.py#L256)

( prompt: typing.Union\[str, typing.List\[str\]\]negative\_prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Nonedo\_classifier\_free\_guidance: bool = Truenum\_videos\_per\_prompt: int = 1prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.Tensor\] = Noneprompt\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonemax\_sequence\_length: int = 128device: typing.Optional\[torch.device\] = Nonedtype: typing.Optional\[torch.dtype\] = None )

Parameters

*   [](#diffusers.LTXPipeline.encode_prompt.prompt)**prompt** (`str` or `List[str]`, _optional_) — prompt to be encoded
*   [](#diffusers.LTXPipeline.encode_prompt.negative_prompt)**negative\_prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
*   [](#diffusers.LTXPipeline.encode_prompt.do_classifier_free_guidance)**do\_classifier\_free\_guidance** (`bool`, _optional_, defaults to `True`) — Whether to use classifier free guidance or not.
*   [](#diffusers.LTXPipeline.encode_prompt.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
*   [](#diffusers.LTXPipeline.encode_prompt.prompt_embeds)**prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.
*   [](#diffusers.LTXPipeline.encode_prompt.negative_prompt_embeds)**negative\_prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated negative text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, negative\_prompt\_embeds will be generated from `negative_prompt` input argument.
*   [](#diffusers.LTXPipeline.encode_prompt.device)**device** — (`torch.device`, _optional_): torch device
*   [](#diffusers.LTXPipeline.encode_prompt.dtype)**dtype** — (`torch.dtype`, _optional_): torch dtype

Encodes the prompt into text encoder hidden states.

[](#diffusers.LTXImageToVideoPipeline)LTXImageToVideoPipeline
-------------------------------------------------------------

### class diffusers.LTXImageToVideoPipeline

[](#diffusers.LTXImageToVideoPipeline)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx/pipeline_ltx_image2video.py#L162)

( scheduler: FlowMatchEulerDiscreteSchedulervae: AutoencoderKLLTXVideotext\_encoder: T5EncoderModeltokenizer: T5TokenizerFasttransformer: LTXVideoTransformer3DModel )

Parameters

*   [](#diffusers.LTXImageToVideoPipeline.transformer)**transformer** ([LTXVideoTransformer3DModel](/docs/diffusers/main/en/api/models/ltx_video_transformer3d#diffusers.LTXVideoTransformer3DModel)) — Conditional Transformer architecture to denoise the encoded video latents.
*   [](#diffusers.LTXImageToVideoPipeline.scheduler)**scheduler** ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/main/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) — A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
*   [](#diffusers.LTXImageToVideoPipeline.vae)**vae** ([AutoencoderKLLTXVideo](/docs/diffusers/main/en/api/models/autoencoderkl_ltx_video#diffusers.AutoencoderKLLTXVideo)) — Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
*   [](#diffusers.LTXImageToVideoPipeline.text_encoder)**text\_encoder** (`T5EncoderModel`) — [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/t5-v1\_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
*   [](#diffusers.LTXImageToVideoPipeline.tokenizer)**tokenizer** (`CLIPTokenizer`) — Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
*   [](#diffusers.LTXImageToVideoPipeline.tokenizer)**tokenizer** (`T5TokenizerFast`) — Second Tokenizer of class [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).

Pipeline for image-to-video generation.

Reference: [https://github.com/Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video)

#### \_\_call\_\_

[](#diffusers.LTXImageToVideoPipeline.__call__)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx/pipeline_ltx_image2video.py#L559)

( image: typing.Union\[PIL.Image.Image, numpy.ndarray, torch.Tensor, typing.List\[PIL.Image.Image\], typing.List\[numpy.ndarray\], typing.List\[torch.Tensor\]\] = Noneprompt: typing.Union\[str, typing.List\[str\]\] = Nonenegative\_prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Noneheight: int = 512width: int = 704num\_frames: int = 161frame\_rate: int = 25num\_inference\_steps: int = 50timesteps: typing.List\[int\] = Noneguidance\_scale: float = 3num\_videos\_per\_prompt: typing.Optional\[int\] = 1generator: typing.Union\[torch.\_C.Generator, typing.List\[torch.\_C.Generator\], NoneType\] = Nonelatents: typing.Optional\[torch.Tensor\] = Noneprompt\_embeds: typing.Optional\[torch.Tensor\] = Noneprompt\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonedecode\_timestep: typing.Union\[float, typing.List\[float\]\] = 0.0decode\_noise\_scale: typing.Union\[float, typing.List\[float\], NoneType\] = Noneoutput\_type: typing.Optional\[str\] = 'pil'return\_dict: bool = Trueattention\_kwargs: typing.Optional\[typing.Dict\[str, typing.Any\]\] = Nonecallback\_on\_step\_end: typing.Optional\[typing.Callable\[\[int, int, typing.Dict\], NoneType\]\] = Nonecallback\_on\_step\_end\_tensor\_inputs: typing.List\[str\] = \['latents'\]max\_sequence\_length: int = 128 ) → export const metadata = 'undefined';`~pipelines.ltx.LTXPipelineOutput` or `tuple`

Expand 23 parameters

Parameters

*   [](#diffusers.LTXImageToVideoPipeline.__call__.image)**image** (`PipelineImageInput`) — The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.prompt)**prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.height)**height** (`int`, defaults to `512`) — The height in pixels of the generated image. This is set to 480 by default for the best results.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.width)**width** (`int`, defaults to `704`) — The width in pixels of the generated image. This is set to 848 by default for the best results.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.num_frames)**num\_frames** (`int`, defaults to `161`) — The number of video frames to generate
*   [](#diffusers.LTXImageToVideoPipeline.__call__.num_inference_steps)**num\_inference\_steps** (`int`, _optional_, defaults to 50) — The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.timesteps)**timesteps** (`List[int]`, _optional_) — Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.guidance_scale)**guidance\_scale** (`float`, defaults to `3` ) — Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — The number of videos to generate per prompt.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.generator)**generator** (`torch.Generator` or `List[torch.Generator]`, _optional_) — One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.latents)**latents** (`torch.Tensor`, _optional_) — Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will ge generated by sampling using the supplied random `generator`.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.prompt_embeds)**prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.prompt_attention_mask)**prompt\_attention\_mask** (`torch.Tensor`, _optional_) — Pre-generated attention mask for text embeddings.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.negative_prompt_embeds)**negative\_prompt\_embeds** (`torch.FloatTensor`, _optional_) — Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not provided, negative\_prompt\_embeds will be generated from `negative_prompt` input argument.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.negative_prompt_attention_mask)**negative\_prompt\_attention\_mask** (`torch.FloatTensor`, _optional_) — Pre-generated attention mask for negative text embeddings.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.decode_timestep)**decode\_timestep** (`float`, defaults to `0.0`) — The timestep at which generated video is decoded.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.decode_noise_scale)**decode\_noise\_scale** (`float`, defaults to `None`) — The interpolation factor between random noise and denoised latents at the decode timestep.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.output_type)**output\_type** (`str`, _optional_, defaults to `"pil"`) — The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.return_dict)**return\_dict** (`bool`, _optional_, defaults to `True`) — Whether or not to return a `~pipelines.ltx.LTXPipelineOutput` instead of a plain tuple.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.attention_kwargs)**attention\_kwargs** (`dict`, _optional_) — A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
*   [](#diffusers.LTXImageToVideoPipeline.__call__.callback_on_step_end)**callback\_on\_step\_end** (`Callable`, _optional_) — A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.callback_on_step_end_tensor_inputs)**callback\_on\_step\_end\_tensor\_inputs** (`List`, _optional_) — The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.
*   [](#diffusers.LTXImageToVideoPipeline.__call__.max_sequence_length)**max\_sequence\_length** (`int` defaults to `128` ) — Maximum sequence length to use with the `prompt`.

Returns

export const metadata = 'undefined';

`~pipelines.ltx.LTXPipelineOutput` or `tuple`

export const metadata = 'undefined';

If `return_dict` is `True`, `~pipelines.ltx.LTXPipelineOutput` is returned, otherwise a `tuple` is returned where the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

[](#diffusers.LTXImageToVideoPipeline.__call__.example)

Examples:

Copied

\>>> import torch
\>>> from diffusers import LTXImageToVideoPipeline
\>>> from diffusers.utils import export\_to\_video, load\_image

\>>> pipe = LTXImageToVideoPipeline.from\_pretrained("Lightricks/LTX-Video", torch\_dtype=torch.bfloat16)
\>>> pipe.to("cuda")

\>>> image = load\_image(
...     "https://huggingface.co/datasets/a-r-r-o-w/tiny-meme-dataset-captioned/resolve/main/images/8.png"
... )
\>>> prompt = "A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background. Flames engulf the structure, with smoke billowing into the air. Firefighters in protective gear rush to the scene, a fire truck labeled '38' visible behind them. The girl's neutral expression contrasts sharply with the chaos of the fire, creating a poignant and emotionally charged scene."
\>>> negative\_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

\>>> video = pipe(
...     image=image,
...     prompt=prompt,
...     negative\_prompt=negative\_prompt,
...     width=704,
...     height=480,
...     num\_frames=161,
...     num\_inference\_steps=50,
... ).frames\[0\]
\>>> export\_to\_video(video, "output.mp4", fps=24)

#### encode\_prompt

[](#diffusers.LTXImageToVideoPipeline.encode_prompt)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx/pipeline_ltx_image2video.py#L279)

( prompt: typing.Union\[str, typing.List\[str\]\]negative\_prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Nonedo\_classifier\_free\_guidance: bool = Truenum\_videos\_per\_prompt: int = 1prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.Tensor\] = Noneprompt\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonemax\_sequence\_length: int = 128device: typing.Optional\[torch.device\] = Nonedtype: typing.Optional\[torch.dtype\] = None )

Parameters

*   [](#diffusers.LTXImageToVideoPipeline.encode_prompt.prompt)**prompt** (`str` or `List[str]`, _optional_) — prompt to be encoded
*   [](#diffusers.LTXImageToVideoPipeline.encode_prompt.negative_prompt)**negative\_prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
*   [](#diffusers.LTXImageToVideoPipeline.encode_prompt.do_classifier_free_guidance)**do\_classifier\_free\_guidance** (`bool`, _optional_, defaults to `True`) — Whether to use classifier free guidance or not.
*   [](#diffusers.LTXImageToVideoPipeline.encode_prompt.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
*   [](#diffusers.LTXImageToVideoPipeline.encode_prompt.prompt_embeds)**prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.
*   [](#diffusers.LTXImageToVideoPipeline.encode_prompt.negative_prompt_embeds)**negative\_prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated negative text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, negative\_prompt\_embeds will be generated from `negative_prompt` input argument.
*   [](#diffusers.LTXImageToVideoPipeline.encode_prompt.device)**device** — (`torch.device`, _optional_): torch device
*   [](#diffusers.LTXImageToVideoPipeline.encode_prompt.dtype)**dtype** — (`torch.dtype`, _optional_): torch dtype

Encodes the prompt into text encoder hidden states.

[](#diffusers.pipelines.ltx.pipeline_output.LTXPipelineOutput)LTXPipelineOutput
-------------------------------------------------------------------------------

### class diffusers.pipelines.ltx.pipeline\_output.LTXPipelineOutput

[](#diffusers.pipelines.ltx.pipeline_output.LTXPipelineOutput)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx/pipeline_output.py#L8)

( frames: Tensor )

Parameters

*   [](#diffusers.pipelines.ltx.pipeline_output.LTXPipelineOutput.frames)**frames** (`torch.Tensor`, `np.ndarray`, or List\[List\[PIL.Image.Image\]\]) — List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for LTX pipelines.

[< \> Update on GitHub](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/ltx_video.md)

[←LEDITS++](/docs/diffusers/main/en/api/pipelines/ledits_pp) [Lumina 2.0→](/docs/diffusers/main/en/api/pipelines/lumina2)