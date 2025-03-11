[](#wan)Wan
===========

[Wan 2.1](https://github.com/Wan-Video/Wan2.1) by the Alibaba Wan Team.

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

Recommendations for inference:

*   VAE in `torch.float32` for better decoding quality.
*   `num_frames` should be of the form `4 * k + 1`, for example `49` or `81`.
*   For smaller resolution videos, try lower values of `shift` (between `2.0` to `5.0`) in the [Scheduler](https://huggingface.co/docs/diffusers/main/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler.shift). For larger resolution videos, try higher values (between `7.0` and `12.0`). The default value is `3.0` for Wan.

### [](#using-a-custom-scheduler)Using a custom scheduler

Wan can be used with many different schedulers, each with their own benefits regarding speed and generation quality. By default, Wan uses the `UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)` scheduler. You can use a different scheduler as follows:

Copied

from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler, WanPipeline

scheduler\_a = FlowMatchEulerDiscreteScheduler(shift=5.0)
scheduler\_b = UniPCMultistepScheduler(prediction\_type="flow\_prediction", use\_flow\_sigmas=True, flow\_shift=4.0)

pipe = WanPipeline.from\_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", scheduler=<CUSTOM\_SCHEDULER\_HERE>)

\# or,
pipe.scheduler = <CUSTOM\_SCHEDULER\_HERE>

### [](#using-single-file-loading-with-wan)Using single file loading with Wan

The `WanTransformer3DModel` and `AutoencoderKLWan` models support loading checkpoints in their original format via the `from_single_file` loading method.

Copied

import torch
from diffusers import WanPipeline, WanTransformer3DModel

ckpt\_path = "https://huggingface.co/Comfy-Org/Wan\_2.1\_ComfyUI\_repackaged/blob/main/split\_files/diffusion\_models/wan2.1\_t2v\_1.3B\_bf16.safetensors"
transformer = WanTransformer3DModel.from\_single\_file(ckpt\_path, torch\_dtype=torch.bfloat16)

pipe = WanPipeline.from\_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", transformer=transformer)

[](#diffusers.WanPipeline)WanPipeline
-------------------------------------

### class diffusers.WanPipeline

[](#diffusers.WanPipeline)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py#L93)

( tokenizer: AutoTokenizertext\_encoder: UMT5EncoderModeltransformer: WanTransformer3DModelvae: AutoencoderKLWanscheduler: FlowMatchEulerDiscreteScheduler )

Parameters

*   [](#diffusers.WanPipeline.tokenizer)**tokenizer** (`T5Tokenizer`) — Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
*   [](#diffusers.WanPipeline.text_encoder)**text\_encoder** (`T5EncoderModel`) — [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
*   [](#diffusers.WanPipeline.transformer)**transformer** ([WanTransformer3DModel](/docs/diffusers/main/en/api/models/wan_transformer_3d#diffusers.WanTransformer3DModel)) — Conditional Transformer to denoise the input latents.
*   [](#diffusers.WanPipeline.scheduler)**scheduler** ([UniPCMultistepScheduler](/docs/diffusers/main/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) — A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
*   [](#diffusers.WanPipeline.vae)**vae** ([AutoencoderKLWan](/docs/diffusers/main/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) — Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

Pipeline for text-to-video generation using Wan.

This model inherits from [DiffusionPipeline](/docs/diffusers/main/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### \_\_call\_\_

[](#diffusers.WanPipeline.__call__)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py#L359)

( prompt: typing.Union\[str, typing.List\[str\]\] = Nonenegative\_prompt: typing.Union\[str, typing.List\[str\]\] = Noneheight: int = 480width: int = 832num\_frames: int = 81num\_inference\_steps: int = 50guidance\_scale: float = 5.0num\_videos\_per\_prompt: typing.Optional\[int\] = 1generator: typing.Union\[torch.\_C.Generator, typing.List\[torch.\_C.Generator\], NoneType\] = Nonelatents: typing.Optional\[torch.Tensor\] = Noneprompt\_embeds: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.Tensor\] = Noneoutput\_type: typing.Optional\[str\] = 'np'return\_dict: bool = Trueattention\_kwargs: typing.Optional\[typing.Dict\[str, typing.Any\]\] = Nonecallback\_on\_step\_end: typing.Union\[typing.Callable\[\[int, int, typing.Dict\], NoneType\], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType\] = Nonecallback\_on\_step\_end\_tensor\_inputs: typing.List\[str\] = \['latents'\]max\_sequence\_length: int = 512 ) → export const metadata = 'undefined';`~WanPipelineOutput` or `tuple`

Expand 16 parameters

Parameters

*   [](#diffusers.WanPipeline.__call__.prompt)**prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.
*   [](#diffusers.WanPipeline.__call__.height)**height** (`int`, defaults to `480`) — The height in pixels of the generated image.
*   [](#diffusers.WanPipeline.__call__.width)**width** (`int`, defaults to `832`) — The width in pixels of the generated image.
*   [](#diffusers.WanPipeline.__call__.num_frames)**num\_frames** (`int`, defaults to `81`) — The number of frames in the generated video.
*   [](#diffusers.WanPipeline.__call__.num_inference_steps)**num\_inference\_steps** (`int`, defaults to `50`) — The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
*   [](#diffusers.WanPipeline.__call__.guidance_scale)**guidance\_scale** (`float`, defaults to `5.0`) — Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
*   [](#diffusers.WanPipeline.__call__.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — The number of images to generate per prompt.
*   [](#diffusers.WanPipeline.__call__.generator)**generator** (`torch.Generator` or `List[torch.Generator]`, _optional_) — A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.
*   [](#diffusers.WanPipeline.__call__.latents)**latents** (`torch.Tensor`, _optional_) — Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.
*   [](#diffusers.WanPipeline.__call__.prompt_embeds)**prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.
*   [](#diffusers.WanPipeline.__call__.output_type)**output\_type** (`str`, _optional_, defaults to `"pil"`) — The output format of the generated image. Choose between `PIL.Image` or `np.array`.
*   [](#diffusers.WanPipeline.__call__.return_dict)**return\_dict** (`bool`, _optional_, defaults to `True`) — Whether or not to return a `WanPipelineOutput` instead of a plain tuple.
*   [](#diffusers.WanPipeline.__call__.attention_kwargs)**attention\_kwargs** (`dict`, _optional_) — A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
*   [](#diffusers.WanPipeline.__call__.callback_on_step_end)**callback\_on\_step\_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, _optional_) — A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of each denoising step during the inference. with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
*   [](#diffusers.WanPipeline.__call__.callback_on_step_end_tensor_inputs)**callback\_on\_step\_end\_tensor\_inputs** (`List`, _optional_) — The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.
*   [](#diffusers.WanPipeline.__call__.autocast_dtype)**autocast\_dtype** (`torch.dtype`, _optional_, defaults to `torch.bfloat16`) — The dtype to use for the torch.amp.autocast.

Returns

export const metadata = 'undefined';

`~WanPipelineOutput` or `tuple`

export const metadata = 'undefined';

If `return_dict` is `True`, `WanPipelineOutput` is returned, otherwise a `tuple` is returned where the first element is a list with the generated images and the second element is a list of `bool`s indicating whether the corresponding generated image contains “not-safe-for-work” (nsfw) content.

The call function to the pipeline for generation.

[](#diffusers.WanPipeline.__call__.example)

Examples:

Copied

\>>> import torch
\>>> from diffusers.utils import export\_to\_video
\>>> from diffusers import AutoencoderKLWan, WanPipeline
\>>> from diffusers.schedulers.scheduling\_unipc\_multistep import UniPCMultistepScheduler

\>>> \# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
\>>> model\_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
\>>> vae = AutoencoderKLWan.from\_pretrained(model\_id, subfolder="vae", torch\_dtype=torch.float32)
\>>> pipe = WanPipeline.from\_pretrained(model\_id, vae=vae, torch\_dtype=torch.bfloat16)
\>>> flow\_shift = 5.0  \# 5.0 for 720P, 3.0 for 480P
\>>> pipe.scheduler = UniPCMultistepScheduler.from\_config(pipe.scheduler.config, flow\_shift=flow\_shift)
\>>> pipe.to("cuda")

\>>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
\>>> negative\_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

\>>> output = pipe(
...     prompt=prompt,
...     negative\_prompt=negative\_prompt,
...     height=720,
...     width=1280,
...     num\_frames=81,
...     guidance\_scale=5.0,
... ).frames\[0\]
\>>> export\_to\_video(output, "output.mp4", fps=16)

#### encode\_prompt

[](#diffusers.WanPipeline.encode_prompt)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py#L181)

( prompt: typing.Union\[str, typing.List\[str\]\]negative\_prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Nonedo\_classifier\_free\_guidance: bool = Truenum\_videos\_per\_prompt: int = 1prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonemax\_sequence\_length: int = 226device: typing.Optional\[torch.device\] = Nonedtype: typing.Optional\[torch.dtype\] = None )

Parameters

*   [](#diffusers.WanPipeline.encode_prompt.prompt)**prompt** (`str` or `List[str]`, _optional_) — prompt to be encoded
*   [](#diffusers.WanPipeline.encode_prompt.negative_prompt)**negative\_prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
*   [](#diffusers.WanPipeline.encode_prompt.do_classifier_free_guidance)**do\_classifier\_free\_guidance** (`bool`, _optional_, defaults to `True`) — Whether to use classifier free guidance or not.
*   [](#diffusers.WanPipeline.encode_prompt.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
*   [](#diffusers.WanPipeline.encode_prompt.prompt_embeds)**prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.
*   [](#diffusers.WanPipeline.encode_prompt.negative_prompt_embeds)**negative\_prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated negative text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, negative\_prompt\_embeds will be generated from `negative_prompt` input argument.
*   [](#diffusers.WanPipeline.encode_prompt.device)**device** — (`torch.device`, _optional_): torch device
*   [](#diffusers.WanPipeline.encode_prompt.dtype)**dtype** — (`torch.dtype`, _optional_): torch dtype

Encodes the prompt into text encoder hidden states.

[](#diffusers.WanImageToVideoPipeline)WanImageToVideoPipeline
-------------------------------------------------------------

### class diffusers.WanImageToVideoPipeline

[](#diffusers.WanImageToVideoPipeline)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_i2v.py#L124)

( tokenizer: AutoTokenizertext\_encoder: UMT5EncoderModelimage\_encoder: CLIPVisionModelimage\_processor: CLIPImageProcessortransformer: WanTransformer3DModelvae: AutoencoderKLWanscheduler: FlowMatchEulerDiscreteScheduler )

Parameters

*   [](#diffusers.WanImageToVideoPipeline.tokenizer)**tokenizer** (`T5Tokenizer`) — Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
*   [](#diffusers.WanImageToVideoPipeline.text_encoder)**text\_encoder** (`T5EncoderModel`) — [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
*   [](#diffusers.WanImageToVideoPipeline.image_encoder)**image\_encoder** (`CLIPVisionModel`) — [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel), specifically the [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large) variant.
*   [](#diffusers.WanImageToVideoPipeline.transformer)**transformer** ([WanTransformer3DModel](/docs/diffusers/main/en/api/models/wan_transformer_3d#diffusers.WanTransformer3DModel)) — Conditional Transformer to denoise the input latents.
*   [](#diffusers.WanImageToVideoPipeline.scheduler)**scheduler** ([UniPCMultistepScheduler](/docs/diffusers/main/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) — A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
*   [](#diffusers.WanImageToVideoPipeline.vae)**vae** ([AutoencoderKLWan](/docs/diffusers/main/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) — Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

Pipeline for image-to-video generation using Wan.

This model inherits from [DiffusionPipeline](/docs/diffusers/main/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### \_\_call\_\_

[](#diffusers.WanImageToVideoPipeline.__call__)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_i2v.py#L441)

( image: typing.Union\[PIL.Image.Image, numpy.ndarray, torch.Tensor, typing.List\[PIL.Image.Image\], typing.List\[numpy.ndarray\], typing.List\[torch.Tensor\]\]prompt: typing.Union\[str, typing.List\[str\]\] = Nonenegative\_prompt: typing.Union\[str, typing.List\[str\]\] = Noneheight: int = 480width: int = 832num\_frames: int = 81num\_inference\_steps: int = 50guidance\_scale: float = 5.0num\_videos\_per\_prompt: typing.Optional\[int\] = 1generator: typing.Union\[torch.\_C.Generator, typing.List\[torch.\_C.Generator\], NoneType\] = Nonelatents: typing.Optional\[torch.Tensor\] = Noneprompt\_embeds: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.Tensor\] = Noneoutput\_type: typing.Optional\[str\] = 'np'return\_dict: bool = Trueattention\_kwargs: typing.Optional\[typing.Dict\[str, typing.Any\]\] = Nonecallback\_on\_step\_end: typing.Union\[typing.Callable\[\[int, int, typing.Dict\], NoneType\], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType\] = Nonecallback\_on\_step\_end\_tensor\_inputs: typing.List\[str\] = \['latents'\]max\_sequence\_length: int = 512 ) → export const metadata = 'undefined';`~WanPipelineOutput` or `tuple`

Expand 20 parameters

Parameters

*   [](#diffusers.WanImageToVideoPipeline.__call__.image)**image** (`PipelineImageInput`) — The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
*   [](#diffusers.WanImageToVideoPipeline.__call__.prompt)**prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.
*   [](#diffusers.WanImageToVideoPipeline.__call__.negative_prompt)**negative\_prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
*   [](#diffusers.WanImageToVideoPipeline.__call__.height)**height** (`int`, defaults to `480`) — The height of the generated video.
*   [](#diffusers.WanImageToVideoPipeline.__call__.width)**width** (`int`, defaults to `832`) — The width of the generated video.
*   [](#diffusers.WanImageToVideoPipeline.__call__.num_frames)**num\_frames** (`int`, defaults to `81`) — The number of frames in the generated video.
*   [](#diffusers.WanImageToVideoPipeline.__call__.num_inference_steps)**num\_inference\_steps** (`int`, defaults to `50`) — The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
*   [](#diffusers.WanImageToVideoPipeline.__call__.guidance_scale)**guidance\_scale** (`float`, defaults to `5.0`) — Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
*   [](#diffusers.WanImageToVideoPipeline.__call__.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — The number of images to generate per prompt.
*   [](#diffusers.WanImageToVideoPipeline.__call__.generator)**generator** (`torch.Generator` or `List[torch.Generator]`, _optional_) — A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.
*   [](#diffusers.WanImageToVideoPipeline.__call__.latents)**latents** (`torch.Tensor`, _optional_) — Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.
*   [](#diffusers.WanImageToVideoPipeline.__call__.prompt_embeds)**prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.
*   [](#diffusers.WanImageToVideoPipeline.__call__.output_type)**output\_type** (`str`, _optional_, defaults to `"pil"`) — The output format of the generated image. Choose between `PIL.Image` or `np.array`.
*   [](#diffusers.WanImageToVideoPipeline.__call__.return_dict)**return\_dict** (`bool`, _optional_, defaults to `True`) — Whether or not to return a `WanPipelineOutput` instead of a plain tuple.
*   [](#diffusers.WanImageToVideoPipeline.__call__.attention_kwargs)**attention\_kwargs** (`dict`, _optional_) — A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
*   [](#diffusers.WanImageToVideoPipeline.__call__.callback_on_step_end)**callback\_on\_step\_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, _optional_) — A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of each denoising step during the inference. with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
*   [](#diffusers.WanImageToVideoPipeline.__call__.callback_on_step_end_tensor_inputs)**callback\_on\_step\_end\_tensor\_inputs** (`List`, _optional_) — The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.
*   [](#diffusers.WanImageToVideoPipeline.__call__.max_sequence_length)**max\_sequence\_length** (`int`, _optional_, defaults to `512`) — The maximum sequence length of the prompt.
*   [](#diffusers.WanImageToVideoPipeline.__call__.shift)**shift** (`float`, _optional_, defaults to `5.0`) — The shift of the flow.
*   [](#diffusers.WanImageToVideoPipeline.__call__.autocast_dtype)**autocast\_dtype** (`torch.dtype`, _optional_, defaults to `torch.bfloat16`) — The dtype to use for the torch.amp.autocast.

Returns

export const metadata = 'undefined';

`~WanPipelineOutput` or `tuple`

export const metadata = 'undefined';

If `return_dict` is `True`, `WanPipelineOutput` is returned, otherwise a `tuple` is returned where the first element is a list with the generated images and the second element is a list of `bool`s indicating whether the corresponding generated image contains “not-safe-for-work” (nsfw) content.

The call function to the pipeline for generation.

[](#diffusers.WanImageToVideoPipeline.__call__.example)

Examples:

Copied

\>>> import torch
\>>> import numpy as np
\>>> from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
\>>> from diffusers.utils import export\_to\_video, load\_image
\>>> from transformers import CLIPVisionModel

\>>> \# Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
\>>> model\_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
\>>> image\_encoder = CLIPVisionModel.from\_pretrained(
...     model\_id, subfolder="image\_encoder", torch\_dtype=torch.float32
... )
\>>> vae = AutoencoderKLWan.from\_pretrained(model\_id, subfolder="vae", torch\_dtype=torch.float32)
\>>> pipe = WanImageToVideoPipeline.from\_pretrained(
...     model\_id, vae=vae, image\_encoder=image\_encoder, torch\_dtype=torch.bfloat16
... )
\>>> pipe.to("cuda")

\>>> image = load\_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
... )
\>>> max\_area = 480 \* 832
\>>> aspect\_ratio = image.height / image.width
\>>> mod\_value = pipe.vae\_scale\_factor\_spatial \* pipe.transformer.config.patch\_size\[1\]
\>>> height = round(np.sqrt(max\_area \* aspect\_ratio)) // mod\_value \* mod\_value
\>>> width = round(np.sqrt(max\_area / aspect\_ratio)) // mod\_value \* mod\_value
\>>> image = image.resize((width, height))
\>>> prompt = (
...     "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
...     "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
... )
\>>> negative\_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

\>>> output = pipe(
...     image=image,
...     prompt=prompt,
...     negative\_prompt=negative\_prompt,
...     height=height,
...     width=width,
...     num\_frames=81,
...     guidance\_scale=5.0,
... ).frames\[0\]
\>>> export\_to\_video(output, "output.mp4", fps=16)

#### encode\_prompt

[](#diffusers.WanImageToVideoPipeline.encode_prompt)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_i2v.py#L228)

( prompt: typing.Union\[str, typing.List\[str\]\]negative\_prompt: typing.Union\[str, typing.List\[str\], NoneType\] = Nonedo\_classifier\_free\_guidance: bool = Truenum\_videos\_per\_prompt: int = 1prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonemax\_sequence\_length: int = 226device: typing.Optional\[torch.device\] = Nonedtype: typing.Optional\[torch.dtype\] = None )

Parameters

*   [](#diffusers.WanImageToVideoPipeline.encode_prompt.prompt)**prompt** (`str` or `List[str]`, _optional_) — prompt to be encoded
*   [](#diffusers.WanImageToVideoPipeline.encode_prompt.negative_prompt)**negative\_prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
*   [](#diffusers.WanImageToVideoPipeline.encode_prompt.do_classifier_free_guidance)**do\_classifier\_free\_guidance** (`bool`, _optional_, defaults to `True`) — Whether to use classifier free guidance or not.
*   [](#diffusers.WanImageToVideoPipeline.encode_prompt.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
*   [](#diffusers.WanImageToVideoPipeline.encode_prompt.prompt_embeds)**prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.
*   [](#diffusers.WanImageToVideoPipeline.encode_prompt.negative_prompt_embeds)**negative\_prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated negative text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, negative\_prompt\_embeds will be generated from `negative_prompt` input argument.
*   [](#diffusers.WanImageToVideoPipeline.encode_prompt.device)**device** — (`torch.device`, _optional_): torch device
*   [](#diffusers.WanImageToVideoPipeline.encode_prompt.dtype)**dtype** — (`torch.dtype`, _optional_): torch dtype

Encodes the prompt into text encoder hidden states.

[](#diffusers.pipelines.wan.pipeline_output.WanPipelineOutput)WanPipelineOutput
-------------------------------------------------------------------------------

### class diffusers.pipelines.wan.pipeline\_output.WanPipelineOutput

[](#diffusers.pipelines.wan.pipeline_output.WanPipelineOutput)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_output.py#L8)

( frames: Tensor )

Parameters

*   [](#diffusers.pipelines.wan.pipeline_output.WanPipelineOutput.frames)**frames** (`torch.Tensor`, `np.ndarray`, or List\[List\[PIL.Image.Image\]\]) — List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for Wan pipelines.

[< \> Update on GitHub](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/wan.md)

CogVideoX

[←Value-guided sampling](/docs/diffusers/main/en/api/pipelines/value_guided_sampling) [Wuerstchen→](/docs/diffusers/main/en/api/pipelines/wuerstchen)