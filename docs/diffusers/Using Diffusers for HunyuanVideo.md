[](#hunyuanvideo)HunyuanVideo
=============================

![LoRA](https://img.shields.io/badge/LoRA-d8b4fe?style=flat)

[HunyuanVideo](https://www.arxiv.org/abs/2412.03603) by Tencent.

_Recent advancements in video generation have significantly impacted daily life for both individuals and industries. However, the leading video generation models remain closed-source, resulting in a notable performance gap between industry capabilities and those available to the public. In this report, we introduce HunyuanVideo, an innovative open-source video foundation model that demonstrates performance in video generation comparable to, or even surpassing, that of leading closed-source models. HunyuanVideo encompasses a comprehensive framework that integrates several key elements, including data curation, advanced architectural design, progressive model scaling and training, and an efficient infrastructure tailored for large-scale model training and inference. As a result, we successfully trained a video generative model with over 13 billion parameters, making it the largest among all open-source models. We conducted extensive experiments and implemented a series of targeted designs to ensure high visual quality, motion dynamics, text-video alignment, and advanced filming techniques. According to evaluations by professionals, HunyuanVideo outperforms previous state-of-the-art models, including Runway Gen-3, Luma 1.6, and three top-performing Chinese video generative models. By releasing the code for the foundation model and its applications, we aim to bridge the gap between closed-source and open-source communities. This initiative will empower individuals within the community to experiment with their ideas, fostering a more dynamic and vibrant video generation ecosystem. The code is publicly available at [this https URL](https://github.com/tencent/HunyuanVideo)._

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

Recommendations for inference:

*   Both text encoders should be in `torch.float16`.
*   Transformer should be in `torch.bfloat16`.
*   VAE should be in `torch.float16`.
*   `num_frames` should be of the form `4 * k + 1`, for example `49` or `129`.
*   For smaller resolution videos, try lower values of `shift` (between `2.0` to `5.0`) in the [Scheduler](https://huggingface.co/docs/diffusers/main/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler.shift). For larger resolution images, try higher values (between `7.0` and `12.0`). The default value is `7.0` for HunyuanVideo.
*   For more information about supported resolutions and other details, please refer to the original repository [here](https://github.com/Tencent/HunyuanVideo/).

[](#available-models)Available models
-------------------------------------

The following models are available for the [`HunyuanVideoPipeline`](text-to-video) pipeline:

Model name

Description

[`hunyuanvideo-community/HunyuanVideo`](https://huggingface.co/hunyuanvideo-community/HunyuanVideo)

Official HunyuanVideo (guidance-distilled). Performs best at multiple resolutions and frames. Performs best with `guidance_scale=6.0`, `true_cfg_scale=1.0` and without a negative prompt.

[`https://huggingface.co/Skywork/SkyReels-V1-Hunyuan-T2V`](https://huggingface.co/Skywork/SkyReels-V1-Hunyuan-T2V)

Skywork’s custom finetune of HunyuanVideo (de-distilled). Performs best with `97x544x960` resolution, `guidance_scale=1.0`, `true_cfg_scale=6.0` and a negative prompt.

The following models are available for the image-to-video pipeline:

Model name

Description

[`Skywork/SkyReels-V1-Hunyuan-I2V`](https://huggingface.co/Skywork/SkyReels-V1-Hunyuan-I2V)

Skywork’s custom finetune of HunyuanVideo (de-distilled). Performs best with `97x544x960` resolution. Performs best at `97x544x960` resolution, `guidance_scale=1.0`, `true_cfg_scale=6.0` and a negative prompt.

[`hunyuanvideo-community/HunyuanVideo-I2V`](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-I2V)

Tecent’s official HunyuanVideo I2V model. Performs best at resolutions of 480, 720, 960, 1280. A higher `shift` value when initializing the scheduler is recommended (good values are between 7 and 20)

[](#quantization)Quantization
-----------------------------

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [HunyuanVideoPipeline](/docs/diffusers/main/en/api/pipelines/hunyuan_video#diffusers.HunyuanVideoPipeline) for inference with bitsandbytes.

Copied

import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
from diffusers.utils import export\_to\_video

quant\_config = DiffusersBitsAndBytesConfig(load\_in\_8bit=True)
transformer\_8bit = HunyuanVideoTransformer3DModel.from\_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder="transformer",
    quantization\_config=quant\_config,
    torch\_dtype=torch.bfloat16,
)

pipeline = HunyuanVideoPipeline.from\_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    transformer=transformer\_8bit,
    torch\_dtype=torch.float16,
    device\_map="balanced",
)

prompt = "A cat walks on the grass, realistic style."
video = pipeline(prompt=prompt, num\_frames=61, num\_inference\_steps=30).frames\[0\]
export\_to\_video(video, "cat.mp4", fps=15)

[](#diffusers.HunyuanVideoPipeline)HunyuanVideoPipeline
-------------------------------------------------------

### class diffusers.HunyuanVideoPipeline

[](#diffusers.HunyuanVideoPipeline)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py#L144)

( text\_encoder: LlamaModeltokenizer: LlamaTokenizerFasttransformer: HunyuanVideoTransformer3DModelvae: AutoencoderKLHunyuanVideoscheduler: FlowMatchEulerDiscreteSchedulertext\_encoder\_2: CLIPTextModeltokenizer\_2: CLIPTokenizer )

Parameters

*   [](#diffusers.HunyuanVideoPipeline.text_encoder)**text\_encoder** (`LlamaModel`) — [Llava Llama3-8B](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers).
*   [](#diffusers.HunyuanVideoPipeline.tokenizer)**tokenizer** (`LlamaTokenizer`) — Tokenizer from [Llava Llama3-8B](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers).
*   [](#diffusers.HunyuanVideoPipeline.transformer)**transformer** ([HunyuanVideoTransformer3DModel](/docs/diffusers/main/en/api/models/hunyuan_video_transformer_3d#diffusers.HunyuanVideoTransformer3DModel)) — Conditional Transformer to denoise the encoded image latents.
*   [](#diffusers.HunyuanVideoPipeline.scheduler)**scheduler** ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/main/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) — A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
*   [](#diffusers.HunyuanVideoPipeline.vae)**vae** ([AutoencoderKLHunyuanVideo](/docs/diffusers/main/en/api/models/autoencoder_kl_hunyuan_video#diffusers.AutoencoderKLHunyuanVideo)) — Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
*   [](#diffusers.HunyuanVideoPipeline.text_encoder_2)**text\_encoder\_2** (`CLIPTextModel`) — [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
*   [](#diffusers.HunyuanVideoPipeline.tokenizer_2)**tokenizer\_2** (`CLIPTokenizer`) — Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

Pipeline for text-to-video generation using HunyuanVideo.

This model inherits from [DiffusionPipeline](/docs/diffusers/main/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### \_\_call\_\_

[](#diffusers.HunyuanVideoPipeline.__call__)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py#L467)

( prompt: typing.Union\[str, typing.List\[str\]\] = Noneprompt\_2: typing.Union\[str, typing.List\[str\]\] = Nonenegative\_prompt: typing.Union\[str, typing.List\[str\]\] = Nonenegative\_prompt\_2: typing.Union\[str, typing.List\[str\]\] = Noneheight: int = 720width: int = 1280num\_frames: int = 129num\_inference\_steps: int = 50sigmas: typing.List\[float\] = Nonetrue\_cfg\_scale: float = 1.0guidance\_scale: float = 6.0num\_videos\_per\_prompt: typing.Optional\[int\] = 1generator: typing.Union\[torch.\_C.Generator, typing.List\[torch.\_C.Generator\], NoneType\] = Nonelatents: typing.Optional\[torch.Tensor\] = Noneprompt\_embeds: typing.Optional\[torch.Tensor\] = Nonepooled\_prompt\_embeds: typing.Optional\[torch.Tensor\] = Noneprompt\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonenegative\_pooled\_prompt\_embeds: typing.Optional\[torch.Tensor\] = Nonenegative\_prompt\_attention\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_type: typing.Optional\[str\] = 'pil'return\_dict: bool = Trueattention\_kwargs: typing.Optional\[typing.Dict\[str, typing.Any\]\] = Nonecallback\_on\_step\_end: typing.Union\[typing.Callable\[\[int, int, typing.Dict\], NoneType\], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType\] = Nonecallback\_on\_step\_end\_tensor\_inputs: typing.List\[str\] = \['latents'\]prompt\_template: typing.Dict\[str, typing.Any\] = {'template': '<|start\_header\_id|>system<|end\_header\_id|>\\n\\nDescribe the video by detailing the following aspects: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:<|eot\_id|><|start\_header\_id|>user<|end\_header\_id|>\\n\\n{}<|eot\_id|>', 'crop\_start': 95}max\_sequence\_length: int = 256 ) → export const metadata = 'undefined';`~HunyuanVideoPipelineOutput` or `tuple`

Expand 24 parameters

Parameters

*   [](#diffusers.HunyuanVideoPipeline.__call__.prompt)**prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.
*   [](#diffusers.HunyuanVideoPipeline.__call__.prompt_2)**prompt\_2** (`str` or `List[str]`, _optional_) — The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is will be used instead.
*   [](#diffusers.HunyuanVideoPipeline.__call__.negative_prompt)**negative\_prompt** (`str` or `List[str]`, _optional_) — The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is not greater than `1`).
*   [](#diffusers.HunyuanVideoPipeline.__call__.negative_prompt_2)**negative\_prompt\_2** (`str` or `List[str]`, _optional_) — The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
*   [](#diffusers.HunyuanVideoPipeline.__call__.height)**height** (`int`, defaults to `720`) — The height in pixels of the generated image.
*   [](#diffusers.HunyuanVideoPipeline.__call__.width)**width** (`int`, defaults to `1280`) — The width in pixels of the generated image.
*   [](#diffusers.HunyuanVideoPipeline.__call__.num_frames)**num\_frames** (`int`, defaults to `129`) — The number of frames in the generated video.
*   [](#diffusers.HunyuanVideoPipeline.__call__.num_inference_steps)**num\_inference\_steps** (`int`, defaults to `50`) — The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
*   [](#diffusers.HunyuanVideoPipeline.__call__.sigmas)**sigmas** (`List[float]`, _optional_) — Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.
*   [](#diffusers.HunyuanVideoPipeline.__call__.true_cfg_scale)**true\_cfg\_scale** (`float`, _optional_, defaults to 1.0) — When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
*   [](#diffusers.HunyuanVideoPipeline.__call__.guidance_scale)**guidance\_scale** (`float`, defaults to `6.0`) — Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. Note that the only available HunyuanVideo model is CFG-distilled, which means that traditional guidance between unconditional and conditional latent is not applied.
*   [](#diffusers.HunyuanVideoPipeline.__call__.num_videos_per_prompt)**num\_videos\_per\_prompt** (`int`, _optional_, defaults to 1) — The number of images to generate per prompt.
*   [](#diffusers.HunyuanVideoPipeline.__call__.generator)**generator** (`torch.Generator` or `List[torch.Generator]`, _optional_) — A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.
*   [](#diffusers.HunyuanVideoPipeline.__call__.latents)**latents** (`torch.Tensor`, _optional_) — Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.
*   [](#diffusers.HunyuanVideoPipeline.__call__.prompt_embeds)**prompt\_embeds** (`torch.Tensor`, _optional_) — Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.
*   [](#diffusers.HunyuanVideoPipeline.__call__.pooled_prompt_embeds)**pooled\_prompt\_embeds** (`torch.FloatTensor`, _optional_) — Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, pooled text embeddings will be generated from `prompt` input argument.
*   [](#diffusers.HunyuanVideoPipeline.__call__.negative_prompt_embeds)**negative\_prompt\_embeds** (`torch.FloatTensor`, _optional_) — Pre-generated negative text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, negative\_prompt\_embeds will be generated from `negative_prompt` input argument.
*   [](#diffusers.HunyuanVideoPipeline.__call__.negative_pooled_prompt_embeds)**negative\_pooled\_prompt\_embeds** (`torch.FloatTensor`, _optional_) — Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, _e.g._ prompt weighting. If not provided, pooled negative\_prompt\_embeds will be generated from `negative_prompt` input argument.
*   [](#diffusers.HunyuanVideoPipeline.__call__.output_type)**output\_type** (`str`, _optional_, defaults to `"pil"`) — The output format of the generated image. Choose between `PIL.Image` or `np.array`.
*   [](#diffusers.HunyuanVideoPipeline.__call__.return_dict)**return\_dict** (`bool`, _optional_, defaults to `True`) — Whether or not to return a `HunyuanVideoPipelineOutput` instead of a plain tuple.
*   [](#diffusers.HunyuanVideoPipeline.__call__.attention_kwargs)**attention\_kwargs** (`dict`, _optional_) — A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention\_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
*   [](#diffusers.HunyuanVideoPipeline.__call__.clip_skip)**clip\_skip** (`int`, _optional_) — Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that the output of the pre-final layer will be used for computing the prompt embeddings.
*   [](#diffusers.HunyuanVideoPipeline.__call__.callback_on_step_end)**callback\_on\_step\_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, _optional_) — A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of each denoising step during the inference. with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
*   [](#diffusers.HunyuanVideoPipeline.__call__.callback_on_step_end_tensor_inputs)**callback\_on\_step\_end\_tensor\_inputs** (`List`, _optional_) — The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

Returns

export const metadata = 'undefined';

`~HunyuanVideoPipelineOutput` or `tuple`

export const metadata = 'undefined';

If `return_dict` is `True`, `HunyuanVideoPipelineOutput` is returned, otherwise a `tuple` is returned where the first element is a list with the generated images and the second element is a list of `bool`s indicating whether the corresponding generated image contains “not-safe-for-work” (nsfw) content.

The call function to the pipeline for generation.

[](#diffusers.HunyuanVideoPipeline.__call__.example)

Examples:

Copied

\>>> import torch
\>>> from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
\>>> from diffusers.utils import export\_to\_video

\>>> model\_id = "hunyuanvideo-community/HunyuanVideo"
\>>> transformer = HunyuanVideoTransformer3DModel.from\_pretrained(
...     model\_id, subfolder="transformer", torch\_dtype=torch.bfloat16
... )
\>>> pipe = HunyuanVideoPipeline.from\_pretrained(model\_id, transformer=transformer, torch\_dtype=torch.float16)
\>>> pipe.vae.enable\_tiling()
\>>> pipe.to("cuda")

\>>> output = pipe(
...     prompt="A cat walks on the grass, realistic",
...     height=320,
...     width=512,
...     num\_frames=61,
...     num\_inference\_steps=30,
... ).frames\[0\]
\>>> export\_to\_video(output, "output.mp4", fps=15)

#### disable\_vae\_slicing

[](#diffusers.HunyuanVideoPipeline.disable_vae_slicing)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py#L425)

( )

Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to computing decoding in one step.

#### disable\_vae\_tiling

[](#diffusers.HunyuanVideoPipeline.disable_vae_tiling)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py#L440)

( )

Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to computing decoding in one step.

#### enable\_vae\_slicing

[](#diffusers.HunyuanVideoPipeline.enable_vae_slicing)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py#L418)

( )

Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.

#### enable\_vae\_tiling

[](#diffusers.HunyuanVideoPipeline.enable_vae_tiling)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py#L432)

( )

Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow processing larger images.

[](#diffusers.pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput)HunyuanVideoPipelineOutput
-----------------------------------------------------------------------------------------------------------

### class diffusers.pipelines.hunyuan\_video.pipeline\_output.HunyuanVideoPipelineOutput

[](#diffusers.pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput)[< source \>](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuan_video/pipeline_output.py#L8)

( frames: Tensor )

Parameters

*   [](#diffusers.pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput.frames)**frames** (`torch.Tensor`, `np.ndarray`, or List\[List\[PIL.Image.Image\]\]) — List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for HunyuanVideo pipelines.

[< \> Update on GitHub](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/hunyuan_video.md)

LTX Video

[←Hunyuan-DiT](/docs/diffusers/main/en/api/pipelines/hunyuandit) [I2VGen-XL→](/docs/diffusers/main/en/api/pipelines/i2vgenxl)