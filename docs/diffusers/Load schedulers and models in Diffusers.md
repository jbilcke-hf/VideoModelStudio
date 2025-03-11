[](#load-schedulers-and-models)Load schedulers and models
=========================================================

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)

Diffusion pipelines are a collection of interchangeable schedulers and models that can be mixed and matched to tailor a pipeline to a specific use case. The scheduler encapsulates the entire denoising process such as the number of denoising steps and the algorithm for finding the denoised sample. A scheduler is not parameterized or trained so they don’t take very much memory. The model is usually only concerned with the forward pass of going from a noisy input to a less noisy sample.

This guide will show you how to load schedulers and models to customize a pipeline. You’ll use the [stable-diffusion-v1-5/stable-diffusion-v1-5](https://hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5) checkpoint throughout this guide, so let’s load it first.

Copied

import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from\_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch\_dtype=torch.float16, use\_safetensors=True
).to("cuda")

You can see what scheduler this pipeline uses with the `pipeline.scheduler` attribute.

Copied

pipeline.scheduler
PNDMScheduler {
  "\_class\_name": "PNDMScheduler",
  "\_diffusers\_version": "0.21.4",
  "beta\_end": 0.012,
  "beta\_schedule": "scaled\_linear",
  "beta\_start": 0.00085,
  "clip\_sample": false,
  "num\_train\_timesteps": 1000,
  "set\_alpha\_to\_one": false,
  "skip\_prk\_steps": true,
  "steps\_offset": 1,
  "timestep\_spacing": "leading",
  "trained\_betas": null
}

[](#load-a-scheduler)Load a scheduler
-------------------------------------

Schedulers are defined by a configuration file that can be used by a variety of schedulers. Load a scheduler with the [SchedulerMixin.from\_pretrained()](/docs/diffusers/main/en/api/schedulers/overview#diffusers.SchedulerMixin.from_pretrained) method, and specify the `subfolder` parameter to load the configuration file into the correct subfolder of the pipeline repository.

For example, to load the [DDIMScheduler](/docs/diffusers/main/en/api/schedulers/ddim#diffusers.DDIMScheduler):

Copied

from diffusers import DDIMScheduler, DiffusionPipeline

ddim = DDIMScheduler.from\_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler")

Then you can pass the newly loaded scheduler to the pipeline.

Copied

pipeline = DiffusionPipeline.from\_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", scheduler=ddim, torch\_dtype=torch.float16, use\_safetensors=True
).to("cuda")

[](#compare-schedulers)Compare schedulers
-----------------------------------------

Schedulers have their own unique strengths and weaknesses, making it difficult to quantitatively compare which scheduler works best for a pipeline. You typically have to make a trade-off between denoising speed and denoising quality. We recommend trying out different schedulers to find one that works best for your use case. Call the `pipeline.scheduler.compatibles` attribute to see what schedulers are compatible with a pipeline.

Let’s compare the [LMSDiscreteScheduler](/docs/diffusers/main/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler), [EulerDiscreteScheduler](/docs/diffusers/main/en/api/schedulers/euler#diffusers.EulerDiscreteScheduler), [EulerAncestralDiscreteScheduler](/docs/diffusers/main/en/api/schedulers/euler_ancestral#diffusers.EulerAncestralDiscreteScheduler), and the [DPMSolverMultistepScheduler](/docs/diffusers/main/en/api/schedulers/multistep_dpm_solver#diffusers.DPMSolverMultistepScheduler) on the following prompt and seed.

Copied

import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from\_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch\_dtype=torch.float16, use\_safetensors=True
).to("cuda")

prompt = "A photograph of an astronaut riding a horse on Mars, high resolution, high definition."
generator = torch.Generator(device="cuda").manual\_seed(8)

To change the pipelines scheduler, use the [from\_config()](/docs/diffusers/main/en/api/configuration#diffusers.ConfigMixin.from_config) method to load a different scheduler’s `pipeline.scheduler.config` into the pipeline.

LMSDiscreteScheduler

EulerDiscreteScheduler

EulerAncestralDiscreteScheduler

DPMSolverMultistepScheduler

[LMSDiscreteScheduler](/docs/diffusers/main/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler) typically generates higher quality images than the default scheduler.

Copied

from diffusers import LMSDiscreteScheduler

pipeline.scheduler = LMSDiscreteScheduler.from\_config(pipeline.scheduler.config)
image = pipeline(prompt, generator=generator).images\[0\]
image

![](https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_lms.png)

LMSDiscreteScheduler

![](https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_euler_discrete.png)

EulerDiscreteScheduler

![](https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_euler_ancestral.png)

EulerAncestralDiscreteScheduler

![](https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_dpm.png)

DPMSolverMultistepScheduler

Most images look very similar and are comparable in quality. Again, it often comes down to your specific use case so a good approach is to run multiple different schedulers and compare the results.

### [](#flax-schedulers)Flax schedulers

To compare Flax schedulers, you need to additionally load the scheduler state into the model parameters. For example, let’s change the default scheduler in [FlaxStableDiffusionPipeline](/docs/diffusers/main/en/api/pipelines/stable_diffusion/text2img#diffusers.FlaxStableDiffusionPipeline) to use the super fast `FlaxDPMSolverMultistepScheduler`.

The `FlaxLMSDiscreteScheduler` and `FlaxDDPMScheduler` are not compatible with the [FlaxStableDiffusionPipeline](/docs/diffusers/main/en/api/pipelines/stable_diffusion/text2img#diffusers.FlaxStableDiffusionPipeline) yet.

Copied

import jax
import numpy as np
from flax.jax\_utils import replicate
from flax.training.common\_utils import shard
from diffusers import FlaxStableDiffusionPipeline, FlaxDPMSolverMultistepScheduler

scheduler, scheduler\_state = FlaxDPMSolverMultistepScheduler.from\_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    subfolder="scheduler"
)
pipeline, params = FlaxStableDiffusionPipeline.from\_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    scheduler=scheduler,
    variant="bf16",
    dtype=jax.numpy.bfloat16,
)
params\["scheduler"\] = scheduler\_state

Then you can take advantage of Flax’s compatibility with TPUs to generate a number of images in parallel. You’ll need to make a copy of the model parameters for each available device and then split the inputs across them to generate your desired number of images.

Copied

\# Generate 1 image per parallel device (8 on TPUv2-8 or TPUv3-8)
prompt = "A photograph of an astronaut riding a horse on Mars, high resolution, high definition."
num\_samples = jax.device\_count()
prompt\_ids = pipeline.prepare\_inputs(\[prompt\] \* num\_samples)

prng\_seed = jax.random.PRNGKey(0)
num\_inference\_steps = 25

\# shard inputs and rng
params = replicate(params)
prng\_seed = jax.random.split(prng\_seed, jax.device\_count())
prompt\_ids = shard(prompt\_ids)

images = pipeline(prompt\_ids, params, prng\_seed, num\_inference\_steps, jit=True).images
images = pipeline.numpy\_to\_pil(np.asarray(images.reshape((num\_samples,) + images.shape\[-3:\])))

[](#models)Models
-----------------

Models are loaded from the [ModelMixin.from\_pretrained()](/docs/diffusers/main/en/api/models/overview#diffusers.ModelMixin.from_pretrained) method, which downloads and caches the latest version of the model weights and configurations. If the latest files are available in the local cache, [from\_pretrained()](/docs/diffusers/main/en/api/models/overview#diffusers.ModelMixin.from_pretrained) reuses files in the cache instead of re-downloading them.

Models can be loaded from a subfolder with the `subfolder` argument. For example, the model weights for [stable-diffusion-v1-5/stable-diffusion-v1-5](https://hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5) are stored in the [unet](https://hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/unet) subfolder.

Copied

from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from\_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet", use\_safetensors=True)

They can also be directly loaded from a [repository](https://huggingface.co/google/ddpm-cifar10-32/tree/main).

Copied

from diffusers import UNet2DModel

unet = UNet2DModel.from\_pretrained("google/ddpm-cifar10-32", use\_safetensors=True)

To load and save model variants, specify the `variant` argument in [ModelMixin.from\_pretrained()](/docs/diffusers/main/en/api/models/overview#diffusers.ModelMixin.from_pretrained) and [ModelMixin.save\_pretrained()](/docs/diffusers/main/en/api/models/overview#diffusers.ModelMixin.save_pretrained).

Copied

from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from\_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet", variant="non\_ema", use\_safetensors=True
)
unet.save\_pretrained("./local-unet", variant="non\_ema")

[< \> Update on GitHub](https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/schedulers.md)

[←Load community pipelines and components](/docs/diffusers/main/en/using-diffusers/custom_pipeline_overview) [Model files and layouts→](/docs/diffusers/main/en/using-diffusers/other-formats)