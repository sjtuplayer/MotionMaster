# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import os
import torch
import json

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import math
import cv2 as cv

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)



def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module

pretrained_model_path = "models/StableDiffusion"
vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from ..models.unet import UNet3DConditionModel
from ..models.sparse_controlnet import SparseControlNetModel
import pdb
from animatediff.utils.util import save_videos_grid

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class motionconvert_PipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class motionconvert_Pipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        controlnet: Union[SparseControlNetModel, None] = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def encode_video(self, sub_video):
        video_length = sub_video.shape[1]
        sub_video = (sub_video - 0.5) * 2
        sub_video = rearrange(sub_video, "b f c h w -> (b f) c h w")
        sub_latent = vae.encode(sub_video).latent_dist
        sub_latent = sub_latent.sample()
        sub_latent = rearrange(sub_latent, "(b f) c h w -> b c f h w", f = video_length)
        sub_latent = sub_latent * 0.18215
        return sub_latent

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def next_step(self, model_output, timestep, sample):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    @torch.no_grad()
    def __call__(
        self,
        video: torch.Tensor,
        prompt1: Union[str],
        prompt2: Union[str],
        savedir: str,
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,

        # support controlnet
        controlnet_images: torch.FloatTensor = None,
        controlnet_image_index: list = [0],
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,

        **kwargs,
    ):
        with open('params.json', 'r') as file:
            data = json.load(file)

        inversion_step = data['inversion_step']
        replace_step = data['replace_step']
        savedir = data.get("savedir")
        mask_save_path = data.get("mask_save_path")
        ##################################################################################
        ##############################video1 inverse process##############################
        with open('params.json', 'r') as file:
            data = json.load(file)
        dir = os.path.join(savedir, 'tmp-save')
        if not os.path.exists(dir):
            latents = self.encode_video(video)
            # Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

            # Check inputs. Raise error if not correct
            self.check_inputs(prompt1, height, width, callback_steps)

            # Define call parameters
            # batch_size = 1 if isinstance(prompt, str) else len(prompt)
            batch_size = 1
            if latents is not None:
                batch_size = latents.shape[0]
            if isinstance(prompt1, list):
                batch_size = len(prompt1)

            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            # Encode input prompt
            prompt1 = prompt1 if isinstance(prompt1, list) else [prompt1] * batch_size
            if negative_prompt is not None:
                negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size
            text_embeddings = self._encode_prompt(
                prompt1, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
            )

            # Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # Prepare latent variables
            num_channels_latents = self.unet.in_channels
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
            )
            latents_dtype = latents.dtype

            # Prepare extra step kwargs.
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
            latent = latents
            # Define call parameters
            # batch_size = 1 if isinstance(prompt, str) else len(prompt)
            batch_size = 1
            if latents is not None:
                batch_size = latents.shape[0]
            if isinstance(prompt1, list):
                batch_size = len(prompt1)

            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            # Encode input prompt
            prompt1 = prompt1 if isinstance(prompt1, list) else [prompt1] * batch_size
            if negative_prompt is not None:
                negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size
            text_embeddings = self._encode_prompt(
                prompt1, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
            )

            # Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # Prepare latent variables
            num_channels_latents = self.unet.in_channels
            latents_dtype = latents.dtype

            # Prepare extra step kwargs.
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)


            with open('params.json', 'r') as file:
                data = json.load(file)

            inversion_step = data['inversion_step']
            replace_step = data['replace_step']
            savedir = data.get("savedir")
            mask_save_path = data.get("mask_save_path")

            timesteps = timesteps[-inversion_step:]

            # DDIM Inversion loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            latents = latents.clone().detach()
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    import torch
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    down_block_additional_residuals = mid_block_additional_residual = None
                    if (getattr(self, "controlnet", None) != None) and (controlnet_images != None):
                        assert controlnet_images.dim() == 5

                        controlnet_noisy_latents = latent_model_input
                        controlnet_prompt_embeds = text_embeddings

                        controlnet_images = controlnet_images.to(latents.device)

                        controlnet_cond_shape    = list(controlnet_images.shape)
                        controlnet_cond_shape[2] = video_length
                        controlnet_cond = torch.zeros(controlnet_cond_shape).to(latents.device)

                        controlnet_conditioning_mask_shape    = list(controlnet_cond.shape)
                        controlnet_conditioning_mask_shape[1] = 1
                        controlnet_conditioning_mask          = torch.zeros(controlnet_conditioning_mask_shape).to(latents.device)

                        assert controlnet_images.shape[2] >= len(controlnet_image_index)
                        controlnet_cond[:,:,controlnet_image_index] = controlnet_images[:,:,:len(controlnet_image_index)]
                        controlnet_conditioning_mask[:,:,controlnet_image_index] = 1
                        down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
                            controlnet_noisy_latents, t,
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=controlnet_cond,
                            conditioning_mask=controlnet_conditioning_mask,
                            conditioning_scale=controlnet_conditioning_scale,
                            guess_mode=False, return_dict=False,
                        )
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input, t,
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_additional_residuals,
                        mid_block_additional_residual=mid_block_additional_residual,
                    ).sample.to(dtype=latents_dtype)

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t-1 -> x_t
                    latents = self.next_step(noise_pred, t, latents)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)
    ############################################video1 diffuse process#################################################
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    down_block_additional_residuals = mid_block_additional_residual = None
                    if (getattr(self, "controlnet", None) != None) and (controlnet_images != None):
                        assert controlnet_images.dim() == 5

                        controlnet_noisy_latents = latent_model_input
                        controlnet_prompt_embeds = text_embeddings

                        controlnet_images = controlnet_images.to(latents.device)

                        controlnet_cond_shape = list(controlnet_images.shape)
                        controlnet_cond_shape[2] = video_length
                        controlnet_cond = torch.zeros(controlnet_cond_shape).to(latents.device)

                        controlnet_conditioning_mask_shape = list(controlnet_cond.shape)
                        controlnet_conditioning_mask_shape[1] = 1
                        controlnet_conditioning_mask = torch.zeros(controlnet_conditioning_mask_shape).to(
                            latents.device)

                        assert controlnet_images.shape[2] >= len(controlnet_image_index)
                        controlnet_cond[:, :, controlnet_image_index] = controlnet_images[:, :,
                                                                        :len(controlnet_image_index)]
                        controlnet_conditioning_mask[:, :, controlnet_image_index] = 1

                        down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
                            controlnet_noisy_latents, t,
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=controlnet_cond,
                            conditioning_mask=controlnet_conditioning_mask,
                            conditioning_scale=controlnet_conditioning_scale,
                            guess_mode=False, return_dict=False,
                        )

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input, t,
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_additional_residuals,
                        mid_block_additional_residual=mid_block_additional_residual,
                    ).sample.to(dtype=latents_dtype)

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

###############################################video2 diffuse process##################################################

        import math
        from torchvision import transforms
        import scipy.sparse as sp
        from scipy.sparse.linalg import splu
        ###poission process
        import torch
        from einops import rearrange
        from torchvision import transforms
        #



        latents = None
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt2, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt2, str) else len(prompt2)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt2, list):
            batch_size = len(prompt2)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt2 = prompt2 if isinstance(prompt2, list) else [prompt2] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [
                                                                                            negative_prompt] * batch_size
        text_embeddings = self._encode_prompt(
            prompt2, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        latent_list = []
        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                down_block_additional_residuals = mid_block_additional_residual = None
                if (getattr(self, "controlnet", None) != None) and (controlnet_images != None):
                    assert controlnet_images.dim() == 5

                    controlnet_noisy_latents = latent_model_input
                    controlnet_prompt_embeds = text_embeddings

                    controlnet_images = controlnet_images.to(latents.device)

                    controlnet_cond_shape = list(controlnet_images.shape)
                    controlnet_cond_shape[2] = video_length
                    controlnet_cond = torch.zeros(controlnet_cond_shape).to(latents.device)

                    controlnet_conditioning_mask_shape = list(controlnet_cond.shape)
                    controlnet_conditioning_mask_shape[1] = 1
                    controlnet_conditioning_mask = torch.zeros(controlnet_conditioning_mask_shape).to(
                        latents.device)

                    assert controlnet_images.shape[2] >= len(controlnet_image_index)
                    controlnet_cond[:, :, controlnet_image_index] = controlnet_images[:, :,
                                                                    :len(controlnet_image_index)]
                    controlnet_conditioning_mask[:, :, controlnet_image_index] = 1

                    down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
                        controlnet_noisy_latents, t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=controlnet_cond,
                        conditioning_mask=controlnet_conditioning_mask,
                        conditioning_scale=controlnet_conditioning_scale,
                        guess_mode=False, return_dict=False,
                    )


                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_additional_residuals,
                    mid_block_additional_residual=mid_block_additional_residual,
                ).sample.to(dtype=latents_dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                # latent_list.append(latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

            # Post-processing
            video = self.decode_latents(latents)

            # Convert to tensor
            if output_type == "tensor":
                video = torch.from_numpy(video)

            if not return_dict:
                return video
            return motionconvert_PipelineOutput(videos=video)


