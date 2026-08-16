import math
import re

import comfy.model_patcher
import comfy.samplers
import torch
from comfy.k_diffusion.sampling import sigma_to_half_log_snr

from raylight.comfy_extra_dist.ray_patch_decorator import ray_patch


def optimized_scale(positive, negative):
    positive_flat = positive.reshape(positive.shape[0], -1)
    negative_flat = negative.reshape(negative.shape[0], -1)
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
    return (dot_product / squared_norm).reshape([positive.shape[0]] + [1] * (positive.ndim - 1))


def score_tangential_damping(cond_score: torch.Tensor, uncond_score: torch.Tensor) -> torch.Tensor:
    batch_num = cond_score.shape[0]
    cond_score_flat = cond_score.reshape(batch_num, 1, -1).float()
    uncond_score_flat = uncond_score.reshape(batch_num, 1, -1).float()
    score_matrix = torch.cat((uncond_score_flat, cond_score_flat), dim=1)
    try:
        _, _, vh = torch.linalg.svd(score_matrix, full_matrices=False)
    except RuntimeError:
        _, _, vh = torch.linalg.svd(score_matrix.cpu(), full_matrices=False)
    v1 = vh[:, 0:1, :].to(uncond_score_flat.device)
    uncond_score_td = (uncond_score_flat @ v1.transpose(-2, -1)) * v1
    return uncond_score_td.reshape_as(uncond_score).to(uncond_score.dtype)


def fourier_filter(x, scale_low=1.0, scale_high=1.5, freq_cutoff=20):
    dtype, device = x.dtype, x.device
    x = x.to(torch.float32)
    x_freq = torch.fft.fftn(x, dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
    mask = torch.ones(x_freq.shape, device=device) * scale_high
    m = mask
    for d in range(len(x_freq.shape) - 2):
        dim = d + 2
        cc = x_freq.shape[dim] // 2
        f_c = min(freq_cutoff, cc)
        m = m.narrow(dim, cc - f_c, f_c * 2)
    m[:] = scale_low
    x_freq = torch.fft.ifftshift(x_freq * mask, dim=(-2, -1))
    return torch.fft.ifftn(x_freq, dim=(-2, -1)).real.to(dtype)


def compute_tsr_rescaling_factor(snr: torch.Tensor, tsr_k: float, tsr_variance: float) -> torch.Tensor:
    posinf_mask = torch.isposinf(snr)
    rescaling_factor = (snr * tsr_variance + 1) / (snr * tsr_variance / tsr_k + 1)
    return torch.where(posinf_mask, tsr_k, rescaling_factor)


def perp_neg(x, noise_pred_pos, noise_pred_neg, noise_pred_nocond, neg_scale, cond_scale):
    pos = noise_pred_pos - noise_pred_nocond
    neg = noise_pred_neg - noise_pred_nocond
    perp = neg - ((torch.mul(neg, pos).sum()) / (torch.norm(pos) ** 2)) * pos
    return noise_pred_nocond + cond_scale * (pos - perp * neg_scale)


class RayCFGZeroStar:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ray_actors": ("RAY_ACTORS",)}}

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra/guidance"

    @ray_patch
    def patch(self, model):
        m = model.clone()

        def cfg_zero_star(args):
            guidance_scale = args["cond_scale"]
            x = args["input"]
            cond_p = args["cond_denoised"]
            uncond_p = args["uncond_denoised"]
            out = args["denoised"]
            alpha = optimized_scale(x - cond_p, x - uncond_p)
            return out + uncond_p * (alpha - 1.0) + guidance_scale * uncond_p * (1.0 - alpha)

        m.set_model_sampler_post_cfg_function(cfg_zero_star)
        return m


class RayCFGNorm:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "pre_cfg": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra/guidance"

    @ray_patch
    def patch(self, model, strength, pre_cfg=False):
        m = model.clone()
        if pre_cfg:
            def cfg_norm_pre(args):
                cond = args["cond"]
                uncond = args["uncond"]
                cond_scale = args["cond_scale"]
                comb = uncond + cond_scale * (cond - uncond)
                cond_norm = torch.linalg.vector_norm(cond, dim=1, keepdim=True)
                comb_norm = torch.linalg.vector_norm(comb, dim=1, keepdim=True)
                rescale = torch.where(comb_norm > 0, cond_norm / comb_norm.clamp_min(1e-12), torch.ones_like(comb_norm))
                rescaled = comb * rescale
                if strength != 1.0:
                    rescaled = strength * rescaled + (1.0 - strength) * comb
                return rescaled

            m.set_model_sampler_cfg_function(cfg_norm_pre)
        else:
            def cfg_norm(args):
                cond_p = args["cond_denoised"]
                pred_text = args["denoised"]
                norm_full_cond = torch.norm(cond_p, dim=1, keepdim=True)
                norm_pred_text = torch.norm(pred_text, dim=1, keepdim=True)
                scale = (norm_full_cond / (norm_pred_text + 1e-8)).clamp(min=0.0, max=1.0)
                return pred_text * scale * strength

            m.set_model_sampler_post_cfg_function(cfg_norm)
        return m


class RayEpsilonScaling:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "scaling_factor": ("FLOAT", {"default": 1.005, "min": 0.5, "max": 1.5, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra/guidance"

    @ray_patch
    def patch(self, model, scaling_factor):
        if scaling_factor == 0:
            scaling_factor = 1e-9
        m = model.clone()

        def epsilon_scaling_function(args):
            denoised = args["denoised"]
            x = args["input"]
            return x - ((x - denoised) / scaling_factor)

        m.set_model_sampler_post_cfg_function(epsilon_scaling_function)
        return m


class RayTemporalScoreRescaling:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "tsr_k": ("FLOAT", {"default": 0.95, "min": 0.01, "max": 100.0, "step": 0.001}),
                "tsr_sigma": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra/guidance"

    @ray_patch
    def patch(self, model, tsr_k, tsr_sigma):
        tsr_variance = tsr_sigma ** 2
        m = model.clone()

        def temporal_score_rescaling(args):
            denoised = args["denoised"]
            x = args["input"]
            sigma = args["sigma"]
            curr_model = args["model"]
            if tsr_k == 1 or sigma == 0:
                return denoised
            model_sampling = curr_model.current_patcher.get_model_object("model_sampling")
            half_log_snr = sigma_to_half_log_snr(sigma, model_sampling)
            snr = (2 * half_log_snr).exp()
            if snr == 0:
                return denoised
            rescaling_r = compute_tsr_rescaling_factor(snr, tsr_k, tsr_variance)
            alpha = sigma * half_log_snr.exp()
            return torch.lerp(x / alpha, denoised, rescaling_r)

        m.set_model_sampler_post_cfg_function(temporal_score_rescaling)
        return m


class RayTCFG:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ray_actors": ("RAY_ACTORS",)}}

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra/guidance"

    @ray_patch
    def patch(self, model):
        m = model.clone()

        def tangential_damping_cfg(args):
            x = args["input"]
            conds_out = args["conds_out"]
            if len(conds_out) <= 1 or None in args["conds"][:2]:
                return conds_out
            cond_pred = conds_out[0]
            uncond_pred = conds_out[1]
            uncond_td = score_tangential_damping(x - cond_pred, x - uncond_pred)
            return [cond_pred, x - uncond_td] + conds_out[2:]

        m.set_model_sampler_pre_cfg_function(tangential_damping_cfg)
        return m


class RayFreSca:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "scale_low": ("FLOAT", {"default": 1.0, "min": 0, "max": 10, "step": 0.01}),
                "scale_high": ("FLOAT", {"default": 1.25, "min": 0, "max": 10, "step": 0.01}),
                "freq_cutoff": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra/guidance"

    @ray_patch
    def patch(self, model, scale_low, scale_high, freq_cutoff):
        m = model.clone()

        def custom_cfg_function(args):
            conds_out = args["conds_out"]
            if len(conds_out) <= 1 or None in args["conds"][:2]:
                return conds_out
            cond, uncond = conds_out[0], conds_out[1]
            filtered_cond = fourier_filter(cond - uncond, scale_low, scale_high, freq_cutoff) + uncond
            return [filtered_cond, uncond] + conds_out[2:]

        m.set_model_sampler_pre_cfg_function(custom_cfg_function)
        return m


class RayRenormCFG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "cfg_trunc": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "renorm_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra/guidance"

    @ray_patch
    def patch(self, model, cfg_trunc, renorm_cfg):
        m = model.clone()
        in_channels = model.model.diffusion_model.in_channels

        def renorm_cfg_func(args):
            cond_denoised = args["cond_denoised"]
            uncond_denoised = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            timestep = args["timestep"]
            x_orig = args["input"]
            cond_eps = cond_denoised[:, :in_channels]
            uncond_eps = uncond_denoised[:, :in_channels]
            cond_rest = cond_denoised[:, in_channels:]
            if timestep[0] < cfg_trunc:
                half_eps = uncond_eps + cond_scale * (cond_eps - uncond_eps)
                if float(renorm_cfg) > 0.0:
                    ori_pos_norm = torch.linalg.vector_norm(cond_eps, dim=tuple(range(1, len(cond_eps.shape))), keepdim=True)
                    max_new_norm = ori_pos_norm * float(renorm_cfg)
                    new_pos_norm = torch.linalg.vector_norm(half_eps, dim=tuple(range(1, len(half_eps.shape))), keepdim=True)
                    half_eps = torch.where(new_pos_norm >= max_new_norm, half_eps * (max_new_norm / new_pos_norm), half_eps)
            else:
                half_eps = cond_eps
            return x_orig - torch.cat([half_eps, cond_rest], dim=1)

        m.set_model_sampler_cfg_function(renorm_cfg_func)
        return m


class RayVideoLinearCFGGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ray_actors": ("RAY_ACTORS",), "min_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.5})}}

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra/guidance"

    @ray_patch
    def patch(self, model, min_cfg):
        m = model.clone()

        def linear_cfg(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            scale = torch.linspace(min_cfg, cond_scale, cond.shape[0], device=cond.device).reshape((cond.shape[0], 1, 1, 1))
            return uncond + scale * (cond - uncond)

        m.set_model_sampler_cfg_function(linear_cfg)
        return m


class RayVideoTriangleCFGGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ray_actors": ("RAY_ACTORS",), "min_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.5})}}

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra/guidance"

    @ray_patch
    def patch(self, model, min_cfg):
        m = model.clone()

        def triangle_cfg(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            values = torch.linspace(0, 1, cond.shape[0], device=cond.device)
            values = 2 * (values - torch.floor(values + 0.5)).abs()
            scale = (values * (cond_scale - min_cfg) + min_cfg).reshape((cond.shape[0], 1, 1, 1))
            return uncond + scale * (cond - uncond)

        m.set_model_sampler_cfg_function(triangle_cfg)
        return m


class RayPerturbedAttentionGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ray_actors": ("RAY_ACTORS",), "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01})}}

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra/guidance"

    @ray_patch
    def patch(self, model, scale):
        m = model.clone()

        def perturbed_attention(q, k, v, extra_options, mask=None):
            return v

        def post_cfg_function(args):
            if scale == 0:
                return args["denoised"]
            model_options = comfy.model_patcher.set_model_options_patch_replace(args["model_options"].copy(), perturbed_attention, "attn1", "middle", 0)
            (pag,) = comfy.samplers.calc_cond_batch(args["model"], [args["cond"]], args["input"], args["sigma"], model_options)
            return args["denoised"] + (args["cond_denoised"] - pag) * scale

        m.set_model_sampler_post_cfg_function(post_cfg_function)
        return m


class RaySkipLayerGuidanceDiT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "double_layers": ("STRING", {"default": "7, 8, 9"}),
                "single_layers": ("STRING", {"default": "7, 8, 9"}),
                "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "start_percent": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.001}),
                "rescaling_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra/guidance"

    @ray_patch
    def patch(self, model, double_layers, single_layers, scale, start_percent, end_percent, rescaling_scale):
        def skip(args, extra_args):
            return args

        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)
        double_layers = [int(i) for i in re.findall(r"\d+", double_layers)]
        single_layers = [int(i) for i in re.findall(r"\d+", single_layers)]
        if len(double_layers) == 0 and len(single_layers) == 0:
            return model
        m = model.clone()

        def post_cfg_function(args):
            cfg_result = args["denoised"]
            model_options = args["model_options"].copy()
            for layer in double_layers:
                model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, skip, "dit", "double_block", layer)
            for layer in single_layers:
                model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, skip, "dit", "single_block", layer)
            sigma_ = args["sigma"][0].item()
            if scale > 0 and sigma_end <= sigma_ <= sigma_start:
                (slg,) = comfy.samplers.calc_cond_batch(args["model"], [args["cond"]], args["input"], args["sigma"], model_options)
                cfg_result = cfg_result + (args["cond_denoised"] - slg) * scale
                if rescaling_scale != 0:
                    factor = args["cond_denoised"].std() / cfg_result.std()
                    cfg_result *= rescaling_scale * factor + (1 - rescaling_scale)
            return cfg_result

        m.set_model_sampler_post_cfg_function(post_cfg_function)
        return m


class RaySkipLayerGuidanceDiTSimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "double_layers": ("STRING", {"default": "7, 8, 9"}),
                "single_layers": ("STRING", {"default": "7, 8, 9"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra/guidance"

    @ray_patch
    def patch(self, model, double_layers, single_layers, start_percent, end_percent):
        def skip(args, extra_args):
            return args

        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)
        double_layers = [int(i) for i in re.findall(r"\d+", double_layers)]
        single_layers = [int(i) for i in re.findall(r"\d+", single_layers)]
        if len(double_layers) == 0 and len(single_layers) == 0:
            return model
        m = model.clone()

        def calc_cond_batch_function(args):
            model_options = args["model_options"]
            slg_model_options = model_options.copy()
            for layer in double_layers:
                slg_model_options = comfy.model_patcher.set_model_options_patch_replace(slg_model_options, skip, "dit", "double_block", layer)
            for layer in single_layers:
                slg_model_options = comfy.model_patcher.set_model_options_patch_replace(slg_model_options, skip, "dit", "single_block", layer)
            cond, uncond = args["conds"]
            sigma_ = args["sigma"][0].item()
            if sigma_end <= sigma_ <= sigma_start and uncond is not None:
                cond_out, _ = comfy.samplers.calc_cond_batch(args["model"], [cond, None], args["input"], args["sigma"], model_options)
                _, uncond_out = comfy.samplers.calc_cond_batch(args["model"], [None, uncond], args["input"], args["sigma"], slg_model_options)
                return [cond_out, uncond_out]
            return comfy.samplers.calc_cond_batch(args["model"], args["conds"], args["input"], args["sigma"], model_options)

        m.set_model_sampler_calc_cond_batch_function(calc_cond_batch_function)
        return m


class RayNAGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "nag_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "nag_tau": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra/guidance"

    @ray_patch
    def patch(self, model, nag_scale, nag_alpha, nag_tau):
        m = model.clone()

        def nag_attention_output_patch(out, extra_options):
            cond_or_uncond = extra_options.get("cond_or_uncond", None)
            if cond_or_uncond is None or not (1 in cond_or_uncond and 0 in cond_or_uncond):
                return out
            img_slice = extra_options.get("img_slice", None)
            orig_out = out
            if img_slice is not None:
                out = out[:, img_slice[0]:img_slice[1]]
            half_size = out.shape[0] // len(cond_or_uncond)
            ind_neg = cond_or_uncond.index(1)
            ind_pos = cond_or_uncond.index(0)
            z_pos = out[half_size * ind_pos:half_size * (ind_pos + 1)]
            z_neg = out[half_size * ind_neg:half_size * (ind_neg + 1)]
            guided = z_pos * nag_scale - z_neg * (nag_scale - 1.0)
            norm_pos = torch.norm(z_pos, p=1, dim=-1, keepdim=True).clamp_min(1e-6)
            norm_guided = torch.norm(guided, p=1, dim=-1, keepdim=True).clamp_min(1e-6)
            scale_factor = torch.minimum(norm_guided / norm_pos, torch.full_like(norm_guided, nag_tau)) / (norm_guided / norm_pos)
            z_final = (guided * scale_factor) * nag_alpha + z_pos * (1.0 - nag_alpha)
            if img_slice is not None:
                orig_out[half_size * ind_neg:half_size * (ind_neg + 1), img_slice[0]:img_slice[1]] = z_final
                orig_out[half_size * ind_pos:half_size * (ind_pos + 1), img_slice[0]:img_slice[1]] = z_final
                return orig_out
            out[half_size * ind_pos:half_size * (ind_pos + 1)] = z_final
            return out

        m.set_model_attn1_output_patch(nag_attention_output_patch)
        m.disable_model_cfg1_optimization()
        return m


NODE_CLASS_MAPPINGS = {
    "RayCFGZeroStar": RayCFGZeroStar,
    "RayCFGNorm": RayCFGNorm,
    "RayEpsilonScaling": RayEpsilonScaling,
    "RayTemporalScoreRescaling": RayTemporalScoreRescaling,
    "RayTCFG": RayTCFG,
    "RayFreSca": RayFreSca,
    "RayRenormCFG": RayRenormCFG,
    "RayVideoLinearCFGGuidance": RayVideoLinearCFGGuidance,
    "RayVideoTriangleCFGGuidance": RayVideoTriangleCFGGuidance,
    "RayPerturbedAttentionGuidance": RayPerturbedAttentionGuidance,
    "RaySkipLayerGuidanceDiT": RaySkipLayerGuidanceDiT,
    "RaySkipLayerGuidanceDiTSimple": RaySkipLayerGuidanceDiTSimple,
    "RayNAGuidance": RayNAGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayCFGZeroStar": "CFGZeroStar (Ray)",
    "RayCFGNorm": "CFGNorm (Ray)",
    "RayEpsilonScaling": "Epsilon Scaling (Ray)",
    "RayTemporalScoreRescaling": "TSR - Temporal Score Rescaling (Ray)",
    "RayTCFG": "Tangential Damping CFG (Ray)",
    "RayFreSca": "FreSca (Ray)",
    "RayRenormCFG": "RenormCFG (Ray)",
    "RayVideoLinearCFGGuidance": "VideoLinearCFGGuidance (Ray)",
    "RayVideoTriangleCFGGuidance": "VideoTriangleCFGGuidance (Ray)",
    "RayPerturbedAttentionGuidance": "PerturbedAttentionGuidance (Ray)",
    "RaySkipLayerGuidanceDiT": "SkipLayerGuidanceDiT (Ray)",
    "RaySkipLayerGuidanceDiTSimple": "SkipLayerGuidanceDiTSimple (Ray)",
    "RayNAGuidance": "Normalized Attention Guidance (Ray)",
}
