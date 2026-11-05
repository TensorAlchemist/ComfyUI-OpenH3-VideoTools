# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Tensor Alchemist
"""Own a native Open-H3 refinement model only for one sampling call."""

import gc
import logging
import sys

import folder_paths
import nodes
import comfy.model_management
import comfy.model_prefetch
from comfy_api.latest import io


def _file_choices(folder, preferred):
    try:
        options = [name for name in folder_paths.get_filename_list(folder)
                   if name.lower().endswith(".safetensors")]
    except KeyError:
        options = []
    options.sort(key=lambda name: (name.replace("\\", "/").rsplit("/", 1)[-1] != preferred,
                                   name.lower()))
    return options or ["(missing)"]


def _native_classes(enable_tiling):
    names = ["OpenH3RegenerateModelPatch", "ModelAttentionBackend", "MiniMaxLowVRAMAttention",
             "MiniMaxChunkFeedForward", "BasicGuider", "SamplerCustomAdvanced"]
    if enable_tiling:
        names.append("TAOpenH3TiledDiffusion")
    missing = [name for name in names if name not in nodes.NODE_CLASS_MAPPINGS]
    if missing:
        raise RuntimeError("Required native/custom node is unavailable: " + ", ".join(missing)
                           + ". Install the matching Open-H3, KJNodes, and OpenH3 VideoTools packs and restart.")
    return {name: nodes.NODE_CLASS_MAPPINGS[name] for name in names}


def _run_native_refinement(native, owned, av_latent, conditioning, noise, sampler, sigmas,
                           unet_name, main_lora, curve_branches, strength, head_chunks,
                           ff_chunks, seq_threshold, enable_tiling, method,
                           tile_width, tile_height, tile_overlap):
    # Retain every owned patcher until synchronized managed unload, then let this
    # frame disappear before outer GC. No MODEL or GUIDER enters a node output.
    model = guider = sample_result = None
    try:
        model = nodes.UNETLoader().load_unet(unet_name, "default")[0]
        owned.append(model)
        model, report = native["OpenH3RegenerateModelPatch"]().patch(
            model, main_lora, curve_branches, strength)
        owned.append(model)
        model = native["ModelAttentionBackend"].execute(
            model, "comfy kitchen attention").result[0]
        owned.append(model)
        model = native["MiniMaxLowVRAMAttention"].execute(model, head_chunks).result[0]
        owned.append(model)
        model = native["MiniMaxChunkFeedForward"].execute(
            model, ff_chunks, seq_threshold).result[0]
        owned.append(model)
        if enable_tiling:
            model = native["TAOpenH3TiledDiffusion"].execute(
                model, method, tile_width, tile_height, tile_overlap).result[0]
            owned.append(model)
        guider = native["BasicGuider"].execute(model, conditioning).result[0]
        sample_result = native["SamplerCustomAdvanced"].execute(
            noise, guider, sampler, sigmas, av_latent).result
        return sample_result[1], report
    finally:
        model = guider = sample_result = None


class TAOpenH3RefineRelease(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TAOpenH3RefineRelease",
            display_name="H3 Refine and Release",
            category="Open-H3/memory",
            description=(
                "Runs the native H3 loader, Open-H3 adapter, Kitchen/KJ patches, optional "
                "TA tiling, BasicGuider and SamplerCustomAdvanced inside one model lifetime. "
                "Returns native denoised AV latent and adapter report only, then releases "
                "all locally owned model patchers. Noise, sampler and sigmas remain external."
            ),
            inputs=[
                io.Latent.Input("av_latent"),
                io.Conditioning.Input("conditioning"),
                io.Noise.Input("noise"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Combo.Input("unet_name", options=_file_choices(
                    "diffusion_models", "Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors")),
                io.Combo.Input("main_lora", options=_file_choices(
                    "loras", "open_h3_regenerate_2k_lora.safetensors")),
                io.Combo.Input("curve_branches", options=_file_choices(
                    "open_h3_regenerate", "sharp_curve_branches.safetensors")),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.5, step=0.01),
                io.Int.Input("head_chunks", default=4, min=1, max=56, step=1),
                io.Int.Input("ff_chunks", default=4, min=1, max=64, step=1),
                io.Int.Input("seq_threshold", default=4096, min=256, max=262144, step=256),
                io.Boolean.Input("enable_tiling", default=True),
                io.Combo.Input("method", options=["gaussian", "uniform"], default="gaussian"),
                io.Int.Input("tile_width", default=1024, min=64, max=8192, step=32),
                io.Int.Input("tile_height", default=768, min=64, max=8192, step=32),
                io.Int.Input("tile_overlap", default=256, min=0, max=8192, step=32),
            ],
            outputs=[io.Latent.Output(display_name="denoised_av_latent"),
                     io.String.Output(display_name="adapter_report")],
        )

    @classmethod
    def execute(cls, av_latent, conditioning, noise, sampler, sigmas, unet_name,
                main_lora, curve_branches, strength, head_chunks, ff_chunks,
                seq_threshold, enable_tiling, method, tile_width, tile_height, tile_overlap) -> io.NodeOutput:
        native = _native_classes(enable_tiling)
        owned = []
        target = None
        completed = False
        try:
            denoised, report = _run_native_refinement(
                native, owned, av_latent, conditioning, noise, sampler, sigmas,
                unet_name, main_lora, curve_branches, strength, head_chunks, ff_chunks,
                seq_threshold, enable_tiling, method, tile_width, tile_height, tile_overlap)
            completed = True
            return io.NodeOutput(denoised, report)
        finally:
            try:
                if owned:
                    target = owned[-1]
                    # The inner call's model/guider/result aliases have been cleared.
                    # Keep the owned chain alive until all device work is drained.
                    with comfy.model_management.cuda_device_context(target.load_device):
                        comfy.model_management.synchronize()
                        comfy.model_prefetch.cleanup_prefetch_queues()
                        comfy.model_management.unload_model_and_clones(target)
            except Exception as cleanup_error:
                if completed:
                    raise RuntimeError(
                        "Native H3 refinement finished, but its local model cleanup failed."
                    ) from cleanup_error
                logging.exception("Local H3 model cleanup failed while handling a native refinement error.")
            finally:
                target = None
                owned.clear()
                pending_error = sys.exc_info()[0] is not None
                try:
                    gc.collect()
                except Exception:
                    if not pending_error:
                        raise
                    logging.exception("H3 model garbage collection failed while preserving an earlier error.")
