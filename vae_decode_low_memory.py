# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Tensor Alchemist
"""Native H3 decoding with an execution-local float16 CPU output buffer."""

import copy
import gc
import logging
import types

import torch
import nodes
import comfy.model_management
import comfy.model_prefetch
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
from comfy_api.latest import io


def _fp16_output_dtype(self):
    return torch.float16


def _validate_native_vae(vae):
    model = getattr(vae, "first_stage_model", None)
    if type(model) is not MiniMaxH3VideoVAE or model.noncausal:
        raise ValueError("This decoder requires the native causal MiniMax H3 video VAE.")
    if not model.comfy_has_chunked_io:
        raise ValueError("This decoder requires native H3 direct-buffer output support.")
    if getattr(getattr(vae, "output_device", None), "type", None) != "cpu":
        raise ValueError("This decoder requires the native CPU intermediate output device.")
    if not callable(getattr(vae, "vae_output_dtype", None)):
        raise ValueError("This decoder requires the native per-VAE output-dtype method.")


class TAOpenH3VAEDecodeLowMemory(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TAOpenH3VAEDecodeLowMemory",
            display_name="H3 VAE Decode, FP16 Host Output",
            category="Open-H3/memory",
            description=(
                "Runs native causal H3 decoding with a local float16 CPU output "
                "buffer. Preserves native decoder tiles, batches, temporal logic "
                "and computation. Only returned image storage changes to float16; "
                "no global intermediate or sampler precision is changed."
            ),
            inputs=[io.Vae.Input("vae"), io.Latent.Input("samples")],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, vae, samples) -> io.NodeOutput:
        _validate_native_vae(vae)
        local_vae = None
        decoded = False
        cleanup_failed = False
        try:
            local_vae = copy.copy(vae)
            local_vae.vae_output_dtype = types.MethodType(_fp16_output_dtype, local_vae)
            images = nodes.VAEDecode().decode(local_vae, samples)[0]
            decoded = True
            return io.NodeOutput(images)
        finally:
            try:
                if local_vae is not None:
                    with comfy.model_management.cuda_device_context(local_vae.patcher.load_device):
                        comfy.model_management.synchronize()
                        comfy.model_prefetch.cleanup_prefetch_queues()
                        comfy.model_management.unload_model_and_clones(local_vae.patcher)
            except Exception as cleanup_error:
                cleanup_failed = True
                if decoded:
                    raise RuntimeError("Native H3 decoding finished, but decoder cleanup failed.") from cleanup_error
                logging.exception("H3 decoder cleanup failed while handling a native decode error.")
            finally:
                # The MethodType points back to this temporary wrapper. Remove it
                # explicitly; the original VAE and its output-dtype method were
                # never modified. Model patcher / weights are managed natively.
                if local_vae is not None:
                    del local_vae.vae_output_dtype
                local_vae = None
                try:
                    gc.collect()
                except Exception as gc_error:
                    if decoded and not cleanup_failed:
                        raise RuntimeError("Native H3 decoding finished, but decoder GC failed.") from gc_error
                    logging.exception("Decoder GC failed while handling an earlier native/cleanup error.")
