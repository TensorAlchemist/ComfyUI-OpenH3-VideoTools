# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Tensor Alchemist
"""Native H3 prompt encoding with an execution-local encoder lifetime."""

import gc
import logging

import folder_paths
import nodes
import comfy.model_management
import comfy.model_prefetch
from comfy_api.latest import io


class TAOpenH3PromptEncodeRelease(io.ComfyNode):
    """Return native conditioning without publishing or retaining a CLIP model."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TAOpenH3PromptEncodeRelease",
            display_name="H3 Prompt Encode and Release",
            category="Open-H3/memory",
            description=(
                "Loads the selected encoder through native CLIPLoader in MiniMax mode, "
                "runs native CLIPTextEncode, then unloads its local encoder. Returns "
                "conditioning only. This avoids retaining a CLIP output in the graph; "
                "actual host-memory recovery depends on the runtime and allocator."
            ),
            inputs=[
                io.Combo.Input("clip_name", options=folder_paths.get_filename_list("text_encoders")),
                io.String.Input("text", default="", multiline=True, dynamic_prompts=True),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning")],
        )

    @classmethod
    def execute(cls, clip_name, text) -> io.NodeOutput:
        clip = None
        encoded = False
        try:
            clip = nodes.CLIPLoader().load_clip(clip_name, "minimax", "default")[0]
            conditioning = nodes.CLIPTextEncode().encode(clip, text)[0]
            encoded = True
            return io.NodeOutput(conditioning)
        finally:
            try:
                if clip is not None:
                    # Native execution normally drains prefetch AFTER a node returns.
                    # This node releases its encoder BEFORE returning, so follow the
                    # target core's writable_model cleanup ordering first: synchronize
                    # in the encoder's device context, drain prefetch, managed unload.
                    # Do not close mapped weight handles or mutate prompt/node caches.
                    with comfy.model_management.cuda_device_context(clip.patcher.load_device):
                        comfy.model_management.synchronize()
                        comfy.model_prefetch.cleanup_prefetch_queues()
                        comfy.model_management.unload_model_and_clones(clip.patcher)
            except Exception as cleanup_error:
                if encoded:
                    raise RuntimeError(
                        "Native H3 prompt encoding finished, but its local encoder cleanup failed."
                    ) from cleanup_error
                # Preserve the native load/encode/cancellation exception if cleanup
                # also fails. Normal core node-finalization still runs afterward.
                logging.exception("Local H3 encoder cleanup failed while handling a native encoding error.")
            finally:
                clip = None
                gc.collect()
