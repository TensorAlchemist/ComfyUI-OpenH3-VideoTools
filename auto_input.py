# SPDX-License-Identifier: GPL-3.0-only
"""Native-uploader V3 boundary for the staged H3 input preparer."""

import folder_paths
from comfy_api.latest import InputImpl, io
import comfy.model_management

from .media_prepare import VERSION, prepare_file, resolve_input, validate_frame_limit, validate_source_scale, validate_working_resolution


def _uploaded_source(video):
    if type(video) is not InputImpl.VideoFromFile:
        raise ValueError("Connect core Load Video directly. In-memory, concatenated or generated VIDEO objects are unsupported.")
    path = resolve_input(video.get_stream_source(), folder_paths.get_input_directory())
    # The node intentionally reads the complete underlying upload, not VIDEO
    # object edits. There is no public crop getter; calling its dimensions or
    # negative-trim getter can itself probe/decode synchronously. Require raw
    # LoadVideo in the UI contract and report the full-file interpretation.
    return path


class TAOpenH3AutoInput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TAOpenH3AutoInput", display_name="H3 Auto Input | Fit, CFR and Native Padding",
            category="Open-H3/input",
            description="Connect raw core Load Video directly. Prepares the complete uploaded file on CPU: timestamp CFR24, source-aspect geometry with minimal VAE alignment padding, native last-frame padding and aligned stereo audio. VIDEO-object crop/trim edits are not applied. max_frames=0 means no user frame cap, not unlimited memory. Set output width and height manually in the downstream latent upscale node.",
            inputs=[io.Video.Input("video"), io.Int.Input("max_frames", default=0, min=0, max=2**31 - 1, step=1,
                        tooltip="0 = no user frame cap. A positive cap rejects longer actual CFR24 input, never silently trims. Full frames still occupy RAM; long/high-resolution input is not an 8 GB guarantee."),
                    io.Combo.Input("working_resolution", options=["source"], default="source", optional=True,
                        tooltip="Keep the displayed source aspect at source_scale, with only up to 15 pixels of VAE alignment padding per dimension. No fixed landscape canvas."),
                    io.Float.Input("source_scale", default=1.0, min=0.01, max=16.0, step=0.01, optional=True,
                        tooltip="1 keeps native displayed pixel size, 0.5 halves it, and 2 doubles it before alignment padding. Large inputs/scales can exceed RAM/VRAM.")],
            outputs=[io.Image.Output("prepared_images"), io.Audio.Output("conditioning_audio"),
                     io.Audio.Output("source_audio"), io.Int.Output("actual_frame_count"),
                     io.Float.Output("fps"), io.Float.Output("duration"), io.String.Output("report")],
        )

    @classmethod
    def execute(cls, video, max_frames=0, working_resolution="source", source_scale=1.0):
        path = _uploaded_source(video)
        result = prepare_file(str(path), folder_paths.get_input_directory(), max_frames,
                              working_resolution=working_resolution, source_scale=source_scale,
                              interrupt=comfy.model_management.throw_exception_if_processing_interrupted)
        return io.NodeOutput(*result)

    @classmethod
    def fingerprint_inputs(cls, video, max_frames=0, working_resolution="source", source_scale=1.0):
        validate_working_resolution(working_resolution)
        validate_frame_limit(max_frames)
        validate_source_scale(source_scale)
        path = _uploaded_source(video)
        stat = path.stat()
        return (VERSION, str(path), stat.st_size, stat.st_mtime_ns, max_frames, working_resolution, source_scale)
