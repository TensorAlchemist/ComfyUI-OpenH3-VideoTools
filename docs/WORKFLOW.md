# Workflow behavior

Import `workflows/OpenH3_Regenerate2K.json` after installing the dependencies and restarting ComfyUI. Connect raw core **Load Video** directly to **Prepare Video**. The helper processes the complete uploaded local file; VIDEO-object crop or trim edits are not applied.

## Source preparation

The source mode normalizes display rotation and sample aspect ratio, applies the explicit `source_scale`, and rounds each working dimension upward to the VAE's 16-pixel grid. Only the resulting small alignment border is added. There is no fixed 1280 x 720 or 16:9 source canvas.

The helper converts timestamps to a constant 24 FPS cadence using duplication or dropping rather than retiming by nominal source FPS. It repeats the final frame to the existing H3 temporal lattice, `max(124, 17*ceil((N-5)/17)+5)`, then the workflow trims decoded output back to the actual prepared count. Model cadence and temporal preparation are distinct from a user duration limit.

`max_frames=0` means no user frame cap. A positive cap fails clearly when exceeded; it does not silently return the beginning of a longer video. Frames and conditioning remain materialized in RAM. There is no long-clip streaming implementation, and no memory guarantee for arbitrary dimensions or duration.

The existing decoder budget is 8192 pixels per side and 32 megapixels for coded and derived working geometry. Oversized derived content or alignment canvases fail rather than being silently reduced. A CPU decoding deadline, local-file containment, container restrictions and cancellation handling remain. PQ/HLG sources are rejected; tone-map HDR externally before use.

## Manual output

Set **Upscale Latent** width and height directly. They are independent editable fields with no automatic-size connections. The initial values are 2560 x 1440. Use the H3 refinement canvas's 32-pixel grid and enough memory for the requested dimensions. A portrait source needs a deliberately selected portrait output, for example 1440 x 2560; input preparation does not automatically choose output aspect or add final output letterboxing. Matching the prepared source's aspect ratio avoids intentional aspect changes; the model is still generative and can alter details or padding.

The supplied configuration keeps two refinement steps and the `res_multistep` sampler. It does not claim equivalence to the separate upstream three-step Euler reference or universal improvement from additional steps. The original seven input-helper outputs are retained; there are no automatic output-width/height sockets or `upscale_factor` control.

## Audio and export

The first source audio track is aligned to the first video timestamp, converted to 32 kHz stereo for conditioning, and padded or trimmed to the actual frame span. Conditioning extends through native frame padding; the final mux receives the source-duration audio. Silent input returns no source soundtrack and uses silent conditioning. A present but corrupt audio track is an error rather than a silent fallback.

The selected export uses VideoHelperSuite NVENC H.264 MP4. NVENC handles compression; input preparation, frame handling, audio conversion and mux work still use CPU. Check the active output node's filename prefix for the output location. Existing diagnostic latent-save nodes remain disabled.

## Validation boundary

This is the version 0.1.8 pre-release configuration. No tests, imports, compilation, inference, frontend round trip or CI were run for this publication copy. Earlier CPU evidence for 0.1.6 is not a current test result. No full render of this latest manual-output configuration is claimed. Review complete outputs before using them in a production deliverable.
