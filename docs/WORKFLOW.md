# Workflow behavior

Import `workflows/OpenH3_Regenerate2K.json` after installing the dependencies and restarting ComfyUI. Connect raw core **Load Video** directly to **Prepare Video**. The helper processes the complete uploaded local file; VIDEO-object crop or trim edits are not applied.

## Source preparation

Source mode normalizes display rotation and sample aspect ratio, applies `source_scale`, and rounds each working dimension upward to the VAE's 16-pixel grid. Only the small alignment border is added. There is no fixed working resolution or forced landscape source canvas.

The helper converts timestamps to a constant 24 FPS cadence using duplication or dropping rather than retiming by nominal FPS. It repeats the final frame to the H3 temporal lattice, `max(124, 17*ceil((N-5)/17)+5)`, and the workflow trims decoded output back to the actual prepared count.

`max_frames=0` disables the user frame cap. A positive cap fails when exceeded; it does not return only the beginning of a longer clip. Frames remain materialized in RAM. This is not temporal streaming or an arbitrary-duration memory guarantee.

The existing decoder budget is 8192 pixels per side and 32 megapixels for coded and derived working geometry. Oversized input fails rather than being silently reduced. A CPU decoding deadline, local-file containment, container restrictions and cancellation handling remain. PQ/HLG sources are rejected; tone-map HDR externally before use.

## Manual output

Set width and height directly in **Upscale Latent**. The initial values are 2560 x 1440, and neither field has an automatic-size connection. Use the H3 32-pixel refinement grid and enough memory for the chosen dimensions. For portrait input, deliberately select a portrait output such as 1440 x 2560; the workflow does not automatically change output aspect or add final letterboxing.

The supplied configuration uses two refinement steps and the `res_multistep` sampler. It does not claim equivalence to the separate upstream three-step Euler reference. Source style preservation is requested by the prompt, but the model is generative and can change detail, identity, text, motion or padding.

## Audio and export

The first source audio track is aligned to the first video timestamp and prepared as 32 kHz stereo. Conditioning extends through native frame padding; the final mux receives source-duration audio. Silent input returns no source soundtrack and uses silent conditioning. A present but corrupt audio track fails clearly.

The selected export uses VideoHelperSuite NVENC H.264 MP4. Input preparation, frame handling, audio processing and muxing still use CPU. Check the active output node's filename prefix for the destination. Diagnostic latent-save nodes remain disabled.
