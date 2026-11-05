# Models and Custom Nodes
Paths below are relative to your ComfyUI folder. Weights are downloaded separately.

## models/diffusion_models/
[Singularity V2 pruned INT8](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity/resolve/1259cf16cd0b655ba691a32fb46a409142762972/Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors?download=true)

## models/text_encoders/
[Qwen3-VL 32B MiniMax H3 NVFP4 AWQ](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors?download=true)

## models/vae/
[MiniMax H3 video VAE INT8 ConvRot](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors?download=true)

[MiniMax H3 audio VAE FP32](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors?download=true)

## models/latent_upscale_models/
[Native learned latent upscaler BF16](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/resolve/5809c33730c18b6b1c827b40974855a4288641b3/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors?download=true)

Publisher filename: minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors. Select it in the loader, or rename these identical bytes to the selected local alias: minimax_h3_latent_upscaler_3d_bf16.safetensors. No precision conversion is needed for this file.

## models/loras/
[Main refinement adapter: original FP32](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/resolve/5809c33730c18b6b1c827b40974855a4288641b3/lora/open_h3_regenerate_2k_lora.safetensors?download=true)

## models/open_h3_regenerate/lora/
[Sharp curve branches: original FP32](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/resolve/5809c33730c18b6b1c827b40974855a4288641b3/lora/sharp_curve_branches.safetensors?download=true)

## Required FP16 adapter copies
This workflow selects open_h3_regenerate_2k_lora_fp16.safetensors and lora/sharp_curve_branches_fp16.safetensors. The two links above download the publisher's FP32 originals. Renaming them does not convert them.

Use [OpenH3 VideoTools](https://github.com/TensorAlchemist/ComfyUI-OpenH3-VideoTools): docs/REPRODUCE_FP16.txt and tools/convert_fp16.py. Run the converter with your ComfyUI Python as described there. It checks the exact original hashes and preserves the source files. Existing valid FP16 copies can be reused. 

## Required custom nodes
[OpenH3-Regenerate2K by Songssx](https://github.com/Songssx/ComfyUI-OpenH3-Regenerate2K)

[KJNodes](https://github.com/kijai/ComfyUI-KJNodes)

[VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

[Current ComfyUI with native MiniMax H3 support](https://github.com/Comfy-Org/ComfyUI)

Install [OpenH3 VideoTools 0.1.8 pre-release](https://github.com/TensorAlchemist/ComfyUI-OpenH3-VideoTools) as ComfyUI/custom_nodes/ComfyUI-OpenH3-VideoTools, with __init__.py directly inside that repository folder. Keep one active copy. Python 3.12+, PyAV and imageio-ffmpeg are required; see its requirements.txt. The selected export also requires working NVIDIA NVENC support.

## Prompt
The Prompt node contains a general preservation prompt for photo, anime, CG and other source styles. You can run it as supplied. For stronger scene guidance, add a short, accurate description inside integrated_multimodal_description. This text encoder does not analyze the upload. Generative reconstruction can still change details, text or motion; review the result.

Set OUTPUT width and height directly in Upscale Latent. Both fields are editable and have no automatic-size connections. Input sizing and duration remain automatic and unchanged. source_scale adjusts working input size; max_frames = 0 disables the optional input-frame limit. Timing follows the model's 24 FPS cadence. Version 0.1.8 is a pre-release; no new tests or full render of this manual-output configuration are claimed.

## Component sources and terms
[Official OpenH3 project](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/)

[Singularity model card and terms](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity)

[MiniMax H3 community license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/6818f6c32d12b210915e44ad56a4228c2608f160/LICENSE)

Each component retains its own license.

## Optional original FP32 adapter selection

The supplied workflow selects converted FP16 storage copies. To use the publisher originals instead, explicitly select `open_h3_regenerate_2k_lora.safetensors` as `main_lora` and `lora/sharp_curve_branches.safetensors` as `curve_branches` in **Refine Latent**. Keep their destination folders exactly as listed above. This is an optional loader-code-compatible route; its RAM use and full execution have not been verified for this version. It is not a measured alternative or a recommendation for limited-memory systems.
