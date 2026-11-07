# Models and Custom Nodes

Paths are relative to your ComfyUI folder. Download the publisher files separately and keep their filenames. No weights are included.

| Folder | Download |
|---|---|
| `models/diffusion_models/` | [Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity/resolve/1259cf16cd0b655ba691a32fb46a409142762972/Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors?download=true) |
| `models/text_encoders/` | [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors?download=true) |
| `models/vae/` | [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors?download=true) |
| `models/vae/` | [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors?download=true) |
| `models/latent_upscale_models/` | [minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/resolve/5809c33730c18b6b1c827b40974855a4288641b3/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors?download=true) |
| `models/loras/` | [open_h3_regenerate_2k_lora.safetensors](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/resolve/5809c33730c18b6b1c827b40974855a4288641b3/lora/open_h3_regenerate_2k_lora.safetensors?download=true) |
| `models/loras/` | [sharp_curve_branches.safetensors](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/resolve/5809c33730c18b6b1c827b40974855a4288641b3/lora/sharp_curve_branches.safetensors?download=true) |

**Refine Latent** uses `open_h3_regenerate_2k_lora.safetensors` for `main_lora` and `sharp_curve_branches.safetensors` for `curve_branches`. OpenH3 VideoTools 0.1.11 finds both files in `models/loras/`.

In **Latent Upscale Model**, select `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`.

## Required custom nodes

- [OpenH3 VideoTools](https://github.com/TensorAlchemist/ComfyUI-OpenH3-VideoTools): install this repository as `ComfyUI/custom_nodes/ComfyUI-OpenH3-VideoTools`, with `__init__.py` directly inside that folder.
- [OpenH3-Regenerate2K by Songssx](https://github.com/Songssx/ComfyUI-OpenH3-Regenerate2K).
- [KJNodes](https://github.com/kijai/ComfyUI-KJNodes).
- [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite).
- [ComfyUI with native MiniMax H3 support](https://github.com/Comfy-Org/ComfyUI).

Use ComfyUI's Python environment to install the helper's `requirements.txt`. Python 3.12+, PyAV and imageio-ffmpeg are required. The selected export needs working NVIDIA NVENC support. Start ComfyUI with your normal launcher. See the [installation guide](https://github.com/TensorAlchemist/ComfyUI-OpenH3-VideoTools#installation).

## Source and terms

- [Official OpenH3 project](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/).
- [OpenH3 model repository](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K).
- [Singularity model card](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity).
- [MiniMax H3 community license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/6818f6c32d12b210915e44ad56a4228c2608f160/LICENSE).

Each component retains its own license.
