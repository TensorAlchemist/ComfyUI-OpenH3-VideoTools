# OpenH3 VideoTools

## Quick start

1. Download [OpenH3 VideoTools v0.1.11](https://github.com/TensorAlchemist/ComfyUI-OpenH3-VideoTools/releases/tag/v0.1.11). Extract `ComfyUI-OpenH3-VideoTools` into `ComfyUI/custom_nodes`, then install the requirements and the other nodes listed below.
2. Download the weights from the [model guide](docs/DOWNLOAD_LINKS.md). Put both `open_h3_regenerate_2k_lora.safetensors` and `sharp_curve_branches.safetensors` in `ComfyUI/models/loras`.
3. Restart ComfyUI with your normal launcher and import the current [OpenH3_Regenerate2K.json workflow](https://raw.githubusercontent.com/TensorAlchemist/ComfyUI-OpenH3-VideoTools/main/workflows/OpenH3_Regenerate2K.json). Upload a local SDR video, enter the target dimensions in **Upscale Latent**, and run the workflow.

## Installation

Requires Python 3.12 or newer and a ComfyUI build supporting native MiniMax H3, core Load Video, and the `comfy_api.latest` extension API.

### Windows portable

Open Command Prompt in the portable folder that contains `python_embeded` and `ComfyUI`.

```bat
python_embeded\python.exe -m pip install -r "ComfyUI\custom_nodes\ComfyUI-OpenH3-VideoTools\requirements.txt"
```

Keep one active copy of OpenH3 VideoTools in `custom_nodes`.

### Other Python installations

Clone or extract the repository into `ComfyUI/custom_nodes/ComfyUI-OpenH3-VideoTools`, activate ComfyUI's Python environment, and install its requirements:

```text
python -m pip install -r ComfyUI/custom_nodes/ComfyUI-OpenH3-VideoTools/requirements.txt
```

Adjust the relative path for your current directory.

### Other required nodes

- [Songssx/ComfyUI-OpenH3-Regenerate2K](https://github.com/Songssx/ComfyUI-OpenH3-Regenerate2K): refinement adapter, audio lock and sigma schedule.
- [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes): attention/feed-forward helpers and final image-range selection.
- [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite): MP4 export.
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI): native H3 model, VAE, latent upscaler and video nodes.

Python requirements include PyAV and imageio-ffmpeg. The current workflow uses VideoHelperSuite H.264 MP4 export and requires an FFmpeg build with `libx264` support. No executable binaries, model weights or media are included.

## Controls

- **Prepare Video / source_scale:** `1.0` keeps the source's displayed dimensions after rotation and sample-aspect normalization. Smaller values reduce working input size. Only VAE alignment padding is added; no fixed 16:9 source canvas is imposed.
- **max_frames:** `0` disables the optional user frame cap. A positive value rejects an over-limit input instead of returning a shortened clip. Full frames still occupy RAM; this is not temporal streaming or unlimited-memory processing.
- **Upscale Latent / width and height:** manually editable output dimensions. Initial values are `2560 x 1440`. Use suitable multiples of 32, such as `1280 x 736`, `1920 x 1088`, or `2560 x 1440`. For portrait input, choose a matching portrait output such as `1440 x 2560`; input preparation does not automatically select output aspect.
- **Prompt:** the supplied text requests content and style preservation. It does not analyze the upload. Add accurate scene guidance if useful; generative reconstruction can still change details, text or motion.

The workflow uses timestamp-based 24 FPS preparation, native last-frame padding and final trimming to the actual frame count. Read [workflow behavior and limits](docs/WORKFLOW.md).

## Included helpers

| Technical node ID | Purpose |
|---|---|
| `TAOpenH3AutoInput` | Local source preparation, geometry, aligned audio and native frame padding |
| `TAOpenH3PromptEncodeRelease` | Prompt encoding and model lifetime management |
| `TAOpenH3RefineRelease` | Native refinement with spatial tiling and model release |
| `TAOpenH3TiledDiffusion` | Audio-locked spatial tiled refinement |
| `TAOpenH3VAEDecodeLowMemory` | Native VAE decoding and host image output |

Technical IDs retain their existing prefixes for workflow compatibility.

## License and credits

The helper source is GPL-3.0-only. Read [LICENSE](LICENSE), the retained [upstream MIT notice](LICENSE-MIT-UPSTREAM.txt), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Existing Tensor Alchemist copyright notices remain intact. Model weights and external dependencies retain their separate licenses, including the MiniMax H3 community terms where applicable.

This community integration is not an official NVIDIA, MiniMax or ComfyUI release.
