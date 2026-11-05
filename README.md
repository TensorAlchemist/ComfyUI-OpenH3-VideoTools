# OpenH3 VideoTools

ComfyUI helpers and an editable OpenH3 refinement workflow for local video input. The workflow prepares the uploaded source, performs learned latent upscaling and two-step refinement, and exports an MP4 with the source soundtrack.

**Version 0.1.8 is a pre-release.** The latest manual-output configuration has not completed a full render. No new tests, imports, compilation, inference, frontend checks or CI were run for this publication copy. Earlier 40 CPU checks belonged to version 0.1.6 and do not validate version 0.1.8. See [release status](docs/RELEASE_STATUS.md).

## Quick start

1. Use your normal ComfyUI installation and normal NVIDIA launcher, such as `run_nvidia_gpu.bat` for Windows portable. No special launcher or memory flags are required by this repository.
2. Install this repository as `ComfyUI/custom_nodes/ComfyUI-OpenH3-VideoTools`. Its `__init__.py` must be directly inside that folder. Install the other node dependencies listed below.
3. Install this helper's Python requirements with the same Python environment that runs ComfyUI. Keep ComfyUI's existing PyTorch installation.
4. Download the model files separately using [Models and Custom Nodes](docs/DOWNLOAD_LINKS.md). The supplied workflow selects two FP16 adapter storage copies; follow [REPRODUCE_FP16.txt](docs/REPRODUCE_FP16.txt) to create them from the publisher's exact FP32 originals. Renaming a file does not convert its precision.
5. Restart ComfyUI after installation, refresh the browser, then import [workflows/OpenH3_Regenerate2K.json](workflows/OpenH3_Regenerate2K.json).
6. Upload a local SDR video. In **Upscale Latent**, enter the output width and height manually. Select the installed model filenames, read the preparation report, and review the result before publishing it.

## Installation

Requires Python 3.12 or newer and a ComfyUI build with native MiniMax H3, core Load Video, and the `comfy_api.latest` extension API. The development target was ComfyUI 0.43.0; compatibility with every other release is not established.

### Windows portable

From the portable distribution folder containing `python_embeded` and `ComfyUI`:

```powershell
git clone https://github.com/TensorAlchemist/ComfyUI-OpenH3-VideoTools.git .\ComfyUI\custom_nodes\ComfyUI-OpenH3-VideoTools
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-OpenH3-VideoTools\requirements.txt
```

Alternatively, choose **Code > Download ZIP** on [the repository](https://github.com/TensorAlchemist/ComfyUI-OpenH3-VideoTools). Extract the top-level repository folder into `ComfyUI/custom_nodes`, rename it `ComfyUI-OpenH3-VideoTools`, and confirm that `__init__.py` is at its root rather than one folder deeper.

Keep one active copy. If replacing an older `ComfyUI-TensorAlchemist-OpenH3-TiledDiffusion` or duplicate VideoTools folder, preserve its backup outside `custom_nodes`; do not leave both active. Technical node IDs are retained for workflow compatibility.

Start ComfyUI normally after setup. This repository does not replace an existing launcher or change operating-system paging settings.

### Other Python installations

Clone or extract the repository into `ComfyUI/custom_nodes/ComfyUI-OpenH3-VideoTools`, activate the Python environment that runs ComfyUI, then run:

```text
python -m pip install -r ComfyUI/custom_nodes/ComfyUI-OpenH3-VideoTools/requirements.txt
```

Adjust the relative path for your current directory. Platform-specific runtime compatibility has not been tested for this pre-release.

### Other required nodes

- [Songssx/ComfyUI-OpenH3-Regenerate2K](https://github.com/Songssx/ComfyUI-OpenH3-Regenerate2K): refinement adapter, audio lock and sigma schedule.
- [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes): attention/feed-forward helpers and final image-range selection.
- [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite): MP4 export.
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI): native H3 model, VAE, latent upscaler and video nodes.

Python requirements include PyAV and imageio-ffmpeg. The selected MP4 preset also needs an FFmpeg build with working `h264_nvenc` and a compatible NVIDIA driver. No FFmpeg executable, model weights or media are included.

## Input and output controls

- **Prepare Video / source_scale:** `1.0` keeps the source's displayed pixel dimensions after rotation and sample-aspect normalization. Smaller values reduce working input dimensions. Source mode adds only the padding required for the VAE's 16-pixel grid; it does not impose a 16:9 canvas.
- **max_frames:** `0` disables the optional user frame cap. A positive value rejects an over-limit input instead of returning a truncated clip. Full frames still occupy host memory; this is not temporal streaming or an unlimited-memory mode.
- **Upscale Latent / width and height:** editable manual output dimensions. The initial values are `2560 x 1440`, not automatically connected limits. Use suitable multiples of 32, for example `1280 x 736`, `1920 x 1088`, or `2560 x 1440`. For portrait input, select an appropriate portrait output such as `1440 x 2560`. Source-aspect input preparation does not select the output aspect automatically; a mismatch can change source geometry.
- **Prompt:** the supplied text requests content and style preservation. It does not caption or analyze the uploaded video. Add accurate scene guidance if needed. Reconstruction can change identities, fine detail, text or motion.

The helper uses timestamp-based 24 FPS preparation, native last-frame padding and final trimming to the actual prepared frame count. Silent sources use silent conditioning without inventing a final soundtrack. Existing decoder geometry/security limits remain, and large inputs or outputs can exceed available RAM or VRAM. Read [workflow behavior and limits](docs/WORKFLOW.md).

## Models and FP16 conversion

The example selects Singularity V2 pruned INT8, a Qwen3-VL H3 text encoder, stock video/audio VAEs, a learned latent upscaler, and OpenH3 refinement components. Exact filenames and destination folders are in [the download guide](docs/DOWNLOAD_LINKS.md).

`open_h3_regenerate_2k_lora_fp16.safetensors` and `sharp_curve_branches_fp16.safetensors` are locally converted storage copies. They are not upstream downloads with `_fp16` appended to their names. Use the supplied converter only after obtaining the exact original files and reading the reproduction instructions. The converter does not download or run a model.

The original FP32 components may instead be selected in the corresponding adapter fields. This is an optional loader-code-compatible route, not a RAM-verified configuration or a measured alternative for this release. Keep the Sharp branches in their dedicated `models/open_h3_regenerate/lora` folder rather than loading them as an ordinary LoRA.

## Included helpers

| Technical node ID | Purpose |
|---|---|
| `TAOpenH3AutoInput` | Local source preparation, geometry, audio alignment and native frame padding |
| `TAOpenH3PromptEncodeRelease` | Prompt encoding with model lifetime management |
| `TAOpenH3RefineRelease` | Native refinement with optional spatial tiling and model release |
| `TAOpenH3TiledDiffusion` | Audio-locked spatial tiled refinement |
| `TAOpenH3VAEDecodeLowMemory` | Native decode with an FP16 host image buffer |

The `TA` prefix in technical IDs is retained so existing workflows can resolve the same classes. It is separate from the neutral display labels.

## License and credits

The helper source is GPL-3.0-only. Read [LICENSE](LICENSE), the retained [upstream MIT notice](LICENSE-MIT-UPSTREAM.txt), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Existing Tensor Alchemist copyright notices remain intact. Model weights and external dependencies retain their separate licenses, including the MiniMax H3 community terms where applicable.

This community integration is not an official NVIDIA, MiniMax or ComfyUI release. Research and component sources are credited in the download guide and third-party notices.
