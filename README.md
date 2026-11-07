# OpenH3 VideoTools

## Quick start

1. Use your normal ComfyUI installation and NVIDIA launcher, such as `run_nvidia_gpu.bat` for Windows portable. No special launcher or startup flags are required by this repository.
2. Install this repository as `ComfyUI/custom_nodes/ComfyUI-OpenH3-VideoTools`, with `__init__.py` directly inside that folder. Install the other required nodes listed below.
3. Install the helper requirements using the same Python environment that runs ComfyUI. Keep ComfyUI's existing PyTorch installation.
4. Download the files in [Models and Custom Nodes](docs/DOWNLOAD_LINKS.md) and place them in the listed folders. Keep the publisher filenames.
5. Restart ComfyUI, refresh the browser, and import [workflows/OpenH3_Regenerate2K.json](workflows/OpenH3_Regenerate2K.json).
6. Upload a local SDR video, select your installed models, and set width and height manually in **Upscale Latent**. Read the preparation report and review the complete result before publishing it.

## Installation

Requires Python 3.12 or newer and a ComfyUI build supporting native MiniMax H3, core Load Video, and the `comfy_api.latest` extension API. The November 5 local render used ComfyUI 0.44.0 and Python 3.12.10 on an NVIDIA RTX 5060 Laptop GPU with 8 GB VRAM and 32 GB system RAM. Compatibility with other environments is not established by that test.

### Windows portable

From the portable distribution folder containing `python_embeded` and `ComfyUI`:

```powershell
git clone https://github.com/TensorAlchemist/ComfyUI-OpenH3-VideoTools.git .\ComfyUI\custom_nodes\ComfyUI-OpenH3-VideoTools
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-OpenH3-VideoTools\requirements.txt
```

Alternatively, choose **Code > Download ZIP** on [the repository](https://github.com/TensorAlchemist/ComfyUI-OpenH3-VideoTools). Extract its top-level folder into `ComfyUI/custom_nodes`, rename it `ComfyUI-OpenH3-VideoTools`, and ensure `__init__.py` is at its root rather than one folder deeper.

Keep one active copy. Preserve any older VideoTools or `ComfyUI-TensorAlchemist-OpenH3-TiledDiffusion` backup outside `custom_nodes` to avoid duplicate node registrations. Start ComfyUI normally after setup; this repository does not replace launchers or change operating-system paging settings.

### Other Python installations

Clone or extract the repository into `ComfyUI/custom_nodes/ComfyUI-OpenH3-VideoTools`, activate ComfyUI's Python environment, and install its requirements:

```text
python -m pip install -r ComfyUI/custom_nodes/ComfyUI-OpenH3-VideoTools/requirements.txt
```

Adjust the relative path for your current directory. The recorded render used Windows portable; other platforms have not been validated by that test.

### Other required nodes

- [Songssx/ComfyUI-OpenH3-Regenerate2K](https://github.com/Songssx/ComfyUI-OpenH3-Regenerate2K): refinement adapter, audio lock and sigma schedule.
- [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes): attention/feed-forward helpers and final image-range selection.
- [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite): MP4 export.
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI): native H3 model, VAE, latent upscaler and video nodes.

Python requirements include PyAV and imageio-ffmpeg. The selected export also requires an FFmpeg build with working `h264_nvenc` and a compatible NVIDIA driver. No executable binaries, model weights or media are included.

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
