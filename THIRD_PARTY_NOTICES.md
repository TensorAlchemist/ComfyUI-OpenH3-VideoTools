# Third-party notices

This adapter is distributed under GPL-3.0-only, following upstream's stated GPLv3 license for its tiled diffusion portion.

Derived implementation: bbaudio-2025/Comfyui-MMH3-UltimateUpscale, nodes/tiled_diffusion.py, commit 0a8d5ee3914e8fe8bd842885412ec397d7ca1629.

https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale/blob/0a8d5ee3914e8fe8bd842885412ec397d7ca1629/nodes/tiled_diffusion.py

Copyright (c) 2026 bbaudio-2025. The original repository MIT permission notice is retained in LICENSE-MIT-UPSTREAM.txt. Its README explicitly marks the tiled diffusion code GPLv3 and credits https://github.com/shiimizu/ComfyUI-TiledDiffusion. The combined derived adapter is treated as GPLv3; the MIT notice does not replace those obligations.

Modifications: isolated TA node name and V3 registration; accept only full zero audio mask unchanged; validate H3 stream shapes and controls; refuse an existing model function wrapper; check host cancellation between tiles; remove retained tensor/payload cache. Full-frame global-position math and prediction blend are unchanged.

MiniMax H3 and Open-H3 model weights retain their separate model licenses. No weights are included.


## Automatic local input preparation
The new auto_input.py and media_prepare.py helper code is GPL-3.0-only. It invokes the FFmpeg executable supplied by the separately installed imageio-ffmpeg package; no FFmpeg binary is distributed in this source bundle. Preserve and consult that runtime distribution's license and notices. PyAV, NumPy and PyTorch remain external dependencies.
