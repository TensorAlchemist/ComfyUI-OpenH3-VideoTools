# Local validation, November 5, 2026

## Scope

A seven-second workflow render completed on an existing Windows portable installation using the same inference implementation and settings as the public workflow. The test used a warm session: 13 nodes executed and eight were cached, including prompt conditioning. Source preparation, encoding, refinement, decoding and export executed for this run.

The exact published 0.1.9 ZIP was checked separately for CPU import, its extension entrypoint and all five V3 node schemas; all passed without initializing CUDA. Version 0.1.10 adds an input cache-fingerprint fix that passed 13 separate CPU contract checks in 0.059 seconds. Its inference implementation is unchanged. A new full GPU render of 0.1.10 was not performed.

## Environment and settings

| Item | Recorded value |
|---|---|
| Launch | Normal Windows portable NVIDIA launcher |
| ComfyUI | 0.44.0 |
| Python | 3.12.10 |
| PyTorch | 2.14.0.dev20260621+cu132 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM |
| System RAM | 32 GB |
| Input | 1056 x 608, 7 seconds, 168 frames at 24 FPS, with audio |
| Output | 2560 x 1440, 7 seconds, 168 frames at 24 FPS |
| Sampler and steps | `res_multistep`, two refinement steps |
| Tiling | Gaussian, 1024 x 832 tiles, 224-pixel overlap |
| Memory patches | Four attention chunks, four feed-forward chunks, threshold 4096 |
| Export | NVENC H.264 MP4, 35 Mbps setting |
| Total execution time | 1379.471 seconds, or 22 minutes 59.471 seconds |

## Output checks

- All 168 video frames decoded successfully at 2560 x 1440. Frame timestamps were strictly increasing at the expected 24 FPS interval, covering seven seconds.
- The exported audio was 32 kHz stereo. A bounded decoded-waveform comparison estimated zero offset relative to the source, with channel correlations of 0.99927 and 0.99912. These are waveform consistency measurements, not an auditory-quality or lip-sync judgment.
- Seven matching source/output frame pairs sampled across the clip were visually reviewed. The action remained present in the samples. This does not establish artifact-free motion at every frame or quality for other subjects and styles.
- The render completed without a reported out-of-memory failure.

## Memory observations

Resources were sampled every 30 seconds. The highest sampled GPU memory reading was 7551 MiB, and the lowest sampled available system RAM was 1.977 GiB. These are sampled system readings affected by other processes; they are not precise allocator peaks or a memory requirement for other machines. RAM use was close to the capacity of this 32 GB system.

## Cache-fingerprint correction

ComfyUI can pass `None` for a linked video while computing cache fingerprints. The previous strict file-video guard raised a warning in that case, causing ComfyUI to assign a nonreusable fingerprint. Version 0.1.10 returns the helper version and preparation controls for that cache-check case. Normal ancestor tracking still accounts for the Load Video selection and file modification time.

Execution-time validation and the real-video file-stat branch are unchanged. This is not content hashing, and it does not alter the automatic input dimensions, source duration, manually selected output dimensions or render settings.

The corrected helper was then installed on the same machine and ComfyUI restarted with the existing normal NVIDIA launcher and unchanged startup arguments. Two input-preparation-only runs reported version 0.1.10 and the expected 168 frames and seven-second duration. The first took 3.922 seconds; the second reused all three probe nodes, including preparation, in 0.062 seconds. The misleading cache warning was absent. These timings concern input preparation only, not video generation or upscaling.

## Limits

This is one render on one existing installation, plus separate package-import and cache-contract checks. It does not establish clean installation on another computer, other operating systems, all clip lengths, portrait rendering, arbitrary output resolutions, cold-start timing, or universal visual quality and memory compatibility. No test media or local logs are distributed with this report.
