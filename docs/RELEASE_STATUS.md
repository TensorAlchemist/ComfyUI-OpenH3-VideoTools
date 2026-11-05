# Version 0.1.10 pre-release

This release fixes linked-video cache fingerprint handling. A missing linked value during ComfyUI's cache check returns a stable helper-version and control tuple, allowing normal Load Video selection and file-modification-time invalidation to work. Source validation and the execution path remain unchanged.

The workflow retains automatic source preparation, manually editable output width and height, the optional input-frame cap at zero, existing technical node IDs and seven preparation outputs.

## Verified scope

- On November 5, 2026, a seven-second local render using the shared inference implementation completed at 2560 x 1440 and 24 FPS in 22 minutes 59.471 seconds, with no reported out-of-memory failure.
- All 168 output video frames decoded with consistent dimensions and timestamps. The output contained 32 kHz stereo audio; decoded-waveform comparisons found zero estimated lag and correlations of 0.99927 and 0.99912.
- Seven matching source/output frame pairs were visually reviewed. The action was retained in those samples; this was not an exhaustive temporal or perceptual quality assessment.
- The exact published 0.1.9 ZIP passed a separate CPU package import, extension-entrypoint and five-node schema check.
- The 0.1.10 input-fingerprint change passed 13 CPU contract checks in 0.059 seconds. Its inference path is unchanged; a new full GPU render of 0.1.10 was not performed.
- After a normal-launcher restart, two live preparation-only checks confirmed version 0.1.10, correct source timing and cache reuse without the misleading warning. No additional video was generated.

The render used an existing Windows portable installation in a warm session. It does not prove fresh-install behavior, other hardware, arbitrary clip lengths or resolutions, or universal quality and memory compatibility. See [TEST_REPORT.md](TEST_REPORT.md) for the detailed environment and measurements.

No weights, media, local logs, private receipts, backups, user profile information or CI pipelines are included. Helper licenses and third-party attributions are retained.
