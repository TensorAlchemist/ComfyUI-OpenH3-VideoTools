# Version 0.1.8 pre-release

This publication packages the installed 0.1.8 helper source and the current editable manual-output workflow. Runtime compatibility of the publication copy remains unverified.

- Source-aspect preparation uses only 16-pixel VAE alignment padding.
- Output width and height are editable in the latent upscale node.
- The input helper retains seven outputs and has no output-size multiplier.
- The optional input frame cap is disabled at zero; memory and decoder limits remain.
- No new tests, imports, compilation, inference, UI checks, model conversions or CI checks were run for publication.
- The earlier 40 CPU checks were performed on version 0.1.6, not this version.
- The latest 0.1.8/manual-output configuration has not completed a full render. Earlier render evidence does not establish its quality, memory use or speed.

Publication-only source changes neutralize display labels and a missing-dependency message, add repository metadata, and preserve technical node IDs and legal attribution. The converter copy replaces a depth-dependent fallback path and uses exclusive hardlink publication to prevent replacement of an existing destination on Windows or POSIX. These converter changes received source review only; they were not executed for this release.

No model weights, media, local logs, private receipts, backups, user profile information or CI pipelines are part of the repository.
