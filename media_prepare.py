# SPDX-License-Identifier: GPL-3.0-only
"""Bounded CPU preparation of a local uploaded clip; no model or GPU calls."""

from fractions import Fraction
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import imageio_ffmpeg
import numpy as np
import torch

WIDTH, HEIGHT, FPS, SAMPLE_RATE = 1024, 576, 24, 32000
WORKING_RESOLUTIONS = {"source": None, "1024x576": (1024, 576), "1280x720": (1280, 720)}
MAX_FRAMES = 2**31 - 1
VERSION = "0.1.10"
# Force a real media demuxer before probing: never let a renamed playlist
# fetch URLs or follow a concat manifest's local paths.
FORMATS = {
    ".mp4": "mov", ".mov": "mov", ".m4v": "mov", ".3gp": "mov",
    ".mkv": "matroska", ".webm": "matroska", ".avi": "avi",
    ".ts": "mpegts", ".mts": "mpegts", ".m2ts": "mpegts",
    ".mpg": "mpeg", ".mpeg": "mpeg", ".wmv": "asf",
}


def padded_length(count):
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("The clip must contain at least one decoded CFR frame.")
    return max(124, 17 * math.ceil((count - 5) / 17) + 5)


def sample_count(count):
    return round(Fraction(count * SAMPLE_RATE, FPS))


def validate_working_resolution(working_resolution):
    if not isinstance(working_resolution, str) or working_resolution not in WORKING_RESOLUTIONS:
        raise ValueError("working_resolution must be source, 1024x576 or 1280x720.")


def displayed_dimensions(info):
    """Square-pixel display geometry; normalize SAR before applying rotation."""
    width, height = info["coded_width"], info["coded_height"]
    if (isinstance(width, bool) or isinstance(height, bool) or
            not isinstance(width, int) or not isinstance(height, int) or min(width, height) < 1):
        raise ValueError("Invalid source dimensions.")
    sar = Fraction(info["sample_aspect_ratio"])
    rotation = float(info["rotation_degrees"])
    if sar <= 0 or not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > 0.001:
        raise ValueError("Invalid source sample aspect ratio or display rotation.")
    # Round only the square-pixel SAR conversion, never scale to fill a canvas.
    width = max(1, math.floor(width * sar + Fraction(1, 2)))
    if int(round(rotation / 90)) % 2:
        width, height = height, width
    return width, height


def validate_source_scale(source_scale):
    if (isinstance(source_scale, bool) or not isinstance(source_scale, (int, float)) or
            not math.isfinite(source_scale) or not 0.01 <= source_scale <= 16.0):
        raise ValueError("source_scale must be finite and between 0.01 and 16.0.")


def validate_working_geometry(width, height, label="Prepared geometry"):
    # Reuse the existing decoder/probe budget for derived square-pixel content
    # and padded canvases; never silently resize to satisfy this safety check.
    if (isinstance(width, bool) or isinstance(height, bool) or
            not isinstance(width, int) or not isinstance(height, int) or
            min(width, height) < 1 or max(width, height) > 8192 or width * height > 33554432):
        raise ValueError(f"{label} exceeds the existing decoder geometry budget "
                         "(8192 per side / 32 megapixels). Choose a smaller source_scale "
                         "or an explicit preset; no automatic reduction was applied.")


def scaled_displayed_dimensions(info, source_scale=1.0):
    validate_source_scale(source_scale)
    width, height = displayed_dimensions(info)
    scale = Fraction(str(source_scale))
    dimensions = (max(1, math.floor(width * scale + Fraction(1, 2))),
                  max(1, math.floor(height * scale + Fraction(1, 2))))
    validate_working_geometry(*dimensions, label="Scaled displayed content")
    return dimensions


def working_dimensions(working_resolution, info=None, source_scale=1.0):
    validate_working_resolution(working_resolution)
    if working_resolution != "source":
        dimensions = WORKING_RESOLUTIONS[working_resolution]
        validate_working_geometry(*dimensions, label="Working canvas")
        return dimensions
    if info is None:
        raise ValueError("Source mode needs probed display geometry.")
    width, height = scaled_displayed_dimensions(info, source_scale)
    # Preserve source aspect; add only the VAE's 16-pixel spatial alignment.
    dimensions = (((width + 15) // 16) * 16, ((height + 15) // 16) * 16)
    validate_working_geometry(*dimensions, label="Working canvas")
    return dimensions


def validate_frame_limit(max_frames):
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or not 0 <= max_frames <= MAX_FRAMES:
        raise ValueError("max_frames must be an integer from 0 to 2147483647; 0 means no user frame cap.")


def frame_limit_exceeded(count, max_frames):
    return max_frames > 0 and count > max_frames


def video_filter(info, width, height, working_resolution, source_scale=1.0):
    prefix = "setpts=PTS-STARTPTS,fps=fps=24:start_time=0:round=near:eof_action=pass,"
    if working_resolution == "source":
        # FFmpeg autorotates before this filter. Explicit dimensions normalize
        # SAR to square pixels, then retain that native display size inside PAD.
        source_width, source_height = scaled_displayed_dimensions(info, source_scale)
        scale = f"scale=w={source_width}:h={source_height}:flags=lanczos,"
    else:
        # Backward-compatible, explicitly selected preset FIT behavior.
        scale = (f"scale=w='max(1,trunc(min({width},{height}*dar)))':"
                 f"h='max(1,trunc(min({height},{width}/dar)))':flags=lanczos,")
    return prefix + scale + f"setsar=1,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,format=rgb24"


def resolve_input(source, input_root):
    if not isinstance(source, str) or not source or "://" in source or source.startswith("\\\\"):
        raise ValueError("Use core Load Video with a local uploaded file, not a URL or memory stream.")
    root = Path(input_root).resolve(strict=True)
    path = Path(source).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError("Video must be an existing uploaded file inside ComfyUI/input.")
    if path.suffix.lower() not in FORMATS:
        raise ValueError("Unsupported uploaded container. Remux to MP4, MOV, MKV, WebM, AVI, TS or MPEG first.")
    return path


def find_ffmpeg():
    # Only use the existing bundled binary. No installer, PATH shell, download,
    # user-provided executable or environment-selected override is invoked.
    binaries = Path(imageio_ffmpeg.__file__).resolve().parent / "binaries"
    candidates = sorted(p for p in binaries.glob("ffmpeg-*") if p.is_file() and
                        (p.suffix.lower() == ".exe" if os.name == "nt" else os.access(p, os.X_OK)))
    if len(candidates) != 1:
        raise RuntimeError("A single bundled imageio-ffmpeg binary is required; no automatic installation is performed.")
    return candidates[0]


def probe(path, interrupt):
    # PyAV demux/first-frame decoding can block in native code. Own it in a
    # tiny CPU-only subprocess too, so cancellation/deadline can always reap it.
    command = [sys.executable, "-X", "utf8", "-B", str(Path(__file__).with_name("media_probe.py")),
               str(path), FORMATS[path.suffix.lower()]]
    iterator = _stream_units(command, 4096, interrupt, "Probe", timeout=30, allow_partial=True)
    try:
        blocks = []
        for block in iterator:
            blocks.append(block)
            if sum(map(len, blocks)) > 65536:
                raise ValueError("Unexpected oversized probe report.")
        envelope = json.loads(b"".join(blocks))
    finally:
        iterator.close()
    if not envelope["ok"]:
        raise ValueError(envelope["error"])
    return envelope["info"]


def _stream_units(command, unit_bytes, interrupt, label, timeout=300, allow_partial=False):
    """Drain bounded stdout/stderr queues, checking cancellation every 0.1s.

    The only process that may be terminated is this exact Popen child. No PID
    search, shell invocation, inherited stdin, temporary media or model loads.
    """
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(command, shell=False, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=flags)
    pending = queue.Queue(maxsize=2)
    stop = threading.Event()

    def send(value):
        while not stop.is_set():
            try:
                pending.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def read_stdout():
        try:
            while not stop.is_set():
                block = process.stdout.read(unit_bytes)
                if not block:
                    break
                if len(block) != unit_bytes and not allow_partial:
                    send(("error", RuntimeError(f"{label}: incomplete decoded output unit.")))
                    return
                send(("data", block))
        except (OSError, ValueError) as error:
            if not stop.is_set():
                send(("error", error))
        finally:
            send(("end", None))

    def read_stderr():
        try:
            while True:
                block = process.stderr.read(4096)
                if not block:
                    break
        except (OSError, ValueError):
            pass

    workers = [threading.Thread(target=read_stdout, daemon=True),
               threading.Thread(target=read_stderr, daemon=True)]
    for worker in workers:
        worker.start()
    started = time.monotonic()
    try:
        while True:
            interrupt()
            if time.monotonic() - started > timeout:
                raise RuntimeError(f"{label}: CPU decoding exceeded {timeout} seconds; use a simpler remuxed source.")
            try:
                kind, value = pending.get(timeout=0.1)
            except queue.Empty:
                continue
            if kind == "error":
                raise RuntimeError(f"{label}: could not read decoded media.") from value
            if kind == "end":
                break
            yield value
        code = process.wait(timeout=10)
        workers[1].join(timeout=2)
        if code:
            # Avoid publishing source paths in ffmpeg's diagnostics.
            raise RuntimeError(f"{label}: FFmpeg failed (exit {code}); present audio/video cannot be decoded safely.")
    finally:
        stop.set()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        for pipe in (process.stdout, process.stderr):
            pipe.close()
        for worker in workers:
            worker.join(timeout=2)


def _base_command(binary, path, info):
    return [str(binary), "-hide_banner", "-loglevel", "error", "-nostdin", "-xerror",
            "-err_detect", "explode", "-threads", "2", "-filter_threads", "1",
            "-protocol_whitelist", "file,pipe", "-copyts", "-f", info["demuxer"], "-i", str(path)]


def prepare_file(source, input_root, max_frames=0, *, working_resolution="source", source_scale=1.0, interrupt=lambda: None, ffmpeg_binary=None):
    validate_frame_limit(max_frames)
    validate_working_resolution(working_resolution)
    validate_source_scale(source_scale)
    path = resolve_input(source, input_root)
    info = probe(path, interrupt)
    width, height = working_dimensions(working_resolution, info, source_scale)
    estimate = info["probe_cfr_frame_estimate"]
    # Probe duration metadata is advisory (especially VFR/Matroska). Even an
    # overestimate is validated by actual bounded decode rather than rejecting
    # a valid clip. With an explicit positive cap, read at most cap+1 before failing.
    binary = Path(ffmpeg_binary) if ffmpeg_binary else find_ffmpeg()
    fit = video_filter(info, width, height, working_resolution, source_scale)
    command = _base_command(binary, path, info) + [
        "-map", f"0:{info['video_stream_index']}", "-an", "-sn", "-dn", "-vf", fit,
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    frames = []
    iterator = _stream_units(command, width * height * 3, interrupt, "Video")
    try:
        for raw in iterator:
            frames.append(raw)
            if frame_limit_exceeded(len(frames), max_frames):
                raise ValueError(f"Actual CFR24 decode exceeds max_frames={max_frames}. No shortened clip is returned. "
                                 "Shorten externally or explicitly raise the limit for stronger hardware.")
    finally:
        iterator.close()
    count = len(frames)
    if not count:
        raise ValueError("Video decoded to zero CFR24 frames. No prepared clip was returned.")
    length = padded_length(count)
    interrupt()
    images = torch.empty((length, height, width, 3), dtype=torch.float32, device="cpu")
    target = images.numpy()
    for index in range(count):
        interrupt()
        frame = np.frombuffer(frames[index], np.uint8).reshape(height, width, 3)
        np.multiply(frame, 1.0 / 255.0, out=target[index], casting="unsafe")
        frames[index] = None
    frames.clear()
    for index in range(count, length):
        interrupt()
        target[index] = target[count - 1]

    wanted = sample_count(count)
    conditioning = torch.zeros((1, 2, sample_count(length)), dtype=torch.float32, device="cpu")
    source_audio = None
    warnings = []
    if info["audio_stream_index"] is not None:
        # copyts retains the original timeline. Offset to the FIRST VIDEO PTS;
        # async resampling trims early audio / inserts silence before late audio,
        # then pads or trims to the actual CFR frame span (nearest sample).
        start = format(info["video_first_pts_seconds"], ".12g")
        af = (f"asetpts=PTS-({start})/TB,aresample=32000:async=1:first_pts=0,"
              f"apad=whole_len={wanted},atrim=end_sample={wanted},asetpts=N/SR/TB")
        audio_command = _base_command(binary, path, info) + [
            "-map", f"0:{info['audio_stream_index']}", "-vn", "-sn", "-dn", "-af", af,
            "-ac", "2", "-ar", "32000", "-f", "f32le", "pipe:1"]
        audio_iterator = _stream_units(audio_command, wanted * 2 * 4, interrupt, "Audio")
        try:
            blocks = list(audio_iterator)
        finally:
            audio_iterator.close()
        if len(blocks) != 1:
            raise ValueError("The first audio track is present but failed exact-length decode. It was not treated as silent video.")
        waveform = np.frombuffer(blocks[0], dtype="<f4").reshape(wanted, 2).T.copy()
        if not np.isfinite(waveform).all():
            raise ValueError("Decoded audio contains non-finite values.")
        source_audio = {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": SAMPLE_RATE}
        conditioning[:, :, :wanted].copy_(source_audio["waveform"])
        channels = info["audio_channels"]
        if channels == 1:
            warnings.append("Mono source duplicated to stereo by FFmpeg.")
        elif channels > 2:
            warnings.append(f"First audio track ({channels} channels) downmixed to stereo by FFmpeg default coefficients.")
        else:
            warnings.append("First stereo audio track resampled to 32 kHz; no gain normalization or compression requested.")
    else:
        warnings.append("No audio track: final source_audio=None; conditioning uses 32 kHz stereo zeros.")
    if info["transfer_characteristic"] in (0, 2):
        warnings.append("Source transfer tags unspecified; treated as SDR. This is not HDR detection from pixel values.")
    if max_frames == 0:
        warnings.append("No user frame cap. The full clip and native padding still materialize in host RAM; available RAM/VRAM and the decoder deadline remain finite.")
    elif max_frames > 175:
        warnings.append("Raised frame limit is for stronger hardware; no 8 GB or long-pipeline execution warranty.")
    if working_resolution == "source":
        warnings.append("Source aspect is retained with only 16-pixel alignment padding at the explicit source_scale; large inputs/scales can greatly increase RAM/VRAM use.")
    if estimate is not None and estimate != count:
        warnings.append(f"Probe duration estimated {estimate} CFR frames; actual timestamp-aware decode produced {count}. Actual count is authoritative.")
    report = {
        "schema_version": "1.0", "implementation_version": VERSION,
        "source_name": path.name, "input_probe": info,
        "cfr": {"fps": FPS, "actual_frame_count": count, "duration_seconds": count / FPS,
                "timestamp_rule": "Normalize first video PTS to zero; FFmpeg fps24 round-near/eof-pass preserves the last frame's display interval; duplicate/drop by timestamps, never retime via nominal FPS."},
        "geometry": {"width": width, "height": height, "working_resolution": working_resolution,
                     "displayed_source_dimensions": list(displayed_dimensions(info)),
                     "source_scale": source_scale if working_resolution == "source" else None,
                     "scaled_content_dimensions": (list(scaled_displayed_dimensions(info, source_scale))
                                                   if working_resolution == "source" else None),
                     "mode": ("SAR-normalized display size at source_scale + minimal 16-pixel alignment PAD; no additional fit-to-canvas upscale"
                              if working_resolution == "source" else
                              "SAR-aware full-frame FIT + centered black PAD; default FFmpeg autorotation"),
                     "cropped": False},
        "native_padding": {"prepared_frame_count": length, "last_frame_repeats": length - count,
                           "rule": "max(124,17*ceil((N-5)/17)+5)", "conditioning_audio_samples": sample_count(length),
                           "source_audio_samples": wanted if source_audio is not None else None},
        "audio": {"present": source_audio is not None, "sample_rate": SAMPLE_RATE, "channels": 2,
                  "offset_alignment": "Original audio PTS minus first video PTS; early trim/late silence; nearest-sample duration and zero tail padding."},
        "max_frames": max_frames, "warnings": warnings,
        "limits": ["CPU IMAGE batch is materialized after geometry preparation; this is not streaming temporal H3 refinement or an unlimited-memory path.",
                   "Reads the complete underlying upload; connect raw LoadVideo directly. VIDEO-object trim/crop edits are not applied.",
                   "PQ/HLG rejected; unknown tags are not a guarantee of SDR mastering.",
                   "FIT/PAD applies to input preparation only. A generative refinement model may redraw black bars; exact final letterboxing is not guaranteed.",
                   "Output width and height are selected manually in the downstream latent upscale node; this node does not run a model."]}
    interrupt()
    return images, {"waveform": conditioning, "sample_rate": SAMPLE_RATE}, source_audio, count, float(FPS), count / FPS, json.dumps(report, indent=2)
