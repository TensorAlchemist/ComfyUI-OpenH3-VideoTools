# SPDX-License-Identifier: GPL-3.0-only
"""Bounded-child PyAV probe. Standalone CLI is private to media_prepare."""
import json
import math
from fractions import Fraction
import sys
import av


def seconds(value, base):
    return None if value is None or base is None else float(value * base)


def geometry(width, height):
    if min(width, height) < 1 or max(width, height) > 8192 or width * height > 33554432:
        raise ValueError("Input exceeds the bounded decoder geometry (8192 per side / 32 megapixels).")


def inspect(path, demuxer):
    if demuxer not in {"mov", "matroska", "avi", "mpegts", "mpeg", "asf"}:
        raise ValueError("Unsupported media demuxer.")
    options = {"protocol_whitelist": "file,pipe"}
    try:
        with av.open(path, format=demuxer, options=options) as container:
            if not container.streams.video:
                raise ValueError("The uploaded file contains no video track.")
            stream = container.streams.video[0]
            geometry(stream.width, stream.height)  # BEFORE any full-size decode
            stream.thread_type = "SLICE"
            stream.codec_context.thread_count = 2
            first = next(container.decode(stream), None)
            if first is None or first.pts is None:
                raise ValueError("Video has no decodable timestamped frame; remux to a timestamped MP4 first.")
            geometry(first.width, first.height)
            transfers = {int(first.color_trc or 0), int(stream.codec_context.color_trc or 0)}
            if transfers & {16, 18}:
                raise ValueError("HDR PQ/HLG is not supported by this SDR workflow. Tone-map to SDR before uploading.")
            transfer = next((value for value in transfers if value not in (0, 2)), int(first.color_trc or 0))
            rotation = float(first.rotation or 0)
            if abs(rotation / 90 - round(rotation / 90)) > 0.001:
                raise ValueError("Only 0/90/180/270-degree display rotation is supported. Normalize rotation first.")
            sar = stream.sample_aspect_ratio or Fraction(1, 1)
            if not math.isfinite(float(sar)) or sar <= 0:
                raise ValueError("Invalid sample-aspect ratio. Normalize it before uploading.")
            video_start = seconds(first.pts, first.time_base)
            duration = seconds(stream.duration, stream.time_base)
            basis = "stream.duration"
            if duration is None and "DURATION" in stream.metadata:
                try:
                    h, m, s = stream.metadata["DURATION"].split(":")
                    duration = int(h)*3600 + int(m)*60 + float(s) - video_start
                    basis = "Matroska stream DURATION end tag minus first video PTS"
                except (ValueError, TypeError):
                    duration = None
            if not math.isfinite(video_start) or (duration is not None and (not math.isfinite(duration) or duration <= 0)):
                raise ValueError("Invalid video timestamps/duration; remux first.")
            audio = container.streams.audio[0] if container.streams.audio else None
            info = {"demuxer": demuxer, "video_stream_index": stream.index,
                    "coded_width": first.width, "coded_height": first.height,
                    "sample_aspect_ratio": str(sar), "rotation_degrees": rotation,
                    "video_first_pts_seconds": video_start, "video_duration_estimate_seconds": duration,
                    "duration_estimate_basis": basis if duration else None,
                    "probe_cfr_frame_estimate": round(duration * 24) if duration else None,
                    "nominal_rate": str(stream.average_rate), "transfer_characteristic": transfer,
                    "audio_stream_index": audio.index if audio is not None else None}
        if info["audio_stream_index"] is not None:
            with av.open(path, format=demuxer, options=options) as container:
                audio = container.streams.audio[0]
                audio.codec_context.thread_count = 1
                first = next(container.decode(audio), None)
                if first is None or first.pts is None:
                    raise ValueError("First audio track is present but empty or has no usable timestamps.")
                start = seconds(first.pts, first.time_base)
                offset = start-video_start
                if not math.isfinite(offset) or abs(offset)>60:
                    raise ValueError("Audio/video start offset exceeds supported +/-60 seconds; remux first.")
                info.update(audio_channels=len(first.layout.channels),audio_layout=first.layout.name,
                            audio_input_sample_rate=first.sample_rate,audio_first_pts_seconds=start,
                            audio_start_offset_seconds=offset)
        return info
    except av.error.FFmpegError as error:
        raise ValueError("Could not decode uploaded video/first audio track. Repair or remux first.") from error


if __name__ == "__main__":
    try:
        result = {"ok":True,"info":inspect(sys.argv[1],sys.argv[2])}
    except Exception as error:
        result = {"ok":False,"error":str(error) if isinstance(error,ValueError) else "Could not probe uploaded media."}
    print(json.dumps(result))
