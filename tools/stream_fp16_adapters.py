"""Cast the two authorized Open-H3 adapter files with bounded CPU memory.

No safetensors/torch imports, GPU work, inference, downloads, or source writes.
The original file is hashed while its payload is read. A destination is published
only after finite/range checks, the expected source hash, and file verification.
"""

import datetime
import hashlib
import json
import math
import os
import pathlib
import struct
import time

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
# The supported entry point supplies --comfyui-root before calling cast_one.
# A repository-relative fallback also permits shallow checkout locations.
ROOT = HERE.parent
CHUNK_BYTES = 4 * 1024 * 1024
SPECS = [
    {
        'source': 'ComfyUI/models/loras/open_h3_regenerate_2k_lora.safetensors',
        'destination': 'ComfyUI/models/loras/open_h3_regenerate_2k_lora_fp16.safetensors',
        'sha256': '8e36a0912ab72f7f3b00f41e3c7c97c7dfdec21e66c88ab22f08b9b0c681bb04',
        'tensor_count': 624,
        'modules': 312,
        'label': 'main',
    },
    {
        'source': 'ComfyUI/models/open_h3_regenerate/lora/sharp_curve_branches.safetensors',
        'destination': 'ComfyUI/models/open_h3_regenerate/lora/sharp_curve_branches_fp16.safetensors',
        'sha256': '26ff1c4ea50fc5b0cea252912b67ec308a65d534cef026ef56f415ae92102e4e',
        'tensor_count': 100,
        'modules': 50,
        'label': 'sharp',
    },
]


def header(path):
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Incomplete safetensors prefix')
        size = struct.unpack('<Q', prefix)[0]
        if size > 100_000_000:
            raise ValueError('Unreasonable safetensors header')
        raw = stream.read(size)
    if len(raw) != size:
        raise ValueError('Incomplete safetensors header')
    parsed = json.loads(raw)
    tensors = {key: value for key, value in parsed.items() if key != '__metadata__'}
    offset = 0
    for key, value in sorted(tensors.items(), key=lambda pair: pair[1]['data_offsets'][0]):
        start, end = value['data_offsets']
        if value['dtype'] not in ['F32', 'F16']:
            raise ValueError(f'Unsupported source tensor dtype: {key}: {value["dtype"]}')
        itemsize = 4 if value['dtype'] == 'F32' else 2
        if start != offset or end - start != math.prod(value['shape']) * itemsize:
            raise ValueError(f'Invalid tensor packing or byte size: {key}')
        offset = end
    if path.stat().st_size != 8 + size + offset:
        raise ValueError('Declared payload does not match physical file size')
    return parsed, prefix + raw, 8 + size


def read_rows(path, h, base, key, rows):
    info = h[key]
    shape = info['shape']
    if len(shape) != 2:
        raise ValueError(f'Expected a matrix: {key}')
    dtype = '<f4' if info['dtype'] == 'F32' else '<f2'
    row_bytes = shape[1] * np.dtype(dtype).itemsize
    values = []
    with path.open('rb') as stream:
        for row in rows:
            stream.seek(base + info['data_offsets'][0] + int(row) * row_bytes)
            raw = stream.read(row_bytes)
            if len(raw) != row_bytes:
                raise ValueError(f'Incomplete matrix row: {key}')
            values.append(np.frombuffer(raw, dtype=dtype).astype(np.float32))
    return np.stack(values)


def sampled_delta_checks(source, destination, label):
    sh, _, sb = header(source)
    dh, _, db = header(destination)
    a_suffix = '.lora_A.weight' if label == 'main' else '.curve_A.weight'
    b_suffix = '.lora_B.weight' if label == 'main' else '.curve_B.weight'
    roots = sorted(key[:-len(a_suffix)] for key in sh if key.endswith(a_suffix))
    # Sample complete factor columns/rows from roots spread across the file.
    indices = sorted(set(np.linspace(0, len(roots) - 1, 16 if label == 'main' else 5, dtype=int)))
    checks = []
    for index in indices:
        root = roots[index]
        a_key, b_key = root + a_suffix, root + b_suffix
        rank, inputs = sh[a_key]['shape']
        outputs, b_rank = sh[b_key]['shape']
        if rank != b_rank:
            raise ValueError(f'Invalid adapter factor ranks: {root}')
        rows = np.unique(np.linspace(0, outputs - 1, min(8, outputs), dtype=int))
        cols = np.unique(np.linspace(0, inputs - 1, min(8, inputs), dtype=int))
        # A is at most ~12MB for these files; only one factor is read at a time.
        sa = read_rows(source, sh, sb, a_key, range(rank))[:, cols]
        da = read_rows(destination, dh, db, a_key, range(rank))[:, cols]
        s_b = read_rows(source, sh, sb, b_key, rows)
        d_b = read_rows(destination, dh, db, b_key, rows)
        original = s_b @ sa
        derived = d_b @ da
        error = derived.astype(np.float64) - original.astype(np.float64)
        base_rms = float(np.sqrt(np.mean(original.astype(np.float64) ** 2)))
        error_rms = float(np.sqrt(np.mean(error ** 2)))
        if not np.isfinite(derived).all():
            raise ValueError(f'Nonfinite sampled effective delta: {root}')
        checks.append({
            'root': root, 'rank': rank, 'sample_rows': rows.tolist(), 'sample_columns': cols.tolist(),
            'source_delta_rms': base_rms, 'difference_rms': error_rms,
            'difference_max_absolute': float(np.max(np.abs(error))),
            'relative_rms': error_rms / base_rms if base_rms else None,
        })
    return {
        'method': 'Selected entries of B@A in float32, comparing original F32 factors with F16 factors upcast to F32. Strength and alpha/rank are already folded in source metadata.',
        'scope': 'Numerical storage-cast check, not a render or a guarantee of runtime BF16 equivalence.',
        'sampled_modules': len(checks), 'checks': checks,
    }


def cast_one(spec):
    source, destination = ROOT / spec['source'], ROOT / spec['destination']
    stage = destination.with_name(destination.name + '.incomplete')
    if destination.exists() or stage.exists():
        raise FileExistsError(f'Refusing to overwrite destination or staging file: {destination.name}')
    source_stat = source.stat()
    sh, source_prefix, data_base = header(source)
    tensors = {key: value for key, value in sh.items() if key != '__metadata__'}
    if len(tensors) != spec['tensor_count'] or any(v['dtype'] != 'F32' for v in tensors.values()):
        raise ValueError('Source tensor count or all-F32 contract changed')
    dh = {}
    metadata = dict(sh.get('__metadata__', {}))
    if metadata.get('dtype') == 'float32':
        metadata['dtype'] = 'float16'
    metadata['tensor_alchemist.storage_cast'] = 'F32 to F16 storage; original sources preserved; inference equivalence requires runtime verification'
    metadata['tensor_alchemist.source_sha256'] = spec['sha256']
    dh['__metadata__'] = metadata
    offset = 0
    ordered = sorted(tensors.items(), key=lambda pair: pair[1]['data_offsets'][0])
    for key, info in ordered:
        count = math.prod(info['shape'])
        dh[key] = {'dtype': 'F16', 'shape': info['shape'], 'data_offsets': [offset, offset + count * 2]}
        offset += count * 2
    encoded = json.dumps(dh, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    encoded += b' ' * (-len(encoded) % 8)
    output_prefix = struct.pack('<Q', len(encoded)) + encoded
    src_hash, dst_hash = hashlib.sha256(source_prefix), hashlib.sha256(output_prefix)
    count_total = 0
    sum_error_squared = sum_source_squared = 0.0
    maximum_error, minimum, maximum = 0.0, math.inf, -math.inf
    underflows = 0
    per_tensor = []
    started = time.monotonic()
    last_progress = started
    with source.open('rb') as reader, stage.open('xb') as writer:
        reader.seek(data_base)
        writer.write(output_prefix)
        for key, info in ordered:
            remaining = info['data_offsets'][1] - info['data_offsets'][0]
            tc = tu = 0
            te = ts = 0.0
            tmax, tmin, terr = -math.inf, math.inf, 0.0
            while remaining:
                block = reader.read(min(CHUNK_BYTES, remaining))
                if not block or len(block) % 4:
                    raise ValueError(f'Incomplete source tensor data: {key}')
                src_hash.update(block)
                values = np.frombuffer(block, dtype='<f4')
                if not np.isfinite(values).all():
                    raise ValueError(f'Nonfinite source values: {key}')
                low, high = float(values.min()), float(values.max())
                if max(abs(low), abs(high)) > 65504:
                    raise ValueError(f'FP16 overflow range: {key}: {low}, {high}')
                halves = values.astype('<f2')
                restored = halves.astype(np.float32)
                error = restored.astype(np.float64) - values.astype(np.float64)
                error_sum = float(np.sum(error * error, dtype=np.float64))
                source_sum = float(np.sum(values.astype(np.float64) ** 2, dtype=np.float64))
                max_error = float(np.max(np.abs(error)))
                under = int(np.count_nonzero((values != 0) & (halves == 0)))
                out_block = halves.tobytes()
                writer.write(out_block)
                dst_hash.update(out_block)
                remaining -= len(block)
                tc += values.size
                tu += under
                te += error_sum
                ts += source_sum
                tmin, tmax, terr = min(tmin, low), max(tmax, high), max(terr, max_error)
            count_total += tc
            underflows += tu
            sum_error_squared += te
            sum_source_squared += ts
            minimum, maximum, maximum_error = min(minimum, tmin), max(maximum, tmax), max(maximum_error, terr)
            per_tensor.append({'key': key, 'elements': tc, 'minimum': tmin, 'maximum': tmax,
                               'error_rms': math.sqrt(te / tc) if tc else 0,
                               'error_max_absolute': terr, 'underflow_to_zero_count': tu})
            if time.monotonic() - last_progress >= 10:
                progress = {'label': spec['label'], 'phase': 'streaming', 'source_bytes_read': reader.tell(),
                            'source_bytes_total': source_stat.st_size, 'tensors_completed': len(per_tensor),
                            'elapsed_seconds': round(time.monotonic() - started, 2)}
                (HERE / 'conversion_progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
                print(json.dumps(progress), flush=True)
                last_progress = time.monotonic()
        if reader.read(1):
            raise ValueError('Unexpected trailing source bytes')
        writer.flush()
        os.fsync(writer.fileno())
    after_stat = source.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (after_stat.st_size, after_stat.st_mtime_ns):
        raise ValueError('Source file changed during conversion')
    if src_hash.hexdigest() != spec['sha256']:
        raise ValueError(f'Source SHA mismatch: {source.name}: {src_hash.hexdigest()}')
    parsed, _, _ = header(stage)
    if set(parsed) - {'__metadata__'} != set(tensors):
        raise ValueError('Derived key set mismatch')
    for key, info in tensors.items():
        if parsed[key]['shape'] != info['shape'] or parsed[key]['dtype'] != 'F16':
            raise ValueError(f'Derived shape/dtype mismatch: {key}')
    with stage.open('rb') as verify_stream:
        physical_hash = hashlib.file_digest(verify_stream, 'sha256').hexdigest()
    if physical_hash != dst_hash.hexdigest():
        raise ValueError('Written output SHA mismatch')
    effective = sampled_delta_checks(source, stage, spec['label'])
    # Same-directory hardlink publication is exclusive on Windows and POSIX.
    # Fail closed on existing targets or filesystems without hardlink support;
    # never fall back to a replacement-capable rename.
    os.link(stage, destination)
    stage.unlink()
    receipt = {
        'source': str(source), 'destination': str(destination), 'source_sha256': src_hash.hexdigest(),
        'expected_source_sha256_matches': True, 'source_bytes': source_stat.st_size,
        'output_sha256': physical_hash, 'output_bytes': destination.stat().st_size,
        'tensor_count': len(tensors), 'same_keys_and_shapes': True, 'source_dtype': 'F32', 'output_dtype': 'F16',
        'all_source_values_finite': True, 'fp16_overflow_count': 0, 'source_minimum': minimum, 'source_maximum': maximum,
        'elements': count_total, 'difference_rms': math.sqrt(sum_error_squared / count_total),
        'difference_max_absolute': maximum_error, 'relative_rms': math.sqrt(sum_error_squared / sum_source_squared),
        'underflow_to_zero_count': underflows, 'underflow_fraction': underflows / count_total,
        'effective_delta_samples': effective, 'per_tensor': per_tensor,
        'seconds': round(time.monotonic() - started, 2),
        'original_preserved': (source.stat().st_size, source.stat().st_mtime_ns) == (source_stat.st_size, source_stat.st_mtime_ns),
        'core_loader_validation': 'Pending separate header-only core-loader check; no full torch materialization performed.',
        'runtime_patch_and_render_validation': 'Not performed by this converter.',
    }
    (HERE / (spec['label'] + '_fp16_conversion_receipt.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return {key: value for key, value in receipt.items() if key not in ['per_tensor', 'effective_delta_samples']}


if __name__ == '__main__':
    HERE.mkdir(parents=True, exist_ok=True)
    completed = []
    for spec in SPECS:
        print(json.dumps({'phase': 'starting', 'source': spec['source'], 'destination': spec['destination']}), flush=True)
        completed.append(cast_one(spec))
        print(json.dumps({'phase': 'published', **completed[-1]}), flush=True)
    result = {'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'method': 'CPU NumPy4MiB chunks; original source SHA checked while reading; no native get_tensor calls.',
              'outputs': completed}
    (HERE / 'FP16_CONVERSION_SUMMARY.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
