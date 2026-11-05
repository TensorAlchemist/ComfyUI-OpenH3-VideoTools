"""Portable entry point for the reviewed, bounded CPU converter."""
import argparse
import datetime
import importlib.util
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--comfyui-root',type=Path,required=True,help='ComfyUI folder containing models and custom_nodes.')
parser.add_argument('--receipts-dir',type=Path,default=Path(__file__).resolve().parents[1]/'conversion_receipts')
args=parser.parse_args()
comfyui=args.comfyui_root.expanduser().resolve()
if not (comfyui/'models').is_dir() or not (comfyui/'custom_nodes').is_dir():
    parser.error('The supplied folder must contain models and custom_nodes.')
receipts=args.receipts_dir.expanduser().resolve()
receipts.mkdir(parents=True,exist_ok=True)
source=Path(__file__).resolve().with_name('stream_fp16_adapters.py')
spec=importlib.util.spec_from_file_location('ta_bounded_fp16_converter',source)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT=comfyui
module.HERE=receipts
completed=[]
for original_spec in module.SPECS:
    item=dict(original_spec)
    item['source']=item['source'].removeprefix('ComfyUI/')
    item['destination']=item['destination'].removeprefix('ComfyUI/')
    print(json.dumps({'phase':'starting','source':item['source'],'destination':item['destination']}),flush=True)
    completed.append(module.cast_one(item))
result={'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'CPU NumPy bounded chunks; originals preserved; no GPU or inference.','outputs':completed}
(receipts/'FP16_CONVERSION_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
