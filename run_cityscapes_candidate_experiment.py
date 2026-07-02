"""Candidate-constrained Cityscapes experiment.

Uses oracle GT-present classes as candidates on 3 val images. The VLM may emit
local IDs or original Cityscapes train IDs; both are accepted and mapped back.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ugvu.configs.config import UGVUConfig
from ugvu.datasets.dataset import CITYSCAPES_CLASSES, CITYSCAPES_COLORMAP, build_dataset
from ugvu.generators.sampler import build_sampler
from ugvu.uncertainty.uncertainty_map import build_uncertainty_map
from ugvu.consensus.consensus_fusion import consensus_fusion
from ugvu.metrics.metrics import evaluate_dataset, segmentation_metrics
from ugvu.calibration.ece import compute_ece_from_uncertainty
from ugvu.calibration.correlation import correlation_report
from ugvu.visualization.visualize import save_uncertainty_map
from ugvu.diagnostics import (
    aggregate_failure_tags,
    attach_k_sweep_summary,
    prediction_diagnostics,
    semantic_trivial_baselines,
)

ROOT = r'D:\BaiduNetdiskDownload\cityscapes'
OUT_ROOT = Path('outputs/cityscapes_candidate')
MAX_SAMPLES = 3
RIGHT_CODE_MODELS = ('gpt-image-2-vip', 'gpt-image-2', 'nano-banana', 'nano-banana-2', 'nano-banana-pro')

def right_code_model(model_version):
 return {'api_endpoint':'https://www.right.codes/draw/v1','api_key':'${IMAGE_API_KEY}','model_version':model_version,'api_mode':'chat_completions','class':'rightcode','timeout_sec':300,'max_retries':1,'image_size':[1024,1024],'size':'1K','temperature':0.2}

MODELS = {
 'qwen-vl': {'api_endpoint':'https://dashscope.aliyuncs.com/compatible-mode/v1','api_key':'${DASHSCOPE_API_KEY}','model_version':'qwen-vl-max','api_mode':'chat','timeout_sec':120,'max_retries':1,'image_size':[1024,512],'temperature':0.2},
 'doubao': {'api_endpoint':'https://ark.cn-beijing.volces.com/api/v3','api_key':'${DOUBAO_API_KEY}','model_version':'ep-m-20260616184522-wp9rq','api_mode':'vision_chat','timeout_sec':120,'max_retries':1,'image_size':[1024,512],'temperature':0.2},
 **{model_name: right_code_model(model_name) for model_name in RIGHT_CODE_MODELS},
}

def load_keys():
 cfg=UGVUConfig.from_yaml('configs/default.yaml')
 return {name: model.api_key for name, model in cfg.models.models.items()}

def prompt(candidates):
 spec='; '.join(f'{i}={CITYSCAPES_CLASSES[c]} (Cityscapes train id {c})' for i,c in enumerate(candidates))
 return ('Perform Cityscapes semantic segmentation using ONLY the candidate classes below. '
         'Output local candidate IDs, not color names. Candidate local IDs: '+spec+'. '
         'Every grid value must be one of these local IDs. Return dense class-index grid only; no 255, no unknown, no text labels. '
         'Use coherent regions and preserve road, building, sky, vegetation, person, and vehicle boundaries.')

def decode_candidate_stack(stack, candidates):
 local_stack=[]; unc_stack=[]
 for generated in stack:
  values=(generated[...,0] if generated.ndim==3 else generated).astype(np.float32)
  nearest=np.rint(values).astype(np.int64)
  local=np.full(nearest.shape,255,dtype=np.int64)
  for local_id, train_id in enumerate(candidates):
   local[nearest==local_id]=local_id
   local[nearest==int(train_id)]=local_id
  unc=np.clip(np.abs(values-np.rint(values)),0,1).astype(np.float32); unc[local==255]=1.0
  local_stack.append(local); unc_stack.append(unc)
 return np.stack(local_stack,axis=0), np.stack(unc_stack,axis=0)

def map_local(local_pred,candidates):
 out=np.full(local_pred.shape,255,dtype=np.int64)
 for local_id, train_id in enumerate(candidates): out[local_pred==local_id]=train_id
 return out

def save_fig(image,pred,gt,unc,path):
 def color(mask):
  rgb=np.zeros((*mask.shape,3),dtype=np.uint8)
  for c in range(len(CITYSCAPES_COLORMAP)): rgb[mask==c]=CITYSCAPES_COLORMAP[c]
  return rgb
 fig,axes=plt.subplots(1,4,figsize=(16,4))
 axes[0].imshow(image); axes[0].set_title('Input'); axes[0].axis('off')
 axes[1].imshow(color(pred)); axes[1].set_title('Prediction'); axes[1].axis('off')
 axes[2].imshow(color(gt)); axes[2].set_title('GT'); axes[2].axis('off')
 im=axes[3].imshow(unc,cmap='hot',vmin=0,vmax=1); axes[3].set_title('Uncertainty'); axes[3].axis('off')
 fig.colorbar(im,ax=axes[3],fraction=0.046,pad=0.04); fig.tight_layout(); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)

def run_case(model_name,k,dataset,key):
 out_dir=OUT_ROOT/f'{model_name}_k{k}'; out_dir.mkdir(parents=True,exist_ok=True)
 model_cfg=MODELS[model_name].copy(); model_cfg['api_key']=key
 sampler=build_sampler({model_name:model_cfg},k_samples=k,parallel=False,max_concurrent=1,seed=42,cache_dir=str(out_dir/'cache'))
 predictions=[]; gts=[]; uncs=[]; per=[]
 for idx in range(len(dataset)):
  sample=dataset[idx]; image_arr=sample['image']; gt=sample['label'].astype(np.int64)
  candidates=sorted(int(v) for v in np.unique(gt) if int(v)!=255)
  collection=sampler.sample(Image.fromarray(image_arr),prompts=[prompt(candidates)]*k,models=[model_name],image_id=f'city_{idx:04d}')
  local_stack,dec_unc=decode_candidate_stack(collection.mask_stack,candidates)
  unc=build_uncertainty_map(local_stack,task='semantic_segmentation',uncertainty_type='entropy',num_classes=len(candidates),ignore_index=255)
  unc=np.maximum(unc,dec_unc.mean(axis=0))
  per_unc=np.maximum(dec_unc,np.tile(unc[None,...],(local_stack.shape[0],1,1)))
  local_pred=consensus_fusion(local_stack,per_unc,method='majority' if k==1 else 'ugcf',num_classes=len(candidates),ignore_index=255,temperature=0.5)
  pred=map_local(local_pred,candidates)
  if pred.shape[:2]!=gt.shape[:2]:
   pred=np.asarray(Image.fromarray(pred.astype(np.int32)).resize((gt.shape[1],gt.shape[0]),resample=Image.NEAREST),dtype=np.int64)
   unc=np.asarray(Image.fromarray(unc.astype(np.float32)).resize((gt.shape[1],gt.shape[0]),resample=Image.BILINEAR),dtype=np.float32)
  predictions.append(pred); gts.append(gt); uncs.append(unc)
  np.save(out_dir/f'pred_{idx:04d}.npy',pred); np.save(out_dir/f'unc_{idx:04d}.npy',unc); save_uncertainty_map(unc,str(out_dir/f'unc_{idx:04d}.png')); save_fig(image_arr,pred,gt,unc,out_dir/f'comparison_{idx:04d}.png')
  vals,cnts=np.unique(pred[pred!=255],return_counts=True)
  per.append({'image':idx,'candidates':[{'id':c,'name':CITYSCAPES_CLASSES[c]} for c in candidates],'pred_classes':[{'id':int(v),'name':CITYSCAPES_CLASSES[int(v)],'pixels':int(n)} for v,n in zip(vals,cnts)]})
 metrics=evaluate_dataset(predictions,gts,task='semantic_segmentation',num_classes=19,ignore_index=255)
 eces=[]; corrs=[]
 for pred,gt,unc in zip(predictions,gts,uncs):
  err=(pred!=gt)&(gt!=255); eces.append(compute_ece_from_uncertainty(unc,err,num_bins=15,strategy='uniform')); corrs.append(correlation_report(unc,err))
 diag_per=[prediction_diagnostics(pred,gt,unc,ignore_index=255,num_classes=19) for pred,gt,unc in zip(predictions,gts,uncs)]
 per_image_calibration=[{'image':i,'ECE':float(e['ece']),'MCE':float(e['mce']),'Spearman':float(c['spearman_r']),'AUROC':float(c['auroc'])} for i,(e,c) in enumerate(zip(eces,corrs))]
 result={'dataset':'cityscapes','protocol':'oracle_candidates','model':model_name,'k':k,'max_samples':len(dataset),'ok':True,'metrics':{'mIoU':metrics.get('mIoU'),'ECE':float(np.mean([e['ece'] for e in eces])),'Spearman':float(np.nanmean([c['spearman_r'] for c in corrs])),'AUROC':float(np.nanmean([c['auroc'] for c in corrs]))},'per_image':per,'diagnostics':{'per_image':diag_per,'per_image_calibration':per_image_calibration,'failure_tag_counts':aggregate_failure_tags(diag_per),'trivial_baselines':semantic_trivial_baselines(gts,num_classes=19,ignore_index=255)}}
 return result

def _parse_csv(value, cast):
 items=[]
 for part in value.split(','):
  part=part.strip()
  if part: items.append(cast(part))
 return items

def main():
 global OUT_ROOT
 import argparse
 parser=argparse.ArgumentParser(description='Run candidate-constrained Cityscapes API diagnostics.')
 parser.add_argument('--max-samples',type=int,default=MAX_SAMPLES,help='Number of Cityscapes val images to evaluate.')
 parser.add_argument('--models',default='gpt-image-2-vip,gpt-image-2,nano-banana,nano-banana-2,nano-banana-pro',help='Comma-separated model names from the MODELS registry.')
 parser.add_argument('--k-values',default='1,3',help='Comma-separated K values for repeated sampling.')
 parser.add_argument('--output-dir',default=str(OUT_ROOT),help='Output directory for predictions and JSON reports.')
 args=parser.parse_args()
 models=_parse_csv(args.models,str)
 k_values=_parse_csv(args.k_values,int)
 bad=[m for m in models if m not in MODELS]
 if bad: raise ValueError(f'Unknown models: {bad}. Available: {list(MODELS)}')
 OUT_ROOT=Path(args.output_dir)
 OUT_ROOT.mkdir(parents=True,exist_ok=True)
 dataset=build_dataset('cityscapes',root=ROOT,split='val',max_samples=args.max_samples)
 keys=load_keys(); results=[]
 for model in models:
  for k in k_values:
   results.append(run_case(model,k,dataset,keys[model]))
 with (OUT_ROOT/'cityscapes_candidate_results.json').open('w',encoding='utf-8') as f: json.dump(results,f,indent=2,ensure_ascii=False)
 report={'dataset':'cityscapes','protocol':'oracle_candidates','max_samples':args.max_samples,'models':models,'k_values':k_values,'paper_framing':'Candidate-constrained diagnostic probing of zero-shot spatial reliability.','k_sweep_summary':attach_k_sweep_summary(results),'results':results}
 with (OUT_ROOT/'cityscapes_candidate_diagnostic_report.json').open('w',encoding='utf-8') as f: json.dump(report,f,indent=2,ensure_ascii=False)
 print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__': main()



