import torch
import numpy as np
import json
import re
import os
import pickle
import math
import torchvision
from PIL import Image
from models.AMG_PC import AMG_PC

import glob, re as re2
ckpts = [f for f in glob.glob('./log/AMG_PC_64_four*/ckpt_*.pt') if 'uniform' not in f]
ckpt_path = min(ckpts, key=lambda f: float(re2.search(r'ckpt_\d+_([\d.]+)\.pt', f).group(1)))
epoch_num = int(re2.search(r'ckpt_(\d+)_', ckpt_path).group(1))
cd_val = float(re2.search(r'ckpt_\d+_([\d.]+)\.pt', ckpt_path).group(1))

device = torch.device('cuda:0')
model = AMG_PC()
ckpt = torch.load(ckpt_path, map_location=device)
state = {}
for k, v in ckpt['model_state_dict'].items():
    k = k.replace('module.', '')
    k = k.replace('fusion_token_embedding', 'fusion_module.token_embedding')
    k = k.replace('fusion_blocks', 'fusion_module.blocks')
    state[k] = v
model.load_state_dict(state)
model = torch.nn.DataParallel(model).to(device)
model.eval()

DATA_ROOT = '/scratch/tbalamur/vipc_data/ShapeNetViPC-Dataset'

# Unseen categories (zero-shot)
UNSEEN = [
    ('02828884', 'Bench'),
    ('03211117', 'Monitor'),
    ('03691459', 'Speaker'),
    ('04090263', 'Firearm'),
    ('04401088', 'Cellphone'),
]

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor()
])

def rotation_y(pts, theta):
    c, s = np.cos(theta), np.sin(theta)
    return pts @ np.array([[c,0,-s],[0,1,0],[s,0,c]]).T

def rotation_x(pts, theta):
    c, s = np.cos(theta), np.sin(theta)
    return pts @ np.array([[1,0,0],[0,c,-s],[0,s,c]]).T

def resample(pcd, n):
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]

def get_sample(cat_id):
    cat_dir = f'{DATA_ROOT}/ShapeNetViPC-View/{cat_id}'
    import random
    objs = os.listdir(cat_dir)
    random.shuffle(objs)
    for obj_id in objs:
        view_id = '05'
        pc_part_path = f'{DATA_ROOT}/ShapeNetViPC-Partial/{cat_id}/{obj_id}/{view_id}.dat'
        pc_path      = f'{DATA_ROOT}/ShapeNetViPC-GT/{cat_id}/{obj_id}/{view_id}.dat'
        view_path    = f'{DATA_ROOT}/ShapeNetViPC-View/{cat_id}/{obj_id}/rendering/{view_id}.png'
        text_path    = f'{DATA_ROOT}/ShapeNetViPC-View/{cat_id}/{obj_id}/text_embed/{view_id}.npy'
        meta_path    = f'{DATA_ROOT}/ShapeNetViPC-View/{cat_id}/{obj_id}/rendering/rendering_metadata.txt'
        if not all(os.path.exists(p) for p in [pc_part_path, pc_path, view_path, meta_path]):
            continue
        try:
            with open(pc_path, 'rb') as f: pc = pickle.load(f).astype(np.float32)
            with open(pc_part_path, 'rb') as f: pc_part = pickle.load(f).astype(np.float32)
            gt_mean = pc.mean(axis=0)
            pc = (pc - gt_mean)
            pc_L_max = np.max(np.sqrt(np.sum(pc**2, axis=-1)))
            if pc_L_max < 1e-6: continue
            pc = pc / pc_L_max
            pc_part = (pc_part - gt_mean) / pc_L_max
            meta = np.loadtxt(meta_path)
            vid = int(view_id)
            pc_part = rotation_y(rotation_x(pc_part, -math.radians(meta[vid,1])), np.pi+math.radians(meta[vid,0]))
            pc_part = rotation_x(rotation_y(pc_part, np.pi-math.radians(meta[vid,0])), math.radians(meta[vid,1]))
            pc_part = resample(pc_part, 2048)
            pc      = resample(pc, 2048)
            img = transform(Image.open(view_path))[:3]
            text_embed = np.load(text_path).reshape(512).astype(np.float32) if os.path.exists(text_path) else np.zeros(512, dtype=np.float32)
            return pc_part, pc, img, text_embed
        except: continue
    return None

samples = []
with torch.no_grad():
    for cat_id, label in UNSEEN:
        result = get_sample(cat_id)
        if result is None:
            print(f'No sample for {label}')
            continue
        pc_part, pc, img, text_embed = result
        partial_t    = torch.from_numpy(pc_part.copy()).float().unsqueeze(0).to(device)
        image_t      = img.float().unsqueeze(0).to(device)
        text_embed_t = torch.from_numpy(text_embed.copy()).float().unsqueeze(0).to(device)
        out, gate_weights = model(partial_t, image_t, text_embed_t)
        pred_np = out[-1][0].cpu().numpy()
        gates = [gw[0].cpu().numpy().tolist() for gw in gate_weights]
        samples.append({
            'label': label,
            'partial': pc_part.tolist(),
            'pred': pred_np.tolist(),
            'gt': pc.tolist(),
            'gates': gates,
        })
        print(f'Collected: {label}, gates={[[f"{g:.3f}" for g in gw] for gw in gates]}')

print(f'Total: {len(samples)} zero-shot samples')

# Compute CD for each sample
from cuda.ChamferDistance import L2_ChamferDistance
loss_cd = L2_ChamferDistance()

print("\nZero-shot CD per category:")
for s in samples:
    pred_t = torch.from_numpy(np.array(s['pred'])).float().unsqueeze(0).to(device)
    gt_t   = torch.from_numpy(np.array(s['gt'])).float().unsqueeze(0).to(device)
    cd = loss_cd(pred_t, gt_t) * 1e3
    print(f"  {s['label']}: CD = {cd.item():.4f}")



# Update labels with CD
from cuda.ChamferDistance import L2_ChamferDistance
loss_cd = L2_ChamferDistance()
for s in samples:
    pred_t = torch.from_numpy(np.array(s['pred'])).float().unsqueeze(0).to(device)
    gt_t   = torch.from_numpy(np.array(s['gt'])).float().unsqueeze(0).to(device)
    cd = loss_cd(pred_t, gt_t) * 1e3
    s['label'] = f"{s['label']}  CD={cd.item():.3f}"

MOD_COLORS = ['#58a6ff','#ffa657','#a78bfa']
STAGE_LABELS = ['Stage 1 (256→512)','Stage 2 (512→1024)','Stage 3 (1024→2048)']
MOD_LABELS = ['PC','IMG','TXT']

html = '''<!DOCTYPE html><html><head><meta charset="utf-8">
<title>AMG-PC Zero-Shot</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#05050d;--bg2:#0c0c18;--bg3:#111120;--border:#1a1a2e;--text:#dde1f0;--muted:#40406a;--accent:#7c6af7;--blue:#58a6ff;--orange:#ffa657;--green:#3fb950;}
*{margin:0;padding:0;box-sizing:border-box;}body{background:var(--bg);font-family:'Inter',sans-serif;color:var(--text);}
header{padding:32px 52px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}
.logo h1{font-size:22px;font-weight:600;letter-spacing:-0.03em;}.logo h1 span{color:var(--accent);}
.logo p{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-top:6px;}
.stats{display:flex;border:1px solid var(--border);border-radius:10px;overflow:hidden;background:var(--bg2);}
.stat-box{padding:10px 20px;border-right:1px solid var(--border);text-align:center;}.stat-box:last-child{border-right:none;}
.stat-val{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:500;color:var(--accent);display:block;}
.stat-lbl{font-size:9px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;display:block;margin-top:2px;}
.legend{display:flex;align-items:center;gap:6px;padding:10px 52px;border-bottom:1px solid var(--border);background:var(--bg2);}
.leg{display:flex;align-items:center;gap:7px;padding:4px 12px;border-radius:20px;border:1px solid;font-family:'JetBrains Mono',monospace;font-size:10px;}
.hint{margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);}
.samples{padding:24px 52px 48px;display:flex;flex-direction:column;gap:20px;}
.sample-card{border:1px solid var(--border);border-radius:14px;overflow:hidden;background:var(--bg2);}
.card-header{padding:12px 18px;border-bottom:1px solid var(--border);background:var(--bg3);display:flex;align-items:center;gap:10px;}
.card-idx{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--accent);background:rgba(124,106,247,.1);border:1px solid rgba(124,106,247,.2);padding:2px 8px;border-radius:4px;}
.card-name{font-size:13px;font-weight:500;}
.card-body{display:grid;grid-template-columns:1fr 1fr 1fr 280px;gap:1px;background:var(--border);}
.cv-wrap{background:var(--bg2);position:relative;}
canvas{display:block;width:100%;height:260px;cursor:grab;}canvas:active{cursor:grabbing;}
.cv-tag{position:absolute;bottom:12px;left:12px;font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.09em;text-transform:uppercase;padding:3px 9px;border-radius:5px;border:1px solid;pointer-events:none;z-index:5;}
.tag-p{color:var(--blue);border-color:rgba(88,166,255,.3);background:rgba(88,166,255,.08);}
.tag-r{color:var(--orange);border-color:rgba(255,166,87,.3);background:rgba(255,166,87,.08);}
.tag-g{color:var(--green);border-color:rgba(63,185,80,.3);background:rgba(63,185,80,.08);}
.gate-panel{background:var(--bg2);padding:20px 16px;display:flex;flex-direction:column;gap:4px;}
.gate-title{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px;}
.gate-stage-label{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--muted);margin-bottom:3px;margin-top:6px;}
.gate-bar-row{display:flex;align-items:center;gap:6px;margin-bottom:2px;}
.gate-bar-label{font-family:'JetBrains Mono',monospace;font-size:8px;color:var(--muted);width:24px;}
.gate-bar-bg{flex:1;height:8px;background:rgba(255,255,255,0.05);border-radius:4px;overflow:hidden;}
.gate-bar-fill{height:100%;border-radius:4px;}
.gate-bar-val{font-family:'JetBrains Mono',monospace;font-size:8px;color:var(--muted);width:36px;text-align:right;}
footer{padding:16px 52px;border-top:1px solid var(--border);display:flex;justify-content:space-between;background:var(--bg2);}
footer span{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:.07em;text-transform:uppercase;}
</style></head><body>
<header>
  <div class="logo"><h1>AMG-PC <span>Zero-Shot</span> Generalization</h1>
  <p>Epoch ''' + str(epoch_num) + ''' &nbsp;·&nbsp; Unseen Categories &nbsp;·&nbsp; ShapeNet-ViPC</p></div>
  <div class="stats">
    <div class="stat-box"><span class="stat-val">''' + str(epoch_num) + '''</span><span class="stat-lbl">Epoch</span></div>
    <div class="stat-box"><span class="stat-val">5</span><span class="stat-lbl">Unseen Cats</span></div>
    <div class="stat-box"><span class="stat-val">2048</span><span class="stat-lbl">Points</span></div>
  </div>
</header>
<div class="legend">
  <div class="leg" style="color:var(--blue);border-color:rgba(88,166,255,.25);background:rgba(88,166,255,.06)"><div style="width:7px;height:7px;border-radius:50%;background:var(--blue);display:inline-block;margin-right:4px"></div>Partial Input</div>
  <div class="leg" style="color:var(--orange);border-color:rgba(255,166,87,.25);background:rgba(255,166,87,.06)"><div style="width:7px;height:7px;border-radius:50%;background:var(--orange);display:inline-block;margin-right:4px"></div>Predicted</div>
  <div class="leg" style="color:var(--green);border-color:rgba(63,185,80,.25);background:rgba(63,185,80,.06)"><div style="width:7px;height:7px;border-radius:50%;background:var(--green);display:inline-block;margin-right:4px"></div>Ground Truth</div>
  <span class="hint">↔ drag to rotate all &nbsp;·&nbsp; scroll to zoom</span>
</div>
<div class="samples" id="samples"></div>
<footer>
  <span>AMG-PC &nbsp;·&nbsp; CSC 449 &nbsp;·&nbsp; University of Rochester &nbsp;·&nbsp; 2025</span>
  <span>Tejaswini BK &amp; Luke Liu</span>
</footer>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
''' + f'const SAMPLES={json.dumps(samples)};' + '''
const COLS={partial:0x58a6ff,pred:0xffa657,gt:0x3fb950};
const LBLS=['partial','pred','gt'];const LNAMES=['Partial Input','Predicted','Ground Truth'];const LTAGS=['tag-p','tag-r','tag-g'];
const MOD_LABELS=['PC','IMG','TXT'];const MOD_COLORS=['#58a6ff','#ffa657','#a78bfa'];
const STAGE_LABELS=['Stage 1 (256→512)','Stage 2 (512→1024)','Stage 3 (1024→2048)'];
const allO=[],allR=[],allS=[],allC=[];
function norm(pts){let cx=0,cy=0,cz=0;pts.forEach(p=>{cx+=p[0];cy+=p[1];cz+=p[2];});cx/=pts.length;cy/=pts.length;cz/=pts.length;let mx=0;pts.forEach(p=>{const d=Math.hypot(p[0]-cx,p[1]-cy,p[2]-cz);if(isFinite(d)&&d>mx)mx=d;});if(mx<1e-6)mx=1;return pts.map(p=>{const x=(p[0]-cx)/mx,y=(p[1]-cy)/mx,z=(p[2]-cz)/mx;return[isFinite(x)?x:0,isFinite(y)?y:0,isFinite(z)?z:0];});}
function makePC(pts,color,big){const g=new THREE.BufferGeometry();const n=norm(pts);const pos=new Float32Array(n.length*3);n.forEach((p,i)=>{pos[i*3]=p[0];pos[i*3+1]=p[1];pos[i*3+2]=p[2];});g.setAttribute('position',new THREE.BufferAttribute(pos,3));return new THREE.Points(g,new THREE.PointsMaterial({color,size:big?0.05:0.038,transparent:true,opacity:0.95,sizeAttenuation:true,depthWrite:false}));}
function makeScene(pts,color,big){const s=new THREE.Scene();s.background=new THREE.Color(0x0c0c18);s.add(makePC(pts,color,big));return s;}
function makeCamera(){const c=new THREE.PerspectiveCamera(40,1,0.01,100);c.position.set(2.6,1.6,2.6);c.lookAt(0,0,0);return c;}
class Orbit{constructor(cam,el){this.cam=cam;this.theta=0.75;this.phi=0.62;this.r=3.6;this.down=false;this.lx=0;this.ly=0;this.cb=null;el.addEventListener('mousedown',e=>{this.down=true;this.lx=e.clientX;this.ly=e.clientY;e.preventDefault();});window.addEventListener('mousemove',e=>{if(!this.down)return;const dx=e.clientX-this.lx,dy=e.clientY-this.ly;this.lx=e.clientX;this.ly=e.clientY;this.theta-=dx*0.004;this.phi=Math.max(0.08,Math.min(Math.PI-0.08,this.phi-dy*0.004));this._u();if(this.cb)this.cb(this.theta,this.phi,this.r);});window.addEventListener('mouseup',()=>{this.down=false;});el.addEventListener('wheel',e=>{this.r=Math.max(1,Math.min(9,this.r+e.deltaY*0.005));this._u();if(this.cb)this.cb(this.theta,this.phi,this.r);e.preventDefault();},{passive:false});this._u();}_u(){const s=Math.sin(this.phi);this.cam.position.set(this.r*s*Math.sin(this.theta),this.r*Math.cos(this.phi),this.r*s*Math.cos(this.theta));this.cam.lookAt(0,0,0);}sync(t,p,r){this.theta=t;this.phi=p;this.r=r;this._u();}}
function makeGatePanel(gates){const div=document.createElement('div');div.className='gate-panel';const title=document.createElement('div');title.className='gate-title';title.textContent='Modality Gate Weights';div.appendChild(title);gates.forEach((gw,si)=>{const sl=document.createElement('div');sl.className='gate-stage-label';sl.textContent=STAGE_LABELS[si];div.appendChild(sl);gw.forEach((w,mi)=>{const row=document.createElement('div');row.className='gate-bar-row';const lbl=document.createElement('div');lbl.className='gate-bar-label';lbl.textContent=MOD_LABELS[mi];const bg=document.createElement('div');bg.className='gate-bar-bg';const fill=document.createElement('div');fill.className='gate-bar-fill';fill.style.width=`${(w*100).toFixed(1)}%`;fill.style.background=MOD_COLORS[mi];bg.appendChild(fill);const val=document.createElement('div');val.className='gate-bar-val';val.textContent=w.toFixed(3);row.appendChild(lbl);row.appendChild(bg);row.appendChild(val);div.appendChild(row);});});return div;}
const cont=document.getElementById('samples');
SAMPLES.forEach((sam,si)=>{const card=document.createElement('div');card.className='sample-card';const hdr=document.createElement('div');hdr.className='card-header';hdr.innerHTML=`<span class="card-idx">#${String(si+1).padStart(2,'0')}</span><span class="card-name">${sam.label}</span>`;card.appendChild(hdr);const body=document.createElement('div');body.className='card-body';card.appendChild(body);cont.appendChild(card);LBLS.forEach((lbl,li)=>{const wrap=document.createElement('div');wrap.className='cv-wrap';const tag=document.createElement('div');tag.className=`cv-tag ${LTAGS[li]}`;tag.textContent=LNAMES[li];wrap.appendChild(tag);const canvas=document.createElement('canvas');wrap.appendChild(canvas);body.appendChild(wrap);const r=new THREE.WebGLRenderer({canvas,antialias:true});r.setPixelRatio(Math.min(window.devicePixelRatio,2));const sc=makeScene(sam[lbl],COLS[lbl],lbl==='partial');const cam=makeCamera();const orb=new Orbit(cam,canvas);allO.push(orb);allR.push(r);allS.push(sc);allC.push(cam);orb.cb=(t,p,rv)=>allO.forEach(o=>{if(o!==orb)o.sync(t,p,rv);});});body.appendChild(makeGatePanel(sam.gates));});
function resize(){allR.forEach((r,i)=>{const c=r.domElement,w=c.clientWidth,h=c.clientHeight;if(c.width!==w||c.height!==h){r.setSize(w,h,false);allC[i].aspect=w/h;allC[i].updateProjectionMatrix();}});}
(function loop(){requestAnimationFrame(loop);resize();allR.forEach((r,i)=>r.render(allS[i],allC[i]));})();
</script></body></html>'''

with open('/tmp/amgpc_zeroshot.html', 'w') as f:
    f.write(html)
print('Saved: /tmp/amgpc_zeroshot.html')
