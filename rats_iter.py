"""Iterative register decomposition + fixed-K grouping (self-contained, single figure).

Pipeline (image goes through the model ONCE):
  1) ONE model forward -> register/patch similarity S, patch feats, normalized patch feats
  2) recursive decomposition (drop + recompute + top_n + clean) to depth D
       -> per-iteration level maps L1, L2, ..., Ld  (Ld = over-seg leaf map)
  3) per leaf segment embedding = normalized mean patch feature
  4) agglomerative clustering into exactly K clusters (cosine) -> grouped map

One output figure visualizes BOTH:
  TOP row     : IMAGE | L1 | L2 | ... | Ld (over-seg) | fixed K        (global progression)
  BOTTOM rows : one row per L1 mask -> isolated mask | L2 | L3 | ... | Ld
                (that mask's recursive breakdown, rest greyed)
"""
import os, sys, argparse
import numpy as np, torch
from scipy.ndimage import gaussian_filter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
import draw_dominant_map as ddm, rats_partwhole as rp

NUM_REG=192; SIG=1.0; TAU=0.1; CLEAN_FRAC=0.005; SHOW_REGIONS=8
OUT=os.path.join(HERE,'rats_iter')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAL=np.array(ddm.get_distinct_colors(40))

def features(model, img_path, size):
    """ONE model forward -> register/patch sim matrix S, patch feats, normalized patch feats, PIL, grid."""
    img_tensor,pil,(h,w)=ddm.load_image(img_path,[size,size],16)
    cap={}
    hh=model.blocks[-1].register_forward_hook(lambda m,i,o: cap.__setitem__('t',(o[0] if isinstance(o,tuple) else o).detach()))
    with torch.no_grad(): model(img_tensor.to(device))
    hh.remove()
    tok=cap['t']; reg=tok[0,:NUM_REG].float().cpu().numpy(); patch=tok[0,NUM_REG:].float().cpu().numpy()
    rn=reg/(np.linalg.norm(reg,axis=1,keepdims=True)+1e-8); pn=patch/(np.linalg.norm(patch,axis=1,keepdims=True)+1e-8)
    return rn@pn.T, patch, pn, pil, (h//16,w//16)

def _pick(scores, top_n, rel, cap):
    """indices of selected registers: top_n strongest (fixed int), OR all with
    peak >= rel*max (capped at `cap`) when top_n=='auto' -> adaptive branching factor."""
    order=np.argsort(-scores)
    if isinstance(top_n,str) and top_n.lower()=='auto':
        keep=order[scores[order]>=rel*scores[order[0]]][:cap]
        return keep if len(keep)>0 else order[:1]
    return order[:int(top_n)]

def fine_map(S,patch,pn,hf,wf,top_n,depth,return_levels=False,topn_rel=0.5,topn_cap=10):
    """Recursive decomposition: each region re-competes among UNUSED registers (drop),
    with registers re-projected onto the region (recompute), winner-take-all + denoise.
    top_n is an int (fixed branching) or 'auto' (adaptive: keep registers >= topn_rel*max).

    Greedy allocation: within a level regions are processed strongest-first, and each
    register is claimed by at most one region (claimed registers are dropped immediately),
    so two different regions never share the same register."""
    L=hf*wf
    def smooth(M):
        K=M.shape[0]; q=M.reshape(K,hf,wf)
        return np.stack([gaussian_filter(q[k],SIG) for k in range(K)]).reshape(K,-1)
    def softmax(x,ax): e=np.exp(x-x.max(ax,keepdims=True)); return e/(e.sum(ax,keepdims=True)+1e-8)
    def clean(l2d): return ddm.clean_isolated_patches(l2d,min_patches_frac=CLEAN_FRAC)
    def recompute(ridx):
        w=softmax(S[:,ridx]/TAU,1); nr=w@patch[ridx]; nr=nr/(np.linalg.norm(nr,axis=1,keepdims=True)+1e-8)
        return nr@pn.T
    min_region=max(8,int(0.01*L))
    peak=S.max(1); top=_pick(peak,top_n,topn_rel,topn_cap)
    S_top=smooth(S[top]); l1=(S_top-S_top.mean(0,keepdims=True)).argmax(0)
    cur=clean(top[l1].reshape(hf,wf)).reshape(-1)
    levels=[cur.copy()]                                 # per-iteration decomposition structure (L1, L2, ...)
    used=set(int(x) for x in np.unique(cur))
    for d in range(2,depth+1):
        # GREEDY (no register reuse across regions): recompute() is pool-independent, so it is
        # computed ONCE per region; then regions are processed strongest-first and as soon as a
        # region claims its registers they are dropped, so no other region this level can reuse them.
        cand=set(r for r in range(NUM_REG) if r not in used)
        if not cand: break
        new=np.full(L,-1,int); gid=0
        info={}                                          # region -> (mask, ridx, rec, per-register peak)
        for val in [v for v in np.unique(cur) if v>=0]:
            m=(cur==val)
            if m.sum()<min_region: new[m]=gid; gid+=1; continue
            ridx=np.where(m)[0]; rec=recompute(ridx)
            info[val]=(m,ridx,rec,rec[:,ridx].max(1))
        remaining=set(info)
        while remaining and cand:
            cand_arr=np.fromiter(cand,int)
            best=max(remaining,key=lambda v: info[v][3][cand_arr].max())   # strongest available activation
            m,ridx,rec,pk_full=info[best]
            sel=cand_arr[_pick(pk_full[cand_arr],top_n,topn_rel,topn_cap)]
            sm=smooth(rec[sel]); rel=sm-sm.mean(0,keepdims=True); ll=rel.argmax(0)
            tmp=np.full(L,-1,int); tmp[m]=ll[m]; tmp=clean(tmp.reshape(hf,wf)).reshape(-1)
            winners=set()
            for s in np.unique(tmp[m]):
                if s<0: continue
                new[m&(tmp==s)]=gid; gid+=1; winners.add(int(sel[s]))
            cand-=winners; used|=winners
            remaining.discard(best)
        for val in remaining: new[info[val][0]]=gid; gid+=1   # registers exhausted -> keep region as-is
        cur=new
        levels.append(cur.copy())
    if return_levels: return cur, levels                # cur = final over-seg leaf map; levels = all L1..Ld
    return cur  # global fine leaf labels [L]

def _seg_emb(leaf, patch, ids):
    emb=[]
    for v in ids:
        f=patch[leaf==v].mean(0); f=f/(np.linalg.norm(f)+1e-8); emb.append(f)
    return np.stack(emb)

def _apply(leaf, ids, grp, L):
    remap={v:int(grp[i]) for i,v in enumerate(ids)}
    out=np.full(L,-1,int)
    for v in ids: out[leaf==v]=remap[v]
    return out, len(set(grp))

def group_fixed_k(leaf, patch, hf, wf, K):
    """agglomerative into exactly K clusters (by feature similarity)."""
    L=hf*wf; ids=[v for v in np.unique(leaf) if v>=0]
    if len(ids)<=K: return leaf.copy(), len(ids)
    emb=_seg_emb(leaf,patch,ids)
    grp=fcluster(linkage(pdist(emb,'cosine'),'average'), t=K, criterion='maxclust')
    return _apply(leaf,ids,grp,L)

def _silhouette(D, lab):
    lab=np.asarray(lab); uniq=np.unique(lab); n=len(lab)
    if len(uniq)<2: return -1.0
    s=np.zeros(n)
    for i in range(n):
        same=lab==lab[i]; same[i]=False
        a=D[i,same].mean() if same.any() else 0.0
        b=min(D[i,lab==c].mean() for c in uniq if c!=lab[i])
        s[i]=(b-a)/max(a,b) if max(a,b)>0 else 0.0
    return float(s.mean())

def group_auto(leaf, patch, hf, wf, kmin=6, kmax=12, method='silhouette'):
    """auto-pick #groups within [kmin,kmax]: 'silhouette' (best K) or 'gap' (largest dendrogram gap).
    kmin prevents collapse to a coarse fg/bg (K=2) split."""
    from scipy.spatial.distance import squareform
    L=hf*wf; ids=[int(v) for v in np.unique(leaf) if v>=0]
    if len(ids)<=kmin: return leaf.copy(), len(ids)
    emb=_seg_emb(leaf,patch,ids); cd=pdist(emb,'cosine'); Z=linkage(cd,'average'); n=len(ids)
    lo=max(2,kmin); hi=min(kmax,n-1)
    if method=='gap':
        h=Z[:,2]; gaps=np.diff(h); idx=int(np.argmax(gaps)); K=n-(idx+1)
        K=max(lo,min(K,hi)); grp=fcluster(Z,K,'maxclust')
    else:                                   # silhouette over [lo,hi]
        D=squareform(cd); best=(-2,lo,None)
        for K in range(lo,hi+1):
            lab=fcluster(Z,K,'maxclust')
            if len(set(lab))<2: continue
            sc=_silhouette(D,lab)
            if sc>best[0]: best=(sc,K,lab)
        grp=best[2] if best[2] is not None else fcluster(Z,lo,'maxclust')
    return _apply(leaf,ids,grp,L)

def group_by_similarity(leaf, patch, hf, wf, merge_sim):
    """Agglomerative merge by cosine similarity (no spatial constraint). Segment
    similarities are all bunched up, so we min-max normalize the pairwise cosine
    distances to [0,1] first (data-adaptive), then merge pairs whose normalized
    similarity >= merge_sim."""
    L=hf*wf; ids=[v for v in np.unique(leaf) if v>=0]
    if len(ids)<=2: return leaf.copy(), len(ids)
    emb=_seg_emb(leaf,patch,ids)
    d=pdist(emb,'cosine')
    d=(d-d.min())/(d.max()-d.min()+1e-8)                 # normalize distances to [0,1]
    grp=fcluster(linkage(d,'average'), t=1.0-merge_sim, criterion='distance')
    return _apply(leaf,ids,grp,L)

def group(leaf, patch, hf, wf, merge_sim):
    """Merge leaf segments whose cosine similarity >= merge_sim. Returns (grouped_map, n_groups, label)."""
    g,nk=group_by_similarity(leaf,patch,hf,wf,merge_sim); return g,nk,f'sim>{merge_sim}'

def nseg(x): return len([v for v in np.unique(x) if v>=0])

def render_iter(pil,hf,wf,levels,grouped,nk,top_n,depth,klabel,img_id,size,out):
    """ONE figure: top row = global L1..Ld + grouping; bottom = per-L1-mask hierarchy."""
    L=hf*wf
    base=np.asarray(pil.resize((wf*16,hf*16))).astype(np.float32)/255.
    grey=(base.mean(axis=2,keepdims=True)*np.ones((1,1,3)))*0.55
    def up(g): return np.kron(np.asarray(g).reshape(hf,wf),np.ones((16,16)))
    def relabel(x):
        vals=[v for v in np.unique(x) if v>=0]; u={v:i for i,v in enumerate(vals)}
        return np.vectorize(lambda z:u.get(z,-1))(x)
    def full_ov(lab):                       # full-image colored overlay (top row)
        rl=relabel(lab); cc=PAL[np.clip(up(rl),0,None).astype(int)%len(PAL)]; return 0.6*cc+0.4*base
    def iso_overlay(mask,color):            # isolated mask, rest greyed
        img=grey.copy(); vm=up(mask).astype(bool); img[vm]=0.5*np.array(color)+0.5*base[vm]; return img
    def iso_disc(lab,mask):                 # discrete sub-parts inside mask, rest greyed
        rl=relabel(lab); cc=PAL[np.clip(up(rl),0,None).astype(int)%len(PAL)]
        img=grey.copy(); vm=up(mask).astype(bool); img[vm]=0.62*cc[vm]+0.38*base[vm]; return img

    # top-row panels: IMAGE | L1..Ld | fixed K
    top=[('IMAGE',None)]+[(f'L{i+1}: {nseg(lv)}',lv) for i,lv in enumerate(levels)]+[(f'group {klabel}: {nk}',grouped)]
    ncol_top=len(top)
    # bottom: one row per L1 mask, columns = isolated | L2..Ld
    L1reg=levels[0]; sub=levels[1:]
    regs=[int(r) for r in np.unique(L1reg) if r>=0]
    regs=sorted(regs,key=lambda r:-(L1reg==r).sum())
    regs=[r for r in regs if (L1reg==r).sum()>0.01*L][:SHOW_REGIONS]
    ncol_bot=1+len(sub); Nmask=max(len(regs),1)
    ncol=max(ncol_top,ncol_bot)

    fig=plt.figure(figsize=(3.1*ncol, 3.0*(1+Nmask)))
    gs=fig.add_gridspec(2,1,height_ratios=[1,Nmask],hspace=0.12)
    gtop=gs[0].subgridspec(1,ncol_top,wspace=0.04)
    gbot=gs[1].subgridspec(Nmask,ncol_bot,wspace=0.04,hspace=0.18)
    # ---- top row ----
    for i,(t,mp) in enumerate(top):
        ax=fig.add_subplot(gtop[0,i]); ax.axis('off'); ax.set_title(t,fontsize=10)
        if mp is None:
            ax.imshow(base); ax.text(0.5,0.5,img_id,transform=ax.transAxes,ha='center',va='center',fontsize=7,color='yellow',bbox=dict(facecolor='black',alpha=0.6,pad=2))
        else:
            ax.imshow(full_ov(mp))
    # ---- bottom: per-mask hierarchy ----
    rl_full=relabel(L1reg); cols=[f'L{d}' for d in range(2,depth+1)]
    for ri,r in enumerate(regs):
        m=(L1reg==r).reshape(hf,wf); mf=m.reshape(-1)
        l1col=PAL[int(rl_full[L1reg==r][0])%len(PAL)]
        ax=fig.add_subplot(gbot[ri,0]); ax.axis('off'); ax.imshow(iso_overlay(m,l1col))
        ax.set_title(f'mask{ri} reg{r} ({int(m.sum())}p)',fontsize=9)
        for ci,lev in enumerate(sub):
            ax=fig.add_subplot(gbot[ri,ci+1]); ax.axis('off')
            ax.imshow(iso_disc(np.where(mf,lev,-1),m))
            if ri==0 and ci<len(cols): ax.set_title(cols[ci],fontsize=10)
            ns=len([v for v in np.unique(lev[mf]) if v>=0])
            ax.text(0.98,0.04,f'{ns}',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color='white',bbox=dict(facecolor='black',alpha=0.5,pad=1))
    fig.suptitle(f'{img_id} | res={size} top_n={top_n} depth{depth} {klabel} | decompose (L1->L{depth}) + group',fontsize=13,color='red',y=0.998)
    plt.savefig(out,dpi=105,bbox_inches='tight'); plt.close()

def process(m, image, size, top_n, depth, merge_sim, tag, out_dir,
            topn_rel=0.5, topn_cap=10):
    os.makedirs(out_dir,exist_ok=True)
    S,patch,pn,pil,(hf,wf)=features(m,image,size)
    leaf,levels=fine_map(S,patch,pn,hf,wf,top_n,depth,return_levels=True,topn_rel=topn_rel,topn_cap=topn_cap)
    grouped,nk,klabel=group(leaf,patch,hf,wf,merge_sim)
    ktag=f'sim{merge_sim}'
    ttag=f'top{top_n}' if str(top_n).lower()!='auto' else 'topAuto'
    out=f'{out_dir}/{tag}__res{size}__{ttag}__d{depth}__{ktag}.png'
    render_iter(pil,hf,wf,levels,grouped,nk,top_n,depth,klabel,tag,size,out)
    return out,nseg(leaf),nk

def _topn(v): return v if str(v).lower()=='auto' else int(v)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--image',required=True); ap.add_argument('--size',type=int,default=1024)
    ap.add_argument('--top_n',type=_topn,default='auto',help="'auto' (adaptive, default) or int for fixed branching")
    ap.add_argument('--topn_rel',type=float,default=0.5,help="keep registers with peak >= rel*max when --top_n auto")
    ap.add_argument('--topn_cap',type=int,default=10,help='max registers per region when --top_n auto')
    ap.add_argument('--depth',type=int,default=4)
    ap.add_argument('--merge_sim',type=float,default=0.85,help='merge leaf segments with cosine similarity >= this')
    ap.add_argument('--tag',default=None); ap.add_argument('--out_dir',default=OUT)
    args=ap.parse_args()
    m,_=ddm.load_model(rp.build_args())
    tag=args.tag or os.path.splitext(os.path.basename(args.image))[0]
    out,nf,nk=process(m,args.image,args.size,args.top_n,args.depth,args.merge_sim,tag,args.out_dir,
                      args.topn_rel,args.topn_cap)
    print('saved',out,'| fine',nf,'groups',nk)

if __name__=='__main__':
    main()
