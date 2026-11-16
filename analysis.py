#computes sdt and computes interaction. also estimates uncertainty using a paired bootstrap over histories and test items.
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtri

CONDITIONS = [
    (p, f)
    for p in ["low", "balanced"]
    for f in ["A", "C"]
]
METRICS=['dprime','criterion','hit_rate','false_alarm_rate','false_negative_rate','accuracy']

def validate_bank(bank):
    tests=bank['tests']
    ids=[c['item_id'] for c in tests]
    if not tests or len(set(ids))!=len(ids): 
        raise ValueError('Empty or duplicate test-item bank')
    if any(type(c['truth']) is not int or c['truth'] not in [0,1] for c in tests):
        raise ValueError('Nonbinary bank truth')
    if {c['truth'] for c in tests}!={0,1}: 
        raise ValueError('Both true classes are required')
    paired=['item_block' in c for c in tests]
    if any(paired) and not all(paired): 
        raise ValueError('Partial paired-item metadata')
    if all(paired):
        groups={}
        for c in tests:
            block=c['item_block']
            if not isinstance(block,str) or not block.startswith(('complete-','incomplete-')):
                raise ValueError('Unknown item-block stratum')
            groups.setdefault(block,[]).append(c['truth'])
        if any(len(ys)!=2 or sorted(ys)!=[0,1] for ys in groups.values()):
            raise ValueError('Each item block must pair one signal and one noise item')

def read_trials(path, bank, n_histories):
    validate_bank(bank)
    if type(n_histories) is not int or n_histories<1: raise ValueError('Positive history count required')
    rows=[json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
    rows=[r for r in rows if r.get('phase')=='test']
    lookup={c['item_id']:c for c in bank['tests']}
    expected={(h,p,f,i) for h in range(n_histories) for p,f in CONDITIONS for i in lookup}
    seen=set()
    for r in rows:
        key=(r['history_id'],r['prevalence'],r['feedback'],r['item_id'])
        if key in seen: raise ValueError('Duplicate test record')
        if key not in expected: raise ValueError('Unexpected record identity')
        if r['truth']!=lookup[r['item_id']]['truth']: raise ValueError('Ground truth mismatch')
        if type(r['decision']) is not int or r['decision'] not in [0,1]: raise ValueError('Invalid/missing decision')
        seen.add(key)
    if seen!=expected: raise ValueError(f'Incomplete coverage {len(seen)}/{len(expected)}')
    return rows

def assemble(rows, bank, n_histories):
    ids=[c['item_id'] for c in bank['tests']]; index={x:i for i,x in enumerate(ids)}
    cube = np.empty((len(CONDITIONS), n_histories, len(ids)),dtype=np.int8)    
    for r in rows:
        cube[CONDITIONS.index((r['prevalence'],r['feedback'])),r['history_id'],index[r['item_id']]]=r['decision']
    return cube,np.array([c['truth'] for c in bank['tests']],dtype=int)

def metrics(cube,y,hw=None,iw=None):
    hw=np.ones(cube.shape[1]) if hw is None else hw
    iw=np.ones(cube.shape[2]) if iw is None else iw
    sums=np.einsum('chi,h,i->c',cube,hw,iw*y)
    falses=np.einsum('chi,h,i->c',cube,hw,iw*(1-y))
    ns=hw.sum()*np.dot(iw,y); nn=hw.sum()*np.dot(iw,1-y)
    h=(sums+.5)/(ns+1); f=(falses+.5)/(nn+1)
    zh,zf=ndtri(h),ndtri(f)
    return np.stack([zh-zf,-.5*(zh+zf),sums/ns,falses/nn,1-sums/ns,(sums+nn-falses)/(ns+nn)],axis=-1)



def contrasts(m):
    """Prevalence × feedback interaction, with C compared against A."""
    # Conditions: low_A, low_C, balanced_A, balanced_C.
    prevalence_A = m[0] - m[2]
    prevalence_C = m[1] - m[3]

    return np.stack([prevalence_C - prevalence_A])


def analyze(path,bank,n_histories,repeats=5000,seed=71932):
    rows=read_trials(path,bank,n_histories)
    cube,y=assemble(rows,bank,n_histories); point=metrics(cube,y)
    rng = np.random.default_rng(seed)

    boots = np.empty((repeats, len(CONDITIONS), len(METRICS)))
    cb = np.empty((repeats, 1, len(METRICS)))

    classes = [np.flatnonzero(y == a) for a in [0, 1]]
    paired = all("item_block" in c for c in bank["tests"])

    if paired:
        block_names = sorted({c["item_block"] for c in bank["tests"]})
        block_ix = np.array([
            block_names.index(c["item_block"])
            for c in bank["tests"]
        ])
        strata = {
            s: [j for j, b in enumerate(block_names)
                if b.startswith(s + "-")]
            for s in ["complete", "incomplete"]
        }

    # Resample histories and matched stimuli, preserving their pairing.
    for b in range(repeats):
        hw = rng.multinomial(
            n_histories, np.ones(n_histories) / n_histories
        )
        iw = np.zeros(len(y))

        if paired:
            bw = np.zeros(len(block_names))
            for ix in strata.values():
                if ix:
                    bw[ix] = rng.multinomial(
                        len(ix), np.ones(len(ix)) / len(ix)
                    )
            iw = bw[block_ix]
        else:
            for ix in classes:
                iw[ix] = rng.multinomial(
                    len(ix), np.ones(len(ix)) / len(ix)
                )

        boots[b] = metrics(cube, y, hw, iw)
        cb[b] = contrasts(boots[b])

    cells={}

    for k,(p,f) in enumerate(CONDITIONS):
        cells[f'{p}_{f}']={m:{'estimate':float(point[k,j]),'lower':float(np.quantile(boots[:,k,j],.025)),
                             'upper':float(np.quantile(boots[:,k,j],.975))} for j,m in enumerate(METRICS)}
        cells[f'{p}_{f}']['counts']={'hits':int(cube[k,:,y==1].sum()),'false_alarms':int(cube[k,:,y==0].sum()),
                                    'signals':int((y==1).sum()*n_histories),'noise':int((y==0).sum()*n_histories)}
    
    cs={}
    
    for k, name in enumerate(["C_minus_A"]):
        cs[name]={m:{'estimate':float(contrasts(point)[k,j]),'lower':float(np.quantile(cb[:,k,j],.025)),
                     'upper':float(np.quantile(cb[:,k,j],.975))} for j,m in enumerate(METRICS)}
    return {'n_trials':len(rows),'n_histories':n_histories,'n_items':len(y),'bootstrap_repeats':repeats,
            'bootstrap_seed':seed,
            'item_resampling':'matched_pairs_within_telemetry_stratum' if paired else 'items_within_truth_class',
            'n_item_blocks':len(block_names) if paired else len(y),
            'cells':cells,'interactions':cs},cube,y

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('run'); parser.add_argument('--repeats',type=int,default=5000)
    args=parser.parse_args(); run=Path(args.run)
    bank=json.loads((run/'bank.json').read_text()); manifest=json.loads((run/'manifest.json').read_text())
    result,cube,y=analyze(run/'trials.jsonl',bank,manifest['config']['n_histories'],args.repeats)
    (run/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(run/'analysis_arrays.npz',decisions=cube,truth=y)
    print(json.dumps(result['interactions']['C_minus_A'],indent=2))

if __name__=='__main__': main()
