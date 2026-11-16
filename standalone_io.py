# new custom functions for this hackathon based on my psyeval scripts


import hashlib
import json
from pathlib import Path
import sqlite3
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def output_text(raw):
    if isinstance(raw.get('output_text'), str) and raw['output_text']:
        return raw['output_text']
    return '\n'.join(p.get('text', '') for i in raw.get('output', []) if i.get('type') == 'message'
                     for p in i.get('content', []) if p.get('type') == 'output_text')


class AdaptionProvider:
    def __init__(self):
        from adaption import Adaption
        self.client = Adaption(max_retries=0, timeout=120.0)

    def __call__(self, payload):
        return self.client.responses.create(**payload).model_dump(mode='json')


class LocalCalls:
    """Single-process cache and durable conservative attempted-call ledger.

    Each attempt permanently reserves its full input/output allowance. Failed
    calls and crashes keep that reservation. Cache hits reserve nothing further.
    This can stop earlier than actual billing. No automatic retries occur.
    """
    def __init__(self, path, provider, config, interval=3.1):
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, raw TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, key TEXT, upper REAL)')
        self.db.commit()
        self.provider, self.config, self.interval = provider, config, interval
        self.last = 0.0

    def spent(self):
        return self.db.execute('SELECT COALESCE(SUM(upper),0) FROM attempts').fetchone()[0]

    def complete(self, messages, max_tokens):
        payload = {'model': self.config['model'], 'input': [
            {'role': r, 'content': t} for r,t in messages], 'temperature': self.config['temperature'],
            'max_output_tokens': max_tokens, 'agentic_memory': False, 'store': False,
            'tools': [], 'tool_choice': 'none', 'previous_response_id': None}
        key = digest({'standalone_cache_schema': 1, 'payload': payload})
        found = self.db.execute('SELECT raw FROM cache WHERE key=?', (key,)).fetchone()
        if found:
            return json.loads(found[0]), key, True
        size = len(canonical(payload).encode())+512
        if size > (120000 if max_tokens == 1024 else 20000):
            raise ValueError('Conservative input allowance exceeded')
        upper = (size*self.config['input_credits_per_million'] +
                 max_tokens*self.config['output_credits_per_million'])/1e6
        self.db.execute('BEGIN IMMEDIATE')
        if self.spent()+upper > self.config['max_credits']:
            self.db.rollback()
            raise RuntimeError('Credit ceiling would be exceeded')
        self.db.execute('INSERT INTO attempts(key,upper) VALUES(?,?)', (key,upper))
        self.db.commit()  # durable before any paid call
        time.sleep(max(0, self.interval-(time.monotonic()-self.last)))
        self.last = time.monotonic()
        raw = self.provider(payload)
        if not isinstance(raw, dict):
            raise ValueError('Provider returned non-object response')
        self.db.execute('INSERT INTO cache VALUES(?,?)', (key,canonical(raw)))
        self.db.commit()
        return raw, key, False

    def close(self):
        self.db.close()


def validate_release():
    """Small offline known-answer and deliberate-corruption acceptance checks."""
    from contextlib import nullcontext
    from uuid import uuid4
    from statistics import NormalDist
    from analysis import analyze, read_trials
    from run_study import CONFIG, REPAIR, Study, prepare
    from stimuli import RESET, make_bank, render, verify_bank
    from scoring import parse_response
    checks = []
    bank = make_bank(CONFIG['seed'])
    assert verify_bank(bank) == 704
    assert [sum(c['truth'] for c in bank['histories']['0'][p]) for p in ['low','balanced']] == [2,10]
    checks.append('generated bank and independent policy oracles')
    completed = {'status':'completed'}
    assert parse_response('CLEAR',completed) == (0,'output_text')
    assert parse_response('**VIOLATION**.',completed)[0] == 1
    for text,raw in [('CLEAR or VIOLATION',completed),('CLEAR',{'status':'incomplete'}),
                     ('CLEAR',{'status':'completed','error':{'message':'bad'}})]:
        assert parse_response(text,raw)[0] is None
    reasoning = {'status':'completed','output':[{'type':'reasoning','content':[
        {'type':'reasoning_text','text':'Considered the policy.\n**Decision:**\nCLEAR'}]}]}
    assert parse_response('',reasoning) == (0,'reasoning_terminal_decision')
    reasoning['output'][0]['content'][0]['text'] = 'Maybe CLEAR'
    assert parse_response('',reasoning)[0] is None
    checks.append('strict final/terminal parsing and rejection')


    # Construct known-answer data to test the analysis offline 
    test_items = [
        {
            "item_id": str(i),
            "truth": i % 2,
            "item_block": (
                "complete-" if i < 4 else "incomplete-"
            ) + str(i // 2),
        }
        for i in range(8)
    ]

    fixture = {
        "bank": {"tests": test_items},
        "rows": [
            {
                "phase": "test",
                "history_id": h,
                "prevalence": p,
                "feedback": f,
                "item_id": item["item_id"],
                "truth": item["truth"],
                "decision": item["truth"],
            }
            for h in range(2)
            for p in ("low", "balanced")
            for f in ("A", "C")
            for item in test_items
        ],
    }

    
    scratch = Path(__file__).parent/'validation'
    scratch.mkdir(exist_ok=True)
    directory = scratch/('offline-'+uuid4().hex)
    directory.mkdir()
    with nullcontext(directory) as directory:
        root = Path(directory)
        path = root/'synthetic.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in fixture['rows']),encoding='utf-8')
        result,_,_ = analyze(path,fixture['bank'],2,repeats=32)
        expected_d = 2*NormalDist().inv_cdf(8.5/9)
        assert all(abs(c['dprime']['estimate']-expected_d)<1e-12 and abs(c['criterion']['estimate'])<1e-12
                   for c in result['cells'].values())
        assert all(v['estimate']==v['lower']==v['upper']==0
                   for contrast in result['interactions'].values() for v in contrast.values())
        for kind in ['missing','duplicate','truth','decision']:
            damaged = json.loads(json.dumps(fixture['rows']))
            if kind == 'missing': damaged.pop()
            if kind == 'duplicate': damaged.append(damaged[0])
            if kind == 'truth': damaged[0]['truth'] = 1-damaged[0]['truth']
            if kind == 'decision': damaged[0]['decision'] = None
            path.write_text(''.join(json.dumps(r)+'\n' for r in damaged),encoding='utf-8')
            try: read_trials(path,fixture['bank'],2)
            except ValueError: pass
            else: raise AssertionError('Corruption escaped: '+kind)
        checks.append('miniature known-answer analysis, exact null and four corrupted inputs')
        payloads = []
        def fake(payload):
            payloads.append(payload)
            return {'status':'completed','output_text':'CLEAR','output':[]}
        calls = LocalCalls(root/'fake.sqlite',fake,CONFIG,interval=0)
        folder = prepare(root/'run')
        study = Study(folder,calls)
        histories = {
            (p, f): study.history(0, p, f)
            for p in ["low", "balanced"]
            for f in ["A", "C"]
            }
        for (p,f),messages in histories.items():
            assert len(messages)==61
            assert sum(role=='assistant' for role,_ in messages)==20
            assert all(text=='CLEAR' for role,text in messages if role=='assistant')
            feedback = [messages[i][1] for i in range(3,61,3)]
            assert all(
                ("Independent audit label:" in text) == (f == "C")
                for text in feedback
            )
            assert all(('Your decision was' in text)==(f=='C') for text in feedback)
        count = len(payloads)
        calls.close()
        calls = LocalCalls(root/'fake.sqlite',fake,CONFIG,interval=0)
        resumed = Study(folder,calls)
        assert resumed.history(0,'low','C') == histories['low','C']
        assert len(payloads)==count
        # Exercise cached request recovery independent of row-level checkpoints.
        raw,key,cached = calls.complete([('user','cache probe')],16384)
        spent = calls.spent()
        assert calls.complete([('user','cache probe')],16384)[2]
        assert calls.spent()==spent
        for case in bank['tests'][:2]:
            messages = histories['low','A']+[('user',RESET+'\n'+render(case))]
            resumed.call(messages,{'phase':'test','history_id':0,'prevalence':'low','feedback':'A',
                'item_id':case['item_id'],'truth':case['truth'],'stratum':case['stratum'],'item_block':case['item_block']})
            assert len(payloads[-1]['input'])==62
            assert payloads[-1]['input'][-1]['content'].startswith(RESET)
        assert all(p['agentic_memory'] is False and p['store'] is False and p['tools']==[]
                   and p['tool_choice']=='none' and p['previous_response_id'] is None for p in payloads)
        calls.close()
        checks.append(
            "four sequential histories, feedback, cache reopen, "
            "resume and independent tests"
        )        
        def forbidden(payload):
            raise AssertionError('Budget rejected request reached provider')
        blocked = LocalCalls(root/'budget.sqlite',forbidden,{**CONFIG,'max_credits':0},interval=0)
        try: blocked.complete([('user','budget probe')],16384)
        except RuntimeError: pass
        else: raise AssertionError('Budget did not stop')
        blocked.close()
        checks.append('credit ceiling before inference')
        repair_payloads = []
        def repaired(payload):
            repair_payloads.append(payload)
            return {'status':'completed','output_text':'unparseable' if len(repair_payloads)==1 else 'CLEAR'}
        repair_calls = LocalCalls(root/'repair.sqlite',repaired,CONFIG,interval=0)
        repair_study = Study(prepare(root/'repair'),repair_calls)
        row = repair_study.call([('user','test only')],{'phase':'test','history_id':0,
            'prevalence':'low','feedback':'A','item_id':'repair-probe','truth':1})
        assert row['format_repaired'] and row['decision']==0 and row['response_source']=='output_text'
        assert repair_payloads[1]['max_output_tokens']==1024
        assert repair_payloads[1]['input'][-1]=={'role':'user','content':REPAIR}
        assert repair_payloads[1]['input'][-2]=={'role':'assistant','content':'unparseable'}
        repair_calls.close()
        checks.append('single truth-blind 1024-token final-answer repair')
    print(json.dumps({'passed':True,'provider_calls':0,'checks':checks},indent=2))
