import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from stimuli import POLICY, RESET, make_bank, render, verify_bank
from scoring import parse_response
from standalone_io import AdaptionProvider, LocalCalls, digest, file_hash, output_text, save

ROOT = Path(__file__).resolve().parent
PREVALENCES = ("low", "balanced")
FEEDBACK_ARMS = ("A", "C")
SOURCES = [
    "run_study.py",
    "standalone_io.py",
    "stimuli.py",
    "scoring.py",
    "analysis.py",]
REPAIR = 'Return your classification using exactly one word: CLEAR or VIOLATION. Do not add explanation.'
CONFIG = {'experiment': 'exp-040-standalone-rerun','model': 'qwen3.8-27b','seed': 2026111501,
          'n_histories': 16,'history_length': 20,'n_test': 64,'temperature': 0.0,
          'max_tokens': 16384,'max_credits': 250.0,'input_credits_per_million': .039,
          'output_credits_per_million': 1.25,'bootstrap_repeats': 5000}


def prepare(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'manifest.json').exists():
        return folder
    bank = make_bank(
    CONFIG['seed'],
    CONFIG['n_histories'],
    CONFIG['n_test'],
    CONFIG['history_length'])
    verify_bank(bank)
    save(folder/'bank.json',bank)
    save(folder/'config.json',CONFIG)
    manifest = {'implementation': 'standalone-v1; not original collector','status': 'prepared',
        'created_utc': datetime.now(timezone.utc).isoformat(),'config': CONFIG,
        'bank_sha256': file_hash(folder/'bank.json'),'config_sha256': file_hash(folder/'config.json'),
        'source_sha256': {n:file_hash(ROOT/n) for n in SOURCES},
        'pricing_note': 'Historical listed rates verified 2026-11-15; recheck before use',
        'planned_history_rows': (CONFIG['n_histories'] * 4 * CONFIG['history_length']),'planned_test_rows': (CONFIG['n_histories'] * 4 * CONFIG['n_test'])}
    manifest['run_id'] = 'standalone-'+digest(manifest)[:16]
    save(folder/'manifest.json',manifest)
    return folder

class Study:
    def __init__(self, folder, calls):
        self.folder,self.calls = Path(folder),calls
        self.manifest = json.loads((self.folder/'manifest.json').read_text(encoding='utf-8'))
        self.config = self.manifest['config']
        # Check that the experiment has not changed since preparation.
        if self.config != CONFIG:
            raise ValueError("Configuration changed")

        for name, expected in self.manifest["source_sha256"].items():
            if file_hash(ROOT / name) != expected:
                raise ValueError(f"Source changed: {name}")

        for name in ("bank", "config"):
            path = self.folder / f"{name}.json"
            if file_hash(path) != self.manifest[f"{name}_sha256"]:
                raise ValueError(f"Experiment inputs changed: {name}")
        self.bank = json.loads((self.folder/'bank.json').read_text(encoding='utf-8'))
        verify_bank(self.bank)
                # Reload completed decisions when resuming an experiment.
        self.rows = {}

        if (self.folder / "trials.jsonl").exists():
            for line in (self.folder / "trials.jsonl").read_text().splitlines():
                row = json.loads(line)
                key = self.key(row)

                if key in self.rows:
                    raise ValueError("Duplicate saved trial")

                self.rows[key] = row
        

    @staticmethod
    def key(row):
        return tuple(row[k] for k in ['phase','history_id','prevalence','feedback','item_id'])

    def call(self, messages, meta):
        identity = digest(messages)
        previous = self.rows.get(self.key(meta))
        if previous is not None:
            if previous["message_sha256"] != identity:
                raise ValueError("Saved trial has a different prompt")

            if (previous["decision"] not in (0, 1)
                    or previous["provider_status"] != "completed"):
                raise ValueError("Saved trial has no valid decision")

            return previous
        raw, key, cached = self.calls.complete(
            messages, self.config["max_tokens"])
        text = output_text(raw)
        decision,source = parse_response(text,raw)
        extra = {'format_repaired': False,'initial_request_hash': key}
        
        # Ask for a simple final label if the first response cannot be parsed.
        if (
            decision is None
            and raw.get("status") == "completed"
            and not raw.get("error")
            and not raw.get("incomplete_details")
        ):
            emitted = text or "\n".join(
                str(part.get("text", ""))
                for item in raw.get("output") or []
                for part in item.get("content") or []
                if part.get("type") in ("reasoning_text", "output_text")
            )

            repair_messages = messages + [
                ("assistant", emitted),
                ("user", REPAIR),
            ]

            raw, key, cached = self.calls.complete(
                repair_messages, 1024
            )



            text = output_text(raw)
            decision,source = parse_response(text,raw)
            if source != 'output_text':
                decision,source = None,'unparsed'
            extra['format_repaired'] = True
        row = {**meta,**extra,'decision': decision,'response_source': source,'text': text,
               'request_hash': key,'message_sha256': identity,'provider_status': raw.get('status'),'cached': cached}
        temp = self.folder/'trials.jsonl.tmp'
        temp.write_text(''.join(json.dumps(r)+'\n' for r in [*self.rows.values(),row]),encoding='utf-8')
        temp.replace(self.folder/'trials.jsonl')
        self.rows[self.key(meta)] = row
        if decision is None or raw.get("status") != "completed":
            raise ValueError("Model response could not be classified")
        return row

    def history(self, h, prevalence, feedback):
        messages = [('system',POLICY)]
        for position,case in enumerate(self.bank['histories'][str(h)][prevalence]):
            messages.append(('user',render(case)))
            row = self.call(messages,{'phase': 'history','history_id': h,'prevalence': prevalence,
                'feedback': feedback,'item_id': case['item_id'],'truth': case['truth'],
                'stratum': case['stratum'],'position': position})
            emitted = ('VIOLATION' if row['decision'] else 'CLEAR') if row['response_source'].startswith('reasoning_terminal') else row['text']
            messages.append(('assistant',emitted))
            note = 'Decision recorded.'
            if feedback == "C":
                label = "VIOLATION" if case["truth"] else "CLEAR"
                accuracy = "correct" if row["decision"] == case["truth"] else "incorrect"
                note = f"Independent audit label: {label}. Your decision was {accuracy}."

            messages.append(("user", note))
        return messages

    def execute(self):
        blocks = list(range(CONFIG['n_histories']))
        random.Random(CONFIG['seed']+9).shuffle(blocks)
        for h in blocks:
            branches = [(p, f) for p in PREVALENCES for f in FEEDBACK_ARMS]
            random.Random(CONFIG['seed']+10+h).shuffle(branches)
            histories = {(p,f):self.history(h,p,f) for p,f in branches}
            jobs = [(p,f,c) for p,f in branches for c in self.bank['tests']]
            random.Random(CONFIG['seed']+11+h).shuffle(jobs)
            for p,f,c in jobs:
                self.call(histories[p,f]+[('user',RESET+'\n'+render(c))],
                    {'phase': 'test','history_id': h,'prevalence': p,'feedback': f,
                     'item_id': c['item_id'],'truth': c['truth'],'stratum': c['stratum'],'item_block': c['item_block']})
            save(self.folder/'progress.json',{'last_completed_block': h,'rows': len(self.rows),
                                             'reserved_credit_bound': self.calls.spent()})
        self.manifest['status'] = 'complete'
        save(self.folder/'manifest.json',self.manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/main'))
    parser.add_argument('--prepare',action='store_true',help='offline only')
    parser.add_argument('--run',action='store_true',help='makes paid requests')
    parser.add_argument('--pricing-checked',action='store_true',help='confirm listed rates still apply')
    parser.add_argument('--validate',action='store_true',help='offline release checks; no API calls')
    args = parser.parse_args()
    if args.validate:
        from standalone_io import validate_release
        validate_release()
        return
    if args.run and not args.pricing_checked:
        parser.error('Check current provider rates, then supply --pricing-checked')
    folder = prepare(args.output)
    if args.run:
        lock = folder/'collector.lock'
        with lock.open('x',encoding='utf-8') as marker:
            marker.write('single collector; remove stale marker only after diagnosing stopped process\n')
        calls = None
        try:
            calls = LocalCalls(folder/'responses.sqlite',AdaptionProvider(),CONFIG)
            Study(folder,calls).execute()
        finally:
            if calls is not None:
                calls.close()
            lock.unlink()
    print('Standalone study prepared:',folder)


if __name__ == '__main__':
    main()
