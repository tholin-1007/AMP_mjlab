from pathlib import Path
import hashlib,json
d=Path(__file__).resolve().parent
m=json.loads((d/'SHA256.json').read_text(encoding='utf-8'))
for name,v in m.items():
    p=d/name
    assert p.is_file(),name
    assert p.stat().st_size==v['bytes'],name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'],name
print(f"Verified {len(m)} files")
