import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from make_fake_runs import main as make_fake  # noqa: E402


def test_analyse_runs_on_fake_data(tmp_path):
    runs = tmp_path / "fake.jsonl"
    out = tmp_path / "analysis"
    make_fake(str(runs))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "05_analyse.py"), str(runs), "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    md = (out / "summary.md").read_text()
    assert "H2" in md
    assert "Criterion 1" in md
