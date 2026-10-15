import sys
import zlib
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isa import appraisal  # noqa: E402

CORPUS = ROOT / "data/crowd-enVent/corpus/crowd-enVent_generation.tsv"
pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="crowd-enVent corpus not downloaded")


class FakeLM:
    def reply_hidden(self, messages, reply, layers):
        g = torch.Generator().manual_seed(zlib.crc32(reply.encode()))
        n = 5
        return {L: torch.randn(n, 16, generator=g) for L in layers}


def test_clean_text():
    assert appraisal.clean_text("I felt ... when my dog died") == "I felt something when my dog died"
    assert appraisal.clean_text("I won ... today") == "I felt something when I won something today"


def test_train_and_score(tmp_path):
    layers = [3, 5]
    items = appraisal.load_corpus(CORPUS, n=10, seed=0)
    assert len(items) == 13 * 10 and "messages" in items[0]
    rep = appraisal.train(FakeLM(), layers, tmp_path, CORPUS, n_per_emotion=10)
    assert set(rep) == set(layers)
    a = appraisal.maybe_load(tmp_path, layers)
    assert a is not None
    s = a.score({3: torch.randn(4, 16)})
    assert len(s) == 21 + 13 and "ap_urgency_L3" in s and "emo_fear_L3" in s
    assert abs(sum(v for k, v in s.items() if k.startswith("emo_")) - 1) < 1e-6
    assert not any(k.endswith("_L5") for k in s)
    assert appraisal.maybe_load(tmp_path / "nope", layers) is None
