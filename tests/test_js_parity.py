"""The browser match tool (web/model.js) must reproduce the Python fixture predictions."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from predictor import config, dist

SCRIPT = r"""
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
const dir = process.argv[2];
const { predictMatch } = await import(pathToFileURL(dir + '/model.js'));
const load = (n) => JSON.parse(readFileSync(dir + '/data/' + n + '.json', 'utf8'));
// odds-blended fixtures first, so both code paths are compared
const model = load('model'), teams = load('teams');
const fixtures = load('fixtures').sort((a, b) => (b.market ? 1 : 0) - (a.market ? 1 : 0)).slice(0, 80);
const out = fixtures.map((f) => ({
  id: f.id, pred: predictMatch(model, teams, f.home, f.away, f.comp, f.neutral, f.market ? f.market.xg : null),
}));
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_matches_python(tmp_path):
    fixtures_path = config.WEB_DATA_DIR / "fixtures.json"
    if not fixtures_path.exists():
        pytest.skip("run `python -m predictor update` first")
    fixtures = {f["id"]: f for f in json.loads(fixtures_path.read_text(encoding="utf-8"))}
    if not fixtures:
        pytest.skip("no fixtures to compare")
    script = tmp_path / "parity.mjs"
    script.write_text(SCRIPT, encoding="utf-8")
    res = subprocess.run(["node", str(script), str(config.WEB_DIR)], capture_output=True, text=True, check=True)
    for row in json.loads(res.stdout):
        py = fixtures[row["id"]]
        js = row["pred"]
        for kind in ("goals", "corners", "cards"):
            # ratings are exported rounded to 4 d.p., so allow small drift
            assert js[kind]["total"] == pytest.approx(py[kind]["total"], abs=5e-3)
            for line, p in py[kind]["over"].items():
                assert js[kind]["over"][line] == pytest.approx(p, abs=3e-3), (row["id"], kind, line)
            for a, b in zip(js[kind]["pmf"], py[kind]["pmf"]):
                assert a == pytest.approx(b, abs=2e-3)
        assert js["goals"]["btts"] == pytest.approx(py["goals"]["btts"], abs=3e-3)
        assert js["goals"]["cs"][0] == pytest.approx(py["goals"]["cs"][0], abs=3e-3)
        assert js["corners"]["home_more"] == pytest.approx(py["corners"]["home_more"], abs=3e-3)
        for a, b in zip(js["goals"]["hda"] + js["ht"]["hda"], py["goals"]["hda"] + py["ht"]["hda"]):
            assert a == pytest.approx(b, abs=3e-3)
        for line, p in py["ht"]["over"].items():
            assert js["ht"]["over"][line] == pytest.approx(p, abs=3e-3)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_asian_handicap_settlement(tmp_path):
    m = dist.score_matrix(1.7, 1.0, -0.03, 10)
    script = tmp_path / "ah.mjs"
    script.write_text(
        "import { pathToFileURL } from 'node:url';\n"
        "const { ahSettle } = await import(pathToFileURL(process.argv[2] + '/model.js'));\n"
        f"const m = {json.dumps(m.tolist())};\n"
        "console.log(JSON.stringify([-0.75, -0.5, 0, 0.25, 1].flatMap((l) => ['home', 'away'].map((s) => ahSettle(m, l, s)))));\n",
        encoding="utf-8")
    js = json.loads(subprocess.run(["node", str(script), str(config.WEB_DIR)], capture_output=True, text=True, check=True).stdout)
    d = np.subtract.outer(np.arange(11), np.arange(11))  # home margin
    win = lambda line, s: (m[(d if s == "home" else -d) + (line if s == "home" else -line) > 0]).sum()  # noqa: E731
    lose = lambda line, s: (m[(d if s == "home" else -d) + (line if s == "home" else -line) < 0]).sum()  # noqa: E731
    i = 0
    for line in (-0.75, -0.5, 0, 0.25, 1):
        halves = (line - 0.25, line + 0.25) if line * 4 % 2 else (line, line)
        for side in ("home", "away"):
            want = [sum(win(x, side) for x in halves) / 2, sum(lose(x, side) for x in halves) / 2]
            assert js[i] == pytest.approx(want, abs=1e-9), (line, side)
            i += 1
    # home -0.75: winning by one returns half the stake, so it is neither all won nor all lost
    assert js[0][0] + js[0][1] < 1
