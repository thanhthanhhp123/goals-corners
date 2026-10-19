"""The browser match tool (web/model.js) must reproduce the Python fixture predictions."""
import json
import shutil
import subprocess

import pytest

from predictor import config

SCRIPT = r"""
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
const dir = process.argv[2];
const { predictMatch } = await import(pathToFileURL(dir + '/model.js'));
const load = (n) => JSON.parse(readFileSync(dir + '/data/' + n + '.json', 'utf8'));
const model = load('model'), teams = load('teams'), fixtures = load('fixtures').slice(0, 60);
const out = fixtures.map((f) => ({ id: f.id, pred: predictMatch(model, teams, f.home, f.away, f.comp, f.neutral) }));
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
        for kind in ("goals", "corners"):
            # ratings are exported rounded to 4 d.p., so allow small drift
            assert js[kind]["total"] == pytest.approx(py[kind]["total"], abs=5e-3)
            for line, p in py[kind]["over"].items():
                assert js[kind]["over"][line] == pytest.approx(p, abs=3e-3), (row["id"], kind, line)
            for a, b in zip(js[kind]["pmf"], py[kind]["pmf"]):
                assert a == pytest.approx(b, abs=2e-3)
        assert js["goals"]["btts"] == pytest.approx(py["goals"]["btts"], abs=3e-3)
        assert js["goals"]["cs"][0] == pytest.approx(py["goals"]["cs"][0], abs=3e-3)
        assert js["corners"]["home_more"] == pytest.approx(py["corners"]["home_more"], abs=3e-3)
