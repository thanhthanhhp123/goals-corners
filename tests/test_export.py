"""The single-file export must bundle the ES modules into a script that parses."""
import re
import shutil
import subprocess

import pytest

from predictor import config, export


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("artifact", [False, True])
def test_export_bundle_parses(tmp_path, monkeypatch, artifact):
    if not (config.WEB_DATA_DIR / "meta.json").exists():
        pytest.skip("run `python -m predictor update` first")
    monkeypatch.setattr(export, "embed_logos", lambda teams: None)  # no network in tests
    out = export.export_html(tmp_path / "page.html", artifact=artifact)
    html = out.read_text(encoding="utf-8")
    assert "<title>" in html[:8192]
    assert ("<html" in html) is (not artifact)
    module = re.search(r'<script type="module">\n(.*)\n</script>', html, re.S).group(1)
    assert "import " not in module.split("\n", 1)[0]
    script = tmp_path / "bundle.mjs"
    script.write_text(module, encoding="utf-8")
    subprocess.run(["node", "--check", str(script)], check=True, capture_output=True)
