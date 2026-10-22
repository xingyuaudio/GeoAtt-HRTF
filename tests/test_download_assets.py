"""Offline tests for revision selection, output paths, and integrity failures."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


pytest.importorskip("huggingface_hub")  # Optional [hub] dependency.

# Load the standalone tool without importing the torch-dependent model package.
spec = importlib.util.spec_from_file_location(
    "download_assets", Path(__file__).parents[1] / "tools" / "download_assets.py"
)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


@pytest.fixture
def downloads(tmp_path, monkeypatch):
    payloads = {name: name.encode() for name in assets.ASSETS}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    cached = {}
    for i, (name, data) in enumerate(payloads.items()):
        cached[name] = tmp_path / f"cached-{i}"
        cached[name].write_bytes(data)
    download = Mock(side_effect=lambda **kwargs: str(cached[kwargs["filename"]]))
    monkeypatch.setattr(assets, "hf_hub_download", download)
    monkeypatch.setattr(assets, "PAPER_SHA256", hashes)
    return payloads, cached, download


def test_default_revision_and_output_paths(tmp_path, monkeypatch, downloads):
    payloads, _, download = downloads
    api = Mock(side_effect=AssertionError("Paper assets must use embedded hashes"))
    monkeypatch.setattr(assets, "HfApi", api)
    monkeypatch.chdir(tmp_path)
    assets.main(["--output-dir", "output", "--cache-dir", "hub-cache"])
    expected_outputs = {
        "dataset/H_log.npy": "data/sonicom/H_log.npy",
        "dataset/locs.npy": "data/sonicom/locs.npy",
        "model/geoatt_final_ep440.pt": "weights/geoatt_final_ep440.pt",
    }
    assert download.call_count == 3
    for call, (source, destination) in zip(download.call_args_list, expected_outputs.items()):
        assert call.kwargs == dict(
            repo_id=assets.PAPER_REPO_ID, filename=source, repo_type="model",
            revision=assets.PAPER_REVISION, cache_dir=tmp_path / "hub-cache",
        )
        assert (tmp_path / "output" / destination).read_bytes() == payloads[source]
    api.assert_not_called()


@pytest.mark.parametrize("repo_id", [assets.PAPER_REPO_ID, "example/alternate"])
@pytest.mark.parametrize("revision", ["main", "release-tag", "c" * 40])
def test_override_resolves_once_and_checks_selected_hashes(
    tmp_path, monkeypatch, downloads, capsys, repo_id, revision
):
    payloads, _, download = downloads
    sha = "a" * 40
    siblings = [SimpleNamespace(rfilename=name, lfs=SimpleNamespace(
        sha256=hashlib.sha256(data).hexdigest()
    )) for name, data in payloads.items()]
    api = Mock()
    api.model_info.return_value = SimpleNamespace(sha=sha, siblings=siblings)
    monkeypatch.setattr(assets, "HfApi", lambda: api)
    # An override must not accidentally use the paper checksums.
    monkeypatch.setattr(assets, "PAPER_SHA256", {})
    assets.main([
        "--repo-id", repo_id, "--revision", revision, "--output-dir", str(tmp_path / "out")
    ])
    api.model_info.assert_called_once_with(repo_id, revision=revision, files_metadata=True)
    assert all(call.kwargs["revision"] == sha for call in download.call_args_list)
    assert f"{repo_id}@{sha}" in capsys.readouterr().out
    for source, destination in assets.ASSETS.items():
        assert (tmp_path / "out" / destination).read_bytes() == payloads[source]


@pytest.mark.parametrize("existing", [False, True])
def test_mismatch_stops_before_installing_bad_file(tmp_path, downloads, capsys, existing):
    _, cached, download = downloads
    source, destination = next(iter(assets.ASSETS.items()))
    cached[source].write_bytes(b"corrupt")
    output = tmp_path / "out"
    target = output / destination
    if existing:
        target.parent.mkdir(parents=True)
        target.write_bytes(b"previous output")
    with pytest.raises(ValueError, match="SHA256 mismatch.*expected.*got"):
        assets.main(["--output-dir", str(output)])
    assert download.call_count == 1
    assert "SHA256 verified" not in capsys.readouterr().out
    assert cached[source].read_bytes() == b"corrupt"
    if existing:
        assert target.read_bytes() == b"previous output"
        assert list(target.parent.iterdir()) == [target]
    else:
        assert list(target.parent.iterdir()) == []


@pytest.mark.parametrize("problem", ["missing_file", "missing_lfs", "invalid_digest", "invalid_sha"])
def test_invalid_override_metadata_fails_before_download(monkeypatch, downloads, problem):
    _, _, download = downloads
    siblings = [SimpleNamespace(rfilename=name, lfs=SimpleNamespace(sha256="b" * 64))
                for name in assets.ASSETS]
    sha = "a" * 40
    if problem == "missing_file":
        siblings.pop()
    elif problem == "missing_lfs":
        siblings[0].lfs = None
    elif problem == "invalid_digest":
        siblings[0].lfs.sha256 = "not-a-sha256"
    else:
        sha = "main"
    api = Mock()
    api.model_info.return_value = SimpleNamespace(sha=sha, siblings=siblings)
    monkeypatch.setattr(assets, "HfApi", lambda: api)
    with pytest.raises(ValueError, match="immutable commit SHA|Missing LFS SHA256"):
        assets.main(["--revision", "another-tag"])
    download.assert_not_called()
