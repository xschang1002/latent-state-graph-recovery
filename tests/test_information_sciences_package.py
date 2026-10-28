"""Regression checks for identified export without relaxing privacy or evidence."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_information_sciences_package as package


def author_block(contents: str) -> str:
    return chr(92) + "author" + chr(123) + contents + chr(125)


def test_canonical_author_and_correspondence_have_narrow_contexts():
    full = author_block(package.AUTHOR + chr(92) * 2 + "\n" + chr(92) + "small "
                        + package.AFFILIATION + chr(92) * 2 + "\n" + chr(92)
                        + "small" + chr(92) + "texttt" + chr(123) + package.EMAIL + chr(125))
    assert package.scan_text(full, "paper/main.tex") == []
    assert package.scan_text(author_block(package.AUTHOR), "paper/supplement.tex") == []
    assert package.scan_text(package.EMAIL, "submission_materials/information_sciences/declarations.txt") == []

    extra = author_block(package.AUTHOR + ", Sample Contributor")
    assert any(f["pattern"] == "tex_non_anonymous_author" for f in package.scan_text(extra, "paper/main.tex"))
    unknown_email = "unexpected" + chr(64) + "example.org"
    assert any(f["pattern"] == "email" for f in package.scan_text(unknown_email, "paper/main.tex"))
    assert package.scan_text(package.EMAIL, "audit/other_context.md")


def test_anonymous_archive_is_protected_even_with_overwrite(tmp_path):
    target = tmp_path / "stcg_anonymous_artifact.zip"
    target.write_bytes(b"preserved earlier archive")
    with pytest.raises(ValueError, match="protected"):
        package.write_archive([], {}, target, overwrite=True)
    assert target.read_bytes() == b"preserved earlier archive"
    assert not list(tmp_path.glob("*.tmp"))


def test_confirmed_postal_author_block_does_not_allow_added_identities():
    contents = package.AUTHOR + "\\\\\n\\small " + "\\\\\n\\small ".join(package.POSTAL_AFFILIATION_LINES)
    contents += "\\\\\n\\small\\texttt{" + package.EMAIL + "}"
    full = author_block(contents)
    assert package.scan_text(full, "paper/main.tex") == []
    extra = full.replace(package.AUTHOR, package.AUTHOR + ", Sample Contributor")
    assert any(f["pattern"] == "tex_non_anonymous_author" for f in package.scan_text(extra, "paper/main.tex"))
    assert package.scan_text(full, "audit/unknown.tex")


def test_manifest_repository_metadata_tracks_the_declared_release(tmp_path):
    path = tmp_path / package.MATERIALS / "deposit_metadata.json"
    path.parent.mkdir(parents=True)
    metadata = {"repository_status": "public versioned GitHub release", "doi": None,
                "public_url": "https://github.com/example/research", "version": "v1",
                "release_url": "https://github.com/example/research/releases/tag/v1"}
    path.write_text(json.dumps(metadata), encoding="utf-8")
    actual = package.repository_metadata(tmp_path)
    assert actual == {"status": metadata["repository_status"], "doi": None,
                      "url": metadata["public_url"], "version": "v1",
                      "release_url": metadata["release_url"]}
    metadata.update(repository_status="pending deposit", public_url=None, release_url=None)
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert package.repository_metadata(tmp_path)["url"] is None


def test_local_presentation_does_not_claim_the_prior_public_release_is_current(tmp_path):
    path = tmp_path / package.MATERIALS / "deposit_metadata.json"
    path.parent.mkdir(parents=True)
    metadata = {"repository_status": "local presentation revision", "doi": None,
                "public_url": "https://github.com/example/research", "version": "v2-layout.1",
                "release_url": None, "scientific_content_version": "v1",
                "public_scientific_release_url": "https://github.com/example/research/releases/tag/v1",
                "presentation_revision_published": False}
    path.write_text(json.dumps(metadata), encoding="utf-8")
    actual = package.repository_metadata(tmp_path)
    assert actual["version"] == "v2-layout.1" and actual["release_url"] is None
    assert actual["scientific_content_version"] == "v1"
    assert actual["presentation_revision_published"] is False
    assert actual["public_scientific_release_url"].endswith("/v1")


def test_changed_source_is_rejected_without_a_partial_archive(tmp_path):
    source = tmp_path / "scientific-output.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    entry = package.legacy.snapshot_entry(source, "results/example.csv")
    manifest = {"sha256": {entry.archive_name: package.sha(entry.read_bytes())}}
    source.write_text("value\n2\n", encoding="utf-8")
    target = tmp_path / "identified_fixture.zip"
    with pytest.raises(RuntimeError, match="Source changed after collection"):
        package.write_archive([entry], manifest, target)
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert source.read_text(encoding="utf-8") == "value\n2\n"


def test_new_numerical_outputs_and_supplement_are_collected_without_git_listing(tmp_path, monkeypatch):
    # The legacy collector intentionally does not list the new numerical tree:
    # its explicit traversal must include arrays even when Git ignores them.
    def write(name: str, content: str) -> Path:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        return target

    for name in package.EXTRA_FILES:
        write(name, "identified source\n")
    write(package.MATERIALS + "CITATION.cff", "cff-version: 1.2.0\n")
    write(package.MATERIALS + "repository.gitattributes", "* -text\n")
    write("docs/information_sciences_artifact_readme.md", "current identified README\n")
    old_readme = write("old_readme.txt", "obsolete anonymous README\n")
    old_sheet = write("old_sheet.txt", "obsolete MLJ information sheet\n")
    builder = write("tools/build_information_sciences_supplement.py", "# fixture builder\n")
    retained = tmp_path / "results/information_sciences/example/retained.npz"
    retained.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(retained, scores=np.arange(6).reshape(2, 3))
    write("results/information_sciences/supplement/tab_new.tex", "retained table\n")
    write("paper/sections/abstract.tex", "current article text\n")
    write("paper/supplement_sections/targets.tex", "current definitions\n")
    write("paper/sections/editor.tmp", "not a manuscript dependency\n")
    write("results/information_sciences/aggregation_literature.bib", "% verified citation fixture\n")
    legacy_entries = [
        package.legacy.snapshot_entry(old_readme, "README.md"),
        package.legacy.snapshot_entry(old_sheet, "submission_materials/mlj_contribution_information/main.tex"),
        package.legacy.snapshot_entry(builder, "tools/build_information_sciences_supplement.py"),
    ]
    monkeypatch.setattr(package.legacy, "collect_entries", lambda include_pdf, root: legacy_entries)
    entries = package.collect_entries(include_pdf=False, root=tmp_path)
    names = {entry.archive_name for entry in entries}
    assert "results/information_sciences/example/retained.npz" in names
    assert "results/information_sciences/supplement/tab_new.tex" in names
    assert "results/information_sciences/aggregation_literature.bib" in names
    assert "tools/build_information_sciences_supplement.py" in names
    assert "paper/supplement.tex" in names and "CITATION.cff" in names
    assert {"paper/sections/abstract.tex", "paper/supplement_sections/targets.tex"} <= names
    assert "paper/sections/editor.tmp" not in names
    assert not any(name.startswith("submission_materials/mlj_contribution_information/") for name in names)
    assert next(e for e in entries if e.archive_name == "README.md").read_bytes() == b"current identified README\n"
    assert next(e for e in entries if e.archive_name == ".gitattributes").read_bytes() == b"* -text\n"
    assert package.scan_identity(entries) == []


def test_pdf_freshness_dependencies_follow_nested_sections_and_figures(tmp_path):
    files = {
        "paper/main.tex": r"\input{paper/sections/results.tex}",
        "paper/sections/results.tex": r"\input{results/table.tex}\includegraphics{results/figure.pdf}",
        "results/table.tex": r"\input{paper/sections/results.tex}",
        "results/figure.pdf": "%PDF-fixture",
        "paper/references.bib": "% references",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    dependencies = package.pdf_dependencies(tmp_path, tmp_path / "paper/main.tex")
    assert {p.relative_to(tmp_path).as_posix() for p in dependencies} == set(files)
    (tmp_path / "results/figure.pdf").unlink()
    with pytest.raises(ValueError, match="Missing or escaping"):
        package.pdf_dependencies(tmp_path, tmp_path / "paper/main.tex")
