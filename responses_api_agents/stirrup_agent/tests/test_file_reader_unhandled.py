# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A deliverable that reaches no handler must never be reported as nonexistent.

``convert_deliverables_to_content_blocks`` dispatches on file extension through a
chain of ``if``/``elif``. With no ``else``, an extension without a branch emitted
nothing at all -- and the judge, handed a deliverable list the file was absent
from, graded it as never produced, with ``invalid_judge_response`` still false.

The invariant these tests defend is one line long: **silence must never read as
absence.** Every deliverable is named, whatever its type.
"""

from pathlib import Path

from responses_api_agents.stirrup_agent.file_reader import (
    HANDLED_EXTS,
    LEGACY_OFFICE_EXTS,
    OFFICE_EXTS,
    TEXT_EXTS,
    convert_deliverables_to_content_blocks,
)


def _text_of(blocks) -> str:
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def test_every_deliverable_is_named_whatever_its_extension(tmp_path: Path):
    """The core invariant, over one file of each shape that used to vanish."""
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "proxy.ts").write_text("export const x = 1;\n")
    (d / "pyproject.toml").write_text("[project]\nname='x'\n")
    (d / "notebook.ipynb").write_text('{"cells": []}')
    (d / "diagram.svg").write_text("<svg></svg>")
    (d / "Makefile").write_text("all:\n\tcc x.c\n")  # extensionless, sniffs as text
    (d / "blob.bin").write_bytes(b"\x00\x01\x02" * 500)  # extensionless-ish binary

    joined = _text_of(convert_deliverables_to_content_blocks(str(d), media_mode="images_and_text"))

    for name in ("proxy.ts", "pyproject.toml", "notebook.ipynb", "diagram.svg", "Makefile", "blob.bin"):
        assert name in joined, f"{name} reached the judge as nothing at all"


def test_source_file_contents_reach_the_judge_not_just_the_name(tmp_path: Path):
    """Naming is the floor; source files are readable and must be read."""
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "auth.ts").write_text("export function login(u: string) { return u; }\n")

    joined = _text_of(convert_deliverables_to_content_blocks(str(d), media_mode="images_and_text"))
    assert "export function login" in joined


def test_unreadable_binary_is_announced_as_present_not_omitted(tmp_path: Path):
    """A format we cannot parse still gets a name and a size."""
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "model.bin").write_bytes(b"\x00\xff" * 4096)

    joined = _text_of(convert_deliverables_to_content_blocks(str(d), media_mode="images_and_text"))
    assert "model.bin" in joined
    assert "NOT missing" in joined
    assert "8,192" in joined  # the size, so the judge can tell empty from substantial


def test_zip_deliverable_lists_its_members(tmp_path: Path):
    """"Does the bundle contain X" is a real rubric criterion; the manifest answers it."""
    import zipfile

    d = tmp_path / "repeat_0"
    d.mkdir()
    with zipfile.ZipFile(d / "Bundle.zip", "w") as z:
        z.writestr("fix/patch.diff", "--- a\n+++ b\n")
        z.writestr("fix/NOTES.md", "notes")

    joined = _text_of(convert_deliverables_to_content_blocks(str(d), media_mode="images_and_text"))
    assert "Bundle.zip" in joined
    assert "fix/patch.diff" in joined and "fix/NOTES.md" in joined


def test_legacy_office_is_treated_as_office_not_as_an_unknown_blob(tmp_path: Path):
    """``.doc``/``.ppt``/``.xls`` must use the sibling PDF preconvert already wrote."""
    assert LEGACY_OFFICE_EXTS <= OFFICE_EXTS
    for ext in (".doc", ".ppt", ".xls"):
        assert ext in OFFICE_EXTS

    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "Deck.ppt").write_bytes(b"\xd0\xcf\x11\xe0legacy-ole2")
    # what preconvert would have produced
    (d / "Deck.pdf").write_bytes(b"%PDF-1.4\n%fake\n")

    blocks = convert_deliverables_to_content_blocks(str(d), media_mode="native_pdf")
    joined = _text_of(blocks)
    assert "Deck.ppt" in joined
    # COUNT, not any(): the sibling PDF is consumed as the Office render, so it
    # must not also be emitted standalone. `any()` passes with two copies, which
    # is precisely the defect -- the judge shown the same pages twice.
    assert sum(1 for b in blocks if b.get("type") == "image_url") == 1
    assert "Deck.pdf" not in joined


def test_handled_exts_is_derived_not_hand_maintained(tmp_path: Path):
    """Derived, so the allowlist and the fallback test cannot drift apart."""
    assert TEXT_EXTS <= HANDLED_EXTS
    assert OFFICE_EXTS <= HANDLED_EXTS
    assert ".pdf" in HANDLED_EXTS


def test_office_sidecar_is_not_also_emitted_standalone(tmp_path: Path):
    """``Plan.pptx.pdf`` belongs to ``Plan.pptx``; showing it twice invites double credit."""
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "Plan.pptx").write_bytes(b"PK\x03\x04fake")
    (d / "Plan.pptx.pdf").write_bytes(b"%PDF-1.4\n%fake\n")

    blocks = convert_deliverables_to_content_blocks(str(d), media_mode="native_pdf")
    joined = _text_of(blocks)
    assert "Plan.pptx" in joined
    assert joined.count("Plan.pptx.pdf") == 0
    assert sum(1 for b in blocks if b.get("type") == "image_url") == 1
