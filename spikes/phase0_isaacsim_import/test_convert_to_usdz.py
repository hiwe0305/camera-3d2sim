import zipfile
from pathlib import Path

USDZ_PATH = Path(__file__).parent / "out" / "test_scene.usdz"


def test_usdz_file_exists_and_is_non_trivial():
    assert USDZ_PATH.exists(), f"{USDZ_PATH} was not produced -- run Task 3 Step 1 first"
    assert USDZ_PATH.stat().st_size > 1024, "USDZ file is suspiciously small (near-empty export)"


def test_usdz_is_a_valid_zip_archive():
    # .usdz is a ZIP container (uncompressed, per the OpenUSD spec) -- a corrupt/truncated
    # export would fail zipfile's own CRC check here before Isaac Sim ever sees it.
    assert zipfile.is_zipfile(USDZ_PATH)
    with zipfile.ZipFile(USDZ_PATH) as zf:
        bad_file = zf.testzip()
        assert bad_file is None, f"corrupt member in USDZ archive: {bad_file}"
        names = zf.namelist()
        assert any(name.endswith((".usd", ".usda", ".usdc")) for name in names), (
            f"no USD layer found inside the USDZ archive, contents: {names}"
        )
