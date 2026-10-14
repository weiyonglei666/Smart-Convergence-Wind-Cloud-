from app.nwp.file_selector import select_latest_files


FILES = [
    "tower_sfc_20260920_18_short.grib2",
    "tower_sfc_20260921_00_short.grib2",
    "tower_pl_20260920_12_short.grib2",
    "tower_pl_20260920_18_short.grib2",
]


def test_select_latest_files_separately():
    result = select_latest_files(FILES, require_common_cycle=False)
    assert result.sfc.endswith("tower_sfc_20260921_00_short.grib2")
    assert result.pl.endswith("tower_pl_20260920_18_short.grib2")


def test_select_latest_common_cycle():
    result = select_latest_files(FILES, require_common_cycle=True)
    assert result.sfc.endswith("tower_sfc_20260920_18_short.grib2")
    assert result.pl.endswith("tower_pl_20260920_18_short.grib2")


def test_selector_accepts_windows_ssh_literal_newline_output():
    output = "tower_sfc_20260921_00_short.grib2\\ntower_pl_20260920_18_short.grib2"
    result = select_latest_files(output.splitlines())
    assert result.sfc.endswith("20260921_00_short.grib2")
