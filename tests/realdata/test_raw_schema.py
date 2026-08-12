from pathlib import Path
from pafar_sim.realdata.raw_io import parse_header
from pafar_sim.realdata.schema import EXPECTED_COLUMNS

ROOT=Path(__file__).resolve().parents[2]
def test_actual_file_counts_and_unique_names():
    a=list((ROOT/"data/physionet2019/raw/training_setA").glob("*.psv")); b=list((ROOT/"data/physionet2019/raw/training_setB").glob("*.psv"))
    assert len(a)==20336 and len(b)==20000
    assert len({x.stem for x in a+b})==40336
def test_actual_41_column_schema():
    for directory in (
        ROOT / "data/physionet2019/raw/training_setA",
        ROOT / "data/physionet2019/raw/training_setB",
    ):
        paths = sorted(directory.glob("*.psv"))
        assert paths
        assert parse_header(paths[0]) == EXPECTED_COLUMNS and len(EXPECTED_COLUMNS) == 41
