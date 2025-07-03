import subprocess
import filecmp
import os
from pathlib import Path
import pandas as pd

def assert_csv_equal(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    pd.testing.assert_frame_equal(df1, df2, check_dtype=False, check_like=True)

def assert_excel_equal(file1, file2):
    xls1 = pd.ExcelFile(file1)
    xls2 = pd.ExcelFile(file2)
    assert set(xls1.sheet_names) == set(xls2.sheet_names), "Sheet names differ"
    for sheet in xls1.sheet_names:
        df1 = xls1.parse(sheet)
        df2 = xls2.parse(sheet)
        pd.testing.assert_frame_equal(df1, df2, check_dtype=False, check_like=True)

def test_parse_bruker_logs_output(tmp_path):
    data_dir = Path(__file__).parent.parent / 'data'
    ref_dir = data_dir / 'parsed_log'
    test_dir = tmp_path / 'parsed_log'
    test_dir.mkdir()
    # Run the script
    subprocess.run([
        'python', str(data_dir.parent / 'parse_bruker_logs.py'),
        str(data_dir / 'neo400_log_test.full'),
        '-o', str(test_dir)
    ], check=True)
    # Compare all files in parsed_log
    for fname in os.listdir(ref_dir):
        ref_file = ref_dir / fname
        test_file = test_dir / fname
        assert test_file.exists(), f"Missing output: {fname}"
        if fname.endswith('.csv'):
            assert_csv_equal(ref_file, test_file)
        else:
            assert filecmp.cmp(ref_file, test_file, shallow=False), f"Mismatch in {fname}"

def test_generate_cfs_output(tmp_path):
    data_dir = Path(__file__).parent.parent / 'data'
    ref_dir = data_dir / 'generated_cfs'
    test_dir = tmp_path / 'generated_cfs'
    test_dir.mkdir()
    # Run the script
    subprocess.run([
        'python', str(data_dir.parent / 'generate_cfs.py'),
        str(data_dir / 'parsed_log/neo400_log_test_billable.csv'),
        str(data_dir / 'pif_test.xlsx'),
        '-i', 'neo400',
        '-o', str(test_dir)
    ], check=True)
    # Compare all files in generated_cfs
    for fname in os.listdir(ref_dir):
        ref_file = ref_dir / fname
        test_file = test_dir / fname
        assert test_file.exists(), f"Missing output: {fname}"
        if fname.endswith('.csv'):
            assert_csv_equal(ref_file, test_file)
        elif fname.endswith('.xlsx'):
            assert_excel_equal(ref_file, test_file)
        else:
            assert filecmp.cmp(ref_file, test_file, shallow=False), f"Mismatch in {fname}"
