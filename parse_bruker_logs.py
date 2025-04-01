import pandas as pd
import re
import argparse
from pathlib import Path
import sys
from datetime import datetime
import os
from difflib import get_close_matches

# Version information
VERSION = "1.0.0"

def parse_log_file(file_path: Path) -> tuple:
    """Parse the log file to extract experiment details."""
    
    # Initialize lists to store parsed data
    all_data = []
    
    # Read the log file
    with open(file_path, 'r', encoding='cp1252') as file:
        lines = file.readlines()

    # Variables to store current experiment details
    current_experiment = {}
    failure_flag = False

    # Regex patterns to match relevant lines
    patterns = {
        'experiment': re.compile(r'nameOfExperiment:\s*(.*)'),
        'solvent': re.compile(r'solvent:\s*(.*)'),
        'filename': re.compile(r'fileName:\s*(.*)'),
        'filesize': re.compile(r'fileSizeAcq:\s*(\d+)'),
        'nucleus': re.compile(r'NUCLEUS:\s*(.*)'),
        'advisor': re.compile(r'Advisor\s*(.*)'),
        'grant': re.compile(r'Grant number\s*(.*)'),
        'onyen': re.compile(r'Onyen\s*([\w\.-]+)(?:@.*)?'),
        'start': re.compile(r'timeOfStart:\s*(.*)'),
        'end': re.compile(r'timeOfTermination:\s*(.*)')
    }

    for line in lines:
        for key, pattern in patterns.items():
            match = pattern.search(line)
            if not match:
                continue
            if key in ['start', 'end']:
                parts = match.group(1).rsplit(' ', 1)
                datetime_str = parts[0]
                timestamp_str = parts[1] if len(parts) == 2 else ''
                current_experiment[f"{key}_time"] = datetime_str
                current_experiment[f"{key}_timestamp"] = timestamp_str
            else:
                current_experiment[key] = match.group(1)

        # Check if this line indicates a failure
        if '#Failure' in line:
            failure_flag = True
            current_experiment['status'] = 'failed'
        
        # If we reach the end of an experiment (indicated by '----' separator), store the data
        if '----' in line and current_experiment:
            # Mark the entry as successful or failed
            if not failure_flag:
                current_experiment['status'] = 'completed'
            all_data.append(current_experiment)
            
            # Reset the current_experiment and failure_flag for the next block
            current_experiment = {}
            failure_flag = False

    # Convert list of experiments into a pandas DataFrame
    columns = ['experiment', 'solvent', 'nucleus', 'filename', 'filesize', 'advisor', 'grant', 'onyen', 
               'start_time', 'start_timestamp', 'end_time', 'end_timestamp', 'status']
    
    log = pd.DataFrame(all_data, columns=columns)
    
    log.loc[
        (log['start_time'].isnull() | (log['start_time'] == '') |
         log['end_time'].isnull() | (log['end_time'] == '')),
        'status'
    ] = 'failed'

    # Convert timestamps to datetime objects
    def parse_timestamp(ts):
        if pd.isna(ts) or ts == '':
            return pd.NaT
        try:
            # First try to parse as Unix timestamp (integer)
            if isinstance(ts, str) and ts.isdigit():
                return datetime.fromtimestamp(int(ts))
            # Then try to parse as formatted datetime string
            return pd.to_datetime(ts, format='%Y%m%d%H%M%S', errors='coerce')
        except:
            return pd.NaT

    log['start_timestamp'] = log['start_timestamp'].apply(parse_timestamp)
    log['end_timestamp'] = log['end_timestamp'].apply(parse_timestamp)

    # Create two DataFrames: successful_runs and failed_runs
    completed_runs = log[log['status'] == 'completed'].copy()
    failed_runs = log[log['status'] == 'failed'].copy()

    return log, completed_runs, failed_runs

def create_cfs(row: pd.Series) -> str:
    parts = [
        str(row['fund']) if pd.notna(row['fund']) else '',
        str(row['source']) if pd.notna(row['source']) else '',
        str(row['dept']) if pd.notna(row['dept']) else '',
        str(row['project_ID']) if pd.notna(row['project_ID']) else '',
        str(row['program']) if pd.notna(row['program']) else '',
        str(row['cost_code_1']) if pd.notna(row['cost_code_1']) else '',
        str(row['cost_code_2']) if pd.notna(row['cost_code_2']) else '',
        str(row['cost_code_3']) if pd.notna(row['cost_code_3']) else ''
    ]
    return '-'.join(parts)

def process_ilabs_logs(df: pd.DataFrame, pif: pd.DataFrame) -> tuple:
    def fuzzy_match(value, choices, cutoff=0.8):
        # Try to find the closest match, returning the original value if none are close enough
        matches = get_close_matches(str(value), map(str, choices), n=1, cutoff=cutoff)
        return matches[0] if matches else value

    # Create copies so as not to modify the original data frames
    pif_adjusted = pif.copy()
    df_adjusted = df.copy()

    # Get the valid unique values for 'onyen' and 'grant' from the pif dataframe
    valid_onyen = pif_adjusted['onyen'].unique()
    valid_grant = pif_adjusted['grant'].unique()

    # Correct potential typos in the df onyen and grant columns using fuzzy matching
    df_adjusted['onyen'] = df_adjusted['onyen'].apply(lambda x: fuzzy_match(x, valid_onyen))
    df_adjusted['grant'] = df_adjusted['grant'].apply(lambda x: fuzzy_match(x, valid_grant))

    # Perform a left join between the adjusted df and pif on the corrected 'onyen' and 'grant' columns
    merged_df = df_adjusted.merge(pif_adjusted, on=['onyen', 'grant'], how='left')
    merged_df['CFS'] = merged_df.apply(create_cfs, axis=1)
    
    # Segregate rows based on advisor
    alcon_mask = merged_df['advisor'].str.contains('Alcon', case=False, na=False)
    nmr_core_mask = merged_df['advisor'].str.contains('ter Horst', case=False, na=False)
    
    alcon_df = merged_df[alcon_mask].copy()
    nmr_core_df = merged_df[nmr_core_mask].copy()
    
    # Exclude alcon and nmr_core entries from merged_df
    merged_df = merged_df[~(alcon_mask | nmr_core_mask)].copy()
    
    # Find rows with valid CFS data: using either a complete set of [fund, source, dept, program]
    # or a complete set of [fund, source, dept, project_ID]
    valid_mask = (
        merged_df[['fund', 'source', 'dept', 'program']].notnull().all(axis=1) |
        merged_df[['fund', 'source', 'dept', 'project_ID']].notnull().all(axis=1)
    )
    noCFS_df = merged_df[~valid_mask].copy()
    
    # Drop error rows from CFS_df by removing noCFS rows
    CFS_df = merged_df[valid_mask].copy()
    
    # Keep only the required columns for CFS_df
    CFS_df = CFS_df[['user', 'PI', 'CFS', 'start_time', 'end_time']]
    
    return merged_df, CFS_df, noCFS_df, alcon_df, nmr_core_df

def excel_col_letter(n: int) -> str:
    """Convert a 0-indexed column number to an Excel column letter."""
    # Converts 0-indexed column number to Excel column letter
    letter = ""
    while n >= 0:
        letter = chr(n % 26 + ord('A')) + letter
        n = n // 26 - 1
    return letter

def save_outputs(log_file: Path,
                 output_dir: Path,
                 log: pd.DataFrame,
                 completed_runs: pd.DataFrame,
                 failed_runs: pd.DataFrame,
                 CFS_df: pd.DataFrame,
                 noCFS_df: pd.DataFrame,
                 alcon_df: pd.DataFrame,
                 nmr_core_df: pd.DataFrame,
                 ) -> None:
    """Save the processed data to CSV and Excel files."""
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build output file names using input log file name
    base_name = Path(log_file).stem
    log_csv = output_dir / f"{base_name}_log.csv"
    cfs_csv = output_dir / f"{base_name}_CFS.csv"
    nocfs_csv = output_dir / f"{base_name}_noCFS.csv"
    alcon_csv = output_dir / f"{base_name}_Alcon.csv"
    nmr_core_csv = output_dir / f"{base_name}_NMR_Core.csv"
    excel_file = output_dir / f"{base_name}_processed.xlsx"
    
    # Save CSV files
    log.to_csv(str(log_csv), index=False)
    alcon_df.to_csv(str(alcon_csv), index=False)
    nmr_core_df.to_csv(str(nmr_core_csv), index=False)
    CFS_df.to_csv(str(cfs_csv), index=False)
    noCFS_df.to_csv(str(nocfs_csv), index=False)
    
    # Mapping of sheet names to DataFrames for dynamic range computation
    sheets_data = {
        'Log': log,
        'Successful_Runs': completed_runs,
        'Failed_Runs': failed_runs,
        'Successful_CFS': CFS_df,
        'Failed_CFS': noCFS_df,
        'Alcon': alcon_df,
        'NMR_Core': nmr_core_df
    }
    
    # Save Excel file with formatting
    with pd.ExcelWriter(str(excel_file), engine='xlsxwriter') as writer:
        for sheet_name, df in sheets_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
        workbook = writer.book
        red_font = workbook.add_format({'font_color': 'red'})
        
        # Apply conditional formatting using computed range for each sheet
        for sheet_name, df in sheets_data.items():
            rows = df.shape[0] + 1   # include header row
            cols = df.shape[1]
            last_col_letter = excel_col_letter(cols - 1)
            cell_range = f"A1:{last_col_letter}{rows}"
            worksheet = writer.sheets[sheet_name]
            worksheet.conditional_format(cell_range, {
                'type': 'text',
                'criteria': 'containing',
                'value': '<NA>',
                'format': red_font
            })

def parseArgs():
    """Parse command-line arguments for the script."""
    prs = argparse.ArgumentParser(
        description="Parse Bruker log files to generate billable time and CFS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    prs.add_argument(
        "log_file",
        type=str,
        help="Path to the input Bruker log file."
    )
    prs.add_argument(
        "pif_file",
        type=str,
        help="Path to the people_in_facility CSV file."
    )
    prs.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: directory of input file)"
    )
    prs.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {VERSION}"
    )
    return prs.parse_args()

def main():
    args = parseArgs()
    
    try:
        # Get absolute paths for input files based on current working directory
        log_file = Path(args.log_file).absolute()
        pif_file = Path(args.pif_file).absolute()
        
        # Handle output directory
        if args.output:
            output_dir = Path(args.output).absolute()
        else:
            output_dir = log_file.parent
        
        print(f"Current working directory: {os.getcwd()}\n")
        print(f"Processing log file: {log_file}")
        print(f"Using PIF database: {pif_file}\n")
        
        # Verify files exist before processing
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        if not pif_file.exists():
            raise FileNotFoundError(f"PIF file not found: {pif_file}")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read the files directly from their actual locations
        log, completed_runs, failed_runs = parse_log_file(log_file)
        pif = pd.read_excel(pif_file, dtype='string')
        
        merged_df, CFS_df, noCFS_df, alcon_df, nmr_core_df = process_ilabs_logs(completed_runs, pif)
        
        save_outputs(log_file, output_dir, log, completed_runs, failed_runs, CFS_df, noCFS_df, alcon_df, nmr_core_df)
        
        print(f"\nSuccess! Output saved to: {output_dir}")
        print(f"  - {Path(log_file).stem}_CFS.csv: {len(CFS_df)} completed runs with valid CFS")
        print(f"  - {Path(log_file).stem}_noCFS.csv: {len(noCFS_df)} runs without valid CFS")
        print(f"  - {Path(log_file).stem}_Alcon.csv: {len(alcon_df)} Alcon runs")
        print(f"  - {Path(log_file).stem}_NMR_Core.csv: {len(nmr_core_df)} NMR Core runs")
        print(f"  - {Path(log_file).stem}_processed.xlsx: Full report with all data")
        
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()