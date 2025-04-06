import pandas as pd
import argparse
from pathlib import Path
import sys
import os
from difflib import get_close_matches
import matplotlib
import calplot
import base64
matplotlib.use('Agg')  # Set the backend to Agg (non-interactive)
import matplotlib.pyplot as plt

# Version information
VERSION = "1.1.0"

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
        matches = get_close_matches(str(value), map(str, choices), n=1, cutoff=cutoff)
        return matches[0] if matches else value

    pif_adjusted = pif.copy()
    # Define the required and optional columns
    required_columns = ['grant', 'onyen', 'start_time', 'end_time']
    optional_columns = ['experiment', 'solvent', 'filename', 'filesize', 'advisor', 'start_timestamp', 'end_timestamp']
    
    # Ensure required columns are present
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {', '.join(missing_required)}")
    
    # Filter columns that exist in the DataFrame
    column_filter = required_columns + [col for col in optional_columns if col in df.columns]
    df_adjusted = df[column_filter].copy()

    valid_onyen = pif_adjusted['onyen'].unique()
    valid_grant = pif_adjusted['grant'].unique()

    df_adjusted['onyen'] = df_adjusted['onyen'].apply(lambda x: fuzzy_match(x, valid_onyen))
    df_adjusted['grant'] = df_adjusted['grant'].apply(lambda x: fuzzy_match(x, valid_grant))

    merged_df = df_adjusted.merge(pif_adjusted, on=['onyen', 'grant'], how='left')
    
    # Save merged DataFrame for debugging
    merged_df.to_csv('merged_debug.csv', index=False)
    
    merged_df['CFS'] = merged_df.apply(create_cfs, axis=1)
    
    # New: Do not segregate based on advisor.
    # Determine valid rows using either [fund, source, dept, program] or [fund, source, dept, project_ID]
    valid_mask = (
        merged_df[['fund', 'source', 'dept', 'program']].notnull().all(axis=1) |
        merged_df[['fund', 'source', 'dept', 'project_ID']].notnull().all(axis=1)
    )
    noCFS_df = merged_df[~valid_mask].copy()
    CFS_df = merged_df[valid_mask].copy()
    
    # Keep only required columns and include affiliation from pif (assumed to be in merged_df)
    CFS_df = CFS_df[['user', 'PI', 'CFS', 'start_time', 'end_time', 'affiliation']]
    
    return CFS_df, noCFS_df

def excel_col_letter(n: int) -> str:
    letter = ""
    while n >= 0:
        letter = chr(n % 26 + ord('A')) + letter
        n = n // 26 - 1
    return letter

def prompt_overwrite(file_path: Path) -> bool:
    ans = input(f"File {file_path} already exists. Overwrite? (y/N): ")
    return ans.strip().lower() == "y"

def save_outputs(log_file: Path,
                 output_dir: Path,
                 completed_runs: pd.DataFrame,
                 CFS_df: pd.DataFrame,
                 noCFS_df: pd.DataFrame,
                 overwrite: bool
                 ) -> None:
    # ...existing code to create output_dir...
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(log_file).stem

    # Create subdirectories for CFS and noCFS outputs
    cfs_dir = output_dir / "CFS"
    nocfs_dir = output_dir / "noCFS"
    cfs_dir.mkdir(parents=True, exist_ok=True)
    nocfs_dir.mkdir(parents=True, exist_ok=True)

    # Save overall noCFS file in its subdirectory
    nocfs_csv = nocfs_dir / f"{base_name}_ALL_noCFS.csv"
    if nocfs_csv.exists() and not overwrite and not prompt_overwrite(nocfs_csv):
        sys.exit(f"Overwrite not confirmed for {nocfs_csv}. Exiting.")
    noCFS_df.to_csv(str(nocfs_csv), index=False)
    
    # Group outputs by affiliation (case-insensitive)
    CFS_df['affiliation'] = CFS_df['affiliation'].str.upper().fillna("UNKNOWN")
    noCFS_df['affiliation'] = noCFS_df['affiliation'].str.upper().fillna("UNKNOWN")
    cfs_groups = dict(tuple(CFS_df.groupby('affiliation')))
    nocfs_groups = dict(tuple(noCFS_df.groupby('affiliation')))
    
    # Save one CSV for each affiliation group into respective subdirectories
    for affil, group in cfs_groups.items():
        affil_csv = cfs_dir / f"{base_name}_{affil}_CFS.csv"
        if affil_csv.exists() and not overwrite and not prompt_overwrite(affil_csv):
            sys.exit(f"Overwrite not confirmed for {affil_csv}. Exiting.")
        group.drop(columns=['affiliation'], inplace=True)
        group.to_csv(str(affil_csv), index=False)
        
    for affil, group in nocfs_groups.items():
        affil_csv = nocfs_dir / f"{base_name}_{affil}_noCFS.csv"
        if affil_csv.exists() and not overwrite and not prompt_overwrite(affil_csv):
            sys.exit(f"Overwrite not confirmed for {affil_csv}. Exiting.")
        group.to_csv(str(affil_csv), index=False)
    
    # Prepare Excel sheets: global completed experiments, global failed experiments,
    # and per-affiliation sheets for both successful (CFS) and failed (noCFS) outputs.
    excel_file = output_dir / f"{base_name}_processed.xlsx"
    if excel_file.exists() and not overwrite and not prompt_overwrite(excel_file):
        sys.exit(f"Overwrite not confirmed for {excel_file}. Exiting.")
    sheets_data = {
        'completed_experiments_log': completed_runs,
        'CFS_all': CFS_df
    }
    for affil, group in cfs_groups.items():
        sheets_data[f"CFS_{affil}"] = group
    
    sheets_data['all_noCFS_all'] = noCFS_df
    for affil, group in nocfs_groups.items():
        sheets_data[f"noCFS_{affil}"] = group
    
    with pd.ExcelWriter(str(excel_file), engine='xlsxwriter') as writer:
        for sheet_name, df in sheets_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        workbook = writer.book
        red_font = workbook.add_format({'font_color': 'red'})
        
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
            

def distribute_usage_by_hour(df: pd.DataFrame) -> dict:
    """
    Distribute each experiment’s usage time across hours.
    Returns a dict mapping hour (0–23) to average usage (percentage).
    """
    # Drop rows with missing timestamps
    df = df.dropna(subset=['start_timestamp', 'end_timestamp'])
    usage = {h: 0 for h in range(24)}
    # Determine the total period (in days) across all experiments
    start_date = df['start_timestamp'].min().date()
    end_date = df['end_timestamp'].max().date()
    total_days = (end_date - start_date).days + 1
    for _, row in df.iterrows():
        current = row['start_timestamp']
        end = row['end_timestamp']
        while current < end:
            next_hour = (current + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            if next_hour > end:
                next_hour = end
            usage[current.hour] += (next_hour - current).total_seconds() / 3600
            current = next_hour
    # For each hour, the maximum possible usage is 1 hour per day.
    hourly_util = {h: min((usage[h] / total_days) * 100, 100) for h in range(24)}
    return hourly_util


def distribute_usage_by_day(df: pd.DataFrame) -> dict:
    """
    Apportion experiment usage to each day.
    Returns a dict mapping each day (as a date) to utilization percentage.
    """
    # Drop rows with missing timestamps
    df = df.dropna(subset=['start_timestamp', 'end_timestamp'])
    daily_usage = {}
    for _, row in df.iterrows():
        start = row['start_timestamp']
        end = row['end_timestamp']
        current_day = start.date()
        while current_day <= end.date():
            day_start = pd.Timestamp(current_day)
            day_end = day_start + pd.Timedelta(days=1)
            # Overlap between experiment and the day
            period_start = max(start, day_start)
            period_end = min(end, day_end)
            used = max((period_end - period_start).total_seconds() / 3600, 0)
            daily_usage[current_day] = daily_usage.get(current_day, 0) + used
            current_day = (day_start + pd.Timedelta(days=1)).date()
    # Each day has 24 hours available; cap utilization at 100%
    daily_util = {day: min((hours / 24) * 100, 100) for day, hours in daily_usage.items()}
    return daily_util

def plot_calendar_heatmap(daily_util: dict) -> None:
    """
    Generate a calendar heatmap of daily utilization using calplot.
    Utilization is shown with the inferno_r colormap.
    """
    # Convert dict to pandas Series with a datetime index.
    util_series = pd.Series(daily_util)
    util_series.index = pd.to_datetime(util_series.index)
    start_date = util_series.index.min().strftime("%m/%d/%Y")
    end_date = util_series.index.max().strftime("%m/%d/%Y")
    fig, _ = calplot.calplot(
                util_series,
                cmap="inferno_r",
                suptitle=f"Daily Utilization Heatmap ({start_date} to {end_date})",
                vmin=0, vmax=100
             )
    
    # Set colorbar label manually
    cbar = fig.axes[-1]  # Access the colorbar axis
    cbar.set_ylabel("Utilization (%)")
    
    
def plot_hourly_utilization_bar(hourly_util: dict, daily_util: dict) -> None:
    """
    Generate a bar plot with average hourly utilization for each hour of a typical day.
    Y-axis is fixed to 0–100%.
    """
    hours = list(hourly_util.keys())
    values = [hourly_util[h] for h in hours]
    start_date = pd.to_datetime(min(daily_util.keys())).strftime("%m/%d/%Y")
    end_date = pd.to_datetime(max(daily_util.keys())).strftime("%m/%d/%Y")
    plt.figure(figsize=(10,6))
    bars = plt.bar(hours, values, color='#7BAFD4')  # Carolina blue
    plt.ylim(0, 100)
    plt.xlabel("Hour of Day")
    plt.ylabel("Utilization (%)")
    plt.title(f"Average Hourly Utilization ({start_date} to {end_date})")
    # Add labels inside each bar with average hours of usage
    for bar, value in zip(bars, values):
        avg_min = value * 0.6  # Convert percentage to minutes (60 minutes in an hour)
        plt.text(bar.get_x() + bar.get_width() / 2, value / 2, f"{avg_min:.0f}\nmin",
                 ha='center', va='center', fontsize=8, color='white')


def plot_average_daily_utilization_bar(daily_util: dict) -> None:
    """
    Generate a bar plot with average daily utilization for each weekday.
    Y-axis is fixed to 0–100%.
    Each bar is labeled with the average hours of usage.
    """
    util_series = pd.Series(daily_util)
    # Convert date index to weekday name (0=Monday, 6=Sunday)
    util_series.index = pd.to_datetime(util_series.index)
    start_date = util_series.index.min().strftime("%m/%d/%Y")
    end_date = util_series.index.max().strftime("%m/%d/%Y")
    weekday_means = util_series.groupby(util_series.index.weekday).mean()
    # Map weekday numbers to names
    weekday_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    weekday_means.index = weekday_means.index.map(weekday_names)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(weekday_means.index, weekday_means.values, color='#7BAFD4')  # Carolina blue
    plt.ylim(0, 100)
    plt.xlabel("Day of Week")
    plt.ylabel("Average Utilization (%)")
    plt.title(f"Average Daily Utilization by Weekday ({start_date} to {end_date})")
    
    # Add labels inside each bar with average hours of usage
    for bar, value in zip(bars, weekday_means.values):
        avg_hours = value * 0.24  # Convert percentage to hours (24 hours in a day)
        plt.text(bar.get_x() + bar.get_width() / 2, value / 2, f"{avg_hours:.1f}\nhrs",
                 ha='center', va='center', fontsize=8, color='white')


def plot_actual_daily_utilization_bar(daily_util: dict) -> None:
    """
    Generate a bar plot with the actual daily utilization for each day
    in the reported time period.
    Y-axis is fixed to 0–100%.
    Each bar is labeled with the number of hours of daily usage.
    """
    dates = sorted(daily_util.keys())
    values = [daily_util[d] for d in dates]
    start_date = pd.to_datetime(dates[0]).strftime("%m/%d/%Y")
    end_date = pd.to_datetime(dates[-1]).strftime("%m/%d/%Y")
    plt.figure(figsize=(12,6))
    bars = plt.bar(dates, values, color='#7BAFD4')  # Carolina blue
    plt.ylim(0, 100)
    plt.xlabel("Date")
    plt.ylabel("Utilization (%)")
    plt.title(f"Actual Daily Utilization ({start_date} to {end_date})")
    plt.xticks(rotation=45)
    
    # Add labels inside each bar
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height / 2, f"{value * 0.24:.1f}\nhrs", 
                 ha='center', va='center', fontsize=6, color='white')


def plot_lab_usage_bar(df: pd.DataFrame) -> None:
    """
    Generate a usage bar plot that displays total usage hours by each lab
    (determined by the 'advisor' column) and colors each bar by the contributions
    from each user (determined by the 'onyen' column). No legend is displayed.
    The labs are sorted by total usage, with the most usage on the left.
    """
    # Calculate usage hours per experiment
    df = df.copy()
    df['usage_hours'] = (df['end_timestamp'] - df['start_timestamp']).dt.total_seconds() / 3600
    # Group by lab and onyen, summing the usage hours
    lab_usage = df.groupby(['advisor', 'onyen'])['usage_hours'].sum().unstack(fill_value=0)
    # Sort labs by total usage (sum of all users' usage)
    lab_usage = lab_usage.loc[lab_usage.sum(axis=1).sort_values(ascending=False).index]
    start_date = df['start_timestamp'].min().strftime("%m/%d/%Y")
    end_date = df['end_timestamp'].max().strftime("%m/%d/%Y")
    plt.figure(figsize=(12,8))
    lab_usage.plot(kind='bar', stacked=True, colormap='tab20', legend=False)
    plt.xlabel("Lab (Advisor)")
    plt.ylabel("Usage Hours")
    plt.title(f"Usage by Lab and User ({start_date} to {end_date})")
    plt.xticks(rotation=45, ha='right')


def plot_hourly_heatmap(df: pd.DataFrame) -> None:
    """
    Generate a heatmap plot where rows represent days and columns represent hours,
    with each cell showing the utilization percentage (max 100% per hour).
    Days without experiments are included with zero utilization.
    """
    # Drop rows with missing timestamps
    df = df.dropna(subset=['start_timestamp', 'end_timestamp'])
    usage = {}
    for _, row in df.iterrows():
        current = row['start_timestamp']
        end_time = row['end_timestamp']
        while current < end_time:
            day = current.date()
            hour = current.hour
            next_hour = (current + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            if next_hour > end_time:
                next_hour = end_time
            duration = (next_hour - current).total_seconds()
            if day not in usage:
                usage[day] = {h: 0 for h in range(24)}
            usage[day][hour] += duration
            current = next_hour

    # Ensure all days between the start and end dates are included
    if not df.empty:
        start_date = df['start_timestamp'].min().date()
        end_date = df['end_timestamp'].max().date()
        all_dates = pd.date_range(start=start_date, end=end_date).date
        for date in all_dates:
            if date not in usage:
                usage[date] = {h: 0 for h in range(24)}

    # Build DataFrame from usage dict and convert seconds to percentage (1 hour = 3600 sec)
    heatmap_data = pd.DataFrame(usage).T.fillna(0).sort_index()
    heatmap_perc = heatmap_data.apply(lambda col: col.map(lambda x: min(x / 3600 * 100, 100)))

    # Plot heatmap using imshow
    plt.figure(figsize=(12, 6))
    plt.imshow(heatmap_perc, aspect='auto', cmap='inferno_r', vmin=0, vmax=100)
    plt.colorbar(label="Utilization (%)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Date")
    plt.title("Hourly Utilization Heatmap")
    plt.xticks(ticks=range(24), labels=range(24))
    plt.yticks(ticks=range(len(heatmap_perc.index)), labels=[str(date) for date in heatmap_perc.index])


def plot_failure_hourly_heatmap(df: pd.DataFrame) -> None:
    """
    Generate a heatmap plot where rows represent days and columns represent hours,
    showing the count of failures per hour. Includes all dates between the start
    and end date, even if there are no failures.
    """
    df = df.dropna(subset=['start_timestamp'])
    df['date'] = df['start_timestamp'].dt.date
    df['hour'] = df['start_timestamp'].dt.hour

    # Determine the full range of dates
    if not df.empty:
        start_date = df['date'].min()
        end_date = df['date'].max()
        all_dates = pd.date_range(start=start_date, end=end_date).date
    else:
        all_dates = []

    # Create a pivot table with all dates and hours
    pivot_table = df.groupby(['date', 'hour']).size().unstack(fill_value=0)
    pivot_table = pivot_table.reindex(index=all_dates, fill_value=0).sort_index()

    # Plot the heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(pivot_table, aspect='auto', cmap='Reds')
    plt.colorbar(label="Number of Failures")
    plt.xlabel("Hour of Day")
    plt.ylabel("Date")
    plt.title("Hourly Failures Heatmap")
    plt.xticks(ticks=range(24), labels=range(24))
    plt.yticks(ticks=range(len(pivot_table.index)), labels=[str(d) for d in pivot_table.index])


def plot_top_users_bar(df: pd.DataFrame) -> None:
    """
    Generate a bar plot showing the top 20 users (determined by 'onyen')
    by total experiment time used.
    """
    # Calculate total experiment time for each user
    df['usage_hours'] = (df['end_timestamp'] - df['start_timestamp']).dt.total_seconds() / 3600
    top_users = df.groupby('onyen')['usage_hours'].sum().nlargest(20)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(top_users.index, top_users.values, color='#7BAFD4')
    plt.xlabel("User (onyen)")
    plt.ylabel("Total Experiment Time (hours)")
    plt.title("Top 20 Users by Total Experiment Time")
    plt.xticks(rotation=45, ha='right')
    
    # Add white labels inside each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{height:.1f}\nhrs', 
                 ha='center', va='center', fontsize=8, color='white')


def plot_nucleus_type_distribution(df: pd.DataFrame) -> None:
    """
    Generate two pie charts displaying the distribution of experiment types (from the 'experiment' column)
    during the Day queue (8:00-19:00) and Night queue (19:00-8:00).
    """
    # Create a copy to avoid side-effects and drop rows missing start_timestamp or experiment info.
    data = df.dropna(subset=['start_timestamp', 'nucleus']).copy()
    # Extract start hour from the timestamp
    data['start_hour'] = data['start_timestamp'].dt.hour
    # Define day and night queues based on the start hour
    day_mask = (data['start_hour'] >= 8) & (data['start_hour'] < 19)
    night_mask = ~day_mask
    # Group experiment type counts for day and night
    day_counts = data.loc[day_mask, 'nucleus'].value_counts()
    night_counts = data.loc[night_mask, 'nucleus'].value_counts()
    
    # Create subplots for the two pie charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    ax1.pie(day_counts, labels=day_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title("Nucleus Types - Day Queue (8:00-19:00)")
    ax2.pie(night_counts, labels=night_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title("Nucleus Types - Night Queue (19:00-8:00)")


def generate_summary_report(report_dir: Path) -> None:
    """
    Generate a self-contained HTML summary report that combines all generated plots with basic descriptions.
    The plots are embedded as base64-encoded images, making the report portable.
    """

    # Mapping of plot filenames to descriptions
    report_plots = {
        "hourly_utilization_heatmap.png": "Heatmap showing hourly utilization percentages per day.",
        "average_hourly_utilization_bar.png": "Bar plot depicting average hourly utilization for a typical day.",
        "daily_utilization_heatmap.png": "Calendar heatmap of daily utilization across the time period.",
        "daily_utilization_bar.png": "Bar plot showing actual daily utilization with hours labeled.",
        "average_daily_utilization_bar.png": "Bar plot of average daily utilization by weekday.",
        "lab_usage_bar.png": "Stacked bar plot of usage hours by lab (advisor) and user (onyen).",
        "failure_hourly_heatmap.png": "Heatmap displaying the number of failures per hour.",
        "top_users_bar.png": "Bar plot of the top 20 users by hours.",
        "experiment_type_distribution.png": "Pie charts showing the distribution of nucleus types during the day and night queues."
    }

    html = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Summary Report</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 20px; }}
      .plot {{ margin-bottom: 40px; }}
      .plot img {{ max-width: 100%; height: auto; }}
    </style>
  </head>
  <body>
    <h1>Summary Report</h1>
    <p>This report combines all generated plots from the parse_bruker_logs.py script along with brief descriptions.</p>
"""
    for filename, desc in report_plots.items():
        plot_path = report_dir / filename
        if plot_path.exists():
            # Encode the image as base64
            with open(plot_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
            # Embed the image in the HTML
            html += f"""
    <div class="plot">
      <h2>{filename}</h2>
      <p>{desc}</p>
      <img src="data:image/png;base64,{encoded_image}" alt="{desc}">
    </div>
"""
    html += """
  </body>
</html>
"""
    report_file = report_dir / "summary_report.html"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html)


def parseArgs():
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
        "-o", "--output_dir",
        type=str,
        default="output",
        help="Output directory (default: directory of input file)"
    )
    prs.add_argument(
        "-r", "--report",
        action="store_true",
        default=False,
        help="Generates usage report based on the log file."
    )
    prs.add_argument(
        "-w", "--overwrite",
        action="store_true",
        default=False,
        help="Allow overwriting existing files/directories without prompt."
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
        log_file = Path(args.log_file).absolute()
        pif_file = Path(args.pif_file).absolute()
        output_dir = Path(args.output_dir).absolute() if args.output_dir else log_file.parent
        
        print(f"Current working directory: {os.getcwd()}\n")
        print(f"Processing log file: {log_file}")
        print(f"Using PIF database: {pif_file}\n")
        
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
        if not pif_file.exists():
            raise FileNotFoundError(f"PIF file not found: {pif_file}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        log = pd.read_csv(log_file, dtype='string')
        pif = pd.read_excel(pif_file, dtype='string')
        
        # Updated process_ilabs_logs now returns only CFS_df and noCFS_df after grouping by affiliation.
        CFS_df, noCFS_df = process_ilabs_logs(log, pif)
        save_outputs(log_file, output_dir, log, CFS_df, noCFS_df, args.overwrite)
        
        print(f"\nSuccess! Output saved to: {output_dir}")
        print(f"  - {Path(log_file).stem}_CFS.csv: {len(CFS_df)} completed runs with valid CFS split by affiliation")
        
        for affil, group in CFS_df.groupby(CFS_df['affiliation'].str.upper().fillna("UNKNOWN")):
            print(f"\t  - {Path(log_file).stem}_{affil}.csv: {len(group)} runs for affiliation '{affil}'")
            
        print()
        print(f"  - {Path(log_file).stem}_noCFS.csv: {len(noCFS_df)} runs without valid CFS")
        
        for affil, group in noCFS_df.groupby(noCFS_df['affiliation'].str.upper().fillna("UNKNOWN")):
            print(f"\t  - {Path(log_file).stem}_{affil}.csv: {len(group)} runs for affiliation '{affil}'")
            
        print()
        print(f"  - {Path(log_file).stem}_processed.xlsx: Full report with all sheets\n")
        
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    if args.report:
        try:
            # Convert timestamps to datetime
            completed_runs['start_timestamp'] = pd.to_datetime(completed_runs['start_timestamp'])
            completed_runs['end_timestamp'] = pd.to_datetime(completed_runs['end_timestamp'])
            # Compute utilization distributions
            hourly_util = distribute_usage_by_hour(completed_runs)
            daily_util = distribute_usage_by_day(completed_runs)

            # Create reports directory
            reports_dir = output_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Save utilization data
            pd.DataFrame({'Date': daily_util.keys(), 'Utilization': daily_util.values()}).to_csv(
                reports_dir / "daily_util.csv", index=False)
            pd.DataFrame({'Hour': hourly_util.keys(), 'Utilization': hourly_util.values()}).to_csv(
                reports_dir / "hourly_util.csv", index=False)

            # Generate and save plots
            plots = [
                (plot_hourly_heatmap, (completed_runs,), "hourly_utilization_heatmap.png"),
                (plot_hourly_utilization_bar, (hourly_util, daily_util), "average_hourly_utilization_bar.png"),
                (plot_calendar_heatmap, (daily_util,), "daily_utilization_heatmap.png"),
                (plot_actual_daily_utilization_bar, (daily_util,), "daily_utilization_bar.png"),
                (plot_average_daily_utilization_bar, (daily_util,), "average_daily_utilization_bar.png"),
                (plot_lab_usage_bar, (completed_runs,), "lab_usage_bar.png"),
                (plot_failure_hourly_heatmap, (failed_runs,), "failure_hourly_heatmap.png"),
                (plot_top_users_bar, (log,), "top_users_bar.png"),
                (plot_nucleus_type_distribution, (log,), "experiment_type_distribution.png"),
            ]

            for plot_func, data_args, filename in plots:
                plt.figure()
                plot_func(*data_args)
                plt.savefig(reports_dir / filename, bbox_inches="tight", pad_inches=0.5)
                plt.close()
            
            # NEW: Generate summary report combining all plots
            generate_summary_report(reports_dir)

        except Exception as e:
            print(f"\nError generating reports: {str(e)}", file=sys.stderr)
            sys.exit(1)

        print(f"Reports and plots saved successfully in {reports_dir}!")

if __name__ == "__main__":
    main()