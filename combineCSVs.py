import pandas as pd
import glob
import os
from pathlib import Path

def combine_ball_data_files(directory='.', output_file='combined_ball_data.csv', date_column=None):
    """
    Combine all CSV files containing 'ball_data' in their name, sorted by date.
    
    Args:
        directory: Directory to search for files (default: current directory)
        output_file: Name of the output combined file
        date_column: Column name to sort by (auto-detect if None)
    """
    
    # Find all CSV files with 'ball_data' in the name
    pattern = os.path.join(directory, '*ball_data*.csv')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files matching '*ball_data*.csv' found in {directory}")
        return
    
    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  - {f}")
    
    # Read and combine all files
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            print(f"✓ Read {file} ({len(df)} records)")
            dfs.append(df)
        except Exception as e:
            print(f"✗ Error reading {file}: {e}")
    
    if not dfs:
        print("No valid files to combine")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal records before sorting: {len(combined_df)}")
    
    # Auto-detect date column if not specified
    if date_column is None:
        date_candidates = ['date', 'Date', 'DATE', 'timestamp', 'Timestamp', 'TIMESTAMP', 
                          'created_at', 'created_date', 'match_date', 'game_date']
        for col in date_candidates:
            if col in combined_df.columns:
                date_column = col
                print(f"Auto-detected date column: {date_column}")
                break
    
    # Sort by date if date column exists
    if date_column and date_column in combined_df.columns:
        try:
            combined_df[date_column] = pd.to_datetime(combined_df[date_column])
            combined_df = combined_df.sort_values(by=date_column, ascending=True)
            print(f"Sorted by {date_column}")
        except Exception as e:
            print(f"Warning: Could not sort by {date_column}: {e}")
    else:
        print(f"Warning: Date column '{date_column}' not found. Available columns: {list(combined_df.columns)}")
    
    # Remove duplicates if any (keeping first occurrence)
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(ignore_index=True)
    duplicates_removed = initial_count - len(combined_df)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate record(s)")
    
    # Save combined file
    combined_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved combined file: {output_file}")
    print(f"Final record count: {len(combined_df)}")
    
    return combined_df


if __name__ == "__main__":
    # Combine files in current directory, sorted by 'scheduled' column (ISO 8601 format)
    combine_ball_data_files(directory='.', output_file='combined_ball_data.csv', date_column='scheduled')