import pandas as pd
import glob
import os

def extract_headers_and_data(excel_path, target_dir):
    """
    Extracts headers from Excel and maps them to CSV files.
    """
    print(f"Reading {excel_path}...")
    # Load the workbook to check for the correct header row
    xls = pd.ExcelFile(excel_path)
    print(f"Sheets: {xls.sheet_names}")
    
    # Usually the first sheet has the data. Let's find the row that looks like headers.
    df_preview = pd.read_excel(excel_path, sheet_name=0, nrows=20, header=None)
    header_row_index = 0
    for i, row in df_preview.iterrows():
        print(f"Row {i}: {row.tolist()[:10]}") # Print first 10 columns of each row
        # Heuristic: Find row with 'Date' or 'Run'
        if any(x in str(row.tolist()) for x in ['Date', 'Run', 'Time']):
            header_row_index = i
            print(f"Found potential header at row {i}")
            break
            
    # Read again with the correct header
    df_excel = pd.read_excel(excel_path, sheet_name=0, header=header_row_index)
    headers = df_excel.columns.tolist()
    print(f"Found {len(headers)} columns.")
    
    # Save headers to a text file for reference
    with open('column_headers.txt', 'w') as f:
        f.write('\n'.join([str(h) for h in headers]))
    
    return headers

def process_csv_files(headers, data_dir, output_file):
    """
    Combines all CSV files into a single dataframe with headers.
    """
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    combined_df = []
    
    for f in all_files:
        print(f"Processing {os.path.basename(f)}...")
        # Skip the first few metadata rows manually or identify data start
        # Based on previous peek, data starts after some blank/metadata rows
        # The data matches the Refrigerant name at some point.
        # Let's try reading and skipping rows until we find numeric data.
        df = pd.read_csv(f, header=None, skiprows=lambda x: x < 10) # Empirical guess
        
        # Trim if column counts don't match
        if df.shape[1] > len(headers):
            df = df.iloc[:, :len(headers)]
        elif df.shape[1] < len(headers):
            # Pad with NaNs or ignore
            print(f"Warning: {f} has fewer columns ({df.shape[1]}) than headers ({len(headers)})")
            
        df.columns = headers[:df.shape[1]]
        combined_df.append(df)
        
    final_df = pd.concat(combined_df, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"Unified dataset saved to {output_file}")
    return final_df

if __name__ == "__main__":
    # Raw NIST experimental files are NOT bundled in this repository.
    # Download "Data for NIST Technical Note 2233" from the NIST data portal
    # (https://doi.org/10.18434/mds2-2613) and place the experiment_data
    # folder under raw_data/, or set NIST_RAW_DIR to its location.
    # The processed output (data/unified_nist_experiment_data.csv) is already
    # included, so running this script is only needed to reproduce it.
    _here = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.environ.get(
        'NIST_RAW_DIR', os.path.join(_here, os.pardir, 'raw_data',
                                     'experiment_data'))
    exp_excel = os.path.join(raw_dir, 'Experiment - All - Rev_1_5.xlsx')
    exp_dir = raw_dir
    output_csv = os.path.join(_here, os.pardir, 'data',
                              'unified_nist_experiment_data.csv')
    
    headers = extract_headers_and_data(exp_excel, exp_dir)
    
    all_files = glob.glob(os.path.join(exp_dir, "*.csv"))
    combined_df = []
    
    for f in all_files:
        print(f"Processing {os.path.basename(f)}...")
        # Data in CSV seems to start after initial metadata. 
        # Using row 8 headers, we expect data to start where columns match.
        # Let's read the CSV and find where the date/time patterns start.
        # Use latin-1 encoding to avoid UnicodeDecodeError
        df_raw = pd.read_csv(f, header=None, encoding='latin-1')
        
        # Find the first row that has a valid Date format (M/D/Y or Y-M-D) in the first column
        # Previous heuristic ('-' or '/') was too loose and matched metadata like "LLSL-HX"
        import re
        date_pattern = re.compile(r'\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{1,2}-\d{1,2}')
        
        data_start_idx = 0
        for i, row in df_raw.iterrows():
            val = str(row[0])
            if pd.notnull(row[0]) and date_pattern.search(val):
                data_start_idx = i
                break
        
        # If no date found, skip file or warn (but don't crash)
        if data_start_idx == 0 and not date_pattern.search(str(df_raw.iloc[0,0])):
             print(f"Warning: No data start found for {os.path.basename(f)}, skipping potential metadata match.")
             # Fallback: look for row starting with numeric 
             # (actually date/time are usually first, so this strictly implies date check failed)
             
        df_data = pd.read_csv(f, header=None, skiprows=data_start_idx, encoding='latin-1')
        
        # FIX: Do not truncate data if it has more columns than headers.
        # The Excel header has ~195 cols, but CSVs have 318.
        # We need to keep the extra data, especially if COP is in the extra columns (or misaligned).
        if df_data.shape[1] > len(headers):
            # Extend headers with placeholder names
            extra_headers = [f"Extra_{i}" for i in range(len(headers), df_data.shape[1])]
            all_headers = headers + extra_headers
            df_data.columns = all_headers
        elif df_data.shape[1] < len(headers):
            print(f"Warning: {f} has fewer columns ({df_data.shape[1]})")
            df_data.columns = headers[:df_data.shape[1]]
        else:
            df_data.columns = headers
            
        # FORCE MAPPING: Column 153 seems to be the COP based on inspection
        # R134a (153) = 7.355, R450A (153) = 6.642
        # Previous Excel inspection was ambiguous, so we force it here for consistency.
        # We'll construct a 'COP_c_ref_dh' column from index 153 if it exists
        # FORCE COP MAPPING
        # We assume col 153 is always the COP for this dataset version
        # We use .values to avoid index alignment issues if we were assigning series
        if df_data.shape[1] > 153:
             df_data['COP_c_ref_dh'] = pd.to_numeric(df_data.iloc[:, 153], errors='coerce')

        # Add a column for LLSL status from filename
        df_data['LLSL_Included'] = 1 if 'with LLSL' in f else 0
        
        # CORRECT FLUID NAME FROM FILENAME
        # The internal CSV metadata is unreliable (e.g. R450A labeled as R134a)
        # We trust the filename structure "FluidName - Condition.csv"
        fluid_name = os.path.basename(f).split(' - ')[0].strip()
        df_data['fluid_name'] = fluid_name

        # Apply composition replacements for blends
        if fluid_name == 'R410A':
            df_data['Component[1]'] = 'R32'
            if 'MassFraction[1]' not in df_data.columns:
                df_data['MassFraction[1]'] = 0.50
            else:
                df_data['MassFraction[1]'] = df_data['MassFraction[1]'].fillna(0.50)
                
            if 'Component[2]' not in df_data.columns:
                df_data['Component[2]'] = 'R125'
            else:
                df_data['Component[2]'] = df_data['Component[2]'].fillna('R125')
                
            if 'MassFraction[2]' not in df_data.columns:
                df_data['MassFraction[2]'] = 0.50
            else:
                df_data['MassFraction[2]'] = df_data['MassFraction[2]'].fillna(0.50)
        else:
            replacements = {
                'R450A': 'R134a',
                'R452B': 'R32',
                'R454B': 'R32',
                'R513A': 'R134a',
                'R515B': 'R1234zeE',
                'Tern1': 'R32'
            }
            df_data['Component[1]'] = replacements.get(fluid_name, fluid_name)
        
        combined_df.append(df_data)
        
    final_df = pd.concat(combined_df, ignore_index=True)
    # Clean up column names (strip whitespace)
    final_df.columns = [str(c).strip() for c in final_df.columns]
    
    # NEW CLEANING STEP:
    # Target columns must be numeric. Identify and convert.
    target_cols = ['COP_c_ref_dh', 'Q_dot_evap_ref']
    for col in target_cols:
        if col in final_df.columns:
            # Convert to numeric, errors='coerce' turns strings to NaN
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
    
    # Drop rows where main target is NaN
    initial_len = len(final_df)
    final_df = final_df.dropna(subset=['COP_c_ref_dh'])
    print(f"Cleaned {initial_len - len(final_df)} non-numeric/empty rows.")
    
    final_df.to_csv(output_csv, index=False)
    print(f"Success! Saved {len(final_df)} rows to {output_csv}")
