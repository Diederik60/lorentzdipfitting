import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
import os

class ODMRDataCombiner:
    def __init__(self, processed_dir, raw_dir):
        """
        Initialize with directories containing processed and raw data
        
        Args:
            processed_dir (str): Directory containing lorentzian parameter files (40,40,5)
            raw_dir (str): Directory containing raw ODMR data files (40,40,100)
        """
        self.processed_dir = Path(processed_dir)
        self.raw_dir = Path(raw_dir)

    def flatten_directory(self, directory):
        """Move all .npy files from subdirectories to the main directory"""
        moved_files = 0
        removed_dirs = 0
        
        print(f"Starting directory flattening for: {directory}")
        
        for npy_file in directory.rglob('*.npy'):
            if npy_file.parent == directory:
                continue
                
            new_path = directory / npy_file.name
            if new_path.exists():
                base = new_path.stem
                extension = new_path.suffix
                counter = 1
                while new_path.exists():
                    new_path = directory / f"{base}_{counter}{extension}"
                    counter += 1
            
            print(f"Moving: {npy_file.name}")
            shutil.move(str(npy_file), str(new_path))
            moved_files += 1
        
        for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
            for dirname in dirnames:
                full_path = Path(dirpath) / dirname
                try:
                    full_path.rmdir()
                    removed_dirs += 1
                    print(f"Removed empty directory: {full_path}")
                except OSError:
                    pass
        
        print(f"Directory flattening complete!")
        print(f"Files moved: {moved_files}")
        print(f"Empty directories removed: {removed_dirs}")

    def get_experiment_number(self, filename):
        """Extract experiment number from filename"""
        import re
        if 'lorentzian_params_' in str(filename):
            pattern = r'lorentzian_params_(\d+)\.npy$'
        else:
            pattern = r'2D_ODMR_scan_(\d+)\.npy$'
        match = re.search(pattern, str(filename))
        return int(match.group(1)) if match else None

    def find_matching_files(self):
        """Find matching pairs of processed and raw data files"""
        processed_files = list(self.processed_dir.glob('lorentzian_params_*.npy'))
        raw_files = list(self.raw_dir.glob('2D_ODMR_scan_*.npy'))
        
        matching_pairs = []
        
        for proc_file in processed_files:
            proc_num = self.get_experiment_number(proc_file)
            if proc_num is None:
                continue
                
            raw_file = self.raw_dir / f'2D_ODMR_scan_{proc_num}.npy'
            if raw_file.exists():
                matching_pairs.append((proc_file, raw_file))
            else:
                print(f"No matching raw data found for experiment {proc_num}")
        
        return matching_pairs

    def find_matching_files(self):
        """Find matching pairs of processed and raw data files, searching recursively for raw data"""
        processed_files = list(self.processed_dir.glob('lorentzian_params_*.npy'))
        
        # Dictionary to store raw files by experiment number
        raw_files_dict = {}
        
        # Recursively find all ODMR scan files in raw directory and its subdirectories
        print("Searching for raw data files...")
        for raw_file in tqdm(list(self.raw_dir.rglob('*.npy'))):
            # Skip if not an ODMR scan file
            if not raw_file.name.startswith('2D_ODMR_scan_'):
                continue
            
            # Get experiment number
            exp_num = self.get_experiment_number(raw_file)
            if exp_num is not None:
                raw_files_dict[exp_num] = raw_file
        
        print(f"Found {len(raw_files_dict)} raw data files")
        
        # Match processed files with raw files
        matching_pairs = []
        for proc_file in processed_files:
            proc_num = self.get_experiment_number(proc_file)
            if proc_num is None:
                continue
            
            if proc_num in raw_files_dict:
                matching_pairs.append((proc_file, raw_files_dict[proc_num]))
            else:
                print(f"No matching raw data found for experiment {proc_num}")
        
        print(f"\nFound {len(matching_pairs)} matching pairs out of:")
        print(f"  {len(processed_files)} processed files")
        print(f"  {len(raw_files_dict)} raw files")
        
        return matching_pairs

    def combine_data(self):
        """Combine raw and processed data into a single DataFrame"""
        matching_pairs = self.find_matching_files()
        if not matching_pairs:
            print("No matching file pairs found!")
            return None

        print(f"Found {len(matching_pairs)} matching file pairs")

        all_data = []
        total_pixels = 0

        for proc_file, raw_file in tqdm(matching_pairs, desc="Processing file pairs"):
            try:
                exp_num = self.get_experiment_number(proc_file)
                
                # Load data
                proc_params = np.load(proc_file)  # Shape: (40,40,5)
                raw_data = np.load(raw_file)      # Shape: (40,40,100)
                
                if proc_params.shape[:2] != (40, 40) or raw_data.shape[:2] != (40, 40):
                    print(f"Unexpected shapes in experiment {exp_num}:")
                    print(f"Processed shape: {proc_params.shape}")
                    print(f"Raw shape: {raw_data.shape}")
                    continue

                # Process each pixel
                for i in range(40):
                    for j in range(40):
                        # Get data for this pixel
                        pixel_params = proc_params[i, j]  # 5 parameters
                        pixel_raw = raw_data[i, j]        # 100 frequency points
                        
                        # Create entry
                        entry = {
                            'experiment': exp_num,
                            'pixel_x': i,
                            'pixel_y': j,
                            'I0': pixel_params[0],
                            'A': pixel_params[1],
                            'width': pixel_params[2],
                            'f_center': pixel_params[3],
                            'f_delta': pixel_params[4]
                        }
                        
                        # Add raw data features
                        for k, value in enumerate(pixel_raw):
                            entry[f'raw_freq_{k}'] = value
                        
                        all_data.append(entry)
                        total_pixels += 1

            except Exception as e:
                print(f"Error processing experiment {exp_num}: {e}")
                continue

        if not all_data:
            print("No data processed!")
            return None

        df = pd.DataFrame(all_data)
        
        print(f"\nProcessed {total_pixels} pixels from {len(matching_pairs)} experiments")
        print(f"DataFrame shape: {df.shape}")

        return df

def main():
    while True:
        print("\nODMR Data Combiner")
        print("1. Flatten directory")
        print("2. Combine data")
        print("3. Exit")

        choice = input("\nEnter choice (1-3): ")

        if choice == '1':
            directory = input("Enter directory path to flatten: ")
            tools = ODMRDataCombiner(directory, directory)  # Directory argument is reused since we only need one
            tools.flatten_directory(Path(directory))
            
        elif choice == '2':
            processed_dir = input("Enter processed data directory path: ")
            raw_dir = input("Enter raw data directory path: ")
            combiner = ODMRDataCombiner(processed_dir, raw_dir)
            
            print("\nCombining data...")
            df = combiner.combine_data()
            
            if df is not None:
                while True:
                    output_file = input("\nEnter output filename (e.g., 'combined_data.csv' or 'combined_data.pkl'): ")
                    if output_file.endswith('.csv'):
                        df.to_csv(output_file, index=False)
                        break
                    elif output_file.endswith('.pkl'):
                        df.to_pickle(output_file)
                        break
                    else:
                        print("Please use .csv or .pkl extension")
                
                print(f"\nData saved to {output_file}")
                print("\nColumns:", df.columns.tolist())
                print("\nFirst few rows:")
                print(df.head())
                
                # Show basic statistics for parameter columns
                param_cols = ['I0', 'A', 'width', 'f_center', 'f_delta']
                print("\nParameter statistics:")
                print(df[param_cols].describe())
        
        elif choice == '3':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()