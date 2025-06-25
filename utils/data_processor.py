import pandas as pd
import numpy as np
from typing import Tuple, Optional
import gc

class LargeDataProcessor:
    """Handle large CSV files efficiently"""
    
    def __init__(self, max_memory_mb: int = 2000):
        self.max_memory_mb = max_memory_mb
        self.chunk_size = 50000
    
    def estimate_file_size(self, filepath: str) -> float:
        """Estimate memory usage of CSV file in MB"""
        try:
            # Read first 1000 rows to estimate
            sample = pd.read_csv(filepath, nrows=1000)
            memory_per_row = sample.memory_usage(deep=True).sum() / len(sample)
            
            # Count total rows
            total_rows = sum(1 for _ in open(filepath)) - 1  # -1 for header
            
            estimated_mb = (memory_per_row * total_rows) / (1024 * 1024)
            return estimated_mb
        except:
            return 0
    
    def process_large_csv(self, filepath: str, sample_size: int = 200000) -> pd.DataFrame:
        """Process large CSV files with sampling if needed"""
        try:
            estimated_size = self.estimate_file_size(filepath)
            
            if estimated_size > self.max_memory_mb:
                print(f"Very large file detected ({estimated_size:.1f}MB), sampling {sample_size} records...")
                return self.sample_large_file(filepath, sample_size)
            else:
                print(f"Loading file ({estimated_size:.1f}MB)...")
                return pd.read_csv(filepath, low_memory=False)
                
        except Exception as e:
            print(f"Error processing file: {e}")
            # Fallback to chunked reading
            return self.read_in_chunks(filepath, sample_size)
    
    def sample_large_file(self, filepath: str, sample_size: int) -> pd.DataFrame:
        """Sample large files efficiently"""
        try:
            # Count total lines
            total_lines = sum(1 for _ in open(filepath)) - 1
            
            if total_lines <= sample_size:
                return pd.read_csv(filepath, low_memory=False)
            
            # Calculate skip probability
            skip_prob = 1 - (sample_size / total_lines)
            
            # Random sampling
            df = pd.read_csv(filepath, 
                           skiprows=lambda i: i > 0 and np.random.random() < skip_prob,
                           low_memory=False)
            
            print(f"Sampled {len(df)} records from {total_lines} total records")
            return df
            
        except Exception as e:
            print(f"Sampling failed: {e}, using chunk method")
            return self.read_in_chunks(filepath, sample_size)
    
    def read_in_chunks(self, filepath: str, max_rows: int) -> pd.DataFrame:
        """Read file in chunks and combine"""
        chunks = []
        rows_read = 0
        
        try:
            for chunk in pd.read_csv(filepath, chunksize=self.chunk_size, low_memory=False):
                chunks.append(chunk)
                rows_read += len(chunk)
                
                if rows_read >= max_rows:
                    break
                
                # Memory cleanup
                if len(chunks) % 10 == 0:
                    gc.collect()
            
            if chunks:
                result = pd.concat(chunks, ignore_index=True)
                if len(result) > max_rows:
                    result = result.sample(n=max_rows, random_state=42)
                
                print(f"Loaded {len(result)} records using chunked reading")
                return result
            else:
                raise ValueError("No data could be read from file")
                
        except Exception as e:
            print(f"Chunked reading failed: {e}")
            raise