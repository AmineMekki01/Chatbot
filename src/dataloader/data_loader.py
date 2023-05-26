import pandas as pd

class HealthcareDataLoader:
    """Data Loader class"""
    
    def __init__(self, file_paths):
        self.file_paths = file_paths

    @staticmethod
    def load_data(self):
        
        """Loads dataset from paths"""
        
        # combine all csv files into a single DataFrame
        combined_df = pd.concat([pd.read_csv(f) for f in self.file_paths], ignore_index=True)
        return combined_df