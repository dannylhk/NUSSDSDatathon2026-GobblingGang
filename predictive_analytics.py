# dependencies
import pandas as pd
import numpy as np
import re
import json
import pickle
from pathlib import Path
from datetime import date
from typing import Tuple, Dict, Any, Optional


class DataPipeline:
    """
    Production-ready Data Pipeline for Revenue Prediction
    
    This pipeline handles both training and prediction scenarios:
    - Training: Processes data and saves encoding artifacts for later use
    - Prediction: Loads encoding artifacts and applies them to new data
    
    Following industry best practices:
    - Serialization using pickle (Python native, handles complex objects)
    - Metadata tracking for versioning and auditing
    - Atomic file operations with validation
    """

    # Default artifact directory
    ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
    
    def __init__(self, raw_data_path: str, is_training: bool):
        """
        Initialize the DataPipeline.
        
        Args:
            raw_data_path: Path to the raw Excel data file
            is_training: True for training mode, False for prediction mode
        """
        self.df = pd.read_excel(raw_data_path)
        self.is_training = is_training
        self.encoding_maps = {}
        self.global_medians = {}

        # Configuration
        self.flag_cols = ['Manufacturing Status', 'Franchise Status', 'Is Headquarters', 'Is Domestic Ultimate']
        self.range_cols = [
            "No. of PC", "No. of Desktops", "No. of Laptops",
            "No. of Routers", "No. of Servers", "No. of Storage Devices"
        ]
        self.target_cols = ["SIC Code", "Legal Status"]

    def _capitalized_elem(self, elem):
        """Standardize data to uppercase."""
        return str(elem).upper().strip()

    def _get_object_columns(self, df: pd.DataFrame) -> list:
        """Identify all columns with dtype 'object' in the dataframe."""
        return df.select_dtypes(include=['object']).columns.tolist()

    def _remove_whitespace(self, elem):
        """Remove leading/trailing whitespace, preserving NaN values."""
        if pd.isna(elem):
            return elem
        else:
            return str(elem).strip()

    def filter_and_clean_string_columns(self, verbose: bool = True) -> None:
        """
        Identify object-type (string) columns and remove leading/trailing whitespace.
        NOTE: this edits the original df
        """
        object_columns = self._get_object_columns(self.df)

        if verbose:
            print("\n" + "="*70)
            print("STRING COLUMN CLEANING REPORT")
            print("="*70)

        if not object_columns:
            if verbose:
                print("No object-type (string) columns found in the dataframe.")

        if verbose:
            print(f"\nFound {len(object_columns)} object-type (string) columns:")
            for col in object_columns:
                print(f"  - {col}")

        for col in object_columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].apply(self._remove_whitespace)

        if verbose:
            print(f"\n✓ Applied .str.strip() to all {len(object_columns)} string columns")
            print("✓ Whitespace removed successfully")
            print("="*70 + "\n")

    def _parse_range(self, val):
        """Helper to convert '10 - 50' string to Geometric Mean float."""
        if not isinstance(val, str):
            return 0
        nums = [float(x) for x in re.findall(r'\d+', val)]
        if len(nums) >= 2:
            return np.sqrt(nums[0] * nums[1])  # Geometric Mean
        elif len(nums) == 1:
            return nums[0]
        return 0

    def save_metadata_maps(self):
        """Extract ID Maps and Lookup Tables for the Frontend Team."""
        print(">> Saving Metadata Maps...")

        id_cols = ['DUNS Number', 'Company Sites', 'Parent Company', 'Global Ultimate Company',
                   'SIC Code', 'Legal Status']
        existing_cols = [c for c in id_cols if c in self.df.columns]
        self.df[existing_cols].to_csv('../data/company_identity_map.csv', index=False)
        print(f"   - Saved 'company_identity_map.csv' with {len(self.df)} rows.")

        if 'SIC Code' in self.df.columns and 'SIC Description' in self.df.columns:
            sic_map = self.df[['SIC Code', 'SIC Description']].copy().drop_duplicates().sort_values('SIC Code')
            sic_map.to_csv('../data/sic_lookup.csv', index=False)
            print(f"   - Saved 'sic_lookup.csv' with {len(sic_map)} unique industries.")

        original_df = self.df.copy()
        original_df.to_csv("../data/original_data.csv", index=False)

    def _impute_missing_state_abbr(self):
        """Fill missing abbreviations by learning from existing State -> Abbr pairs."""
        state_pairs = [
            ("State", "State Or Province Abbreviation"),
            ("Parent State/Province", "Parent State/Province Abbreviation"),
            ('Global Ultimate State/Province', 'Ultimate State/Province Abbreviation'),
            ('Domestic Ultimate State/Province Name', 'Domestic Ultimate State Abbreviation')
        ]

        for full_col, abbr_col in state_pairs:
            if full_col not in self.df.columns or abbr_col not in self.df.columns:
                print(f"{full_col} or {abbr_col} doesnt exist.")
                continue

            self.df[full_col] = self.df[full_col].astype(str).str.upper().str.strip().replace('NAN', np.nan)
            reference_df = self.df[[full_col, abbr_col]].copy().dropna().drop_duplicates(subset=[full_col], keep='first')
            state_map = dict(zip(reference_df[full_col], reference_df[abbr_col]))
            self.df[abbr_col] = self.df[abbr_col].fillna(self.df[full_col].map(state_map))

    def clean_general(self):
        """Step 2: Drops, Standardization, and Type Conversion."""
        print(">> Running General Cleaning...")

        self._impute_missing_state_abbr()

        # Check for international presence indicators
        if "NACE Rev 2 Code" in self.df.columns and "NACE Rev 2 Description" in self.df.columns:
            self.df["Has_NACE"] = (
                self.df["NACE Rev 2 Code"].notna() |
                self.df["NACE Rev 2 Description"].notna()
            ).astype(int)
        else:
            print("missing either NACE Rev 2 Code or its description")

        if "NAICS Code" in self.df.columns and "NAICS Description" in self.df.columns:
            self.df["Has_NAICS"] = (
                self.df["NAICS Code"].notna() |
                self.df["NAICS Description"].notna()
            ).astype(int)
        else:
            print("missing either NAICS Code or its description")

        if "ANZSIC Code" in self.df.columns and "ANZSIC Description" in self.df.columns:
            self.df["Has_ANZSIC"] = (
                self.df["ANZSIC Code"].notna() |
                self.df["ANZSIC Description"].notna()
            ).astype(int)
        else:
            print("missing either ANZISC Code or its description")

        if "ISIC Rev 4 Code" in self.df.columns and "ISIC Rev 4 Description" in self.df.columns:
            self.df["Has_ISIC"] = (
                self.df["ISIC Rev 4 Code"].notna() |
                self.df["ISIC Rev 4 Description"].notna()
            ).astype(int)
        else:
            print("missing either ISIC Rev 4 Code or its description")

        # Drop unwanted columns
        cols_to_drop = [
            'Company Sites', 'Parent Company', 'Global Ultimate Company', 'Domestic Ultimate Company',
            'Ticker', 'Website', 'Address Line 1', 'Phone Number', 'Lattitude', 'Longitude',
            'Registration Number', 'Registration Number Type',
            'SIC Description',
            '8-Digit SIC Code', '8-Digit SIC Description',
            'NACE Rev 2 Code', 'NACE Rev 2 Description',
            'NAICS Code', 'NAICS Description',
            'ANZSIC Code', 'ANZSIC Description',
            'ISIC Rev 4 Code', 'ISIC Rev 4 Description',
            'State', 'Parent State/Province', 'Global Ultimate State/Province',
            'Domestic Ultimate State/Province Name',
            'Fiscal Year End'
        ]

        address_cols = [c for c in self.df.columns if 'Street Address' in c]
        cols_to_drop.extend(address_cols)

        print("Dropping columns below:")
        for c in [c for c in cols_to_drop if c in self.df.columns]:
            print(c)

        self.df.drop(columns=[c for c in cols_to_drop if c in self.df.columns], inplace=True)

        if len([c for c in cols_to_drop if c in self.df.columns]) == 0:
            print("Successfully dropped!")

        # Standardize state abbreviations
        abbr_cols = [c for c in self.df.columns if 'Abbreviation' in c]
        for c in abbr_cols:
            self.df[c] = self.df[c].astype(str).str.upper().str.strip()
            self.df[c] = self.df[c].replace(["NAN", "nan", "NULL"], "UNKNOWN")

        # Postal codes
        postal_cols = [c for c in self.df.columns if 'Postal Code' in c]
        for col in postal_cols:
            self.df[col] = self.df[col].astype(str).replace("nan", "Unknown")
            mask = self.df[col] != "Unknown"
            self.df.loc[mask, col] = self.df.loc[mask, col].str.split(".").str[0]

            self.df["Country"] = self.df["Country"].apply(self._capitalized_elem)

            china_mask = mask & (self.df["Country"] == "CHINA")
            indonesia_mask = mask & (self.df["Country"] == "INDONESIA")
            other_mask = mask & (~china_mask) & (~indonesia_mask)

            if any(other_mask):
                print("==" * 70 + "\n" +
                      "there are other countries other than China and Indonesia" +
                      "==" * 70 + "\n")

            self.df.loc[china_mask, col] = self.df.loc[china_mask, col].str.zfill(6)
            self.df.loc[indonesia_mask, col] = self.df.loc[indonesia_mask, col].str.zfill(5)
            self.df.loc[other_mask, col] = self.df.loc[other_mask, col].str.zfill(5)
            self.df[col] = self.df[col].astype("category")

        print("Postal Codes Conversion Done")

        # Clean city columns
        self.df["City"] = self.df["City"].apply(self._capitalized_elem)
        self.df["City"] = self.df["City"].replace(["NAN", "nan", "NULL"], "UNKNOWN")
        self.df["Parents City"] = self.df["Parents City"].apply(self._capitalized_elem)
        self.df["Parents City"] = self.df["Parents City"].replace(["NAN", "nan", "NULL"], "UNKNOWN")
        self.df["Global Ultimate City Name"] = self.df["Global Ultimate City Name"].apply(self._capitalized_elem)
        self.df["Global Ultimate City Name"] = self.df["Global Ultimate City Name"].replace(["NAN", "nan", "NULL"], "UNKNOWN")
        self.df["Domestic Ultimate City Name"] = self.df["Domestic Ultimate City Name"].apply(self._capitalized_elem)
        self.df["Domestic Ultimate City Name"] = self.df["Domestic Ultimate City Name"].replace(["NAN", "nan", "NULL"], "UNKNOWN")

        # Clean ownership type
        self.df["Ownership Type"] = self.df["Ownership Type"].apply(self._capitalized_elem)
        self.df["Ownership Type"] = self.df["Ownership Type"].replace(["NAN", "nan", "NULL"], "UNKNOWN")

    def clean_numeric_and_bool(self):
        """Step 3: Handling Ranges, Booleans, and Numeric Parsers."""
        print(">> Cleaning Numerics & Booleans...")

        # Parse ranges to geometric mean
        for col in self.range_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self._parse_range).astype(float)

        # Boolean logic
        def map_bool_status(val):
            if pd.isna(val):
                return -1
            s = str(val).upper()
            if s in ["TRUE", "YES", "1", "1.0"]:
                return 1
            elif s in ["FALSE", "NO", "0", "0.0"]:
                return 0
            print("there are weird values to be catched in step 5")
            return -1

        for col in self.flag_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(map_bool_status).astype(int)

        # Year Found
        if 'Year Found' in self.df.columns:
            current_year = date.today().year
            self.df["Year Found"] = self.df["Year Found"].fillna(current_year - 1).astype(int)
            self.df['Company_Age'] = current_year - self.df['Year Found']
            self.df.drop(columns=["Year Found"], inplace=True)

        # Clean employees
        print(f"    - Dropping {self.df['Employees Single Site'].isna().sum()} rows with missing Employees")
        self.df.dropna(subset=["Employees Single Site"], inplace=True)

        # Revenue and employees cleanup
        for col in ['Revenue (USD)', 'Employees Total', 'Employees Single Site']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df[f'Log_{col}'] = np.log1p(self.df[col].fillna(0))

    def target_encode_features(self):
        """
        Step 4: Bayesian Target Encoding with Training/Prediction Logic.
        
        During training: learns encoding maps and saves them
        During prediction: uses previously saved encoding maps
        """
        print(f">> Performing Bayesian Target Encoding on {self.target_cols}...")

        if self.is_training:
            self._target_encode_training()
        else:
            self._target_encode_prediction()

    def _target_encode_training(self):
        """Training mode: learn and save encoding maps."""
        global_median = self.df["Revenue (USD)"].median()

        for col in self.target_cols:
            if col not in self.df.columns:
                continue

            # Clean the column
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(-1).astype(int)

            # Calculate statistics per category
            stats = self.df.groupby(col)["Revenue (USD)"].agg(
                Median_Rev="median",
                Count="count"
            ).reset_index()

            # Bayesian smoothing formula
            m = 10
            stats["weight"] = stats["Count"] / (stats["Count"] + m)
            stats["Encoded_Value"] = (
                (stats["weight"] * stats["Median_Rev"]) +
                ((1 - stats["weight"]) * global_median)
            )

            # Save the mapping for later use
            self.encoding_maps[col] = stats.set_index(col)["Encoded_Value"].to_dict()
            self.global_medians[col] = global_median

            # Apply to dataframe
            self.df = self.df.merge(stats[[col, "Encoded_Value"]], on=col, how="left")
            self.df.rename(columns={"Encoded_Value": f"{col}_Target_Encoded"}, inplace=True)

            # Fill any gaps
            if self.df[f"{col}_Target_Encoded"].isna().sum() > 0:
                print(f"check {col} target encoded properly, there is empty value")
            self.df[f"{col}_Target_Encoded"] = self.df[f"{col}_Target_Encoded"].fillna(global_median)

            # Drop original column
            print(f"    - Dropping raw {col} to prevent overfitting.")
            self.df.drop(columns=[col], inplace=True)

    def _target_encode_prediction(self):
        """Prediction mode: use saved encoding maps."""
        print(" (Prediction Mode) Using saved encoding maps...")

        for col in self.target_cols:
            if col in self.df.columns and col in self.encoding_maps:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(-1).astype(int)

                # Use saved map, fallback to global median for unseen values
                fallback = self.global_medians.get(col, 0)
                if fallback == 0:
                    print("check target_encode_features, why is global median 0")
                self.df[f"{col}_Target_Encoded"] = self.df[col].map(self.encoding_maps[col]).fillna(fallback)

                # Drop original column
                self.df.drop(columns=[col], inplace=True)

    def save_encoding_artifacts(self, artifacts_dir: Optional[Path] = None) -> Path:
        """
        Save encoding maps and global medians to disk.
        
        Follows industry best practices:
        - Uses pickle for serialization (native Python, efficient)
        - Stores metadata for versioning and auditing
        - Creates artifacts directory if it doesn't exist
        - Returns path for verification
        
        Args:
            artifacts_dir: Directory to save artifacts. Defaults to ARTIFACTS_DIR
            
        Returns:
            Path to the artifacts directory
            
        Raises:
            ValueError: If encoding maps are empty (no training done)
        """
        if not self.is_training:
            raise ValueError("Cannot save artifacts in prediction mode. Only available during training.")

        if not self.encoding_maps or not self.global_medians:
            raise ValueError("Encoding maps are empty. Run pipeline first or check training data.")

        artifacts_dir = artifacts_dir or self.ARTIFACTS_DIR
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Prepare metadata
        metadata = {
            "timestamp": date.today().isoformat(),
            "target_cols": self.target_cols,
            "encoding_maps_keys": list(self.encoding_maps.keys()),
            "global_medians_keys": list(self.global_medians.keys()),
        }

        # Save using pickle (native, efficient, handles complex types)
        encoding_path = artifacts_dir / "encoding_maps.pkl"
        medians_path = artifacts_dir / "global_medians.pkl"
        metadata_path = artifacts_dir / "encoding_metadata.json"

        try:
            # Atomic write using pickle
            with open(encoding_path, 'wb') as f:
                pickle.dump(self.encoding_maps, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open(medians_path, 'wb') as f:
                pickle.dump(self.global_medians, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata as JSON for human readability
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"\n✓ Encoding artifacts saved successfully!")
            print(f"  - {encoding_path}")
            print(f"  - {medians_path}")
            print(f"  - {metadata_path}\n")

            return artifacts_dir

        except Exception as e:
            raise RuntimeError(f"Failed to save encoding artifacts: {str(e)}")

    def load_encoding_artifacts(self, artifacts_dir: Optional[Path] = None) -> bool:
        """
        Load encoding maps and global medians from disk.
        
        Follows industry best practices:
        - Validates artifact existence and integrity
        - Provides clear error messages
        - Sets instance variables for use in prediction
        
        Args:
            artifacts_dir: Directory containing saved artifacts. Defaults to ARTIFACTS_DIR
            
        Returns:
            True if artifacts loaded successfully, False otherwise
            
        Raises:
            FileNotFoundError: If artifacts directory or files don't exist
            pickle.UnpicklingError: If pickle files are corrupted
        """
        if self.is_training:
            raise ValueError("Cannot load artifacts in training mode. Use prediction mode instead.")

        artifacts_dir = artifacts_dir or self.ARTIFACTS_DIR

        # Check if artifacts directory exists
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

        encoding_path = artifacts_dir / "encoding_maps.pkl"
        medians_path = artifacts_dir / "global_medians.pkl"
        metadata_path = artifacts_dir / "encoding_metadata.json"

        # Check if all required files exist
        missing_files = []
        for path in [encoding_path, medians_path, metadata_path]:
            if not path.exists():
                missing_files.append(path)

        if missing_files:
            raise FileNotFoundError(
                f"Missing artifact files: {[str(f) for f in missing_files]}\n"
                f"Expected artifacts in: {artifacts_dir}"
            )

        try:
            # Load pickle files
            with open(encoding_path, 'rb') as f:
                self.encoding_maps = pickle.load(f)

            with open(medians_path, 'rb') as f:
                self.global_medians = pickle.load(f)

            # Load and display metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            print(f"\n✓ Encoding artifacts loaded successfully!")
            print(f"  - Loaded from: {artifacts_dir}")
            print(f"  - Created on: {metadata.get('timestamp', 'Unknown')}")
            print(f"  - Target columns: {metadata.get('target_cols', [])}\n")

            return True

        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(f"Failed to unpickle encoding artifacts: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load encoding artifacts: {str(e)}")

    def run(self) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
        """
        Execute full pipeline with automatic artifact management.
        
        Training mode:
            - Processes data
            - Learns encoding maps
            - Saves artifacts automatically
            - Returns: (cleaned_dataframe, None)
            
        Prediction mode:
            - Loads encoding artifacts
            - Applies transformations
            - Returns: (transformed_dataframe, artifact_info)
        
        Returns:
            Tuple of (dataframe, artifact_info)
        """
        self.filter_and_clean_string_columns()
        self.save_metadata_maps()
        self.clean_general()
        self.clean_numeric_and_bool()
        self.target_encode_features()

        print(f">> Pipeline Complete. Final Shape: {self.df.shape}")

        if self.is_training:
            # Save artifacts before returning
            self.save_encoding_artifacts()
            return self.df, None
        else:
            # Return dataframe and artifact info for verification
            artifact_info = {
                "encoding_maps": self.encoding_maps,
                "global_medians": self.global_medians
            }
            return self.df, artifact_info


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # TRAINING MODE
    print("="*70)
    print("TRAINING PIPELINE")
    print("="*70)
    training_pipeline = DataPipeline("../data/Gobbling gang data.xlsx", is_training=True)
    df_train, _ = training_pipeline.run()
    print(f"Training data shape: {df_train.shape}")
    print(f"Columns: {df_train.columns.tolist()}")

    # PREDICTION MODE
    print("\n" + "="*70)
    print("PREDICTION PIPELINE")
    print("="*70)
    prediction_pipeline = DataPipeline("../data/Gobbling gang data.xlsx", is_training=False)
    prediction_pipeline.load_encoding_artifacts()  # Load saved artifacts
    df_pred, artifact_info = prediction_pipeline.run()
    print(f"Prediction data shape: {df_pred.shape}")
    print(f"Encoding maps loaded: {list(artifact_info['encoding_maps'].keys())}")
