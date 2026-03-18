import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import pickle
from pathlib import Path
from datetime import date
from typing import Tuple, Dict, Any, Optional
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

class DataPipeline:
    
    def __init__(self, df: Optional[pd.DataFrame], raw_data_path: str = "", is_training: bool = False):
        if raw_data_path.endswith((".xls", ".xlsx")):
            self.df = pd.read_excel(raw_data_path)
        elif raw_data_path == "":
            print("you should have passed in a df")
            self.df = df
        else:
            raise TypeError("File is not in the correct type")
        self.is_training = is_training
        self.encoding_maps = {}
        self.global_medians = {}
        self.employees_medians = {}
        # Configuration
        # Define columns to treat as boolean
        self.flag_cols = ['Manufacturing Status', 'Franchise Status', 'Is Headquarters', 'Is Domestic Ultimate']
        # Define columns with ranges (e.g., "1 - 10")
        self.range_cols = [
            "No. of PC", "No. of Desktops", "No. of Laptops", 
            "No. of Routers", "No. of Servers", "No. of Storage Devices"
        ]
        # Columns we want to Target Encode
        self.target_cols = ["SIC Code", "Legal Status"]
        self.ARTIFACTS_DIR = Path("../artifacts")
    
    def _capitalized_elem(self, elem):
        """
        used to standardize data to make them into capital letter

        argument elem must be an element that can and is intended to be turned into string
        NOTE: NaN will be turned into string as well
        """
        return str(elem).upper().strip()
    
    def _get_object_columns(self, df: pd.DataFrame) -> list:
        """
        Identify all columns with dtype 'object' in the dataframe.
        
        Args:
            df: Input pandas dataframe
        
        Returns:
            List of column names with dtype 'object'
        """
        return df.select_dtypes(include=['object', 'str']).columns.tolist()
    
    def _remove_whitespace(self, elem):
        """
        if the elem is not NaN, then will convert into string and remove leading and trailing whitespace
        else, just leave it there.
        """
        if pd.isna(elem):
            return elem
        else:
            return str(elem).strip()
    
    def filter_and_clean_string_columns(self, verbose: bool = True) -> pd.DataFrame:
        """
        Identify object-type (string) columns and remove leading/trailing whitespace.
        NOTE: this edits the original df
        
        This function:
        1. Finds all columns with dtype 'object'
        2. Applies .str.strip() to each column to remove leading/trailing whitespace
        3. Returns the cleaned dataframe
        
        Args:
            df: Input pandas dataframe
            verbose: If True, prints detailed information about the cleaning process (default: True)
        
        Returns:
            DataFrame with whitespace removed from all string columns
        """
        # Get all object-type columns
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
        
        # Apply .str.strip() to each object column
        for col in object_columns:
            # Only apply strip if the column contains string values
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].apply(self._remove_whitespace)
        
        if verbose:
            print(f"\n✓ Applied .str.strip() to all {len(object_columns)} string columns")
            print("✓ Whitespace removed successfully")
            print("="*70 + "\n")

    def _parse_range(self, val):
        """Helper to convert '10 - 50' string to Geometric Mean float."""
        if not isinstance(val, str): return 0
        nums = [float(x) for x in re.findall(r'\d+', val)]
        if len(nums) >= 2:
            return np.sqrt(nums[0] * nums[1]) # Geometric Mean
        elif len(nums) == 1:
            return nums[0]
        return 0 # Default to 0 if text is garbage

    def save_metadata_maps(self):
        """
        Step 1: Extract ID Maps and Lookup Tables for the Frontend Team.
        Does not modify self.df, just saves side files.
        """
        print(">> Saving Metadata Maps...")
        
        # 1. Master ID Map (For Dashboard Linking)
        # Keeps the human names linked to the unique DUNS ID
        id_cols = ['DUNS Number', 'Company Sites', 'Parent Company', 'Global Ultimate Company',
                   'SIC Code', 'Legal Status']
        # Only keep columns that actually exist in the CSV
        existing_cols = [c for c in id_cols if c in self.df.columns]
        self.df[existing_cols].to_csv('../data/company_identity_map.csv', index=False)
        print(f"   - Saved 'company_identity_map.csv' with {len(self.df)} rows.")

        # 2. SIC Lookup Table (For Dashboard Filters)
        if 'SIC Code' in self.df.columns and 'SIC Description' in self.df.columns:
            sic_map = self.df[['SIC Code', 'SIC Description']].copy().drop_duplicates().sort_values('SIC Code')
            sic_map.to_csv('../data/sic_lookup.csv', index=False)
            print(f"   - Saved 'sic_lookup.csv' with {len(sic_map)} unique industries.")

        # 3. Original Dataset (For reference)
        original_df = self.df.copy()
        original_df.to_csv("../data/original_data.csv", index = False)

    def _impute_missing_state_abbr(self):
        """
        Helper: fill missing abbreviations by learning from existing State -> Abbr pairs.
        Uses a vectorized dictionary map to ensure efficiency.
        """
        state_pairs = [
            ("State", "State Or Province Abbreviation"),
            ("Parent State/Province", "Parent State/Province Abbreviation"),
            ('Global Ultimate State/Province', 'Ultimate State/Province Abbreviation'),
            ('Domestic Ultimate State/Province Name', 'Domestic Ultimate State Abbreviation')
        ]

        for full_col, abbr_col in state_pairs:
            # skip if columns don't exist
            if full_col not in self.df.columns or abbr_col not in self.df.columns:
                print(f"{full_col} or {abbr_col} doesnt exist.")
                continue
            # 1. Standardize the Full Name (Key) so 'New York' == 'NEW YORK'
            # We treat 'nan' string as NaN object to avoid mapping "NAN" -> "NA"
            self.df[full_col] = self.df[full_col].astype(str).str.upper().str.strip().replace('NAN', np.nan)
            
            # 2. Build the Reference Dictionary (The "Truth")
            # We take only rows where BOTH are present.
            # drop_duplicates('keep=first') ensures we stick to the first valid mapping we find.
            reference_df = self.df[[full_col, abbr_col]].copy().dropna().drop_duplicates(subset=[full_col], keep='first')
            
            # Convert to dictionary: {'CALIFORNIA': 'CA', 'TEXAS': 'TX'}
            state_map = dict(zip(reference_df[full_col], reference_df[abbr_col]))

            # 3. Apply the Map to Missing Values Only
            # logical flow: "If Abbr is missing, look at Full Name. If Full Name is in our map, take the Abbr."
            self.df[abbr_col] = self.df[abbr_col].fillna(self.df[full_col].map(state_map))
            self.df[abbr_col] = self.df[abbr_col].fillna("UNKNOWN")
        
    def clean_general(self):
        """Step 2: Drops, Standardization, and Type Conversion."""
        print(">> Running General Cleaning...")

        # 0. Fix Abbreviations BEFORE dropping the Full Name columns
        self._impute_missing_state_abbr()

        # OB. Checking existence of CODE - if company bothered to register that code,
        # means it might have business in other countries -> potentially higher revenue
        # EUROPEAN PRESENCE (NACE)
        if "NACE Rev 2 Code" in self.df.columns and "NACE Rev 2 Description" in self.df.columns:
            self.df["Has_NACE"] = (
                self.df["NACE Rev 2 Code"].notna() | 
                self.df["NACE Rev 2 Description"].notna()
                ).astype(int)
        else:
            print("missing either NACE Rev 2 Code or its description")

        # NORTH AMERICAN PRESENCE (NAICS)
        if "NAICS Code" in self.df.columns and "NAICS Description" in self.df.columns:
            self.df["Has_NAICS"] = (
                self.df["NAICS Code"].notna() | 
                self.df["NAICS Description"].notna()
            ).astype(int)
        else:
            print("missing either NAICS Code or its description")

        # AUSTRALIA / NZ PRESENCE (ANZSIC)
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
        
        # 1. DROP UNWANTED COLUMNS
        # Note: We drop specific names because we saved them in the ID Map above
        cols_to_drop = [
            'Company Sites', 'Parent Company', 'Global Ultimate Company', 'Domestic Ultimate Company',
            'Ticker', 'Website', 'Address Line 1', 'Phone Number', 'Lattitude', 'Longitude',
            'Registration Number', 'Registration Number Type', # only 8 non-null 
            # Drop the redundant Industry Codes/Descriptions
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
        
        # Drop all "Street Address" columns dynamically
        address_cols = [c for c in self.df.columns if 'Street Address' in c]
        cols_to_drop.extend(address_cols)

        print("Dropping columns below:")
        for c in [c for c in cols_to_drop if c in self.df.columns]:
            print(c)
        
        self.df.drop(columns=[c for c in cols_to_drop if c in self.df.columns], inplace=True)

        if len([c for c in cols_to_drop if c in self.df.columns]) == 0:
            print("Successfully dropped!")

        # 2. STATE STANDARDIZATION
        # Standardize all Abbreviation columns to UPPERCASE
        abbr_cols = [c for c in self.df.columns if 'Abbreviation' in c]
        for c in abbr_cols:
            self.df[c] = self.df[c].astype(str).str.upper().str.strip()
            self.df[c] = self.df[c].replace(["NAN", "nan", "NULL"], "UNKNOWN")

        # 3. POSTAL CODES (Keep as String/Category)
        postal_cols = [c for c in self.df.columns if 'Postal Code' in c]
        for col in postal_cols:
            # fillna(-1) ensures missing values are a distinct category
            self.df[col] = self.df[col].fillna(-1)
            self.df[col] = self.df[col].astype(str).replace("-1", "Unknown")
            mask = self.df[col] != "Unknown"
            self.df.loc[mask, col] = self.df.loc[mask, col].str.split(".").str[0]
            
            self.df["Country"] = self.df["Country"].apply(self._capitalized_elem)

            # Apply country-specific padding
            china_mask = mask & (self.df["Country"] == "CHINA")
            indonesia_mask = mask & (self.df["Country"] == "INDONESIA")
            other_mask = mask & (~china_mask) & (~indonesia_mask)

            if any(other_mask):
                print("==" * 70 + "\n" +
                      "there are other countries other than China and Indonesia" +
                      "==" * 70 + "\n")

            # Pad based on country
            self.df.loc[china_mask, col] = self.df.loc[china_mask, col].str.zfill(6)
            self.df.loc[indonesia_mask, col] = self.df.loc[indonesia_mask, col].str.zfill(5)
            self.df.loc[other_mask, col] = self.df.loc[other_mask, col].str.zfill(5)
            self.df[col] = self.df[col].astype("category")
        print("Postal Codes Conversion Done")

        # 4. CLEAN COLUMN CITY (and related cities)
        self.df["City"] = self.df["City"].apply(self._capitalized_elem)
        self.df["City"] = self.df["City"].replace(["NAN", "nan", "NULL"], "UNKNOWN")
        self.df["Parents City"] = self.df["Parents City"].apply(self._capitalized_elem)
        self.df["Parents City"] = self.df["Parents City"].replace(["NAN", "nan", "NULL"], "UNKNOWN")
        self.df["Global Ultimate City Name"] = self.df["Global Ultimate City Name"].apply(self._capitalized_elem)
        self.df["Global Ultimate City Name"] = self.df["Global Ultimate City Name"].replace(["NAN", "nan", "NULL"], "UNKNOWN")
        self.df["Domestic Ultimate City Name"] = self.df["Domestic Ultimate City Name"].apply(self._capitalized_elem)
        self.df["Domestic Ultimate City Name"] = self.df["Domestic Ultimate City Name"].replace(["NAN", "nan", "NULL"], "UNKNOWN")
    

        # 5. CLEAN COLUMN OWNERSHIP TYPE
        self.df["Ownership Type"] = self.df["Ownership Type"].apply(self._capitalized_elem)
        self.df["Ownership Type"] = self.df["Ownership Type"].replace(["NAN", "nan", "NULL"], "UNKNOWN")

    def clean_numeric_and_bool(self):
        """Step 3: Handling Ranges, Booleans, and Numeric Parsers."""
        print(">> Cleaning Numerics & Booleans...")

        # 1. PARSE RANGES (Geometric Mean)
        for col in self.range_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self._parse_range).astype(float)

        # 2. BOOLEAN LOGIC (True=1, False=0, NaN=-1)
        def map_bool_status(val):
            # to keep actual NaN
            if pd.isna(val):
                return -1
            
            # if not NaN, then we only convert to str and perform the operations
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
                print(f"{col} done converting")

        # 3. YEAR FOUND (Impute Median)
        if 'Year Found' in self.df.columns:
            current_year = date.today().year
            # missing values will have -1 for age (different from positive value)
            self.df["Year Found"] = self.df["Year Found"].fillna(current_year + 1).astype(int)
            # Calculate Age - prevent gradient explosion (range is smaller)
            self.df['Company_Age'] = current_year - self.df['Year Found']
            # drop original column - prevent multicollinearity
            self.df.drop(columns = ["Year Found"], inplace = True)

        # 3B. CLEAN EMPLOYEES SINGLE SITE - drop (because only have 2 missing values, don't need to specially create a grp)
        # if cfreate a grp for them will cause overfitting for companies who ahs missing value for that
        # if we fill in using median, we have the risk of creating false trend, so removing can preserve the original insights
        if self.is_training:
            print(f"    - Dropping {self.df["Employees Single Site"].isna().sum()} rows with missing Employees")
            self.df.dropna(subset = ["Employees Single Site"], inplace = True)

        # 4. REVENUE & EMPLOYEES CLEANUP
        # Remove '$' and ',' and convert to numeric
        for col in ['Revenue (USD)', 'Employees Total', 'Employees Single Site']:
            if col in self.df.columns:
                if not self.is_training and col != "Revenue (USD)":
                    self.df[col].fillna(self.employees_medians[col])
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                # Log Transform for Training (Essential for Revenue)
                # because in real world business data, usually have more small businesses
                # and giant companies usually have revenue way more higher than small businesses
                # so by using log(1 + x) it fixes the scale (1 + x) prevent the code from crashing
                # ex. log(0)
                self.df[f'Log_{col}'] = np.log1p(self.df[col].fillna(0))
                print(f"{col} is successfully log-transformed, dropping now")
                if self.is_training and (col != "Revenue (USD)"):
                    self.employees_medians[col] = self.df[col].median()
                self.df.drop(columns = [col], inplace = True)

    def target_encode_features(self):
        """
        Step 4: Bayesian Target Encoding with Training/Prediction Logic.
        
        During training: learns encoding maps and saves them
        During prediction: uses previously saved encoding maps
        """
        print(f">> Performing Bayesian Target Encoding on {self.target_cols}...")
        # Smoothing: If an industry has < 5 companies, use Global Median to prevent overfitting
        # it is better for our user to loss an opportunity by predicting high revenue as low revenue
        # than predicting a low revenue as high revenue which loses money
        # bcs grouping rare companies tgt ex. OHE - model might learn this grp to be high revenue, 
        # but this might not always be the case
        if self.is_training:
            self._target_encode_training()
        else:
            self._target_encode_prediction()
        

    def _target_encode_training(self):
        """
        Training mode: learn and save encoding maps.
        """
        global_median = self.df["Log_Revenue (USD)"].median()

        for col in self.target_cols:
            if col not in self.df.columns:
                continue

            # A. Clean the column (Ensure valid integers, fill NaN with -1)
            self.df[col] = pd.to_numeric(self.df[col], errors = "coerce").fillna(-1).astype(int)

            # B. Calculate Statistics per Category
            stats = self.df.groupby(col)["Log_Revenue (USD)"].agg(
                Median_Rev = "median",
                Count = "count"
            ).reset_index()

            # C. The Bayesian Formula (Smoothing)
            # not doing hard cutoff ex. if < 5 use global, instead take data from both sides
            # ex. 1 sample: 1% industry median, 99% global median
            m = 10
            stats["weight"] = stats["Count"] / (stats["Count"] + m)
            stats["Encoded_Value"] = (
                (stats["weight"] * stats["Median_Rev"]) + 
                ((1 - stats["weight"]) * global_median)
            )

            # D. Save the "Truth" to our Class Dictionary (For future use)
            # we save a simple map: {1: $5M, 2: $10M, ...}
            self.encoding_maps[col] = stats.set_index(col)["Encoded_Value"].to_dict()
            self.global_medians[col] = global_median

            # E. Apply the Map to the DataFrame
            self.df = self.df.merge(stats[[col, "Encoded_Value"]], on = col, how = "left")
            self.df.rename(columns = {"Encoded_Value": f"{col}_Target_Encoded"}, inplace = True)

            # F. Fill any gaps (not sure when will use but defensive programming)
            # bcs if empty should be -1, else, count confirm > 0
            if self.df[f"{col}_Target_Encoded"].isna().sum() > 0:
                print(f"check {col} target encoded properly, thr is empty value")
            self.df[f"{col}_Target_Encoded"] = self.df[f"{col}_Target_Encoded"].fillna(global_median)

            # E. DROP THE ORIGINAL COLUMN
            print(f"    - Dropping raw {col} to prevent overfitting.")
            self.df.drop(columns = [col], inplace = True)

    def _target_encode_prediction(self):
        """
        Prediction mode: use saved encoding maps.
        """
        print(" (Prediction Mode) Using saved encoding maps...")
        for col in self.target_cols:
            if col in self.df.columns and col in self.encoding_maps:
                self.df[col] = pd.to_numeric(self.df[col], errors = "coerce").fillna(-1).astype(int)

                # use value in saved map, if unseen, use global median, ex. not NaN and not those that exist in train
                fallback = self.global_medians.get(col, 0)
                if fallback == 0:
                    print("check target_encode_features, why is global median 0")
                self.df[f"{col}_Target_Encoded"] = self.df[col].map(self.encoding_maps[col]).fillna(fallback)

                # drop the original columns
                self.df.drop(columns = [col], inplace = True)

    def save_encoding_artifacts(self, artifacts_dir: Optional[str] = None) -> Path:
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
        employees_path = artifacts_dir / "employees_medians.pkl"

        try:
            # Atomic write using pickle
            with open(encoding_path, 'wb') as f:
                pickle.dump(self.encoding_maps, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open(medians_path, 'wb') as f:
                pickle.dump(self.global_medians, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata as JSON for human readability
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            with open(employees_path, 'wb') as f:
                pickle.dump(self.employees_medians, f, protocol = pickle.HIGHEST_PROTOCOL)

            print(f"\n✓ Encoding artifacts saved successfully!")
            print(f"  - {encoding_path}")
            print(f"  - {medians_path}")
            print(f"  - {metadata_path}")
            print(f"   - {employees_path}")

            return artifacts_dir

        except Exception as e:
            raise RuntimeError(f"Failed to save encoding artifacts: {str(e)}")
        
    def load_encoding_artifacts(self, artifacts_dir: Optional[Path] = None) -> bool:
        """
        Load encoding maps and global medians from disk.
        Need to be used before running the pipeline
        
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
        employees_path = artifacts_dir / "employees_medians.pkl"

        # Check if all required files exist
        missing_files = []
        for path in [encoding_path, medians_path, metadata_path, employees_path]:
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
            
            with open(employees_path, 'rb') as f:
                self.employees_medians = pickle.load(f)

            print(f"\n✓ Encoding artifacts loaded successfully!")
            print(f"  - Loaded from: {artifacts_dir}")
            print(f"  - Created on: {metadata.get('timestamp', 'Unknown')}")
            print(f"  - Target columns: {metadata.get('target_cols', [])}\n")

            return True

        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(f"Failed to unpickle encoding artifacts: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load encoding artifacts: {str(e)}")
        
    def final_type_change(self):
        obj_cols = self._get_object_columns(self.df)
        for c in obj_cols:
            self.df[c] = self.df[c].astype("category")
        sanity_check = self._get_object_columns(self.df)
        if not sanity_check:
            print("All object columns successfully turned into categorical")
        print("currently testing only to drop Company Description")
        self.df.drop(columns = ["Company Description"], inplace = True)


    def run(self):
        """
        Execute full pipeline.
        save the company identity map to '../data/company_identity_map.csv', consists of
        - own DUNS Number, own Company name, Parent and Global Ultimate Company name
        save the SIC Lookup Table to '../data/sic_lookup.csv' (For Dashboard Filters)
        - original dataset in csv form saved to "../data/original_data.csv"

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
        self.final_type_change()

        print(f">> Pipeline Complete. Final Shape: {self.df.shape}")
        if self.is_training:
            self.save_encoding_artifacts()
            return self.df, None
        else:
            artifact_info = { 
                "encoding_maps": self.encoding_maps,
                "global_medians": self.global_medians
            } 
            return self.df, artifact_info