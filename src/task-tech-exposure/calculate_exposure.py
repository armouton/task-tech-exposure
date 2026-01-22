# IMPORT PACKAGES
import os
import pandas as pd
import numpy as np
import itertools


# ================== CALCULATE EXPOSURE MEASURES =================

# LOAD FILES WITH ERROR HANDLING
def try_load_csv(path, usecols=None, abort=False):
    """
    Load CSV file with error handling.
    
    Args:
        path: Path to CSV file
        usecols: Columns to load (optional)
        abort: Whether to raise exception on error (default: False)
        
    Returns:
        DataFrame if successful, None if file not found and abort=False
        
    Raises:
        FileNotFoundError: If file not found and abort=True
    """
    try:
        data = pd.read_csv(path, usecols=usecols)
        return data
    except FileNotFoundError as e:
        if abort:
            raise FileNotFoundError(f"✗ Required file not found at {path}") from e
        else:
            print(f"Warning: File not found at {path}, skipping")
            return None
    except Exception as e:
        error_msg = f"✗ Error loading CSV from {path}: {e}"
        if abort:
            raise Exception(error_msg) from e
        else:
            print(f"Warning: {error_msg}")
            return None


def try_load_npy(path):
    """
    Load numpy array with error handling.
    
    Args:
        path: Path to .npy file
        
    Returns:
        Numpy array if successful
        
    Raises:
        FileNotFoundError: If file not found
    """
    try:
        data = np.load(path)
        return data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"✗ Required file not found at {path}") from e
    except Exception as e:
        raise Exception(f"✗ Failed to load numpy array from {path}: {e}") from e


# CALCULATE EXPOSURE MEASURES
def measure_exposure(path_to_data, path_to_results, 
                     path_to_tech_classifications = None,
                     path_to_task_classifications = None, 
                     level="aggregate", frequency="annual", weights="both", 
                     measure="exposed", match_cutoff=None, 
                     drop_thresh=None, start_date=None, end_date=None, 
                     num_percentiles=20, crosswalk=None, digits=6):
    """
    Calculate technology exposure measures for occupations.
    
    Args:
        path_to_data: Path to data directory
        path_to_tech_classifications: Path to technology classification files
        path_to_task_classifications: Path to task classification files
        path_to_results: Path to output directory
        level: Aggregation level ('aggregate', 'percentiles', 'occupation')
        frequency: Time aggregation ('annual', 'quarterly', 'monthly', 'all')
        weights: Weighting scheme ('both', 'none', 'occupation', 'task')
        match_cutoff: Similarity threshold for matches
        measure: Type of measure to calculate (default: 'exposed')
        drop_thresh: Threshold for dropping low-exposure categories (optional)
        start_date: Start date filter (optional)
        end_date: End date filter (optional)
        num_percentiles: Number of wage percentiles (default: 20)
        crosswalk: SOC crosswalk year ('2000' or '2019', optional)
        digits: Number of SOC digits for occupation level (default: 6)
    """
    # Load directories from manifest if not specified
    path_to_master = path_to_data + 'tte/'
    if path_to_tech_classifications is None:
        path_to_tech_classifications = path_to_results + 'tech_classification.csv'
        if not os.path.exists(path_to_tech_classifications):
            raise FileNotFoundError("✗ Technology classification file not found, please specify path")
    if path_to_task_classifications is None:
        path_to_task_classifications = path_to_results + 'task_classification.csv'
        if not os.path.exists(path_to_task_classifications):
            raise FileNotFoundError("✗ Task classification file not found, please specify path")
    print(f"\n{'='*60}")
    print(f"TTE Exposure Measurement")
    print(f"{'='*60}")
    print(f"Tech classification: {path_to_tech_classifications}")
    print(f"Task classification: {path_to_task_classifications}")
    print(f"Level: {level}, Frequency: {frequency}, Weights: {weights}")
    print(f"{'='*60}\n")
    
    try:
        # Check if required directories exist
        for path_name, path in [("data", path_to_data), 
                               ("tech classifications", path_to_tech_classifications),
                               ("task classifications", path_to_task_classifications),
                               ("results", path_to_results)]:
            if not os.path.exists(path):
                if path_name == "results":
                    os.makedirs(path, exist_ok=True)
                    print(f"Created results directory at {path}")
                else:
                    raise FileNotFoundError(f"✗ {path_name.capitalize()} directory not found at {path}")
            
        # Load classification files
        task_class_df = try_load_csv(
            f'{path_to_task_classifications}', 
            abort=True)
        tech_class_df = try_load_csv(
            f'{path_to_tech_classifications}', 
            abort=True)
        tech_names = tech_class_df.columns[1:].tolist()
        print(f"Loaded {len(tech_names)} technology categories")
        print(f"Loaded {len(task_class_df['task_cat'].drop_duplicates())} task categories")

        # Load task and occupation files
        onet_tasks = try_load_csv(
            f'{path_to_master}tte_onet_oews/occupations/occupation_tasks.csv', 
            usecols=["occ_id", "task_ref", "version", "task_weight"], 
            abort=True)
        occ_wages = try_load_csv(
            f'{path_to_master}tte_onet_oews/occupations/occupation_data.csv', 
            usecols=["occ_id", "version", "wage_percentile", "occ_weight"], 
            abort=True)
        
        # Create weights file
        occ_weights = onet_tasks.merge(
            occ_wages[["occ_id", "version", "occ_weight"]], 
            how="left", on=["occ_id", "version"])
        occ_wages = occ_wages.drop("occ_weight", axis=1)
        print(f"Loaded {len(onet_tasks)} occupation-task pairs")

        # Determine dates in sample and set temporal aggregation groups
        print("Processing patent dates and setting temporal aggregation...")
        patent_dates = []
        year_dirs = sorted([item for item in os.listdir(path_to_master) 
                           if item.startswith('tte_2')])
        
        if not year_dirs:
            raise FileNotFoundError("✗ No year directories found in master path")
        
        for item in year_dirs:
            # Validate year directory
            year = item.replace('tte_', '').split('.')[0]
            if not year.isdigit() or len(year) != 4:
                print(f"Warning: Skipping invalid year directory: {item}")
                continue

            # Add dates
            try:
                dates = try_load_csv(
                    f'{path_to_master}{item}/patents/patent_text_{year}.csv', 
                    usecols=["patent_id", "date_earliest"])
                if dates is not None:
                    patent_dates.append(dates)
            except Exception as e:
                print(f"Warning: Could not load dates for year {year}: {e}")
                continue
        
        if not patent_dates:
            raise Exception("✗ No patent dates could be loaded")
        
        patent_dates = pd.concat(patent_dates, ignore_index=True)
        patent_dates["agg_date"] = pd.to_datetime(patent_dates["date_earliest"])
        
        # Set frequency-based aggregation
        if frequency == "annual":
            patent_dates["agg_date"] = pd.to_datetime({
                'year': patent_dates["agg_date"].dt.year,
                'month': 7,
                'day': 1
            })
            print("  Aggregating by year")
        elif frequency == "quarterly":
            patent_dates["agg_date"] = pd.to_datetime({
                'year': patent_dates["agg_date"].dt.year,
                'month': patent_dates["agg_date"].dt.quarter * 3 - 2,
                'day': 15
            })
            print("  Aggregating by quarter")
        elif frequency == "monthly":
            patent_dates["agg_date"] = pd.to_datetime({
                'year': patent_dates["agg_date"].dt.year,
                'month': patent_dates["agg_date"].dt.month,
                'day': 15
            })
            print("  Aggregating by month")
        elif frequency == "all":
            patent_dates["agg_date"] = "All Dates"
            print("  Aggregating over all dates")
        else:
            raise ValueError(f"✗ Frequency '{frequency}' not recognized. Must be 'annual', 'quarterly', 'monthly', or 'all'")
        
        # Set occupational aggregation groups
        if level == "aggregate":
            occ_wages["group_var"] = "all occupations"
            print("  Aggregating across all occupations")
        elif level == "percentiles":
            occ_wages["group_var"] = np.ceil(
                occ_wages["wage_percentile"] / int(100 / num_percentiles)
            )
            occ_wages.drop("wage_percentile", axis=1, inplace=True)
            print(f"  Aggregating by {num_percentiles} occupational wage percentiles")
        elif level == "occupation":
            # Apply crosswalk if specified
            soc_crosswalk = None
            if crosswalk is not None:
                if crosswalk == "2000":
                    print("  Loading SOC 2000 crosswalk")
                    soc_crosswalk = try_load_csv(
                        f'{path_to_master}tte_onet_oews/occupations/soc00_crosswalk.csv')
                elif crosswalk == "2019":
                    print("Loading SOC 2019 crosswalk")
                    soc_crosswalk = try_load_csv(
                        f'{path_to_master}tte_onet_oews/occupations/soc19_crosswalk.csv')
                else:
                    print(f"  Warning: Crosswalk year '{crosswalk}' not recognized, ignoring")
            
            if soc_crosswalk is not None:
                occ_wages = occ_wages.merge(
                    soc_crosswalk[["occ_id", "version", "soc_combined"]], 
                    how="left", on=["occ_id", "version"])
                occ_wages["group_var"] = occ_wages["soc_combined"]
                print(f"  Crosswalked occupations into SOC{crosswalk} codes")
            else:
                occ_wages["group_var"] = occ_wages["occ_id"]

            # Aggregate to selected occupation digits
            occdig = digits if digits <= 2 else digits + 1
            occ_wages["group_var"] = occ_wages["group_var"].str[:occdig]
            print(f"  Aggregating by {digits}-digit occupation codes")
        else:
            raise ValueError(f"✗ Aggregation level '{level}' not recognized. Must be 'aggregate', 'percentiles', or 'occupation'")

        # Set occupation weights
        if weights == "both":
            occ_weights["task_weight"] = occ_weights["task_weight"].fillna(1)
            print("  Weighting by employment and task importance")
        elif weights == "none":
            occ_weights["task_weight"] = 1
            occ_weights["occ_weight"] = 1
            print("  No weighting applied")
        elif weights == "occupation":
            occ_weights["task_weight"] = 1
            print("  Weighting by employment only")
        elif weights == "task":
            occ_weights["task_weight"] = occ_weights["task_weight"].fillna(1)
            occ_weights["occ_weight"] = 1
            print("  Weighting by task importance only")
        else:
            raise ValueError(f"✗ Weights '{weights}' not recognized. Must be 'both', 'none', 'occupation', or 'task'")
        
        # Combine task data
        task_data = onet_tasks[["occ_id", "task_ref", "version"]].merge(
            task_class_df, how="left", on=["task_ref"]).merge(
            occ_weights, how="left", on=["occ_id", "task_ref", "version"]).merge(
            occ_wages, how="left", on=["occ_id", "version"])
        task_data["weight"] = task_data["occ_weight"].multiply(task_data["task_weight"])
        
        #### Load and process matches ####
        
        print(f"Loading and processing {len(year_dirs)} annual match files...")
        matches_df = []
        ref_dates = []
        patent_counts = []
        
        for item in year_dirs:
            # Validate year directory
            year = item.replace('tte_', '').split('.')[0]
            if not year.isdigit() or len(year) != 4:
                print(f"Warning: Skipping invalid year directory: {item}")
                continue
            
            try:
                # Load matches
                matches = try_load_csv(
                    f'{path_to_master}{item}/matches_{year}.csv', abort=True)
                file_length = len(matches)
                if match_cutoff is not None:
                    # If specified, filter by match cutoff
                    matches = matches[matches["similarity_score"] >= match_cutoff]
                matches.drop("similarity_score", axis=1, inplace=True)
                
                # Merge with patent dates
                matches = matches.merge(patent_dates, how="left", on=["patent_id"])

                # Generate days in version
                dates = matches[["agg_date", "version", "date_earliest"]].groupby(
                    ["agg_date", "version"], as_index=False).agg(
                    first_date=("date_earliest", "min"),
                    last_date=("date_earliest", "max")).drop_duplicates()
                dates['first_date'] = pd.to_datetime(dates['first_date'])
                dates['last_date'] = pd.to_datetime(dates['last_date'])
                
                # Merge with tech classifications
                matches = matches.merge(
                    tech_class_df, how="left", on=["patent_id"])
                matches["matches"] = 1
                
                # Drop if number of matches is greater than drop_thresh
                if drop_thresh is not None:
                    matches["matches_per_patent"] = matches.groupby(
                        ["patent_id", "version"])["matches"].transform('sum')
                    matches = matches[matches["matches_per_patent"] <= drop_thresh]
                    matches.drop("matches_per_patent", axis=1, inplace=True)
                
                # Compile patent counts
                patents = matches[["patent_id", "agg_date", "task_ref", "version"]].merge(
                    task_class_df, how="left", on=["task_ref"])
                patents = patents.merge(onet_tasks[["occ_id", "task_ref", "version"]],
                                        how="left", on=["task_ref", "version"])
                patents = patents.merge(occ_wages[["occ_id", "version", "group_var"]], 
                                        how="left", on=["occ_id", "version"])
                
                patents["patents_total"] = (patents.groupby(
                    ["agg_date", "patent_id", "group_var"]).cumcount()==0)
                patents_total = patents[["patents_total", "agg_date", "group_var"]].groupby(
                    ["agg_date", "group_var"]).agg('sum').reset_index()
                patents_total["task_cat"] = "all tasks"
                patents["patents_total"] = (patents.groupby(
                    ["agg_date", "patent_id", "group_var", "task_cat"]).cumcount()==0)
                patents_per_task = patents[
                    ["patents_total", "agg_date", "group_var", "task_cat"]].groupby(
                    ["agg_date", "group_var", "task_cat"], as_index=False).agg('sum')

                # Report match counts
                if match_cutoff is None and drop_thresh is None:
                    print(f"  {year}: {len(matches)} total matches")
                else:
                    print(f"  {year}: {len(matches)} matches after filtering, {file_length - len(matches)} dropped")

                # Drop unneeded columns and aggregate by task and date
                matches.drop(["patent_id", "date_earliest"], axis=1, inplace=True)
                matches = matches.groupby(["agg_date", "version", "task_ref"], 
                                            as_index=False).agg('sum')
                        
                # Append dates and match counts
                ref_dates.append(dates)
                matches_df.append(matches)
                patent_counts.append(patents_total)
                patent_counts.append(patents_per_task)
                
            except Exception as e:
                print(f"  Warning: Error processing matches for year {year}: {e}")
                continue
        
        if not matches_df:
            raise Exception("No matches could be loaded and processed")

        #### Aggregate matches by grouping variables ####
            
        # Combine years and aggregate by task and date
        print("Merging all datasets...")
        matches = pd.concat(matches_df, ignore_index=True).groupby(
            ["task_ref", "version", "agg_date"], as_index=False).agg('sum')
        matches.sort_values(["agg_date", "version", "task_ref"], inplace=True)
        patent_counts = pd.concat(patent_counts, ignore_index=True).groupby(
            ["agg_date", "task_cat", "group_var"], as_index=False).agg('sum')

        # Generate date reference file, with weights calculated as days per version
        ref_dates = pd.concat(ref_dates, ignore_index=True).drop_duplicates()
        ref_dates = ref_dates.groupby(["agg_date", "version"], as_index=False).agg(
            first_date=("first_date", "min"), last_date=("last_date", "max"))
        ref_dates["first_date_all"] = pd.to_datetime(
            ref_dates.groupby("agg_date")["first_date"].transform('min'))
        ref_dates["last_date_all"] = pd.to_datetime(
            ref_dates.groupby("agg_date")["last_date"].transform('max'))
        ref_dates["days_total"] = (ref_dates["last_date_all"] - ref_dates[
            "first_date_all"]).dt.days
        ref_dates["days_weight"] = (ref_dates["last_date"] - ref_dates[
            "first_date"]).dt.days.divide(ref_dates["days_total"])
        ref_dates = ref_dates[["agg_date", "version", "days_weight"]
                            ].drop_duplicates()
        ref_dates.sort_values(["agg_date", "version"], inplace=True)

        # Add task categories, occupation codes, weights, and wages
        print("Adding task and occupational data...")
        matches = matches.merge(task_data, how="left", on=["task_ref", "version"])

        #### Calculate task-level exposure ####

        # Calculate ratio of total weighted matches to total tasks by version
        print("Calculating task-level exposure...")
        total_tasks = task_data[["task_ref", "version"]].groupby(
            ["version"]).agg('count').rename(columns={"task_ref": "total_tasks"})
        total_matches = matches[["agg_date", "version", "matches"]].groupby(
            ["agg_date", "version"], as_index=False).agg('sum').rename(
            columns={"matches": "total_matches"})
        total_matches = total_matches.merge(
            total_tasks, how="left", on=["version"])
        total_matches["total_exp"] = total_matches["total_matches"].divide(
            total_matches["total_tasks"])
        matches = matches.merge(
            total_matches[["agg_date", "version", "total_exp"]], how="left", 
            on=["agg_date", "version"])

        # Calculate exposure as matches/task > total matches/total tasks
        tech_names = tech_names + ["allpatents"]
        matches["allpatents"] = matches["matches"]
        for tech in tech_names:
            if measure == "exposed":
                matches[tech + "_exp"] = (matches[tech]>matches["total_exp"]).astype(int)
            if measure == "counts":
                matches[tech + "_exp"] = matches[tech]

        #### Create master task file ####
        
        # Merge date list with task data
        master = ref_dates[["version"]].drop_duplicates()
        master = master.merge(task_data, on=["version"], how="left")

        # Variable for total tasks
        master["tasks"] =  master.groupby(
            ["version", "group_var", "occ_id", "task_ref"]).cumcount()==0
        
        # Add dates
        master = ref_dates[["agg_date", "version"]].merge(master, how="left", 
                                                        on=["version"])
        

        #### Weighted sum across tasks within task category ####
        
        # Weighted aggregation of tasks
        print("Aggregating by level, frequency, task category...")
        master.rename(columns={"weight": "tasks_weighted"}, inplace=True)
        master.drop(["occ_id", "task_ref", "occ_weight", "task_weight"], 
                    axis=1, inplace=True)
        master = master.groupby(["agg_date", "version", "group_var", "task_cat"], 
                                as_index=False).agg('sum')

        # Weighted aggregation of matches
        matches[[tech + "_exp" for tech in tech_names]] = matches[
            [tech + "_exp" for tech in tech_names]].multiply(
            matches["weight"], axis="index")
        matches.drop(["occ_id", "task_ref", "occ_weight", "task_weight", 
                    "weight", "total_exp"], axis=1, inplace=True)
        matches = matches.groupby(["agg_date", "version", "group_var", "task_cat"], 
                                as_index=False).agg('sum')

        #### Calculate exposure measures ####

        # Generate "all tasks" category and calculate aggregate statistics
        print("Calculating exposure indices...")
        master_all = master.drop(["task_cat"], axis=1).groupby(
            ["agg_date", "version", "group_var"], as_index=False).agg('sum')
        master_all["task_cat"] = "all tasks"
        master_all["tasks_group"] = master_all["tasks_weighted"]
        master_all["tasks_aggregate"] = master_all[
            ["agg_date", "version", "tasks_weighted"]].groupby(
                ["agg_date", "version"]).transform('sum')
        matches_all = matches.drop(["task_cat"], axis=1).groupby(
            ["agg_date", "version", "group_var"], as_index=False).agg('sum')
        matches_all["task_cat"] = "all tasks"
        matches_all["allpatents_aggregate"] = matches_all[
            ["agg_date", "version", "allpatents"]].groupby(
                ["agg_date", "version"]).transform('sum')
        matches_all["allpatents_exp_aggregate"] = matches_all[
            ["agg_date", "version", "allpatents_exp"]].groupby(
                ["agg_date", "version"]).transform('sum')
        
        # Calculate group and aggregate statistics for non-aggregated task categories
        master["tasks_group"] = master[
            ["agg_date", "version", "group_var", "tasks_weighted"]].groupby(
                ["agg_date", "version",  "group_var"]).transform('sum')
        master["tasks_aggregate"] = master[
            ["agg_date", "version", "tasks_weighted"]].groupby(
                ["agg_date", "version"]).transform('sum')
        matches["allpatents_aggregate"] = matches[
            ["agg_date", "version", "allpatents"]].groupby(
                ["agg_date", "version"]).transform('sum')
        matches["allpatents_exp_aggregate"] = matches[
            ["agg_date", "version", "allpatents_exp"]].groupby(
                ["agg_date", "version"]).transform('sum')
        master = pd.concat([master, master_all], axis=0).sort_values(
            ["agg_date", "version", "group_var", "task_cat"])
        matches = pd.concat([matches, matches_all], axis=0).sort_values(
            ["agg_date", "version", "group_var", "task_cat"])
        master = master.merge(matches, how="left", 
                            on=["agg_date", "version", "group_var", "task_cat"])

        # Calculate normalized exposure measures
        for tech in tech_names:
            # Populate dates/groups/task categories with no matches
            master["matches"] = master["matches"].astype(float).fillna(0)
            master[tech] = master[tech].astype(float).fillna(0)
            master[tech + "_exp"] = master[tech + "_exp"].astype(float).fillna(0)

            # Calculate exposure as weighted % of tasks exposed
            master[tech + "_exp"] = master[tech + "_exp"].divide(
                master["tasks_weighted"])
            master["agg_exp"] = master["allpatents_exp_aggregate"].divide(
                master["tasks_aggregate"])
            master[tech + "_exp"] = master[tech + "_exp"].divide(
                master["agg_exp"])

        # Keep only desired columns
        val_names = [tech for tech in tech_names] + [
            tech + "_exp" for tech in tech_names]
        master["task_share"] = master["tasks_weighted"].divide(master["tasks_group"])
        master = master[
            ["agg_date", "version", "group_var", "task_cat", "task_share", 
            "tasks", "matches", "tasks_weighted"] + val_names]

        # If multiple versions per date, aggregate based on days of each version
        print("Aggregating across O*NET versions...")
        master = master.merge(ref_dates, how="left", on=["agg_date", "version"])
        master = master.groupby(
            ["agg_date", "group_var", "task_cat"], as_index=False).apply(
                lambda x: pd.Series({
                    "task_share": np.average(x["task_share"], 
                                            weights=x['days_weight']),
                    "tasks": np.round(np.average(x["tasks"], weights=x['days_weight'])),
                    "matches": np.sum(x["matches"]),
                    **{tech + "_count": np.sum(x[tech]) for tech in tech_names},
                    **{tech + "_exp": np.average(
                        x[tech + "_exp"], weights=x['days_weight']
                        ) for tech in tech_names},
            })).reset_index(drop=True) 

        # Merge with patent counts
        master = master.merge(
            patent_counts, how="left", on=["agg_date", "group_var", "task_cat"])
        col_names = master.columns.tolist()
        master = master[['agg_date', 'group_var', 'task_cat', 'task_share', 
                         'tasks', 'matches', 'patents_total'] +
                         [name for name in col_names if name not in
                          ['agg_date', 'group_var', 'task_cat', 'task_share',
                              'tasks', 'matches', 'patents_total']]]
           
        # Add rows with all zeros for dates/groups/task categories with no tasks
        print("Populating cells with no matches...")
        all_combinations = list(itertools.product(
            master['agg_date'].unique(), 
            master['group_var'].unique(), 
            master['task_cat'].unique()))
        all_combinations = pd.DataFrame(
            all_combinations, columns=['agg_date', 'group_var', 'task_cat'])
        all_combinations = all_combinations.sort_values(
            ['agg_date', 'group_var', 'task_cat'])
        master = all_combinations.merge(
            master, how='left', on=['agg_date', 'group_var', 'task_cat'])
        master = master.fillna(0)

        # Apply date filters if specified
        if start_date is not None:
            print(f"Filtering to dates on or after {start_date}...")
            master = master[
                pd.to_datetime(master['agg_date']) >= pd.to_datetime(start_date)]

        if end_date is not None:
            print(f"Filtering to dates up to and including {end_date}...")
            master = master[
                pd.to_datetime(master['agg_date']) <= pd.to_datetime(end_date)]
            
        # If occupation level, remove occupation-years with no tasks
        if level == "occupation":
            print("Removing empty occupation-date combinations...")
            master["occupation_tasks"] = master.groupby(
                ["group_var", "agg_date"])["tasks"].transform('sum')
            master = master[master["occupation_tasks"] > 0]
            master = master.drop("occupation_tasks", axis=1)

        # Rename columns
        rename_dict = {
            'agg_date': 'date_aggregated',
            'task_cat': 'task_category',
            'tasks': 'task_count',
            'matches': 'match_count',
            'patents_total': 'patent_count',
            'allpatents_exp': 'total_exp'
        }
        master = master.rename(columns=rename_dict)
        
        if level == "aggregate":
            master.drop("group_var", axis=1, inplace=True)
        elif level == "percentiles":
            master.rename(columns={"group_var": "wage_quantile"}, inplace=True)
        elif level == "occupation":
            master.rename(columns={"group_var": "SOC_code"}, inplace=True)
            level = f"{digits}dSOC"

        # Export results
        file_dest = f'{path_to_results}exposure_{level}_{frequency}'
        if drop_thresh is not None:
            file_dest += f'_drop{drop_thresh}'
        master.to_csv(f'{file_dest}.csv', index=False)
        
        print(f"Total rows in exposure file: {len(master)}")
        print(f"Results saved to {file_dest}.csv")

        print(f"\n{'='*60}")
        print(f"✓ Exposure measurement complete")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Error during exposure measurement: {e}")
        raise
