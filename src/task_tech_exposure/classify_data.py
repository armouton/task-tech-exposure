# IMPORT PACKAGES
import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


# ================== HELPER FUNCTIONS =================

# DETERMINE DEVICE FOR TORCH OPERATIONS
def get_device():
    """Determine the best available device for PyTorch operations."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

device = get_device()

# CALCULATE SIMILARITY SCORES
def sim_scores(emb1, emb2):
    """Calculate cosine similarity between two embedding tensors."""
    # Normalize embeddings and compute cosine similarity
    emb1_norm = torch.norm(emb1, dim=1, keepdim=True)
    emb2_norm = torch.norm(emb2, dim=1, keepdim=True)
    return torch.mm(emb1, emb2.t()) / (emb1_norm * emb2_norm.t())

# INITIALIZE MODEL AND CHECK DIRECTORIES
def initialize_model(path_to_master, path_to_descriptions, model):
    """
    Initialize SBERT model and verify required directories exist.
    
    Args:
        path_to_master: Path to master data directory
        path_to_descriptions: Path to category descriptions
        model: Path to SBERT model
        
    Returns:
        SentenceTransformer model instance
        
    Raises:
        FileNotFoundError: If SBERT model or required directories are missing
    """
    # Load SBERT model
    if not os.path.exists(model):
        raise FileNotFoundError(f"ERROR: SBERT model not found at {model}")

    try:
        model_name = os.path.basename(model).lower()
        if 'qwen3' in model_name:
            model = SentenceTransformer(model, device=device,
                                        tokenizer_kwargs={"fix_mistral_regex": True})
        else:
            model = SentenceTransformer(model, device=device)
    except Exception as e:
        raise Exception(f"ERROR: Failed to load SBERT model: {e}") from e

    # Check for required directories
    for path_name, path in [("master", path_to_master), ("descriptions", path_to_descriptions)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path_name.capitalize()} directory not found at {path}")

    return model

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
            raise FileNotFoundError(f"ERROR: Required file not found at {path}") from e
        else:
            print(f"Warning: File not found at {path}, skipping")
            return None
    except Exception as e:
        error_msg = f"ERROR: Error loading CSV from {path}: {e}"
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
        Exception: If loading fails
    """
    try:
        data = np.load(path)
        return data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"ERROR: Required file not found at {path}") from e
    except Exception as e:
        raise Exception(f"ERROR: Failed to load numpy array from {path}: {e}") from e


# ================== CLASSIFY DATA =================

# CLASSIFY PATENTS
def classify_patents(path_to_data, path_to_results,
                     path_to_output=None, path_to_descriptions=None,
                     model=None, cutoff=None, groups=None,
                     priority="order"):
    """
    Classify patents into technology categories based on semantic similarity.
    
    Args:
        path_to_data: Path to data directory
        path_to_results: Path to output directory
        path_to_output: Path to output file (optional)
        path_to_descriptions: Path to category descriptions (optional)
        model: Path to SBERT model
        cutoff: Similarity threshold for classification
        groups: List of mutually exclusive technology groups (optional)
        priority: Priority method for groups - "order" or "score" (default: "order")
    """
    # Load directories from manifest if not specified
    path_to_master = path_to_data + 'tte/'
    if path_to_output is None:
        path_to_output = path_to_results + 'tech_classification.csv'
    if path_to_descriptions is None:
        path_to_descriptions = path_to_master + 'tte_models/category_descriptions/tech_categories.csv'
        if not os.path.exists(path_to_descriptions):
            raise FileNotFoundError("ERROR: Technology category descriptions file not found, please specify path")
    # Load manifest to fill defaults and detect model mismatches
    manifest = {}
    try:
        with open(path_to_master + 'dataset_manifest.json', 'r') as f:
            manifest = json.load(f)
    except FileNotFoundError:
        if cutoff is None or model is None:
            raise FileNotFoundError(
                "ERROR: Dataset manifest not found, please specify SBERT model and cutoff")
    cutoff = manifest.get("tech_cutoff") if cutoff is None else cutoff
    if model is None:
        model = path_to_master + 'tte_models/' + manifest.get("tech_model")
    elif manifest.get("tech_model"):
        if os.path.basename(model) != manifest.get("tech_model"):
            print(f"Warning: Specified model differs from the manifest default "
                  f"('{manifest.get('tech_model')}'). Pre-stored patent embeddings"
                  f" were generated with the manifest model; if embedding "
                  f"dimensions differ, live encodings will be silently truncated "
                  f"and similarity scores may be unreliable. Run embed_data() to "
                  f"regenerate patent embeddings with the new model.")

    print(f"\n{'='*60}")
    print(f"TTE Patent Classification")
    print(f"{'='*60}")
    print(f"Category file: {path_to_descriptions}")
    print(f"SBERT model: {model}")
    print(f"Cutoff: {cutoff}")
    print(f"Classification method: {f'Mutually exclusive groups with {priority}' if groups else 'All matching categories'}")
    print(f"Using device: {device}")
    print(f"{'='*60}\n")
    
    try:
        # Initialize model and check directories
        model = initialize_model(path_to_master, path_to_descriptions, model)

        # Load and embed tech categories
        category_path = f'{path_to_descriptions}'
        tech_class = try_load_csv(category_path, abort=True)
        
        if 'name' not in tech_class.columns or 'gpt_description' not in tech_class.columns:
            raise ValueError("ERROR: Category file must contain 'name' and 'gpt_description' columns")
        
        tech_names = tech_class['name'].tolist()
        print(f"Encoded {len(tech_names)} technology categories")
        tech_embed_full = model.encode(tech_class['gpt_description'].tolist(),
                                       convert_to_tensor=True)

        # Resolve cutoff to a per-category list
        if cutoff is None or (isinstance(cutoff, list) and len(cutoff) != len(tech_names)):
            # No usable cutoff yet — try validation thresholds file first
            thresholds_path = f'{path_to_results}tte_samples/tech_thresholds.csv'
            resolved = False
            if os.path.exists(thresholds_path):
                try:
                    thresh_df = pd.read_csv(thresholds_path)
                    thresh_df = thresh_df[thresh_df['name'] != 'avg. threshold']
                    if ('marg_prec_0.5' in thresh_df.columns and
                            all(n in thresh_df['name'].values for n in tech_names)):
                        cutoff = [
                            float(thresh_df.loc[thresh_df['name'] == n,
                                                'marg_prec_0.5'].values[0])
                            for n in tech_names
                        ]
                        print(f"Note: Cutoff resolved from {thresholds_path} (marg_prec_0.5).")
                        resolved = True
                except Exception:
                    pass
            if not resolved:
                if isinstance(cutoff, list):
                    fallback = max(cutoff)
                    print(f"Warning: Cutoff list length ({len(cutoff)}) does not match category "
                          f"count ({len(tech_names)}). Using max manifest cutoff ({fallback:.4f}) "
                          f"as a conservative default. Run validate_model() to obtain "
                          f"category-specific thresholds.")
                    cutoff = [fallback] * len(tech_names)
                else:
                    raise ValueError(
                        "ERROR: No cutoff specified and no default found in dataset manifest. "
                        "Specify a cutoff value, or run validate_model() to generate "
                        "category-specific thresholds.")
        elif isinstance(cutoff, (int, float)):
            cutoff = [cutoff] * len(tech_names)

        # Process groups if provided
        if groups is not None:
            # Validate groups
            for i, group in enumerate(groups):
                if not isinstance(group, (list, tuple)):
                    raise ValueError(f"ERROR: Group {i} must be a list or tuple")
                if any(idx >= len(tech_names) or idx < 0 for idx in group):
                    raise ValueError(f"ERROR: Group {i} contains invalid indices (must be 0-{len(tech_names)-1})")

        # Loop over years to save memory
        patents = []
        year_dirs = sorted([item for item in os.listdir(path_to_master)
                           if item.startswith('tte_2')])

        if not year_dirs:
            raise FileNotFoundError("No year directories found in master path")

        print(f"Classifying {len(year_dirs)} year directories...")

        for item in year_dirs:
            # Validate year directory
            year = item.replace('tte_', '').split('.')[0]
            if not year.isdigit() or len(year) != 4:
                print(f"Warning: Skipping invalid year directory: {item}")
                continue

            # Load patent data and embeddings
            embed_path = f'{path_to_master}{item}/patents/patent_embed_{year}.npy'
            text_path = f'{path_to_master}{item}/patents/patent_text_{year}.csv'

            try:
                pat_embed = torch.tensor(try_load_npy(embed_path), device=device)
                patents_year = try_load_csv(text_path,
                                           usecols=["patent_id", "abstract", "date_earliest"],
                                           abort=True)

                # Truncate tech embeddings to match dimensionality of pre-embedded patents (supports MRL)
                embed_dim = pat_embed.shape[1]
                tech_embed = tech_embed_full[:, :embed_dim]

                # Obtain similarity scores
                similarity_scores = sim_scores(pat_embed, tech_embed).cpu().numpy()

                # If no groups, classify patents into all matching categories
                if groups is None:
                    for i, tech_name in enumerate(tech_names):
                        patents_year[tech_name] = (similarity_scores[:, i] >= 
                                                   cutoff[i]).astype(int)

                # Otherwise, classify patents into mutually exclusive groups
                else:
                    for group in groups:
                        # If score priority, assign patent to highest score
                        if priority == "score":
                            group_scores = similarity_scores[:, group]
                            top_techs = np.argmax(group_scores, axis=1)
                            for i, tech in enumerate(group):
                                patents_year[tech_names[tech]] = (
                                    (top_techs == i).astype(int) *
                                    (group_scores[:, i] >= cutoff[tech])
                                )
                            
                        # If order priority, assign patent to first match
                        elif priority == "order":
                            tech_match = np.zeros(len(patents_year)).astype(int)
                            for tech in group:
                                current_match = (similarity_scores[:, tech] >= cutoff[tech]).astype(int)
                                current_match = current_match * (1 - tech_match)
                                patents_year[tech_names[tech]] = current_match
                                tech_match = tech_match + current_match
                        else:
                            raise ValueError(f"  ERROR: Invalid priority: {priority}. Must be 'order' or 'score'")
                
                # Store results
                patents.append(patents_year)
                print(f"  {year}: {len(patents_year)} patents")
                
            except Exception as e:
                print(f"Warning: Could not process year {year}: {e}")
                continue

        if not patents:
            raise Exception("ERROR: No patents were successfully classified")
        
        patents = pd.concat(patents, ignore_index=True)

        # Save classification file
        patents = patents[["patent_id"] + tech_names]
        output_path = f'{path_to_output}'
        patents.to_csv(output_path, index=False)
        print(f"Total patents classified: {len(patents)}")
        print(f"Technology classification counts:")
        for tech_name in tech_names:
            count = patents[tech_name].sum()
            print(f"  {tech_name}: {count}")
        print(f"Results saved to {output_path}")
        print(f"\n{'='*60}")
        print(f"OK Classification complete")
        print(f"{'='*60}\n")

        
    except Exception as e:
        print(f"Error during patent classification: {e}")
        raise

# CLASSIFY TASK STATEMENTS
def classify_tasks(path_to_data, path_to_results, path_to_output=None,
                   path_to_descriptions=None, model=None):
    """
    Classify O*NET task statements into task categories based on semantic similarity.
    
    Args:
        path_to_data: Path to data directory
        path_to_results: Path to output directory
        path_to_output: Path to output file (optional)
        path_to_descriptions: Path to category descriptions (optional)
        model: Path to SBERT model
    """
    # Load directories from manifest if not specified
    path_to_master = path_to_data + 'tte/'
    if path_to_output is None:
        path_to_output = path_to_results + 'task_classification.csv'
    if path_to_descriptions is None:
        path_to_descriptions = path_to_master + 'tte_models/category_descriptions/task_categories.csv'
        if not os.path.exists(path_to_descriptions):
            raise FileNotFoundError("ERROR: Task category descriptions file not found, please specify path")
    # Load manifest to fill defaults and detect model mismatches
    manifest = {}
    try:
        with open(path_to_master + 'dataset_manifest.json', 'r') as f:
            manifest = json.load(f)
    except FileNotFoundError:
        if model is None:
            raise FileNotFoundError(
                "ERROR: Dataset manifest not found, please specify SBERT model")
    if model is None:
        model = path_to_master + 'tte_models/' + manifest.get("task_model")
    elif manifest.get("task_model"):
        if os.path.basename(model) != manifest.get("task_model"):
            print(f"Warning: Specified model differs from the manifest default "
                  f"('{manifest.get('task_model')}'). Pre-stored task embeddings"
                  f" were generated with the manifest model; if embedding "
                  f"dimensions differ, live encodings will be silently truncated "
                  f"and similarity scores may be unreliable. Run embed_data() to "
                  f"regenerate task embeddings with the new model.")

    print(f"\n{'='*60}")
    print(f"TTE Task Classification")
    print(f"{'='*60}")
    print(f"Category file: {path_to_descriptions}")
    print(f"SBERT model: {model}")
    print(f"{'='*60}\n")
    
    try:
        # Initialize model and check directories
        model = initialize_model(path_to_master, path_to_descriptions, model)

        # Load the descriptions
        desc_path = f'{path_to_descriptions}'
        task_desc = try_load_csv(desc_path, 
                                usecols=['gpt_description', 'name'], 
                                abort=True)
        
        if 'name' not in task_desc.columns or 'gpt_description' not in task_desc.columns:
            raise ValueError("ERROR: Category file must contain 'name' and 'gpt_description' columns")

        # Load O*NET files
        onet_path = f'{path_to_master}tte_onet_oews/tasks/task_text.csv'
        embed_path = f'{path_to_master}tte_onet_oews/tasks/task_embed.npy'
        onet = try_load_csv(onet_path, 
                           usecols=["task_ref", "task"], 
                           abort=True)
        onet_embed = torch.tensor(try_load_npy(embed_path), device=device)
        
        print(f"Loaded {len(onet)} task statements")

        # Load and embed task categories; truncate to match dimensionality of pre-embedded tasks (supports MRL)
        task_dict = dict(zip(task_desc.index, task_desc['name']))
        print(f"Encoded {len(task_desc)} task categories")
        embed_dim = onet_embed.shape[1]
        task_embed = model.encode(task_desc['gpt_description'].tolist(),
                                  convert_to_tensor=True)
        task_embed = task_embed[:, :embed_dim]

        # Classify tasks
        similarity_scores = sim_scores(onet_embed, task_embed).cpu().numpy()
        onet["task_cat"] = np.argmax(similarity_scores, axis=1)
        onet["task_cat"] = onet["task_cat"].map(task_dict)

        # Save classification file
        onet = onet[["task_ref", "task_cat"]]
        output_path = f'{path_to_output}'
        onet.to_csv(output_path, index=False)
        
        print(f"Total tasks classified: {len(onet)}")
        print(f"Task category distribution:")
        for category, count in onet['task_cat'].value_counts().items():
            print(f"  {category}: {count}")
        print(f"Results saved to {output_path}")

        print(f"\n{'='*60}")
        print(f"OK Classification complete")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Error during task classification: {e}")
        raise
