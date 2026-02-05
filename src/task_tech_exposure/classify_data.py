# IMPORT PACKAGES
import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


# ================== CLASSIFY DATA =================

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
def initialize_model(path_to_master, path_to_descriptions, sbert_model):
    """
    Initialize SBERT model and verify required directories exist.
    
    Args:
        path_to_master: Path to master data directory
        path_to_descriptions: Path to category descriptions
        sbert_model: Path to SBERT model
        
    Returns:
        SentenceTransformer model instance
        
    Raises:
        FileNotFoundError: If SBERT model or required directories are missing
    """
    # Load SBERT model
    if not os.path.exists(sbert_model):
        raise FileNotFoundError(f"✗ SBERT model not found at {sbert_model}")
    
    try:
        model = SentenceTransformer(sbert_model, device=device)
    except Exception as e:
        raise Exception(f"✗ Failed to load SBERT model: {e}") from e

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
        Exception: If loading fails
    """
    try:
        data = np.load(path)
        return data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"✗ Required file not found at {path}") from e
    except Exception as e:
        raise Exception(f"✗ Failed to load numpy array from {path}: {e}") from e


# CLASSIFY PATENTS
def classify_patents(path_to_data, path_to_results,
                     path_to_output=None, path_to_descriptions=None,
                     sbert_tech=None, cutoff=None, tech_groups=None,
                     tech_priority="order"):
    """
    Classify patents into technology categories based on semantic similarity.
    
    Args:
        path_to_data: Path to data directory
        path_to_results: Path to output directory
        path_to_output: Path to output file (optional)
        path_to_descriptions: Path to category descriptions (optional)
        sbert_tech: Path to SBERT model
        cutoff: Similarity threshold for classification
        tech_groups: List of mutually exclusive technology groups (optional)
        tech_priority: Priority method for groups - "order" or "score" (default: "order")
    """
    # Load directories from manifest if not specified
    path_to_master = path_to_data + 'tte/'
    if path_to_output is None:
        path_to_output = path_to_results + 'tech_classification.csv'
    if path_to_descriptions is None:
        path_to_descriptions = path_to_master + 'tte_models/category_descriptions/tech_categories.csv'
        if not os.path.exists(path_to_descriptions):
            raise FileNotFoundError("✗ Technology category descriptions file not found, please specify path")
    if cutoff is None or sbert_tech is None:
        """Load local manifest if it exists."""
        try:
            with open(path_to_master + 'dataset_manifest.json', 'r') as f:
                manifest = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError("✗ Dataset manifest not found, please specify SBERT model and cutoff") from e
        
        cutoff = manifest.get("tech_cutoff") if cutoff is None else cutoff
        sbert_tech = path_to_master + 'tte_models/' + manifest.get("tech_model") if sbert_tech is None else sbert_tech

    print(f"\n{'='*60}")
    print(f"TTE Patent Classification")
    print(f"{'='*60}")
    print(f"Category file: {path_to_descriptions}")
    print(f"SBERT model: {sbert_tech}")
    print(f"Cutoff: {cutoff}")
    print(f"Classification method: {f'Mutually exclusive groups with {tech_priority}' if tech_groups else 'All matching categories'}")
    print(f"Using device: {device}")
    print(f"{'='*60}\n")
    
    try:
        # Initialize model and check directories
        model = initialize_model(path_to_master, path_to_descriptions, 
                                 sbert_tech)

        # Load and embed tech categories
        category_path = f'{path_to_descriptions}'
        tech_class = try_load_csv(category_path, abort=True)
        
        if 'name' not in tech_class.columns or 'gpt_description' not in tech_class.columns:
            raise ValueError("✗ Category file must contain 'name' and 'gpt_description' columns")
        
        tech_names = tech_class['name'].tolist()
        print(f"Encoded {len(tech_names)} technology categories")
        tech_embed = model.encode(tech_class['gpt_description'].tolist(), 
                                  convert_to_tensor=True)

        # Process tech_groups if provided
        if tech_groups is not None:
            # Validate tech_groups
            for i, group in enumerate(tech_groups):
                if not isinstance(group, (list, tuple)):
                    raise ValueError(f"✗ Group {i} must be a list or tuple")
                if any(idx >= len(tech_names) or idx < 0 for idx in group):
                    raise ValueError(f"✗ Group {i} contains invalid indices (must be 0-{len(tech_names)-1})")

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
                
                # Obtain similarity scores
                similarity_scores = np.array(sim_scores(pat_embed, tech_embed))

                # If no groups, classify patents into all matching categories
                if tech_groups is None:
                    for i, tech_name in enumerate(tech_names):
                        patents_year[tech_name] = (similarity_scores[:, i] >= cutoff).astype(int)

                # Otherwise, classify patents into mutually exclusive groups
                else:
                    for group in tech_groups:
                        # If score priority, assign patent to highest score
                        if tech_priority == "score":
                            group_scores = similarity_scores[:, group]
                            top_techs = np.argmax(group_scores, axis=1)
                            for i, tech in enumerate(group):
                                patents_year[tech_names[tech]] = (
                                    (top_techs == i).astype(int) * 
                                    (group_scores[:, i] >= cutoff)
                                )
                            
                        # If order priority, assign patent to first match
                        elif tech_priority == "order":
                            tech_match = np.zeros(len(patents_year)).astype(int)
                            for tech in group:
                                current_match = (similarity_scores[:, tech] >= cutoff).astype(int)
                                current_match = current_match * (1 - tech_match)
                                patents_year[tech_names[tech]] = current_match
                                tech_match = tech_match + current_match
                        else:
                            raise ValueError(f"  ✗ Invalid tech_priority: {tech_priority}. Must be 'order' or 'score'")
                
                # Store results
                patents.append(patents_year)
                print(f"  {year}: {len(patents_year)} patents")
                
            except Exception as e:
                print(f"Warning: Could not process year {year}: {e}")
                continue

        if not patents:
            raise Exception("✗ No patents were successfully classified")
        
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
        print(f"✓ Classification complete")
        print(f"{'='*60}\n")

        
    except Exception as e:
        print(f"Error during patent classification: {e}")
        raise


# CLASSIFY TASK STATEMENTS
def classify_tasks(path_to_data, path_to_results, path_to_output=None,
                   path_to_descriptions=None, sbert_task=None):
    """
    Classify O*NET task statements into task categories based on semantic similarity.
    
    Args:
        path_to_data: Path to data directory
        path_to_results: Path to output directory
        path_to_output: Path to output file (optional)
        path_to_descriptions: Path to category descriptions (optional)
        sbert_task: Path to SBERT model
    """
    # Load directories from manifest if not specified
    path_to_master = path_to_data + 'tte/'
    if path_to_output is None:
        path_to_output = path_to_results + 'task_classification.csv'
    if path_to_descriptions is None:
        path_to_descriptions = path_to_master + 'tte_models/category_descriptions/task_categories.csv'
        if not os.path.exists(path_to_descriptions):
            raise FileNotFoundError("✗ Task category descriptions file not found, please specify path")
    if sbert_task is None:
        """Load local manifest if it exists."""
        try:
            with open(path_to_master + 'dataset_manifest.json', 'r') as f:
                manifest = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError("✗ Dataset manifest not found, please specify SBERT model") from e
        
        sbert_task = path_to_master + 'tte_models/' + manifest.get("task_model") if sbert_task is None else sbert_task

    print(f"\n{'='*60}")
    print(f"TTE Task Classification")
    print(f"{'='*60}")
    print(f"Category file: {path_to_descriptions}")
    print(f"SBERT model: {sbert_task}")
    print(f"{'='*60}\n")
    
    try:
        # Initialize model and check directories
        model = initialize_model(path_to_master, path_to_descriptions, 
                                 sbert_task)

        # Load the descriptions
        desc_path = f'{path_to_descriptions}'
        task_desc = try_load_csv(desc_path, 
                                usecols=['gpt_description', 'name'], 
                                abort=True)
        
        if 'name' not in task_desc.columns or 'gpt_description' not in task_desc.columns:
            raise ValueError("✗ Category file must contain 'name' and 'gpt_description' columns")

        # Load O*NET files
        onet_path = f'{path_to_master}tte_onet_oews/tasks/task_text.csv'
        embed_path = f'{path_to_master}tte_onet_oews/tasks/task_embed.npy'
        onet = try_load_csv(onet_path, 
                           usecols=["task_ref", "task"], 
                           abort=True)
        onet_embed = torch.tensor(try_load_npy(embed_path), device=device)
        
        print(f"Loaded {len(onet)} task statements")

        # Load and embed task categories
        task_dict = dict(zip(task_desc.index, task_desc['name']))
        print(f"Encoded {len(task_desc)} task categories")
        task_embed = model.encode(task_desc['gpt_description'].tolist(), 
                                  convert_to_tensor=True)

        # Classify tasks
        similarity_scores = sim_scores(onet_embed, task_embed)
        onet["task_cat"] = np.argmax(np.array(similarity_scores), axis=1)
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
        print(f"✓ Classification complete")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Error during task classification: {e}")
        raise
