# IMPORT PACKAGES
import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

TECH_INSTRUCT = (
    "Instruct: Given a patent abstract, retrieve technology categories that "
    "describe the technologies used in the invention.\nQuery: "
)
TASK_INSTRUCT = (
    "Instruct: Given an occupational task description, find the task category "
    "that best describes the primary activity performed in this task.\nQuery: "
)


# ================== HELPER FUNCTIONS =================

# DETERMINE DEVICE FOR TORCH OPERATIONS
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

# LOAD EMBEDDING MODEL
def load_model(model_path, device):
    """Load SentenceTransformer with Qwen3-specific kwargs when needed."""
    model_name = os.path.basename(model_path).lower()
    if 'qwen3' in model_name:
        return SentenceTransformer(
            model_path, device=device,
            tokenizer_kwargs={"fix_mistral_regex": True})
    return SentenceTransformer(model_path, device=device)

# ENCODE TEXTS WITH OPTIONAL INSTRUCT PROMPT AND MRL TRUNCATION
def encode_texts(model, texts, instruct, use_instruct, embed_dim):
    """Encode texts with optional instruct prefix and MRL truncation."""
    kwargs = dict(convert_to_numpy=True, normalize_embeddings=True)
    if use_instruct and instruct is not None:
        kwargs['prompt'] = instruct
    embs = model.encode(texts, **kwargs)
    if embed_dim is not None and embs.shape[1] > embed_dim:
        embs = embs[:, :embed_dim]
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


# ================== EMBED DATA =================

# MAIN FUNCTION
def embed_data(path_to_data, embed_type='both',
               tech_model=None, task_model=None,
               embed_dim=None, use_instruct=True, force=False):
    """
    Re-embed patent abstracts and/or O*NET task statements for classification.

    Regenerates the .npy classification embedding files used by classify_patents()
    and classify_tasks(). Use this when switching to a custom-trained model —
    the default embeddings shipped with the dataset were produced by the
    technology and task classification models specified in dataset_manifest.json.
    Patents are encoded using the technology model with the tech instruct prompt;
    tasks are encoded using the task model with the task instruct prompt.

    Args:
        path_to_data: Path to directory containing the downloaded dataset.
        embed_type: What to re-embed. 'patents' processes all year files using the
                    tech model; 'tasks' processes the O*NET task file using the task
                    model; 'both' does both. Default 'both'.
        tech_model: Path to the technology classification model for patent embedding.
                    Defaults to the tech_model in dataset_manifest.json. Ignored
                    when embed_type='tasks'.
        task_model: Path to the task classification model for task embedding.
                    Defaults to the task_model in dataset_manifest.json. Ignored
                    when embed_type='patents'.
        embed_dim: Embedding dimension to store. If less than the model's native
                   dimension, MRL truncation is applied. Defaults to
                   class_embedding_dim_mrl in dataset_manifest.json.
        use_instruct: Whether to apply asymmetric instruct prompts to the query side.
                      Default True.
        force: If True, re-embeds files that already exist. Default False.
    """
    if embed_type not in ('patents', 'tasks', 'both'):
        raise ValueError(
            f"ERROR: embed_type must be 'patents', 'tasks', or 'both', "
            f"got '{embed_type}'")

    path_to_master = path_to_data + 'tte/'
    device = get_device()

    print(f"\n{'='*60}")
    print(f"TTE Re-embedding — {embed_type}")
    print(f"{'='*60}")

    # Load manifest
    manifest_path = path_to_master + 'dataset_manifest.json'
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"ERROR: Dataset manifest not found at {manifest_path}. "
            f"Run download_data() first.")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Resolve embedding dimension
    if embed_dim is None:
        embed_dim = manifest.get('class_embedding_dim_mrl')

    # Resolve model paths
    if embed_type in ('patents', 'both'):
        if tech_model is None:
            tech_model_name = manifest.get('tech_model')
            if not tech_model_name:
                raise ValueError(
                    "ERROR: No 'tech_model' entry in dataset_manifest.json.")
            tech_model = path_to_master + 'tte_models/' + tech_model_name
        if not os.path.exists(tech_model):
            raise FileNotFoundError(
                f"ERROR: Tech model not found at {tech_model}.")

    if embed_type in ('tasks', 'both'):
        if task_model is None:
            task_model_name = manifest.get('task_model')
            if not task_model_name:
                raise ValueError(
                    "ERROR: No 'task_model' entry in dataset_manifest.json.")
            task_model = path_to_master + 'tte_models/' + task_model_name
        if not os.path.exists(task_model):
            raise FileNotFoundError(
                f"ERROR: Task model not found at {task_model}.")

    if embed_type == 'patents':
        print(f"Tech model: {tech_model}")
    elif embed_type == 'tasks':
        print(f"Task model: {task_model}")
    else:
        print(f"Tech model: {tech_model}")
        print(f"Task model: {task_model}")
    print(f"Embed dim:  {embed_dim if embed_dim is not None else 'native'}")
    print(f"Instruct:   {use_instruct}")
    print(f"Device:     {device}")
    print(f"Force:      {force}")
    print(f"{'='*60}\n")

    # ---- EMBED TASKS ----
    if embed_type in ('tasks', 'both'):
        task_dir = path_to_master + 'tte_onet_oews/tasks/'
        task_text_path = task_dir + 'task_text.csv'
        task_embed_path = task_dir + 'task_embed.npy'

        if os.path.exists(task_embed_path) and not force:
            print("Tasks: embedding already exists, skipping "
                  "(pass force=True to overwrite).")
        else:
            if not os.path.exists(task_text_path):
                raise FileNotFoundError(
                    f"ERROR: Task text file not found at {task_text_path}.")
            tasks = pd.read_csv(task_text_path, usecols=['task'])
            print("Tasks: loading model...")
            st_task = load_model(task_model, device)
            native_dim = st_task.get_sentence_embedding_dimension()
            eff_dim = min(embed_dim, native_dim) if embed_dim is not None else None
            print(f"Tasks: encoding {len(tasks)} task statements "
                  f"(dim={eff_dim if eff_dim is not None else native_dim})...")
            embs = encode_texts(st_task, tasks['task'].tolist(),
                                TASK_INSTRUCT, use_instruct, eff_dim)
            del st_task
            if device == 'cuda':
                torch.cuda.empty_cache()
            np.save(task_embed_path, embs)
            print(f"Tasks: saved {embs.shape} embeddings to {task_embed_path}")

    # ---- EMBED PATENTS ----
    if embed_type in ('patents', 'both'):
        year_dirs = sorted([d for d in os.listdir(path_to_master)
                            if d.startswith('tte_2') and
                            os.path.isdir(path_to_master + d)])
        if not year_dirs:
            raise FileNotFoundError(
                "ERROR: No year directories found. Run download_data() first.")

        print("Patents: loading model...")
        st_tech = load_model(tech_model, device)
        native_dim = st_tech.get_sentence_embedding_dimension()
        eff_dim = min(embed_dim, native_dim) if embed_dim is not None else None
        print(f"Patents: found {len(year_dirs)} year directories "
              f"(dim={eff_dim if eff_dim is not None else native_dim}).")
        skipped = 0

        for item in year_dirs:
            year = item.replace('tte_', '')
            patent_dir = path_to_master + item + '/patents/'
            text_path = patent_dir + f'patent_text_{year}.csv'
            embed_path = patent_dir + f'patent_embed_{year}.npy'

            if os.path.exists(embed_path) and not force:
                skipped += 1
                continue

            if not os.path.exists(text_path):
                print(f"  {year}: patent text file not found, skipping.")
                continue

            patents = pd.read_csv(text_path, usecols=['abstract'])
            print(f"  {year}: encoding {len(patents)} patents...")
            embs = encode_texts(st_tech, patents['abstract'].tolist(),
                                TECH_INSTRUCT, use_instruct, eff_dim)
            np.save(embed_path, embs)
            print(f"  {year}: saved {embs.shape} embeddings.")

        if skipped:
            print(f"Patents: skipped {skipped} year(s) with existing embeddings "
                  f"(pass force=True to overwrite).")

    print(f"\n{'='*60}")
    print(f"OK Re-embedding complete")
    print(f"{'='*60}\n")
