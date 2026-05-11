# IMPORT PACKAGES
import os
import pandas as pd
import numpy as np
import torch
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score,
    average_precision_score, f1_score, roc_auc_score,
    classification_report
)
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments, losses, evaluation
)
from datasets import Dataset


# ================== DEFAULTS =================

EMBED_VERSION = 'qwen3_0.6b'
BASE_MODEL = 'Qwen/Qwen3-Embedding-0.6B'

# INSTRUCT PROMPTS APPLIED TO QUERY SIDE
# TECH: PATENT ABSTRACT QUERIES. TASK: TASK STATEMENT QUERIES.
# CATEGORY DESCRIPTIONS ARE ALWAYS ENCODED WITHOUT INSTRUCT.
TECH_INSTRUCT = (
    "Instruct: Given a patent abstract, retrieve technology categories that "
    "describe the technologies used in the invention.\nQuery: "
)
TASK_INSTRUCT = (
    "Instruct: Given an occupational task description, find the task category "
    "that best describes the primary activity performed in this task.\nQuery: "
)

# DEFAULT TRAINING HYPERPARAMETERS
TECH_DEFAULTS = dict(
    steps_total=2250, steps_per_eval=125, batch_size=4,
    learn_rate=8e-6, gradient_accumulation_steps=8, weight_decay=0
)
TASK_DEFAULTS = dict(
    steps_total=1000, steps_per_eval=125, batch_size=8,
    learn_rate=1e-5, gradient_accumulation_steps=4, weight_decay=0
)


# ================== HELPER FUNCTIONS =================

# DETERMINE DEVICE FOR TORCH OPERATIONS
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

# CUTOFFS ACHIEVING MARGINAL PRECISION VALUES, EVALUATED USING LOGIT MODEL
def mrg_precision(similarity, classification, thresholds, accuracy):
    """Logit-based threshold achieving target marginal precision."""
    model = sm.GLM(classification, add_constant(similarity),
                   family=sm.families.Binomial()).fit(disp=0)
    predicted_probs = model.predict(add_constant(thresholds))
    if np.all(predicted_probs < accuracy):
        return np.nan
    return np.min(thresholds[predicted_probs >= accuracy])

# ENCODE TEXTS WITH OPTIONAL INSTRUCT PROMPT AND MRL TRUNCATION
def encode(model, texts, use_instruct=False, instruct=None, truncate_dim=None):
    """Encode texts with optional instruct prefix and MRL truncation."""
    kwargs = dict(normalize_embeddings=True, convert_to_numpy=True)
    if use_instruct:
        if instruct is not None:
            kwargs['prompt'] = instruct
        else:
            kwargs['prompt_name'] = 'query'
    embs = model.encode(texts, **kwargs)
    if truncate_dim is not None:
        embs = embs[:, :truncate_dim]
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    return embs

# EVALUATE MODEL PERFORMANCE ON VALIDATION SET
def evaluate_performance(cat_type, val_df, categories, model,
                          use_instruct, instruct, truncate_dim=None):
    """
    Evaluate model on the validation set. Returns a metrics DataFrame.
    Tech: binary classification via pairwise cosine similarity + logit cutoff.
    Task: multi-class classification via argmax over category descriptions.
    """
    cat_descs = categories['gpt_description'].tolist()

    if cat_type[:4] == 'tech':
        patent_embs = encode(model, val_df['text'].tolist(),
                              use_instruct=use_instruct, instruct=instruct,
                              truncate_dim=truncate_dim)
        all_tech_embs = encode(model, cat_descs, truncate_dim=truncate_dim)
        tech_to_idx = {t: i for i, t in enumerate(cat_descs)}
        tech_embs = all_tech_embs[val_df['gpt_description'].map(tech_to_idx).values]

        similarity = np.einsum('ij,ij->i', patent_embs, tech_embs)
        true_values = val_df['label'].values
        thresholds = np.linspace(0, 1, 10001)
        cutoff = mrg_precision(similarity, true_values, thresholds, 0.5)
        predictions = (similarity > cutoff).astype(int) if not np.isnan(cutoff) else np.zeros_like(true_values)

        return pd.DataFrame({
            'category': ['All'],
            'accuracy': [accuracy_score(true_values, predictions)],
            'precision': [precision_score(true_values, predictions, zero_division=0)],
            'recall': [recall_score(true_values, predictions, zero_division=0)],
            'avg_precision': [average_precision_score(true_values, similarity)],
            'f1': [f1_score(true_values, predictions, zero_division=0)],
            'auc': [roc_auc_score(true_values, similarity)],
        })

    elif cat_type == 'task':
        task_embs = encode(model, val_df['anchor'].tolist(),
                            use_instruct=use_instruct, instruct=instruct,
                            truncate_dim=truncate_dim)
        cat_embs = encode(model, cat_descs, truncate_dim=truncate_dim)
        similarity = task_embs @ cat_embs.T
        true_values = val_df['class_gpt'].values
        predictions = 1 + np.argmax(similarity, axis=1)
        return pd.DataFrame(
            classification_report(true_values, predictions, output_dict=True)
        ).transpose()

# FORMAT TECH TRAINING DATA
def format_tech(labeled, categories):
    """
    Convert raw labeled CSV to (text, gpt_description, label) format.

    For each category: filters to confident labels ('0'/'1'), creates a
    balanced positive/negative subsample, and appends the category description.
    Returns a shuffled concatenation across all categories.
    """
    rows = []

    for i, cat_row in categories.iterrows():
        col = f'class_gpt_{i + 1}'
        if col not in labeled.columns:
            print(f"  Warning: Column {col} not found, skipping '{cat_row['name']}'")
            continue

        df = labeled[['text', col]].copy()
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col].isin(['0', '1'])]
        df[col] = df[col].astype(int)

        pos = df[df[col] == 1]
        neg = df[df[col] == 0]
        n = min(len(pos), len(neg))
        if n == 0:
            print(f"  Warning: No balanced examples for '{cat_row['name']}', skipping")
            continue

        balanced = pd.concat([
            pos.sample(n, random_state=100),
            neg.sample(n, random_state=100)
        ])
        balanced = balanced.rename(columns={col: 'label'})
        balanced['gpt_description'] = cat_row['gpt_description']
        rows.append(balanced[['text', 'gpt_description', 'label']])

    if not rows:
        raise ValueError("ERROR: No valid training examples found after filtering confident labels")
    return (pd.concat(rows, ignore_index=True)
            .sample(frac=1, random_state=100)
            .reset_index(drop=True))

# FORMAT TASK TRAINING DATA
def format_task(labeled, categories):
    """
    Convert raw labeled CSV to anchor / positive / negative format.

    The GPT response is a ranked list of category indices: class_gpt_1 holds
    the index of the best-matching category, class_gpt_2 the second-best, etc.
    Positive = description of the best-matching category; negatives = the rest.
    """
    n_cats = len(categories)
    cat_desc = dict(zip(range(1, n_cats + 1), categories['gpt_description']))
    rank_cols = [f'class_gpt_{i + 1}' for i in range(n_cats)]

    df = labeled.copy()
    df[rank_cols] = df[rank_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=rank_cols).reset_index(drop=True)
    df[rank_cols] = df[rank_cols].astype(int)

    rename_map = {'class_gpt_1': 'positive'}
    rename_map.update({f'class_gpt_{i + 1}': f'negative_{i}'
                       for i in range(1, n_cats)})
    df = df.rename(columns=rename_map)
    df['anchor'] = df['text']
    df['class_gpt'] = df['positive']

    df['positive'] = df['positive'].map(cat_desc)
    neg_cols = [f'negative_{i}' for i in range(1, n_cats)]
    for col in neg_cols:
        df[col] = df[col].map(cat_desc)

    keep_cols = ['anchor', 'class_gpt', 'positive'] + neg_cols
    df = df[keep_cols].dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError("ERROR: No valid training examples found after formatting task data")
    return df.sample(frac=1, random_state=100).reset_index(drop=True)


# ================== TRAIN MODEL =================

# MAIN FUNCTION
def train_model(path_to_data, path_to_results, cat_type, val_frac=0.1,
                run_validation=True, steps_total=None, steps_per_eval=None, 
                batch_size=None, learn_rate=None, gradient_accumulation_steps=None,
                matryoshka_dims=[1024, 512, 384, 256], max_seq_length=512):
    """
    Fine-tune Qwen3-Embedding-0.6B for technology or task classification.

    Loads the labeled sample produced by label_sample(), formats it for
    training, splits into train/validation sets, and fine-tunes the model
    using Matryoshka Representation Learning and asymmetric instruct prompts.
    Optionally calls validate_model() after training to find optimal thresholds.

    Args:
        path_to_data: Path to directory containing the downloaded dataset.
        path_to_results: Path where intermediate sample files are saved.
        cat_type: Classification type ('tech' or 'task').
        val_frac: Fraction of formatted data to hold out for validation. Default 0.1.
        run_validation: If True, calls validate_model() after training to compute
                        classification thresholds. Default True.
        steps_total: Total training steps. Defaults to 2250 (tech) or 1000 (task).
        steps_per_eval: Steps between evaluations and checkpoints. Default 125.
        batch_size: Per-device batch size. Defaults to 4 (tech) or 8 (task).
        learn_rate: Learning rate. Defaults to 8e-6 (tech) or 1e-5 (task).
        gradient_accumulation_steps: Gradient accumulation steps, giving an effective
                                     batch of 32 at defaults. Defaults to 8 (tech)
                                     or 4 (task).
        matryoshka_dims: MRL dimensions to train and evaluate. Default [1024,512,384,256].
                         Pass None to disable MRL.
        max_seq_length: Tokenizer sequence length cap. Default 512.
    """
    path_to_master = path_to_data + 'tte/'
    path_to_samples = path_to_results + 'tte_samples/'
    output_dir = f'{path_to_master}tte_models/{cat_type}_custom'

    print(f"\n{'='*60}")
    print(f"TTE Model Training — {cat_type}")
    print(f"{'='*60}")

    # Apply cat_type-specific defaults for any unspecified hyperparameters
    defaults = TECH_DEFAULTS if cat_type[:4] == 'tech' else TASK_DEFAULTS
    steps_total = steps_total or defaults['steps_total']
    steps_per_eval = steps_per_eval or defaults['steps_per_eval']
    batch_size = batch_size or defaults['batch_size']
    learn_rate = learn_rate or defaults['learn_rate']
    gradient_accumulation_steps = (gradient_accumulation_steps
                                   or defaults['gradient_accumulation_steps'])
    weight_decay = defaults['weight_decay']

    instruct = TECH_INSTRUCT if cat_type[:4] == 'tech' else TASK_INSTRUCT

    # Load labeled sample
    labeled_path = f'{path_to_samples}{EMBED_VERSION}_{cat_type}_labeled.csv'
    if not os.path.exists(labeled_path):
        raise FileNotFoundError(
            f"ERROR: Labeled sample not found at {labeled_path}. "
            f"Run create_sample() and label_sample() first.")
    labeled = pd.read_csv(labeled_path)
    print(f"Loaded {len(labeled)} labeled rows from {labeled_path}")

    # Load category descriptions — prefer the snapshot saved by create_sample()
    cat_path = f'{path_to_samples}{EMBED_VERSION}_{cat_type}_categories.csv'
    if not os.path.exists(cat_path):
        fallback = (f'{path_to_master}tte_models/category_descriptions/'
                    f'{cat_type[:4]}_categories.csv')
        print(f"  Warning: No sample categories snapshot found. "
              f"Falling back to {fallback}.")
        cat_path = fallback
    if not os.path.exists(cat_path):
        raise FileNotFoundError(f"ERROR: Category file not found at {cat_path}")
    categories = pd.read_csv(cat_path).reset_index(drop=True)
    print(f"Loaded {len(categories)} categories from {cat_path}")

    # Format for training
    if cat_type[:4] == 'tech':
        formatted = format_tech(labeled, categories)
    elif cat_type == 'task':
        formatted = format_task(labeled, categories)
    else:
        raise ValueError(
            f"ERROR: cat_type must start with 'tech' or equal 'task', got '{cat_type}'")

    print(f"Formatted {len(formatted)} training examples")

    # Train/validation split (random — no per-category balancing of the split)
    val_df = formatted.sample(frac=val_frac, random_state=100)
    train_df = formatted.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Split: {len(train_df)} train / {len(val_df)} validation")

    # Save splits so validate_model() can find the val set
    os.makedirs(path_to_samples, exist_ok=True)
    train_df.to_csv(
        f'{path_to_samples}{EMBED_VERSION}_{cat_type}_train.csv', index=False)
    val_df.to_csv(
        f'{path_to_samples}{EMBED_VERSION}_{cat_type}_val.csv', index=False)

    device = get_device()

    print(f"\nBase model: {BASE_MODEL}")
    print(f"Output:     {output_dir}")
    print(f"Device:     {device}")
    print(f"Steps:      {steps_total} total, eval every {steps_per_eval}")
    print(f"Batch:      {batch_size} × {gradient_accumulation_steps} accumulation "
          f"= {batch_size * gradient_accumulation_steps} effective")
    print(f"LR:         {learn_rate}  |  MRL dims: {matryoshka_dims}")
    print(f"{'='*60}\n")

    # Load model
    os.makedirs(output_dir, exist_ok=True)
    model = SentenceTransformer(
        BASE_MODEL,
        device=device,
        tokenizer_kwargs={"fix_mistral_regex": True}
    )
    model.max_seq_length = max_seq_length

    # Configure loss and evaluator
    if cat_type[:4] == 'tech':
        train_dataset = Dataset.from_pandas(
            train_df[['text', 'gpt_description', 'label']])
        val_dataset = Dataset.from_pandas(
            val_df[['text', 'gpt_description', 'label']])
        base_loss = losses.OnlineContrastiveLoss(model=model)
        evaluator = evaluation.BinaryClassificationEvaluator(
            val_df['text'].tolist(),
            val_df['gpt_description'].tolist(),
            val_df['label'].tolist(),
            name='validation', batch_size=batch_size
        )
        prompt_col = 'text'

    elif cat_type == 'task':
        neg_cols = [c for c in train_df.columns if c.startswith('negative_')]
        train_dataset = Dataset.from_pandas(train_df[['anchor', 'positive']])
        val_dataset = Dataset.from_pandas(
            val_df[['anchor', 'positive'] + neg_cols])
        base_loss = losses.MultipleNegativesRankingLoss(model=model)
        val_samples = [
            {
                "query": row["anchor"],
                "positive": row["positive"],
                "negative": [row[c] for c in neg_cols]
            }
            for _, row in val_df.iterrows()
        ]
        evaluator = evaluation.RerankingEvaluator(
            samples=val_samples, batch_size=batch_size)
        prompt_col = 'anchor'

    # Wrap with MatryoshkaLoss if MRL dimensions specified
    if matryoshka_dims is not None:
        train_loss = losses.MatryoshkaLoss(
            model=model,
            loss=base_loss,
            matryoshka_dims=sorted(matryoshka_dims, reverse=True)
        )
    else:
        train_loss = base_loss

    # Fine-tune
    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            load_best_model_at_end=True,
            dataloader_drop_last=True,
            max_steps=steps_total,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            bf16=True,
            lr_scheduler_type='cosine',
            warmup_steps=steps_per_eval,
            learning_rate=learn_rate,
            weight_decay=weight_decay,
            eval_strategy='steps',
            eval_steps=steps_per_eval,
            eval_on_start=True,
            save_strategy='steps',
            save_steps=steps_per_eval,
            save_total_limit=20,
            logging_strategy='steps',
            logging_steps=steps_per_eval,
            logging_first_step=True,
            prompts={prompt_col: instruct},
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")

    # Post-training evaluation: base vs finetuned, across MRL dimensions
    all_evals = []

    base_model = SentenceTransformer(
        BASE_MODEL, device=device, tokenizer_kwargs={"fix_mistral_regex": True})
    base_model.max_seq_length = max_seq_length

    for label, m in [('base', base_model), ('finetuned', model)]:
        evals = evaluate_performance(
            cat_type, val_df, categories, m,
            use_instruct=False, instruct=None)
        evals['model'] = label
        evals['dim'] = 'full'
        all_evals.append(evals)
        print(f"\n{label.capitalize()} model (full dim):")
        print(evals.to_string())

        if matryoshka_dims is not None and label == 'finetuned':
            for dim in matryoshka_dims:
                evals_dim = evaluate_performance(
                    cat_type, val_df, categories, m,
                    use_instruct=False, instruct=None, truncate_dim=dim)
                evals_dim['model'] = 'finetuned'
                evals_dim['dim'] = dim
                all_evals.append(evals_dim)

        # Asymmetric (instruct) evaluation for finetuned model
        if label == 'finetuned':
            evals_instruct = evaluate_performance(
                cat_type, val_df, categories, m,
                use_instruct=True, instruct=instruct)
            evals_instruct['model'] = 'finetuned_instruct'
            evals_instruct['dim'] = 'full'
            all_evals.append(evals_instruct)
            print(f"\nFinetuned model (instruct, full dim):")
            print(evals_instruct.to_string())

    eval_df = pd.concat(all_evals, axis=0)
    eval_path = f'{output_dir}/validation_stats.csv'
    eval_df.to_csv(eval_path, index=False)
    print(f"\nValidation stats saved to {eval_path}")

    print(f"\n{'='*60}")
    print(f"OK Training complete")
    print(f"  Model saved to: {output_dir}")
    print(f"  Pass this path as the 'model' argument to classify_patents()")
    print(f"  or classify_tasks() to use the fine-tuned model.")
    print(f"{'='*60}\n")

    # Optionally find classification thresholds
    if run_validation:
        from .validate_model import validate_model
        validate_model(path_to_data, path_to_results, cat_type)
