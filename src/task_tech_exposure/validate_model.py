# IMPORT PACKAGES
import os
import json
import pandas as pd
import numpy as np
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from sklearn.metrics import (classification_report, precision_score, recall_score,
                              accuracy_score, f1_score)
from sentence_transformers import SentenceTransformer
from .train_model import get_device, encode, TECH_INSTRUCT, TASK_INSTRUCT


# ================== HELPER FUNCTIONS =================

# FIND EMBEDDING MODEL TO TRAIN
def find_model(path_to_master, cat_type):
    """Return path to the model to evaluate (custom first, then manifest default)."""
    custom_path = f'{path_to_master}tte_models/{cat_type}_custom'
    if os.path.exists(custom_path):
        print(f"Model: {custom_path} (custom)")
        return custom_path

    manifest_path = f'{path_to_master}dataset_manifest.json'
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            "ERROR: No custom model found and dataset_manifest.json is missing. "
            "Run download_data() or train_model() first.")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    model_key = 'tech_model' if cat_type[:4] == 'tech' else 'task_model'
    model_rel = manifest.get(model_key)
    if not model_rel:
        raise ValueError(
            f"ERROR: No '{model_key}' entry in dataset_manifest.json.")
    model_path = f'{path_to_master}tte_models/{model_rel}'
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ERROR: Model not found at {model_path}. Run download_data() first.")
    print(f"Model: {model_path} (manifest default)")
    return model_path

# LOAD EMBEDDING MODEL TO TRAIN
def load_model(model_path, device):
    """Load SentenceTransformer with Qwen3-specific kwargs when needed."""
    model_name = os.path.basename(model_path).lower()
    if 'qwen3' in model_name:
        return SentenceTransformer(
            model_path, device=device,
            tokenizer_kwargs={"fix_mistral_regex": True})
    return SentenceTransformer(model_path, device=device)

# CUTOFFS ACHIEVING AVERAGE PRECISION VALUES
def avg_precision_cutoff(similarity, classification, thresholds, accuracy):
    """Lowest threshold achieving target average precision."""
    imputed = (similarity[:, None] >= thresholds).astype(int)
    tp = np.sum((imputed == 1) & (classification == 1)[:, None], axis=0)
    fp = np.sum((imputed == 1) & (classification == 0)[:, None], axis=0)
    precision = np.divide(tp, tp + fp,
                          out=np.zeros_like(tp, dtype=float),
                          where=(tp + fp) != 0)
    if np.all(precision < accuracy):
        return np.nan
    return np.round(np.min(thresholds[precision >= accuracy]), 4)

# CUTOFFS ACHIEVING MARGINAL PRECISION VALUES, EVALUATED USING LOGIT MODEL
def mrg_precision_cutoff(similarity, classification, thresholds, accuracy):
    """Logit-based threshold achieving target marginal precision."""
    model = sm.GLM(classification, add_constant(similarity),
                   family=sm.families.Binomial()).fit(disp=0)
    predicted_probs = model.predict(add_constant(thresholds))
    if np.all(predicted_probs < accuracy):
        return np.nan
    return np.round(np.min(thresholds[predicted_probs >= accuracy]), 4)

# CUTOFFS MAXIMISING F-SCORE
def f_score_threshold(similarity, classification, thresholds, beta):
    """Threshold maximising F-beta score."""
    imputed = (similarity[:, None] >= thresholds).astype(int)
    tp = np.sum((imputed == 1) & (classification == 1)[:, None], axis=0)
    fp = np.sum((imputed == 1) & (classification == 0)[:, None], axis=0)
    fn = np.sum((imputed == 0) & (classification == 1)[:, None], axis=0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float),
                          where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float),
                       where=(tp + fn) != 0)
    fscore = np.divide(
        (1 + beta ** 2) * precision * recall,
        beta ** 2 * precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0)
    return np.round(thresholds[np.argmax(fscore)], 4)

# FORMAT TECH VALIDATION DATA
def format_tech_val(labeled, categories):
    """
    Format raw labeled data for tech threshold evaluation.

    Unlike the training formatter, this keeps ALL confident labels without
    balancing, so that precision and recall metrics are unbiased.
    Returns a long-format DataFrame with columns:
        text, gpt_description, name, label, similarity_score (added later)
    """
    rows = []
    for _, cat_row in categories.iterrows():
        col = f"class_gpt_{cat_row.name + 1}"
        if col not in labeled.columns:
            continue
        df = labeled[['text', col]].copy()
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col].isin(['0', '1'])].copy()
        df[col] = df[col].astype(int)
        df = df.rename(columns={col: 'label'})
        df['gpt_description'] = cat_row['gpt_description']
        df['name'] = cat_row['name']
        rows.append(df[['text', 'gpt_description', 'name', 'label']])
    if not rows:
        raise ValueError(
            "ERROR: No valid validation rows found. Check that label_sample() "
            "completed successfully.")
    return pd.concat(rows, ignore_index=True)

# FORMAT TASK VALIDATION DATA
def format_task_val(labeled):
    """
    Format raw labeled data for task accuracy evaluation.

    class_gpt_1 contains the 1-based index of the best-matching category
    (GPT rank 1). Returns a DataFrame with columns: anchor, class_gpt.
    """
    df = labeled.copy()
    if 'class_gpt_1' not in df.columns:
        raise ValueError("ERROR: Column 'class_gpt_1' not found in labeled data.")
    df['class_gpt'] = pd.to_numeric(df['class_gpt_1'], errors='coerce')
    df = df.dropna(subset=['class_gpt']).copy()
    df['class_gpt'] = df['class_gpt'].astype(int)
    return df[['text', 'class_gpt']].rename(columns={'text': 'anchor'})


# ================== VALIDATE MODEL =================

# MAIN FUNCTION
def validate_model(path_to_data, path_to_results, cat_type,
                   beta=[0.5, 1],
                   prec_avg=[0.25, 0.5, 0.75],
                   prec_mrg=[0.25, 0.5, 0.75]):
    """
    Evaluate model classification performance and find optimal similarity thresholds.

    Looks for a fine-tuned custom model first, then falls back to the manifest
    default. Uses the validation split from train_model() if available, otherwise
    uses the full labeled sample from label_sample().

    For technology classification, produces per-category similarity thresholds
    at multiple precision targets. For task classification, produces a
    classification accuracy report.

    Args:
        path_to_data: Path to directory containing the downloaded dataset.
        path_to_results: Path where intermediate sample files are saved.
        cat_type: Classification type ('tech' or 'task').
        beta: List of beta values for F-score threshold computation (tech only).
        prec_avg: Average precision targets for threshold computation (tech only).
        prec_mrg: Marginal precision targets (logit-based) for threshold computation
                  (tech only).
    """
    path_to_master = path_to_data + 'tte/'
    if not os.path.isdir(path_to_master) and os.path.isfile(path_to_data + 'dataset_manifest.json'):
        print(f"Note: path_to_data={path_to_data!r} points to the tte/ folder directly; "
              f"pass the parent directory instead. Adjusting automatically.")
        path_to_master = path_to_data
    path_to_samples = path_to_results + 'tte_samples/'
    device = get_device()

    print(f"\n{'='*60}")
    print(f"TTE Model Validation — {cat_type}")
    print(f"{'='*60}")
    print(f"Using device: {device}")

    # Find and load model
    model_path = find_model(path_to_master, cat_type)
    model = load_model(model_path, device)

    instruct = TECH_INSTRUCT if cat_type[:4] == 'tech' else TASK_INSTRUCT

    # Load category descriptions — prefer the snapshot saved by create_sample()
    cat_path = f'{path_to_samples}{cat_type}_categories.csv'
    if not os.path.exists(cat_path):
        fallback = (f'{path_to_master}tte_models/category_descriptions/'
                    f'{cat_type[:4]}_categories.csv')
        print(f"  Warning: No sample categories snapshot found. "
              f"Falling back to {fallback}.")
        cat_path = fallback
    if not os.path.exists(cat_path):
        raise FileNotFoundError(f"ERROR: Category file not found at {cat_path}")
    categories = pd.read_csv(cat_path).reset_index(drop=True)
    print(f"Categories: {len(categories)} loaded from {cat_path}")

    # Load validation data — prefer the training val split, fall back to full labeled
    val_split_path = (f'{path_to_samples}{cat_type}_val.csv')
    labeled_path = (f'{path_to_samples}{cat_type}_labeled.csv')
    pre_formatted = False

    if os.path.exists(val_split_path):
        print(f"Validation data: {val_split_path} (training split)")
        raw = pd.read_csv(val_split_path)
        pre_formatted = True
    elif os.path.exists(labeled_path):
        print(f"Validation data: {labeled_path} (full labeled sample)")
        raw = pd.read_csv(labeled_path)
    else:
        raise FileNotFoundError(
            f"ERROR: No validation data found. Run label_sample() to generate "
            f"labeled data, or train_model() to also produce a validation split.")

    # Format validation data
    if cat_type[:4] == 'tech':
        if pre_formatted:
            # Val split already has text / gpt_description / label;
            # merge in the category name for grouped threshold computation
            val_df = raw.merge(
                categories[['name', 'gpt_description']],
                on='gpt_description', how='left')
        else:
            val_df = format_tech_val(raw, categories)

        print(f"Validation rows: {len(val_df)} "
              f"({val_df['label'].sum()} positive, "
              f"{(val_df['label'] == 0).sum()} negative)")

        # Encode: instruct on abstracts (query side), no instruct on descriptions
        print("Encoding patent abstracts...")
        patent_embs = encode(model, val_df['text'].tolist(),
                              use_instruct=True, instruct=instruct)
        print("Encoding category descriptions...")
        all_cat_embs = encode(model, categories['gpt_description'].tolist())

        # Map each row to its category embedding and compute pairwise similarity
        cat_to_idx = {d: i for i, d in enumerate(categories['gpt_description'])}
        cat_embs = all_cat_embs[val_df['gpt_description'].map(cat_to_idx).values]
        val_df = val_df.copy()
        val_df['similarity_score'] = np.sum(patent_embs * cat_embs, axis=1)

        # Find thresholds per category
        eval_grid = np.linspace(0, 1, 10001)
        cat_names = val_df['name'].unique()

        f_names = [f'f{b}_score' for b in beta]
        ap_names = [f'avg_prec_{a}' for a in prec_avg]
        mp_names = [f'marg_prec_{a}' for a in prec_mrg]
        col_names = ['name'] + f_names + ap_names + mp_names

        rows = []
        for cat_name in cat_names:
            subset = val_df[val_df['name'] == cat_name]
            if subset.empty:
                continue
            sim = subset['similarity_score'].values
            true_vals = subset['label'].values

            f_scores = [f_score_threshold(sim, true_vals, eval_grid, b)
                        for b in beta]
            ap_scores = [avg_precision_cutoff(sim, true_vals, eval_grid, a)
                         for a in prec_avg]
            mp_scores = [mrg_precision_cutoff(sim, true_vals, eval_grid, a)
                         for a in prec_mrg]

            rows.append([cat_name] + f_scores + ap_scores + mp_scores)
            print(f"  {cat_name}: marg_prec_0.5 = "
                  f"{mp_scores[prec_mrg.index(0.5)] if 0.5 in prec_mrg else mp_scores[0]:.4f}")

        # Append average across categories
        numeric = np.array([r[1:] for r in rows], dtype=float)
        avg_row = ['avg. threshold'] + list(np.nanmean(numeric, axis=0))
        rows.append(avg_row)

        thresholds = pd.DataFrame(rows, columns=col_names)
        thresh_path = (f'{path_to_samples}{cat_type}_thresholds.csv')
        thresholds.to_csv(thresh_path, index=False)
        print(f"\nThresholds saved to {thresh_path}")
        print(thresholds.to_string(index=False))

        # Performance summary at the 50% marginal precision threshold
        perf_rows = []
        for cat_name in cat_names:
            subset = val_df[val_df['name'] == cat_name]
            if 0.5 in prec_mrg:
                cutoff_val = thresholds.loc[
                    thresholds['name'] == cat_name, 'marg_prec_0.5'].values
                if len(cutoff_val) and not np.isnan(cutoff_val[0]):
                    sim = subset['similarity_score'].values
                    true_vals = subset['label'].values
                    preds = (sim >= cutoff_val[0]).astype(int)
                    perf_rows.append({
                        'name': cat_name,
                        'cutoff': cutoff_val[0],
                        'n_positive': true_vals.sum(),
                        'n_negative': (true_vals == 0).sum(),
                        'accuracy': accuracy_score(true_vals, preds),
                        'precision': precision_score(true_vals, preds, zero_division=0),
                        'recall': recall_score(true_vals, preds, zero_division=0),
                        'f1': f1_score(true_vals, preds, zero_division=0),
                    })
        if perf_rows:
            perf_df = pd.DataFrame(perf_rows)
            perf_path = (f'{path_to_samples}{cat_type}_performance.csv')
            perf_df.to_csv(perf_path, index=False)
            print(f"\nPerformance summary saved to {perf_path}")
            print(perf_df.to_string(index=False))

    elif cat_type == 'task':
        if pre_formatted:
            val_df = raw[['anchor', 'class_gpt']] if 'anchor' in raw.columns \
                else raw.rename(columns={'text': 'anchor'})[['anchor', 'class_gpt']]
        else:
            val_df = format_task_val(raw)

        print(f"Validation rows: {len(val_df)}")

        # Encode task statements with instruct; category descriptions without
        print("Encoding task statements...")
        task_embs = encode(model, val_df['anchor'].tolist(),
                            use_instruct=True, instruct=instruct)
        print("Encoding category descriptions...")
        cat_embs = encode(model, categories['gpt_description'].tolist())

        similarity = task_embs @ cat_embs.T
        predicted = 1 + np.argmax(similarity, axis=1)
        true_values = val_df['class_gpt'].values

        report_dict = classification_report(
            true_values, predicted,
            target_names=[f"{i+1}: {r['name']}"
                          for i, r in categories.iterrows()],
            output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose()

        perf_path = (f'{path_to_samples}{cat_type}_performance.csv')
        report_df.to_csv(perf_path)
        print(f"\nClassification report saved to {perf_path}")
        print(classification_report(
            true_values, predicted,
            target_names=[f"{i+1}: {r['name']}"
                          for i, r in categories.iterrows()],
            zero_division=0))

    else:
        raise ValueError(
            f"ERROR: cat_type must start with 'tech' or equal 'task', got '{cat_type}'")

    print(f"\n{'='*60}")
    print(f"OK Validation complete")
    print(f"{'='*60}\n")
