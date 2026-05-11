# IMPORT PACKAGES
import os
import json
import pandas as pd
import openai


# ================== DEFAULTS =================

EMBED_VERSION = 'qwen3_0.6b'


# ================== LABEL SAMPLE =================

# MAIN FUNCTION
def label_sample(path_to_results, cat_type, api_key,
                 batch_id=None):
    """
    Retrieve GPT batch results and save the labeled sample.

    Must be called after create_sample() once the OpenAI batch is complete. If the
    batch is still in progress, prints the current status and returns without saving.
    The labeled file saved here is the input for train_model() and validate_model().

    Args:
        path_to_results: Path where intermediate sample files are saved.
        cat_type: Classification type ('tech' or 'task').
        api_key: OpenAI API key.
        batch_id: Batch ID to retrieve. If None, reads from the file saved by
                  create_sample(). Pass explicitly if calling after the file has been
                  deleted or to retrieve a specific earlier batch.
    """
    path_to_samples = path_to_results + 'tte_samples/'

    print(f"\n{'='*60}")
    print(f"TTE Sample Labelling — {cat_type}")
    print(f"{'='*60}")

    # Load batch ID from file if not provided
    if batch_id is None:
        batch_id_path = (f'{path_to_samples}{EMBED_VERSION}_{cat_type}'
                         f'_batch_id.txt')
        if not os.path.exists(batch_id_path):
            raise FileNotFoundError(
                f"ERROR: Batch ID file not found at {batch_id_path}. "
                f"Run create_sample() first, or pass batch_id directly.")
        with open(batch_id_path, 'r') as f:
            batch_id = f.read().strip()

    print(f"Batch ID: {batch_id}")

    # Connect and retrieve batch status
    try:
        client = openai.OpenAI(api_key=api_key)
        batch = client.batches.retrieve(batch_id)
    except openai.NotFoundError:
        raise ValueError(
            f"ERROR: Batch '{batch_id}' not found. Check the batch ID.") from None
    except openai.AuthenticationError:
        raise ValueError("ERROR: Invalid OpenAI API key.") from None

    print(f"Status: {batch.status}")

    # Handle non-terminal states — return gracefully so the user can retry later
    if batch.status in ('in_progress', 'finalizing', 'validating'):
        counts = batch.request_counts
        if counts:
            print(f"Progress: {counts.completed}/{counts.total} requests completed")
        print("Batch not yet complete. Call label_sample() again when ready.")
        print(f"\n{'='*60}\n")
        return

    if batch.status == 'failed':
        raise Exception(f"ERROR: Batch failed. Details: {batch.errors}")

    if batch.status == 'cancelled':
        raise Exception("ERROR: Batch was cancelled.")

    if batch.status not in ('completed', 'expired'):
        raise Exception(f"ERROR: Unexpected batch status: {batch.status}")

    if not batch.output_file_id:
        raise Exception(
            f"ERROR: Batch has status '{batch.status}' but no output file. "
            f"The batch may have expired before any results were written.")

    # Log partial errors from the batch output
    if batch.error_file_id:
        error_text = client.files.content(batch.error_file_id).text
        print(f"Warning: Batch completed with some errors:\n{error_text[:500]}")

    # Download and parse the JSONL output
    response_text = client.files.content(batch.output_file_id).text
    records = []
    for line in response_text.strip().splitlines():
        record = json.loads(line)
        custom_id = record.get("custom_id")
        content = None
        body = record.get("response", {}).get("body", {})
        if body:
            choices = body.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content")
        records.append({"custom_id": custom_id, "class_gpt": content})

    # Split comma-separated GPT response into per-category columns
    # Tech: "0,1,1,0,1,0" → class_gpt_1 ... class_gpt_N (binary labels)
    # Task: "4,2,3,1,5"  → class_gpt_1 ... class_gpt_5 (category rankings)
    df = pd.DataFrame(records)
    df['index'] = (df['custom_id']
                   .str.replace('prompt-', '', regex=False)
                   .astype(int))
    df_split = df['class_gpt'].str.split(',', expand=True)
    df_split.columns = [f'class_gpt_{i+1}' for i in range(df_split.shape[1])]
    df = pd.concat([df[['index']], df_split], axis=1)

    # Load original sample texts and merge on row index
    sample_path = f'{path_to_samples}{EMBED_VERSION}_{cat_type}_sample.csv'
    if not os.path.exists(sample_path):
        raise FileNotFoundError(
            f"ERROR: Original sample file not found at {sample_path}. "
            f"Ensure create_sample() was run with the same path_to_results.")
    sample = pd.read_csv(sample_path).reset_index()
    labeled = pd.merge(sample, df, on='index', how='left').drop(columns=['index'])

    # Report missing responses (API errors on individual requests)
    gpt_cols = [c for c in labeled.columns if c.startswith('class_gpt_')]
    n_missing = labeled[gpt_cols].isnull().any(axis=1).sum()
    if n_missing > 0:
        print(f"Warning: {n_missing}/{len(labeled)} rows have missing GPT labels")

    # Save labeled file
    labeled_path = f'{path_to_samples}{EMBED_VERSION}_{cat_type}_labeled.csv'
    labeled.to_csv(labeled_path, index=False)
    print(f"Labeled {len(labeled)} rows saved to {labeled_path}")

    # Remove batch ID file now that results are saved
    batch_id_path = (f'{path_to_samples}{EMBED_VERSION}_{cat_type}'
                     f'_batch_id.txt')
    if os.path.exists(batch_id_path):
        os.remove(batch_id_path)

    print(f"\n{'='*60}")
    print(f"OK Labelling complete")
    print(f"{'='*60}\n")
