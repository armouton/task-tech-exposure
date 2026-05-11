# IMPORT PACKAGES
import os
import json
import datetime
import pandas as pd
import openai


# ================== DEFAULTS =================

EMBED_VERSION = 'qwen3_0.6b'

TECH_PROMPT = (
    "<instructions> Determine if the patent text falls into one or more of the following {n} "
    "technology categories, and return a comma-separated vector of \"1\"'s, \"0\"'s, and "
    "\"-\"'s with no other text (\"1\"=yes, \"0\"=no, \"-\"=can't be determined).\n"
    "<guidelines>\n"
    "1. Evaluate each category independently. The categories are not mutually exclusive.\n"
    "2. For each category, mark \"1\" if the invention explicitly uses or clearly implies the "
    "use of technologies from the category. Mark \"0\" if the invention is clearly unrelated "
    "to the category. Mark \"-\" if the relationship is genuinely uncertain or indirect "
    "(e.g., vague reference or unclear application).\n"
    "3. Return only a comma-separated vector of \"1\"'s, \"0\"'s, and \"-\"'s "
    "(e.g., \"0,1,1,0,1,0\"), where the Nth entry is the value assigned to category N. "
    "Do not return any other text.\n"
    "4. Definitions: Digital tasks use, process, or transmit digital information. Physical "
    "tasks involve movement or mechanical interaction with the environment. Fixed systems "
    "follow predefined rules. Adaptive systems autonomously sense and react to information "
    "about their environment or state.\n"
    "<patent abstract> {text}\n"
    "<categories>\n{categories}"
)

TASK_PROMPT = (
    "<instructions> Identify the task group that best describes the task statement, and return "
    "a comma-separated vector of task group numbers ordered from best to worst match, with no "
    "other text.\n"
    "<guidelines>\n"
    "1. Determine the primary activity required to perform the task statement.\n"
    "2. If multiple activities are required, the primary activity is the one that most "
    "strongly determines the skills or judgment required.\n"
    "3. Order the task groups from best match to worst match. Each group must appear exactly "
    "once.\n"
    "4. Return only a comma-separated vector of task group numbers in ranked order (best to "
    "worst). The first number is the index of the best-matching group, the second is the "
    "second-best, and so on through to the worst. Do not return any other text.\n"
    "<task statement> {text}\n"
    "<categories>\n{categories}"
)


# ================== HELPER FUNCTIONS =================

# GET MOST RECENT YEAR IN DATASET
def get_most_recent_year(path_to_master):
    """Return the most recent year with a tte_YYYY directory in the master path."""
    years = []
    for d in os.listdir(path_to_master):
        if d.startswith('tte_2'):
            suffix = d.replace('tte_', '')
            if suffix.isdigit() and len(suffix) == 4:
                years.append(int(suffix))
    if not years:
        raise FileNotFoundError("ERROR: No year directories found in master path")
    return max(years)

# LOAD PATENTS WITH DATE FILTERING
def load_patents(path_to_master, from_date, to_date):
    """Load patent abstracts from year-split directories, filtered to the date range."""
    from_dt = datetime.datetime.strptime(from_date, '%Y-%m-%d')
    to_dt = datetime.datetime.strptime(to_date, '%Y-%m-%d')

    dfs = []
    for year in range(from_dt.year, to_dt.year + 1):
        year_path = f'{path_to_master}tte_{year}/patents/patent_text_{year}.csv'
        if not os.path.exists(year_path):
            print(f"  Warning: No patent file found for {year}, skipping")
            continue
        df = pd.read_csv(year_path, usecols=['patent_id', 'abstract', 'date_earliest'])
        dates = pd.to_datetime(df['date_earliest'], errors='coerce')
        mask = (dates >= from_dt) & (dates <= to_dt)
        dfs.append(df[mask].drop(columns=['date_earliest']))

    if not dfs:
        raise FileNotFoundError(
            f"ERROR: No patents found between {from_date} and {to_date}")
    return pd.concat(dfs, ignore_index=True)

# VALIDATE API KEY AND MODEL
def validate_gpt_model(client, gpt_model):
    """Raise a descriptive error if the requested GPT model is not available."""
    try:
        client.models.retrieve(gpt_model)
    except openai.NotFoundError:
        raise ValueError(
            f"ERROR: GPT model '{gpt_model}' not found. Check the model name at "
            f"https://platform.openai.com/docs/models") from None
    except openai.AuthenticationError:
        raise ValueError(
            "ERROR: Invalid OpenAI API key. Check that api_key is correct.") from None


# ================== CREATE SAMPLE =================

# MAIN FUNCTION TO CREATE SAMPLE AND SUBMIT BATCH
def create_sample(path_to_data, path_to_results, cat_type, api_key,
                  from_date=None, to_date=None, sample_size=None,
                  gpt_model='gpt-4.1', path_to_categories=None):
    """
    Create a labelled sample by submitting patent or task texts to the OpenAI batch API.

    Texts are saved locally and a batch job is submitted. Call label_sample() once the
    batch is complete (up to 24 hours) to retrieve results. A snapshot of the category
    descriptions used is saved alongside the sample so that train_model() and
    validate_model() use exactly the same categories.

    Args:
        path_to_data: Path to directory containing the downloaded dataset.
        path_to_results: Path where intermediate sample files will be saved.
        cat_type: Classification type. 'tech' for patent technology categories,
                  'task' for occupational task categories.
        api_key: OpenAI API key for batch submission.
        from_date: Start date for patent sampling ('YYYY-MM-DD'). Tech only. Defaults
                   to January 1 of the most recent year in the dataset.
        to_date: End date for patent sampling ('YYYY-MM-DD'). Tech only. Defaults to
                 December 31 of the most recent year in the dataset.
        sample_size: Number of texts to submit for GPT labelling. None submits all
                     texts in the specified range. Texts are drawn randomly.
        gpt_model: OpenAI model to use for labelling. Default is 'gpt-4.1'.
        path_to_categories: Path to a category descriptions CSV with at least columns
                            'name', 'category', and 'gpt_description'. Defaults to
                            the standard file in the downloaded dataset.
    """
    path_to_master = path_to_data + 'tte/'
    path_to_samples = path_to_results + 'tte_samples/'
    os.makedirs(path_to_samples, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TTE Sample Creation — {cat_type}")
    print(f"{'='*60}")

    # Validate API key and model before doing any work
    client = openai.OpenAI(api_key=api_key)
    validate_gpt_model(client, gpt_model)

    # Load category descriptions
    if path_to_categories is None:
        path_to_categories = (f'{path_to_master}tte_models/category_descriptions/'
                              f'{cat_type[:4]}_categories.csv')
    if not os.path.exists(path_to_categories):
        raise FileNotFoundError(
            f"ERROR: Category descriptions file not found at {path_to_categories}")
    categories = pd.read_csv(path_to_categories).reset_index(drop=True)
    if not {'name', 'category', 'gpt_description'}.issubset(categories.columns):
        raise ValueError(
            "ERROR: Categories CSV must contain columns 'name', 'category', and 'gpt_description'")
    print(f"Loaded {len(categories)} categories from {path_to_categories}")

    # Load texts
    if cat_type[:4] == 'tech':
        if from_date is None or to_date is None:
            most_recent = get_most_recent_year(path_to_master)
            from_date = from_date or f'{most_recent}-01-01'
            to_date = to_date or f'{most_recent}-12-31'
        print(f"Date range: {from_date} to {to_date}")

        patents = load_patents(path_to_master, from_date, to_date)
        print(f"Loaded {len(patents)} patents in date range")
        if sample_size is not None and sample_size < len(patents):
            patents = patents.sample(sample_size, random_state=100)
            print(f"Sampled {sample_size} patents")
        sample = patents[['patent_id', 'abstract']].rename(
            columns={'abstract': 'text'})

    elif cat_type[:4] == 'task':
        onet_path = f'{path_to_master}tte_onet_oews/tasks/task_text.csv'
        if not os.path.exists(onet_path):
            raise FileNotFoundError(f"ERROR: Task file not found at {onet_path}")
        onet = pd.read_csv(onet_path, usecols=['task_ref', 'task'])
        print(f"Loaded {len(onet)} ONET task statements")
        if sample_size is not None and sample_size < len(onet):
            onet = onet.sample(sample_size, random_state=100)
            print(f"Sampled {sample_size} task statements")
        sample = onet.rename(columns={'task': 'text'})

    else:
        raise ValueError(
            f"ERROR: cat_type must start with 'tech' or 'task', got '{cat_type}'")

    sample = sample.reset_index(drop=True)

    # Load classification prompt template from file; fall back to built-in default
    template_path = (f'{path_to_master}tte_models/gpt_prompts/templates/'
                     f'{cat_type[:4]}_classification_template.txt')
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        print(f"Prompt template: {template_path}")
    else:
        prompt_template = TECH_PROMPT if cat_type[:4] == 'tech' else TASK_PROMPT
        print(f"Prompt template: built-in default (no file found at {template_path})")

    # Build category list and format prompt
    n = len(categories)
    categories_text = '\n'.join(
        f"{i + 1}. {cat}" for i, cat in enumerate(categories['category']))

    # Apply template and save sample texts (without prompt column)
    sample['prompt'] = sample['text'].apply(
        lambda t: prompt_template.format(text=t, n=n, categories=categories_text))
    sample_path = f'{path_to_samples}{EMBED_VERSION}_{cat_type}_sample.csv'
    sample.drop(columns=['prompt']).to_csv(sample_path, index=False)
    print(f"Saved {len(sample)} sample texts to {sample_path}")

    # Save a categories snapshot so train_model() and validate_model() use the same categories
    cat_snapshot_path = f'{path_to_samples}{EMBED_VERSION}_{cat_type}_categories.csv'
    categories.to_csv(cat_snapshot_path, index=False)
    print(f"Saved category snapshot to {cat_snapshot_path}")

    # Build JSONL batch input
    print(f"Submitting batch to OpenAI (model: {gpt_model})...")
    jsonl_path = f'{path_to_samples}temp_batchinput.jsonl'
    with open(jsonl_path, 'w') as f:
        for idx, row in sample.iterrows():
            entry = {
                "custom_id": f"prompt-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": gpt_model,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": row['prompt']}],
                    "max_completion_tokens": 2500
                }
            }
            f.write(json.dumps(entry) + "\n")

    # Upload and submit batch; always clean up the temp file
    try:
        with open(jsonl_path, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
    except Exception as e:
        raise Exception(f"ERROR: Failed to submit batch to OpenAI: {e}") from e
    finally:
        try:
            os.remove(jsonl_path)
        except OSError:
            pass

    # Save batch ID so label_sample() can retrieve without requiring the user to
    # track it manually
    batch_id_path = (f'{path_to_samples}{EMBED_VERSION}_{cat_type}_batch_id.txt')
    with open(batch_id_path, 'w') as f:
        f.write(batch.id)

    print(f"Batch submitted. ID: {batch.id}")
    print(f"Batch ID saved to: {batch_id_path}")
    print(f"\nCall label_sample() once the batch is complete (up to 24 hours).")
    print(f"\n{'='*60}")
    print(f"OK Sample creation complete")
    print(f"{'='*60}\n")
