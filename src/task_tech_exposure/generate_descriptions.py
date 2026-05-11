# IMPORT PACKAGES
import os
import pandas as pd


# ================== DEFAULTS =================

# DEFAULT PROMPT TEMPLATES
DEFAULT_TECH_DESC_TEMPLATE = (
    "<instructions> Write a 150-word description of each of the following {n} technology "
    "categories, that will be used by an embedding model classifier to identify patent abstracts "
    "that describe the use of technologies from the category.\n"
    "<description guidelines>\n"
    "1. Use patent-style language throughout the description. Prefer technical language seen in "
    "patent abstracts.\n"
    "2. Avoid abstract or generic descriptions that are unlikely to appear in patents.\n"
    "3. Describe the principle systems, methods, and applications that comprise the category.\n"
    "4. Avoid language that may apply to other, non-targeted technologies. Focus on affirmative "
    "technological characteristics, and do not use contrastive, comparative, or exclusionary "
    "language.\n"
    "5. The description should be in paragraph form, and approximately 150 words in length.\n"
    "6. If important technological subdomains are missing from the description, expand the "
    "description past 150 words. If any of those listed are overly vague or generic, remove "
    "them and shorten the description.\n"
    "<categories>\n"
    "{categories}"
)

DEFAULT_TASK_DESC_TEMPLATE = (
    "<instructions> Write a 150-word description of each of the following {n} occupational task "
    "categories, that will be used by an embedding model classifier to label ONET occupational "
    "task statements.\n"
    "<description guidelines>\n"
    "1. Use ONET-style language throughout the description. Focus on actions and contexts seen "
    "in ONET task statements.\n"
    "2. Avoid abstract or generic actions and contexts, as well as those that are very narrowly "
    "focused.\n"
    "3. Describe common workplace activities that generalize across multiple occupations and "
    "industries.\n"
    "4. Avoid activities that are likely to be associated with other, non-targeted tasks. Do not "
    "mention 'excluded' activities or negative cases.\n"
    "5. The description should be in paragraph form, and approximately 150 words in length.\n"
    "6. If important activities or contexts are missing from the description, expand the "
    "description past 150 words. If any of the listed are overly narrow, vague, or generic, "
    "remove them and shorten the description.\n"
    "<categories>\n"
    "{categories}"
)


# ================== GENERATE DESCRIPTIONS =================

# MAIN FUNCTION
def generate_descriptions(path_to_data, path_to_results, cat_type,
                          path_to_categories=None):
    """
    Generate a GPT prompt for writing gpt_description entries in a categories CSV.

    Outputs a single .txt file containing a prompt ready to paste into a web app
    (e.g., ChatGPT). The response should be the gpt_description text for each
    category, which is then added as the 'gpt_description' column in the categories
    CSV before running create_sample().

    Uses a description prompt template from the dataset's templates folder if present,
    otherwise falls back to the built-in default. To customise the prompt style,
    copy the default template to:
        tte_models/gpt_prompts/templates/{tech|task}_description_template.txt

    Args:
        path_to_data: Path to directory containing the downloaded dataset.
        path_to_results: Path where the output prompt file will be saved.
        cat_type: Classification type ('tech' or 'task').
        path_to_categories: Path to a categories CSV with at least 'name' and
                            'category' columns. Defaults to the standard file in
                            the downloaded dataset.
    """
    path_to_master = path_to_data + 'tte/'

    print(f"\n{'='*60}")
    print(f"TTE Description Prompt Generation — {cat_type}")
    print(f"{'='*60}")

    # Load categories
    if path_to_categories is None:
        path_to_categories = (f'{path_to_master}tte_models/category_descriptions/'
                              f'{cat_type[:4]}_categories.csv')
    if not os.path.exists(path_to_categories):
        raise FileNotFoundError(
            f"ERROR: Category descriptions file not found at {path_to_categories}")
    categories = pd.read_csv(path_to_categories)
    if not {'name', 'category'}.issubset(categories.columns):
        raise ValueError(
            "ERROR: Categories CSV must contain columns 'name' and 'category'")
    print(f"Loaded {len(categories)} categories from {path_to_categories}")

    # Load description prompt template from file; fall back to built-in default
    template_path = (f'{path_to_master}tte_models/gpt_prompts/templates/'
                     f'{cat_type[:4]}_description_template.txt')
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        print(f"Template: {template_path}")
    else:
        template = (DEFAULT_TECH_DESC_TEMPLATE if cat_type[:4] == 'tech'
                    else DEFAULT_TASK_DESC_TEMPLATE)
        print(f"Template: built-in default (no file found at {template_path})")

    # Build category list and format prompt
    n = len(categories)
    categories_text = '\n'.join(
        f"{i + 1}. {cat}" for i, cat in enumerate(categories['category']))
    prompt = template.format(n=n, categories=categories_text)

    # Save output
    out_path = f'{path_to_results}{cat_type[:4]}_description_prompt.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(prompt)

    print(f"Saved description prompt to {out_path}")
    print(f"Paste this file into a web app (e.g., ChatGPT) to generate gpt_description text,")
    print(f"then add the returned descriptions as the 'gpt_description' column in your CSV.")

    print(f"\n{'='*60}")
    print(f"OK Description prompt generation complete")
    print(f"{'='*60}\n")
