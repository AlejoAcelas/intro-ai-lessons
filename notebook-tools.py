from copy import deepcopy
import json
import re
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

import anthropic
import typer

# TODO: Create function to run arbitrary Claude instructions on a notebook

app = typer.Typer(no_args_is_help=True, add_completion=False)
ROOT = Path(__file__).resolve().parent

BADGE_TEMPLATE = """<a href="https://colab.research.google.com/github/EffiSciencesResearch/{repo_path}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>"""
RE_BADGE = re.compile(BADGE_TEMPLATE.format(repo_path=r'[^"]+'), re.MULTILINE)

Notebook = Dict[str, Any]


def gather_ipynbs(files: list[Path]) -> list[Path]:
    """Recursively gather all the notebooks in the given directories and files.

    Raises an error if a file does not exist or is not a notebook.
    Meant for command line arguments.
    """

    ipynbs = []
    for file in files:
        if file.is_dir():
            ipynbs.extend(file.rglob("*.ipynb"))
        elif file.exists():
            assert file.suffix == ".ipynb", f"{file} is not a notebook"
            ipynbs.append(file)
        else:
            raise FileNotFoundError(file)

    return ipynbs


def notebook_to_str(notebook: Notebook) -> str:
    """Convert the notebook to a string, suitable for saving to disc."""
    return json.dumps(notebook, indent=4, ensure_ascii=False) + "\n"


def save_notebook(file: Path, notebook: Notebook):
    """Save the notebook to the given file."""
    file.write_text(notebook_to_str(notebook))


def load_notebook(file: Path) -> Notebook:
    """Load the notebook from the given file."""
    return json.loads(file.read_text())


def clean_notebook(notebook: Notebook) -> Notebook:
    notebook = deepcopy(notebook)

    notebook["metadata"] = dict(
        language_info=dict(
            name="python",
            pygments_lexer="ipython3",
        )
    )

    for cell in notebook["cells"]:
        if "outputs" in cell:
            cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None
        if "metadata" in cell:
            cell["metadata"] = {}

    return notebook


def ask_claude(user_prompt: str) -> str:
    client = anthropic.Client()
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.content[0].text

def format_notebook_cell_for_claude(cell: Notebook) -> str:
    content = "".join(cell["source"]).strip()
    cell_type = cell['cell_type']
    return f"<{cell_type}>\n{content}\n</{cell_type}>"

def build_translation_prompt(cell_type: str, cell: dict, context: list[str]) -> str:
    """Build the translation prompt for Claude."""
    context_text = ""
    if context:
        if len(context) > 6:
            selected_context = context[:3] + context[-3:]
        else:
            selected_context = context
            
        string_context = "\n".join(selected_context)
        context_text = f"""
These are the last {len(selected_context)} cells of the Jupyter Notebook translated to Spanish:
<notebook>
{string_context}
</notebook>"""

    cell_content = format_notebook_cell_for_claude(cell)
    
    if cell_type == 'markdown':
        return f"{context_text}\n\nTranslate the following markdown text to Spanish, preserving any code blocks, links or special formatting. Output only the translated text in <markdown> tags:\n\n{cell_content}"
    else:  # code
        return f"{context_text}\n\nTranslate the comments and variable names in this Python code to Spanish, preserving the code functionality. Output only the translated text in <code> tags:\n\n{cell_content}"

def extract_translation(translation: str, cell_type: str) -> str:
    """Extract translated content from Claude's response."""
    match = re.search(f"<{cell_type}>(.*?)</{cell_type}>", translation, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find {cell_type} tags in Claude's response: {translation}")
    return match.group(1).strip()

# ------------------------------
# CLI commands
# ------------------------------


@app.command()
def clean(files: list[Path]):
    """Clean the given notebooks."""

    for file in gather_ipynbs(files):
        notebook = load_notebook(file)
        cleaned = clean_notebook(notebook)
        save_notebook(file, cleaned)


@app.command()
def translate(files: list[Path], overwrite: bool = False):
    """Translate notebook cells to Spanish using Claude."""
    for file in gather_ipynbs(files):
        spanish_file = file.with_stem(file.stem + '-spanish')
        
        # Skip if translation exists and overwrite not requested
        if spanish_file.exists() and not overwrite:
            print(f"⏭️  Skipping {file} - translation already exists at {spanish_file}")
            continue
            
        notebook = load_notebook(file)
        translated = clean_notebook(notebook)
        
        context = []
        
        for cell in tqdm(translated['cells'], desc=f"Translating {file.name}"):
            # Build prompt based on cell type
            cell_type = cell['cell_type']
            if cell_type not in ('markdown', 'code'):
                continue  # Skip unknown cell types
                
            prompt = build_translation_prompt(cell_type, cell, context)
            
            # Translate and update cell content
            translation = ask_claude(prompt)
            translated_content = extract_translation(translation, cell_type)
            cell['source'] = [line + '\n' for line in translated_content.splitlines()]
            
            # Update context
            context.append(format_notebook_cell_for_claude(cell))
                
        save_notebook(spanish_file, translated)
        print(f"🌍 Translated {file} to Spanish and saved as {spanish_file}")

if __name__ == "__main__":
    app()
