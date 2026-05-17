# dev_tools/

Developer-only utilities. NOT loaded or executed by the plugin at runtime —
ComfyUI ignores this folder. Tools here are for maintainers who want to
grow / update the plugin's data files.

## extract_embedding_names_from_prompt_corpus.py

Scans a corpus of prompt files (workflow `.json`, plain `.txt`, `.png`
with embedded workflow metadata, etc.) for `embedding:NAME` references,
strips folder paths from each captured name, deduplicates, and writes
the sorted result to a text file.

### Purpose

The plugin's runtime cutoff node (`CLIPTextEncodeWithCutoffRegionSeparation`)
auto-filters bare comma-separated tags that match a known embedding name
but are NOT installed on the user's system — preventing other people's
shared A1111-style prompts from accidentally encoding orphan embedding
names as plain text and influencing the image.

For that filter to fire, the node needs a curated list of embedding names
known to be in common use across the prompt-sharing community. This tool
helps the maintainer grow that list by feeding in folders of prompts from
many sources, harvesting all referenced names, and merging them into the
canonical list file at the plugin root:

    ../known_a1111_embedding_names_to_filter_when_not_installed_locally.txt

### Usage

Single source folder, fresh list:

```bash
python3 extract_embedding_names_from_prompt_corpus.py \
    --source-folder /path/to/random_workflows \
    --output-file ../known_a1111_embedding_names_to_filter_when_not_installed_locally.txt
```

Multiple source folders in one run (repeat `--source-folder`):

```bash
python3 extract_embedding_names_from_prompt_corpus.py \
    --source-folder /path/to/workflows_a \
    --source-folder /path/to/workflows_b \
    --source-folder /path/to/workflows_c \
    --output-file ../known_a1111_embedding_names_to_filter_when_not_installed_locally.txt
```

Merge new names into the EXISTING list (updates in place, preserving prior
entries):

```bash
python3 extract_embedding_names_from_prompt_corpus.py \
    --source-folder /path/to/new_workflows \
    --output-file ../known_a1111_embedding_names_to_filter_when_not_installed_locally.txt \
    --existing-list ../known_a1111_embedding_names_to_filter_when_not_installed_locally.txt
```

### Behavior

- Scans the configured file extensions (default: `json txt md yaml yml png`)
  recursively under each source folder.
- Regex `embedding:([\w./\\-]+)` finds every `embedding:NAME` reference.
- Folder paths in matched names are stripped (so `style\foo` and
  `vibes/foo` both contribute `foo` to the list — the runtime filter
  matches on base name only).
- Output is one name per line, sorted case-insensitively. Lines starting
  with `#` in an existing list are treated as comments and preserved on
  re-load.
- The scan does NOT touch ComfyUI's `models/embeddings/` directory or
  load any actual embedding files — it only mines names that appear in
  prompt text.

### When to re-run

Run after collecting more workflows from online prompt-sharing communities
(Civitai posts, OpenArt prompts, Discord prompt-share channels, etc.).
The bigger the corpus, the better the runtime filter at recognizing
"this bare tag is an orphan embedding from someone else's setup".
