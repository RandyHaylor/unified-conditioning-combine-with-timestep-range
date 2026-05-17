#!/usr/bin/env python3
"""
extract_embedding_names_from_prompt_corpus.py

Walks a folder recursively for files that may contain `embedding:NAME`
references (ComfyUI workflow .json files, image .png files with embedded
workflow metadata, plain .txt prompts, etc.), extracts every unique
embedding name found, and writes them as a sorted newline-separated list
to an output file.

Optionally takes an existing list file and merges into it (preserving
existing entries; only new uniques are added).

Usage:
    python extract_embedding_names_from_prompt_corpus.py \\
        --source-folder /path/to/workflows_or_prompts \\
        --output-file ./known_embedding_names.txt

Multiple source folders in one run (repeat --source-folder):
    python extract_embedding_names_from_prompt_corpus.py \\
        --source-folder /path/to/workflows_a \\
        --source-folder /path/to/workflows_b \\
        --source-folder /path/to/workflows_c \\
        --output-file ./known_embedding_names.txt

With merge into existing list:
    python extract_embedding_names_from_prompt_corpus.py \\
        --source-folder /path/to/new_workflows \\
        --output-file ./known_embedding_names.txt \\
        --existing-list ./known_embedding_names.txt

Custom file extensions to scan:
    python extract_embedding_names_from_prompt_corpus.py \\
        --source-folder /path \\
        --output-file ./out.txt \\
        --file-extensions json txt md yaml

Notes:
    - Matches `embedding:NAME` where NAME is alphanumeric / dash /
      underscore / dot / forward-slash / backslash. Matches the same
      syntax ComfyUI's tokenizer reads from prompts.
    - Case-sensitive: `embedding:Foo` and `embedding:foo` are kept as
      separate entries (the plugin's runtime matcher is case-insensitive
      on lookups, but the canonical list preserves the user's casing).
    - PNG files: this script reads them as raw text (errors='replace') so
      any embedded workflow JSON metadata that contains `embedding:NAME`
      references will be picked up. Binary garbage is harmless.
"""

import argparse
import os
import re
import sys

EMBEDDING_REFERENCE_REGEX_PATTERN_MATCHING_COMFYUI_PROMPT_SYNTAX = re.compile(
    r"embedding:([\w./\\-]+)"
)


def _strip_any_folder_path_prefix_keeping_only_the_basename_for_embedding_name(
    captured_name_string_which_may_contain_folder_path_prefix,
):
    """
    Returns just the basename of an extracted embedding name. Handles both
    `/` and `\\` separators so subfolder references like
    `style\\lazyhand` or `style/lazyhand` collapse to `lazyhand` in the
    output list. Embedding files with the same base name in different
    subfolders are intentionally collapsed because the output list is
    intended for use as a name-matching filter, not as a full file index.
    """
    position_of_last_forward_slash = captured_name_string_which_may_contain_folder_path_prefix.rfind("/")
    position_of_last_back_slash = captured_name_string_which_may_contain_folder_path_prefix.rfind("\\")
    last_separator_position = max(position_of_last_forward_slash, position_of_last_back_slash)
    if last_separator_position >= 0:
        return captured_name_string_which_may_contain_folder_path_prefix[last_separator_position + 1:]
    return captured_name_string_which_may_contain_folder_path_prefix

DEFAULT_FILE_EXTENSIONS_TO_SCAN_RECURSIVELY = [
    "json",
    "txt",
    "md",
    "yaml",
    "yml",
    "png",
]


def find_all_files_under_root_folder_with_matching_extensions_recursively(
    root_folder_absolute_path, extension_list_without_leading_dots
):
    matched_file_absolute_paths_list = []
    lowercased_extensions_set_for_membership_test = set(
        extension_string.lower().lstrip(".") for extension_string in extension_list_without_leading_dots
    )
    for current_subdirectory_absolute_path, _ignored_subdir_names_list, file_basenames_in_current_subdir in os.walk(
        root_folder_absolute_path
    ):
        for one_file_basename in file_basenames_in_current_subdir:
            file_extension_lowercased_without_dot = (
                os.path.splitext(one_file_basename)[1].lstrip(".").lower()
            )
            if file_extension_lowercased_without_dot in lowercased_extensions_set_for_membership_test:
                matched_file_absolute_paths_list.append(
                    os.path.join(current_subdirectory_absolute_path, one_file_basename)
                )
    return matched_file_absolute_paths_list


def extract_embedding_names_from_one_file_returning_set_of_unique_names(file_absolute_path):
    embedding_names_found_in_this_file_set = set()
    try:
        with open(file_absolute_path, "r", encoding="utf-8", errors="replace") as input_file_handle:
            full_file_text_contents = input_file_handle.read()
    except OSError as os_error_opening_file:
        print(f"  warning: could not read {file_absolute_path}: {os_error_opening_file}", file=sys.stderr)
        return embedding_names_found_in_this_file_set
    for embedding_reference_regex_match_object in EMBEDDING_REFERENCE_REGEX_PATTERN_MATCHING_COMFYUI_PROMPT_SYNTAX.finditer(
        full_file_text_contents
    ):
        full_captured_name_with_possible_folder_path = embedding_reference_regex_match_object.group(1)
        basename_only_for_unique_list = (
            _strip_any_folder_path_prefix_keeping_only_the_basename_for_embedding_name(
                full_captured_name_with_possible_folder_path
            )
        )
        embedding_names_found_in_this_file_set.add(basename_only_for_unique_list)
    return embedding_names_found_in_this_file_set


def load_existing_embedding_name_list_text_file_into_set(existing_list_file_path_or_none):
    if not existing_list_file_path_or_none:
        return set()
    if not os.path.isfile(existing_list_file_path_or_none):
        print(
            f"  note: --existing-list path does not exist yet, starting from empty: "
            f"{existing_list_file_path_or_none}"
        )
        return set()
    with open(existing_list_file_path_or_none, "r", encoding="utf-8") as input_file_handle:
        return {
            line_text.strip()
            for line_text in input_file_handle
            if line_text.strip() and not line_text.strip().startswith("#")
        }


def write_sorted_embedding_name_set_to_text_list_file(
    full_set_of_unique_embedding_names_to_write, output_file_absolute_path
):
    sorted_names_case_insensitive_alphabetical = sorted(
        full_set_of_unique_embedding_names_to_write, key=lambda name_string: name_string.lower()
    )
    with open(output_file_absolute_path, "w", encoding="utf-8") as output_file_handle:
        for one_embedding_name_to_write in sorted_names_case_insensitive_alphabetical:
            output_file_handle.write(one_embedding_name_to_write + "\n")


def main_cli_entrypoint_for_standalone_script_invocation():
    argument_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    argument_parser.add_argument(
        "--source-folder",
        required=True,
        action="append",
        help="Folder to scan recursively for files containing embedding:NAME references. "
             "Pass --source-folder MULTIPLE TIMES to scan several folders in one run; "
             "their found names are merged into one unique list.",
    )
    argument_parser.add_argument(
        "--output-file",
        required=True,
        help="Path to write the sorted unique embedding name list (newline-separated).",
    )
    argument_parser.add_argument(
        "--existing-list",
        default=None,
        help="Optional existing list file. Its entries are loaded first, then merged with newly-found names. "
             "Can be the same path as --output-file to update-in-place.",
    )
    argument_parser.add_argument(
        "--file-extensions",
        nargs="+",
        default=DEFAULT_FILE_EXTENSIONS_TO_SCAN_RECURSIVELY,
        help=f"File extensions to scan (no dots). "
             f"Default: {DEFAULT_FILE_EXTENSIONS_TO_SCAN_RECURSIVELY}",
    )
    parsed_command_line_arguments_namespace = argument_parser.parse_args()

    list_of_all_source_folder_paths_passed_on_command_line = (
        parsed_command_line_arguments_namespace.source_folder
    )
    for one_source_folder_path_to_validate_exists in list_of_all_source_folder_paths_passed_on_command_line:
        if not os.path.isdir(one_source_folder_path_to_validate_exists):
            print(
                f"ERROR: source folder does not exist or is not a directory: "
                f"{one_source_folder_path_to_validate_exists}",
                file=sys.stderr,
            )
            sys.exit(1)

    existing_embedding_names_loaded_from_optional_list_file = (
        load_existing_embedding_name_list_text_file_into_set(
            parsed_command_line_arguments_namespace.existing_list
        )
    )
    if parsed_command_line_arguments_namespace.existing_list:
        print(
            f"Loaded {len(existing_embedding_names_loaded_from_optional_list_file)} existing "
            f"name(s) from --existing-list."
        )

    newly_found_embedding_names_from_corpus_walk = set()
    for one_source_folder_path_to_scan in list_of_all_source_folder_paths_passed_on_command_line:
        files_to_scan_for_embedding_references_under_this_source_folder = (
            find_all_files_under_root_folder_with_matching_extensions_recursively(
                one_source_folder_path_to_scan,
                parsed_command_line_arguments_namespace.file_extensions,
            )
        )
        print(
            f"Scanning {len(files_to_scan_for_embedding_references_under_this_source_folder)} file(s) under "
            f"{one_source_folder_path_to_scan} "
            f"(extensions: {parsed_command_line_arguments_namespace.file_extensions})..."
        )
        for one_file_absolute_path_to_scan in files_to_scan_for_embedding_references_under_this_source_folder:
            names_extracted_from_one_file = (
                extract_embedding_names_from_one_file_returning_set_of_unique_names(
                    one_file_absolute_path_to_scan
                )
            )
            newly_found_embedding_names_from_corpus_walk |= names_extracted_from_one_file

    full_merged_set_of_unique_embedding_names = (
        existing_embedding_names_loaded_from_optional_list_file | newly_found_embedding_names_from_corpus_walk
    )
    new_entries_count_added_beyond_existing = len(full_merged_set_of_unique_embedding_names) - len(
        existing_embedding_names_loaded_from_optional_list_file
    )

    write_sorted_embedding_name_set_to_text_list_file(
        full_merged_set_of_unique_embedding_names,
        parsed_command_line_arguments_namespace.output_file,
    )

    print(
        f"Found {len(newly_found_embedding_names_from_corpus_walk)} unique embedding name(s) in this scan. "
        f"{new_entries_count_added_beyond_existing} new beyond --existing-list. "
        f"Total names written: {len(full_merged_set_of_unique_embedding_names)}."
    )
    print(f"Wrote sorted list to: {parsed_command_line_arguments_namespace.output_file}")


if __name__ == "__main__":
    main_cli_entrypoint_for_standalone_script_invocation()
