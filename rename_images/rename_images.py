"""Rename images in a specified directory using a visual-LLM."""

import os
import glob
import logging
import itertools
from dataclasses import dataclass

import tyro
import ollama
from tqdm import tqdm
from pydantic import BaseModel

SUPPORTED_IMAGE_EXTENSIONS = [
    "jpg",
    "jpeg",
    "png",
    "gif",
]


def get_extension(filename: str) -> str:
    """Return the file extension in lowercase."""

    _, ext = os.path.splitext(filename)
    return ext.lower()[1:]  # remove fullstop


def has_image_extension(filename: str) -> bool:
    return get_extension(filename) in SUPPORTED_IMAGE_EXTENSIONS


class ImageInfoScheme(BaseModel):
    """Scheme of image information for LLM structured generation."""

    description: str  # Detailed image description for guiding file name selection with LLM
    filename: str  # New file name selected by LLM



def generate_new_filename(model_name: str, imagepath: str) -> str:
    """Generate a new filename using a response of a LLM.

    Args:
        model_name (str): The name of the LLM from Ollama provider.
        imagepath (str): The path to the image file.

    Returns:
        str: A new filename.
    """

    if not os.path.exists(imagepath):
        raise FileNotFoundError(f"The image file {imagepath} does not exist.")

    messages = [
        {
            "role": "user",
            "content": (
                "Describe given image. "
                "Suggest not-trivial filename for the image."
            ),
            "images": [imagepath]
        }
    ]

    response: ollama.ChatResponse = ollama.chat(
        model_name,
        messages,
        format=ImageInfoScheme.model_json_schema(),
    )

    total_duration = (response.total_duration or 0) / 1e9
    eval_count = (response.eval_count or 0)
    logging.debug("LLM call duration: %.3f sec. Number of tokens: %d", total_duration, eval_count)

    generated_json_text = response.message.content
    assert isinstance(generated_json_text, str)
    image_info = ImageInfoScheme.model_validate_json(generated_json_text)

    return image_info.filename


def rename_file(
        path: str,
        filename: str,
        new_filename: str,
        max_attempts: int=1000,
        dry: bool=False,
        ):
    """Rename a file at the given path with the specified new filename.

    Args:
        path (str): The directory path where the file is located.
        filename (str): The new name for the file.
        new_filename (str): The new name for the file.
        max_attempts (int, optional): The maximum number of attempts to rename
            the file in case of conflicts. Defaults to 1000.
        dry (bool, optional): If True, perform a dry run without actually renaming the file.
            Defaults to False.

    Returns:
        None
    """

    if filename == new_filename:
        return

    # Search through all files to validate new file name
    existing_files = glob.glob("*", root_dir=path)

    # Filter only allowed image extensions
    existing_files = [f for f in existing_files if has_image_extension(f)]

    assert filename in existing_files

    _, file_ext = os.path.splitext(filename)
    new_filename, _ = os.path.splitext(new_filename)

    filename_candidate = f"{new_filename}{file_ext}"
    for i in itertools.count(2):
        if filename_candidate not in existing_files:
            new_filename = filename_candidate
            break

        filename_candidate = f"{new_filename}_{i}{file_ext}"

        if i >= max_attempts:
            raise ValueError(f"Exceeded maximum attempts ({max_attempts}) to find correct name.")

    logging.info("Rename: '%s' -> '%s'.", filename, new_filename)

    if dry:
        return

    try:
        filepath = os.path.join(path, filename)
        new_filepath = os.path.join(path, new_filename)
        os.rename(filepath, new_filepath)
    except (OSError, FileExistsError, IsADirectoryError, NotADirectoryError) as e:
        logging.error("Error: %s", e)


def rename_files(
        directory: str,
        pattern: str="*",
        model_name: str="gemma3:27b",
        dry: bool = False,
        ):
    """
    Renames files in the specified directory that match the given pattern.

    Args:
        directory (str): The path to the directory containing the files to be renamed.
        pattern (str, optional): A glob-style pattern to filter files. Defaults to "*".
        model_name (str, optional): The name of the model to use for generating new filenames.
            Defaults to "gemma3:27b".
        dry (bool, optional): If True, perform a dry run without actually renaming any files.
            Defaults to False.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """

    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The specified directory '{directory}' does not exist.")

    # Find images specified with pattern
    filenames = glob.glob(pattern, root_dir=directory)

    # Filter only allowed image extensions
    filenames = [f for f in filenames if has_image_extension(f)]

    logging.info("%d images have been found.", len(filenames))

    # Rename images
    for filename in tqdm(filenames):
        filepath = os.path.join(directory, filename)
        new_filename = generate_new_filename(model_name, filepath)
        rename_file(directory, filename, new_filename, dry=dry)


def main():
    """Create CLI for image renaming."""

    # Read command-line arguments
    args = tyro.cli(Args)

    logging_level = logging.DEBUG if args.verbose else logging.INFO
    # Configure logging to write to a file and the console
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler("log.log")
    file_handler.setLevel(logging_level)

    # Create console handler with the same level
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    rename_files(args.directory, args.pattern, args.model_name, args.dry)


@dataclass
class Args:
    """Arguments for images renaming."""

    directory: str = ""  # Directory with files to rename
    pattern: str = "*"  # Pattern to filter files
    model_name: str = "gemma3:27b"  # Name of LLM model from Ollama
    verbose: bool = False  # Output additional debug information
    dry: bool = False  # Do not rename actual files.


if __name__ == "__main__":
    main()
