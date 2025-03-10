import boto3
from box.exceptions import BoxValueError
from box import ConfigBox
from pathlib import Path


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its content as a ConfigBox object.

    This function opens the specified YAML file, reads its content using the `yaml.safe_load()` function,
    and logs a success message using the `logger.info()` function. If the YAML file is empty,
    it raises a `ValueError` with the message "Yaml file is empty". If any other exception occurs,
    it is raised as is.

    Parameters:
    path_to_yaml (Path): The path to the YAML file to be read.

    Returns:
    ConfigBox: A ConfigBox object containing the content of the YAML file.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Yaml file : {path_to_yaml} loaded successfully")
        return ConfigBox(content)

    except BoxValueError:
        raise ValueError("Yaml file is empty")
    except Exception as e:
        raise e