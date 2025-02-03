# feature_schema_manager.py

import json
from typing import Tuple, List, Dict
from jsonschema import validate, ValidationError

# JSON Schema definition for the feature schema.
FEATURE_SCHEMA_JSON_SCHEMA = {
    "type": "object",
    "description": "Mapping from feature name to its properties",
    "patternProperties": {
        "^.*$": {  # for any feature name
            "type": "object",
            "properties": {
                "feature_type": {
                    "type": "string",
                    "enum": ["categorical", "numerical"],
                    "description": "Type of the feature (categorical or numerical)"
                },
                "values": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Allowed values for categorical features, empty for numerical features"
                }
            },
            "required": ["feature_type", "values"],
            "additionalProperties": False,
            # If the feature is categorical, ensure that there is at least one value.
            "allOf": [
                {
                    "if": {
                        "properties": {"feature_type": {"const": "categorical"}}
                    },
                    "then": {
                        "properties": {"values": {"minItems": 1}}
                    }
                }
            ]
        }
    },
    "additionalProperties": False
}


def load_feature_schema(file_path: str) -> Tuple[Dict, List[str], Dict[str, List[str]]]:
    """
    Reads and validates a feature schema from a JSON file.
    Also extracts the feature name and value vocabularies.
    
    Args:
        file_path (str): Path to the JSON file containing the feature schema.
    
    Returns:
        Tuple containing:
          - feature_schema (dict): Mapping from feature name to its properties.
          - feature_name_vocabulary (List[str]): List of all feature names.
          - feature_value_vocabulary (Dict[str, List[str]]): Dictionary mapping each feature name
            to its list of allowed categorical values (or an empty list for numerical features).
    
    Raises:
        ValueError: If the loaded JSON does not match the expected schema.
    """
    with open(file_path, "r") as f:
        schema = json.load(f)
    try:
        validate(instance=schema, schema=FEATURE_SCHEMA_JSON_SCHEMA)
    except ValidationError as e:
        raise ValueError(f"Invalid feature schema in {file_path}: {e.message}")
    
    # Build the vocabularies.
    feature_name_vocabulary = list(schema.keys())
    feature_value_vocabulary = set()
    for feature_name, properties in schema.items():
        if properties["feature_type"] == "categorical":
            feature_value_vocabulary.update(properties["values"])
    feature_value_vocabulary = sorted(feature_value_vocabulary)                                        

    return schema, feature_name_vocabulary, feature_value_vocabulary


def save_feature_schema(schema: dict, file_path: str) -> None:
    """
    Validates and writes a feature schema to a JSON file.
    
    Args:
        schema (dict): The feature schema to save.
        file_path (str): Path where the JSON file will be written.
    
    Raises:
        ValueError: If the provided schema does not match the expected structure.
    """
    try:
        validate(instance=schema, schema=FEATURE_SCHEMA_JSON_SCHEMA)
    except ValidationError as e:
        raise ValueError(f"Provided schema is invalid: {e.message}")
    
    with open(file_path, "w") as f:
        json.dump(schema, f, indent=2)
        
