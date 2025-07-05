import yaml
import json
import sys
import os

def convert_yaml_to_json(yaml_path, json_path=None):
    if not os.path.isfile(yaml_path):
        print(f"Error: File '{yaml_path}' not found.")
        return

    with open(yaml_path, 'r') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"YAML parse error: {e}")
            return

    if json_path is None:
        json_path = os.path.splitext(yaml_path)[0] + ".json"

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Converted '{yaml_path}' to '{json_path}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python yaml2json.py <input.yaml> [output.json]")
        sys.exit(1)

    yaml_file = sys.argv[1]
    json_file = sys.argv[2] if len(sys.argv) > 2 else None
    convert_yaml_to_json(yaml_file, json_file)
