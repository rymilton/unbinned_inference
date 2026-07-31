import yaml
import os

def LoadJson(file_name, base_path="../configs"):
    JSONPATH = os.path.join(base_path, file_name)
    return yaml.safe_load(open(JSONPATH))


def LoadManifest(manifest_file, base_path="../configs"):
    """Load a dataset manifest: a YAML list of {path, parameters} records."""
    return LoadJson(manifest_file, base_path=base_path)


def GetReferenceFile(manifest):
    """Return the manifest record flagged as the reference dataset (is_reference: true)."""
    refs = [entry for entry in manifest if entry.get("is_reference", False)]
    if len(refs) != 1:
        raise ValueError(
            f"Expected exactly one manifest entry with is_reference: true, found {len(refs)}"
        )
    return refs[0]


def GetManifestEntry(manifest, path):
    """Return the full manifest record (path, parameters, ...) for a given file path."""
    for entry in manifest:
        if entry["path"] == path:
            return entry
    raise KeyError(f"File {path} not found in dataset manifest")


def GetFileParams(manifest, file_names, param_names):
    """Look up parameter values for each file in file_names, in param_names order.
    Returns a list of dicts (one per file), suitable for Dataset's file_params argument."""
    lookup = {entry["path"]: entry["parameters"] for entry in manifest}
    file_params = []
    for f in file_names:
        if f not in lookup:
            raise KeyError(f"File {f} not found in dataset manifest")
        params = lookup[f]
        missing = [p for p in param_names if p not in params]
        if missing:
            raise KeyError(f"File {f} missing manifest parameters: {missing}")
        file_params.append(params)
    return file_params