def word_wrap(text, width=87):
    """
    Wraps the given text to the specified width.

    Args:
    text (str): The text to wrap.
    width (int): The width to wrap the text to.

    Returns:
    str: The wrapped text.
    """
    return "\n".join([text[i : i + width] for i in range(0, len(text), width)])

def project_embeddings(embeddings, umap_transform):
    """
    Projects the given embeddings using the provided UMAP transformer.

    Args:
    embeddings (numpy.ndarray): The embeddings to project.
    umap_transform (umap.UMAP): The trained UMAP transformer.

    Returns:
    numpy.ndarray: The projected embeddings.
    """
    projected_embeddings = umap_transform.transform(embeddings)
    return projected_embeddings

def recursively_strip_dict(d):
    """
    Recursively strips whitespace from all string values in a dictionary.

    Args:
    d (dict): The dictionary to process.

    Returns:
    dict: A new dictionary with all string values stripped of whitespace.
    """
    if isinstance(d, dict):
        return {k: recursively_strip_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [recursively_strip_dict(item) for item in d]
    elif isinstance(d, str):
        return d.strip()
    else:
        return d

