def swap_nested_dict_axes(dictionary: dict) -> dict:
    """
    Perform a simple swapping of axes with nested dictionaries:
        dict_type1[Any, dict_type2] -> dict_type2[Any, dict_type1]
    """
    keys = []
    for nested_dicts in dictionary.values():
        keys += list(nested_dicts.keys())

    keys = set(keys)
    output = {}
    for key in keys:
        output[key] = {
            outer_key: nested_dict[key]
            for outer_key, nested_dict in dictionary.items()
            if key in nested_dict
        }

    return output


def scale_dict_values(dictionary: dict, scaling: int | float) -> dict:
    """
    Multiply all dictionary values by the `scaling` parameter
    """
    return {k: v * scaling for k, v in dictionary.items()}


def reshape_dict_list(dict_list: list[dict]) -> dict:
    """
    Turn a list of dictionaries into a dictionary of lists
    """
    keys = list(dict_list[0].keys())
    output = {i: [] for i in keys}
    for i in dict_list:
        for k, v in i.items():
            output[k].append(v)
    return output
