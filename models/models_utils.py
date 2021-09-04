def rename_ordered_dict_from_parallel(ordered_dict):
    old_keys = list(ordered_dict.keys())
    for key in old_keys:
        key_new = key.replace("module.", "")
        ordered_dict[key_new] = ordered_dict.pop(key)

    return ordered_dict


def rename_ordered_dict_to_parallel(ordered_dict):
    old_keys = list(ordered_dict.keys())
    for key in old_keys:
        key_new = "module." + key
        ordered_dict[key_new] = ordered_dict.pop(key)

    return ordered_dict