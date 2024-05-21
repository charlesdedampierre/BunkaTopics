def wrap_by_word(string, n_words):
    """returns a string where \\n is inserted between every n words"""
    try:
        a = string.split()
        ret = ""
        for i in range(0, len(a), n_words):
            ret += " ".join(a[i : i + n_words]) + "<br>"
    except Exception as e:
        print(e)
    return ret


list_of_colors = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "cyan",
    "magenta",
    "teal",
    "lime",
    "indigo",
    "maroon",
    "navy",
    "olive",
    "coral",
    "lavender",
    "turquoise",
    "slategray",
]


def check_list_type(lst):
    if all(isinstance(item, str) for item in lst):
        return "string"
    elif all(isinstance(item, int) for item in lst):
        return "integer"


def normalize_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    normalized_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized_lst
