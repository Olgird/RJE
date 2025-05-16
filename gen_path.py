import json
import random


def convert_paths_to_str(paths):
    if len(paths) == 0:
        return "None"
    
    formatted_paths = []
    for path in paths:
        formatted_path = ""
        for i, element in enumerate(path):
            if i % 2 == 0:
                formatted_path += f"[{element}] "
            else:
                formatted_path += f"→ ({element}) → "
                
        formatted_path = formatted_path.rstrip(" → ")
        formatted_paths.append(formatted_path)
    
    return "\n".join(formatted_paths)

