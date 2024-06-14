import json
import os


def read_coefs(coefs_dir: str, parcela_id: int) -> tuple:
    with open(os.path.join(coefs_dir, f'parcela_{parcela_id}.json'), 'r') as file:
        coefs = json.load(file)
        params = coefs[1]['params']
        return params[0], params[1], params[2]
