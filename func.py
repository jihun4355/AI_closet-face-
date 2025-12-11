import csv

def load_from_file(filename):
    try:
        with open(filename, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            data = [row for row in reader]
        print(f"Data loaded from {filename}")
        return data
    except FileNotFoundError:
        print(f"{filename} not found.")
        return []