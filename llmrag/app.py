import os

from diagnostic_logic import run_diagnosis
from tutor import start_tutor
from build_index import build_index

def ensure_index():
    if not os.path.exists("index/lexipath_index.json"):
        print("Index not found. Building index now...\n")
        build_index()
        print()

def main():
    ensure_index()
    level = run_diagnosis()
    start_tutor(level)

if __name__ == "__main__":
    main()
