import os

from build_index import build_index
from config import INDEX_PATH
from diagnostic_logic import run_diagnosis
from tutor import start_tutor

def ensure_index():
    if not os.path.exists(INDEX_PATH):
        print("Index not found. Building index now...\n")
        build_index()
        print()

def main():
    ensure_index()
    print("Welcome.")
    print()
    level = run_diagnosis()
    start_tutor(level)

if __name__ == "__main__":
    main()
