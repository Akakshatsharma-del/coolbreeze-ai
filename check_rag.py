"""
check_rag.py
Run with: python check_rag.py
Place this in your project root (same folder as manage.py).
"""

import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dj_ai_employee_main.settings")
django.setup()

from support.rag import collection

count = collection.count()
print("=" * 40)
print("CHUNK COUNT:", count)
print("=" * 40)

if count > 0:
    sample = collection.get(limit=1)
    print("Sample doc id:", sample["ids"][0] if sample["ids"] else "none")
    print("Sample text (first 200 chars):", sample["documents"][0][:200] if sample["documents"] else "none")
else:
    print("Collection is empty. Checking support/documents/ folder...")
    docs_path = "support/documents/"
    if not os.path.exists(docs_path):
        print(f"FOLDER DOES NOT EXIST: {docs_path}")
    else:
        files = os.listdir(docs_path)
        print(f"Files in {docs_path}: {files}")
        pdfs = [f for f in files if f.endswith(".pdf")]
        print(f"PDF files found: {pdfs}")