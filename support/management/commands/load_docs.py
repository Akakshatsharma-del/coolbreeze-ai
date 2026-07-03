from django.core.management.base import BaseCommand
from support.rag import load_documents


class Command(BaseCommand):
    help = "Load company documents into ChromaDB for RAG"

    def handle(self, *args, **options):
        self.stdout.write("Loading documents into ChromaDB...")
        try:
            load_documents()
            self.stdout.write(self.style.SUCCESS("Done."))
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"RAG document load failed (non-fatal): {e}"))