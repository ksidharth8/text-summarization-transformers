"""Text summarization toolkit (Deep Learning course project).

Modules
-------
preprocessing   cleaning of CNN/DailyMail boilerplate, rule-based sentence splitting
extractive      LEAD-k, TextRank and transformer-embedding (centroid + MMR) extractive summarizers
metrics         ROUGE with bootstrap confidence intervals, novel n-grams, length statistics
summarizer      inference wrapper for fine-tuned BART/T5/PEGASUS models, incl. long-document mode
"""

__version__ = "1.0.0"
