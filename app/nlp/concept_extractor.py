import re
from typing import List, Dict, Any, Optional
from collections import Counter
import spacy
from loguru import logger


class ConceptExtractor:
    def __init__(self):
        self.nlp = None
        self._initialize_nlp()

    def _initialize_nlp(self):
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_sci_sm")
                logger.info("Loaded sciSpaCy model")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy small model")
                except OSError:
                    logger.warning("No spaCy model found, using fallback regex extraction")
                    self.nlp = None
        except ImportError:
            logger.warning("spaCy not installed, using regex-based extraction")
            self.nlp = None

    # Domain-specific concept patterns
    DOMAIN_PATTERNS = {
        "protein": r'\b[A-Z][a-z]+(?:\s+\d+)?(?:\s+[A-Z][a-z]+)*\b(?=\s+(?:protein|enzyme|receptor))',
        "gene": r'\b[A-Z]{2,5}\d?\b(?!\s+[a-z])',
        "disease": r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:syndrome|disease|disorder|cancer|tumor)\b',
        "drug": r'\b[A-Z][a-z]*(?:mab|nib|zom|cic|ximab|izumab|tinib)\b',
        "pathway": r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:pathway|signaling|mechanism)\b',
        "chemical": r'\b[A-Z][a-z]*\d*[A-Za-z]*\b',
    }

    def extract_concepts_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract concepts using spaCy NER and noun phrases."""
        if not self.nlp or not text:
            return []

        doc = self.nlp(text[:100000])  # Limit text length
        concepts = []
        seen = set()

        # Named entities
        for ent in doc.ents:
            if ent.label_ in {"PROTEIN", "GENE", "DISEASE", "CHEMICAL", "CELL", "DISEASE", "ORGANISM"}:
                key = ent.text.lower().strip()
                if key not in seen and len(ent.text) > 2:
                    concepts.append({
                        "name": ent.text.strip(),
                        "type": ent.label_.lower(),
                        "confidence": 0.85,
                        "positions": [[ent.start_char, ent.end_char]],
                    })
                    seen.add(key)

        # Noun chunks as generic concepts
        for chunk in doc.noun_chunks:
            key = chunk.text.lower().strip()
            if key not in seen and len(chunk.text) > 3 and chunk.text.strip():
                concepts.append({
                    "name": chunk.text.strip(),
                    "type": "generic",
                    "confidence": 0.6,
                    "positions": [[chunk.start_char, chunk.end_char]],
                })
                seen.add(key)

        return concepts[:50]  # Limit per paper

    def extract_concepts_regex(self, text: str) -> List[Dict[str, Any]]:
        """Fallback: extract concepts using regex patterns."""
        concepts = []
        seen = set()

        for concept_type, pattern in self.DOMAIN_PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                key = match.group().lower().strip()
                if key not in seen and len(match.group()) > 2:
                    concepts.append({
                        "name": match.group().strip(),
                        "type": concept_type,
                        "confidence": 0.5,
                        "positions": [match.span()],
                    })
                    seen.add(key)

        return concepts[:50]

    def extract_relationships(
        self, text: str, concepts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract relationships between concepts based on co-occurrence and syntactic patterns."""
        relationships = []
        if not concepts or len(concepts) < 2:
            return relationships

        name_to_concept = {c["name"]: c for c in concepts}
        concept_names = list(name_to_concept.keys())

        # Relationship patterns
        rel_patterns = [
            (r'\bregulates?\b', 'REGULATES'),
            (r'\binhibits?\b', 'INHIBITS'),
            (r'\bactivates?\b', 'ACTIVATES'),
            (r'\bbinds?\s+to\b', 'BINDS_TO'),
            (r'\binteracts?\s+with\b', 'INTERACTS_WITH'),
            (r'\bcauses?\b', 'CAUSES'),
            (r'\bassociated?\s+with\b', 'ASSOCIATED_WITH'),
            (r'\binduces?\b', 'INDUCES'),
            (r'\bpromotes?\b', 'PROMOTES'),
            (r'\bsuppresses?\b', 'SUPPRESSES'),
        ]

        # Check co-occurrence in sentences
        sentences = re.split(r'[.!?]+', text)
        window_size = 50  # words

        for pattern, rel_type in rel_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pos = match.start()
                # Find concepts near this relationship keyword
                nearby_concepts = []
                for name in concept_names:
                    name_positions = name_to_concept[name].get("positions", [])
                    for name_start, name_end in name_positions:
                        if abs(name_start - pos) < 200:  # Within 200 chars
                            nearby_concepts.append(name)

                if len(nearby_concepts) >= 2:
                    for i in range(len(nearby_concepts)):
                        for j in range(i + 1, len(nearby_concepts)):
                            relationships.append({
                                "source": nearby_concepts[i],
                                "target": nearby_concepts[j],
                                "type": rel_type,
                                "confidence": 0.5,
                                "evidence_snippet": text[max(0, pos - 100):pos + 100],
                            })

        # Deduplicate
        seen_rels = set()
        unique_rels = []
        for rel in relationships:
            key = (rel["source"], rel["target"], rel["type"])
            if key not in seen_rels:
                seen_rels.add(key)
                unique_rels.append(rel)

        return unique_rels[:30]  # Limit

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract concepts and relationships from text."""
        if not text or not text.strip():
            return {"concepts": [], "relationships": [], "method": "none"}

        # Extract concepts
        if self.nlp:
            concepts = self.extract_concepts_spacy(text)
            method = "spacy"
        else:
            concepts = self.extract_concepts_regex(text)
            method = "regex"

        # Extract relationships
        relationships = self.extract_relationships(text, concepts)

        return {
            "concepts": concepts,
            "relationships": relationships,
            "method": method,
        }


concept_extractor = ConceptExtractor()