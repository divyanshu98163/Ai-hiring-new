from __future__ import annotations

import csv
import io
import json
import math
import os
import pickle
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from docx import Document
from PyPDF2 import PdfReader


DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/brackozi/Resume/raw/main/UpdatedResumeDataSet.csv"
)
DEFAULT_KAGGLE_SOURCE = (
    "https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset"
)

KNOWN_SKILLS = [
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "React",
    "Next.js",
    "Node.js",
    "FastAPI",
    "Django",
    "Flask",
    "AWS",
    "Azure",
    "GCP",
    "Docker",
    "Kubernetes",
    "SQL",
    "PostgreSQL",
    "MongoDB",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "TensorFlow",
    "PyTorch",
    "Power BI",
    "Tableau",
    "Figma",
    "UI/UX",
    "Salesforce",
    "SEO",
    "Content Marketing",
    "Project Management",
    "Agile",
]

STOPWORDS = {
    "and",
    "the",
    "with",
    "from",
    "that",
    "this",
    "your",
    "have",
    "will",
    "for",
    "are",
    "you",
    "about",
    "using",
    "build",
    "built",
    "team",
    "teams",
    "role",
    "resume",
    "candidate",
    "experience",
}

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z+#.]{1,}")
MAX_FEATURES = 6000


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def clean_lines(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if re.sub(r"\s+", " ", line).strip()
    ]


def sentence_slice(text: str, max_length: int = 260) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[:max_length].strip()}..."


def keyword_token(token: str) -> str | None:
    cleaned = re.sub(r"[^a-zA-Z+#.]", "", token).lower()
    if len(cleaned) < 3 or cleaned in STOPWORDS:
        return None
    return cleaned


def format_skill(skill: str) -> str:
    parts = []
    for part in skill.split():
        if len(part) <= 3:
            parts.append(part.upper())
        else:
            parts.append(f"{part[0].upper()}{part[1:]}")
    return " ".join(parts)


def extract_skills(text: str, extra_skills: list[str] | None = None) -> list[str]:
    lowered = text.lower()
    discovered = [skill for skill in KNOWN_SKILLS if skill.lower() in lowered]

    keyword_candidates = [
        token
        for token in (keyword_token(chunk) for chunk in re.split(r"[\s,./()\-:;]+", text))
        if token
    ]
    counts: dict[str, int] = {}
    for token in keyword_candidates:
        counts[token] = counts.get(token, 0) + 1

    ranked_keywords = [
        format_skill(token)
        for token, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8]
    ]
    merged = unique([*(extra_skills or []), *discovered, *ranked_keywords])
    return merged[:12]


def extract_section_lines(
    text: str, keywords: list[str], fallback_pattern: str | None = None
) -> list[str]:
    lines = clean_lines(text)
    matches = [
        line for line in lines if any(keyword in line.lower() for keyword in keywords)
    ]

    if matches:
        return unique(matches)[:4]

    if not fallback_pattern:
        return []

    regex = re.compile(fallback_pattern)
    return unique([line for line in lines if regex.search(line)])[:4]


def extract_location(text: str) -> str:
    remote_match = re.search(r"\b(remote|hybrid|onsite)\b", text, re.I)
    if remote_match:
        return remote_match.group(1)

    location_match = re.search(
        r"(?:location[:\s-]+)?([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*,\s?[A-Z][a-zA-Z. ]{2,})",
        text,
    )
    return location_match.group(1) if location_match else ""


def estimate_years_experience(text: str) -> int:
    explicit_years = [
        int(match.group(1))
        for match in re.finditer(r"(\d{1,2})\+?\s*(?:years?|yrs?)", text, re.I)
    ]
    if explicit_years:
        return int(clamp(max(explicit_years), 0, 40))

    timeline_years = [int(match.group(1)) for match in re.finditer(r"\b(20\d{2})\b", text)]
    if len(timeline_years) >= 2:
        return int(clamp(max(timeline_years) - min(timeline_years), 0, 40))

    return 0


def infer_headline(text: str, filename: str) -> str:
    for line in clean_lines(text):
        if 6 < len(line) < 100 and "@" not in line and not re.search(r"\d{7,}", line):
            return line
    return re.sub(r"[_-]+", " ", Path(filename).stem).strip()


def build_suggestions(text: str, skills: list[str]) -> list[str]:
    suggestions: list[str] = []

    if not re.search(r"\d+[%+]?", text):
        suggestions.append("Add measurable outcomes to show hiring impact.")

    if len(skills) < 5:
        suggestions.append("List more tools and technologies to improve match precision.")

    if not re.search(r"lead|mentor|manage|owned|launched", text, re.I):
        suggestions.append("Highlight ownership and leadership moments for stronger screening.")

    return suggestions[:3]


def build_job_text(job: dict[str, Any]) -> str:
    parts = [
        job.get("title") or "",
        job.get("category") or "",
        job.get("description") or "",
        " ".join(job.get("skills") or []),
        job.get("employment_type") or "",
        job.get("location") or "",
    ]
    return " ".join(part for part in parts if part).strip()


def tokenize(text: str) -> list[str]:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    anchor = max(scores)
    exps = [math.exp(score - anchor) for score in scores]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def normalize_vector(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return {}
    return {key: value / norm for key, value in vector.items()}


def dot_product(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


@dataclass
class ResumeInsights:
    category: str
    category_confidence: float
    certifications: list[str]
    confidence: float
    education: list[str]
    experience: list[str]
    extracted_text: str
    headline: str
    location: str
    projects: list[str]
    skills: list[str]
    suggestions: list[str]
    summary: str
    years_experience: int


class ResumeMLService:
    def __init__(self) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "resume_model.joblib"
        self.model: dict[str, Any] | None = None
        self.metadata: dict[str, Any] = {}

    def load(self) -> None:
        if self.model is not None:
            return

        if not self.model_path.exists():
            self.train_from_source()

        with self.model_path.open("rb") as handle:
            artifact = pickle.load(handle)

        self.model = artifact["model"]
        self.metadata = artifact["metadata"]

    def train_from_source(self) -> Path:
        dataset_url = os.environ.get("RESUME_DATASET_URL", DEFAULT_DATASET_URL)
        dataset_path = self.data_dir / "UpdatedResumeDataSet.csv"

        if not dataset_path.exists():
            self.download_dataset(dataset_url, dataset_path)

        labels, texts = self._read_dataset(dataset_path)
        if len(texts) < 20:
            raise ValueError("Resume dataset is too small to train the classifier.")

        indices = list(range(len(texts)))
        random.Random(42).shuffle(indices)
        split_index = max(1, int(len(indices) * 0.8))
        train_indices = indices[:split_index]
        test_indices = indices[split_index:] or indices[:1]

        train_texts = [texts[index] for index in train_indices]
        train_labels = [labels[index] for index in train_indices]
        test_texts = [texts[index] for index in test_indices]
        test_labels = [labels[index] for index in test_indices]

        model = self._build_model(train_texts, train_labels)
        predictions = [self._predict_label(text, model)[0] for text in test_texts]
        correct = sum(1 for predicted, actual in zip(predictions, test_labels) if predicted == actual)
        accuracy = correct / max(1, len(test_labels))

        artifact = {
            "model": model,
            "metadata": {
                "accuracy": accuracy,
                "kaggle_source": DEFAULT_KAGGLE_SOURCE,
                "mirror_source": dataset_url,
                "labels": sorted(set(labels)),
                "record_count": len(texts),
            },
        }

        with self.model_path.open("wb") as handle:
            pickle.dump(artifact, handle)

        self.model = model
        self.metadata = artifact["metadata"]
        return self.model_path

    def download_dataset(self, dataset_url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=90.0, follow_redirects=True) as client:
            response = client.get(dataset_url)
            response.raise_for_status()
            destination.write_bytes(response.content)

    def extract_text(self, file_bytes: bytes, filename: str, content_type: str) -> str:
        lower_name = filename.lower()

        if lower_name.endswith(".pdf") or "pdf" in content_type:
            reader = PdfReader(io.BytesIO(file_bytes))
            return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

        if lower_name.endswith(".docx") or "wordprocessingml" in content_type:
            document = Document(io.BytesIO(file_bytes))
            return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()

        return file_bytes.decode("utf-8", errors="ignore").strip()

    def predict_resume(self, text: str, filename: str) -> ResumeInsights:
        self.load()
        assert self.model is not None

        normalized_text = text.replace("\x00", " ").strip()
        if not normalized_text:
            raise ValueError("We could not extract readable text from this file.")

        category, category_confidence, probabilities = self._predict_label(
            normalized_text, self.model
        )

        skills = extract_skills(normalized_text, [category])
        experience = extract_section_lines(
            normalized_text,
            ["experience", "engineer", "manager", "developer"],
            r"\b20\d{2}\b",
        )
        education = extract_section_lines(
            normalized_text,
            ["education", "university", "college", "bachelor", "master"],
        )
        projects = extract_section_lines(
            normalized_text,
            ["project", "launched", "built", "shipped"],
        )
        certifications = extract_section_lines(
            normalized_text,
            ["certification", "certified", "aws", "google", "microsoft"],
        )
        years_experience = estimate_years_experience(normalized_text)
        summary = sentence_slice(normalized_text)
        headline = infer_headline(normalized_text, filename)
        location = extract_location(normalized_text)
        suggestions = build_suggestions(normalized_text, skills)
        probability_values = list(probabilities.values())
        confidence = float(
            clamp(
                0.42
                + min(len(skills), 8) * 0.04
                + (0.14 if experience else 0)
                + (0.08 if education else 0)
                + max(probability_values, default=0.0) * 0.2,
                0.45,
                0.98,
            )
        )

        return ResumeInsights(
            category=category,
            category_confidence=category_confidence,
            certifications=certifications,
            confidence=confidence,
            education=education,
            experience=experience,
            extracted_text=normalized_text,
            headline=headline,
            location=location,
            projects=projects,
            skills=skills,
            suggestions=suggestions,
            summary=summary,
            years_experience=years_experience,
        )

    def rank_jobs(
        self,
        resume_text: str,
        resume_skills: list[str],
        predicted_category: str,
        jobs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        self.load()
        assert self.model is not None

        if not jobs:
            return []

        resume_vector = self._vectorize_text(resume_text, self.model)
        scored_jobs: list[dict[str, Any]] = []

        for job in jobs:
            job_text = build_job_text(job)
            job_vector = self._vectorize_text(job_text, self.model)
            similarity = float(dot_product(resume_vector, job_vector))
            job_skills = extract_skills(job_text, job.get("skills") or [])
            matched_skills = [skill for skill in job_skills if skill in resume_skills][:6]
            category_boost = (
                0.12
                if predicted_category.lower() in (job.get("category") or "").lower()
                or predicted_category.lower() in (job.get("title") or "").lower()
                else 0.0
            )
            skill_boost = min(len(matched_skills) * 0.04, 0.22)
            combined_score = clamp(similarity + category_boost + skill_boost, 0.08, 1.0)

            scored_jobs.append(
                {
                    "jobId": job["id"],
                    "matchedSkills": matched_skills,
                    "reasonSummary": (
                        f"{', '.join(matched_skills[:3])} align strongly with this role."
                        if matched_skills
                        else "The resume overlaps with the role profile, but needs more targeted skills."
                    ),
                    "score": int(round(clamp(25 + combined_score * 73, 18, 98))),
                    "similarity": similarity,
                }
            )

        return sorted(scored_jobs, key=lambda item: item["score"], reverse=True)

    def export_metadata(self) -> dict[str, Any]:
        self.load()
        return dict(self.metadata)

    def _read_dataset(self, dataset_path: Path) -> tuple[list[str], list[str]]:
        labels: list[str] = []
        texts: list[str] = []

        with dataset_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                category_field = "Category" if "Category" in reader.fieldnames else reader.fieldnames[0]
                resume_field = "Resume" if "Resume" in reader.fieldnames else reader.fieldnames[1]
                for row in reader:
                    label = (row.get(category_field) or "").strip()
                    text = (row.get(resume_field) or "").strip()
                    if label and len(text) > 80:
                        labels.append(label)
                        texts.append(text)

        return labels, texts

    def _build_model(self, texts: list[str], labels: list[str]) -> dict[str, Any]:
        tokenized_docs = [tokenize(text) for text in texts]
        term_counts: Counter[str] = Counter()
        doc_frequencies: Counter[str] = Counter()

        for tokens in tokenized_docs:
            term_counts.update(tokens)
            doc_frequencies.update(set(tokens))

        vocabulary = [
            token
            for token, _ in sorted(
                term_counts.items(),
                key=lambda item: (doc_frequencies[item[0]], item[1]),
                reverse=True,
            )[:MAX_FEATURES]
        ]
        vocabulary_set = set(vocabulary)
        idf = {
            token: math.log((1 + len(tokenized_docs)) / (1 + doc_frequencies[token])) + 1.0
            for token in vocabulary
        }

        centroids: dict[str, dict[str, float]] = defaultdict(dict)
        label_counts: Counter[str] = Counter()

        for tokens, label in zip(tokenized_docs, labels):
            vector = self._vectorize_tokens(tokens, vocabulary_set, idf)
            label_counts[label] += 1
            for token, weight in vector.items():
                centroids[label][token] = centroids[label].get(token, 0.0) + weight

        normalized_centroids = {}
        for label, centroid in centroids.items():
            count = max(1, label_counts[label])
            averaged = {token: weight / count for token, weight in centroid.items()}
            normalized_centroids[label] = normalize_vector(averaged)

        return {
            "centroids": normalized_centroids,
            "idf": idf,
            "vocabulary": vocabulary,
        }

    def _predict_label(
        self, text: str, model: dict[str, Any]
    ) -> tuple[str, float, dict[str, float]]:
        vector = self._vectorize_text(text, model)
        centroids = model["centroids"]
        labels = sorted(centroids.keys())
        scores = [dot_product(vector, centroids[label]) for label in labels]
        probabilities = softmax(scores)
        probability_map = {label: probability for label, probability in zip(labels, probabilities)}
        best_label = max(probability_map, key=probability_map.get)
        return best_label, probability_map[best_label], probability_map

    def _vectorize_text(self, text: str, model: dict[str, Any]) -> dict[str, float]:
        return self._vectorize_tokens(tokenize(text), set(model["vocabulary"]), model["idf"])

    def _vectorize_tokens(
        self,
        tokens: list[str],
        vocabulary: set[str],
        idf: dict[str, float],
    ) -> dict[str, float]:
        filtered = [token for token in tokens if token in vocabulary]
        if not filtered:
            return {}

        counts = Counter(filtered)
        total = sum(counts.values()) or 1
        raw_vector = {
            token: (count / total) * idf.get(token, 1.0)
            for token, count in counts.items()
        }
        return normalize_vector(raw_vector)


def write_metadata_file(target: Path, metadata: dict[str, Any]) -> None:
    target.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
