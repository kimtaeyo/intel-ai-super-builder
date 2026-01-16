"""
Semantic Routing Sample Code

Requirements:
pip install openai numpy
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from openai import OpenAI


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "C:/ProgramData/IntelAIA/local_models/bge-base-en-v1.5-int8-ov",
        base_url="http://localhost:8101/v3/embeddings",
    ):
        self.model = model_name
        self.client = OpenAI(base_url=base_url, api_key="unused")

    def encode(self, texts: List[str]) -> np.ndarray:
        responses = self.client.embeddings.create(input=texts, model=self.model)
        result = [np.array(res.embedding) for res in responses.data]
        return result

    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Normalize embeddings
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)

        # Calculate cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        return float(similarity)


@dataclass
class TestCase:
    """A labeled test query for evaluation"""

    query: str
    expected_route: str


class Route:
    """Represents a semantic route with precomputed embeddings"""

    def __init__(self, name: str, examples: List[str], embedding_model: EmbeddingModel):
        self.name = name
        self.examples = examples
        self.embedding_model = embedding_model
        self.example_embeddings = embedding_model.encode(examples)

    def score(self, query_embedding: np.ndarray) -> float:
        """Calculate how well this route matches the query embedding"""
        similarities = [
            self.embedding_model.cosine_similarity(query_embedding, example_emb)
            for example_emb in self.example_embeddings
        ]
        return max(similarities) if similarities else 0.0


class SemanticRouter:
    """Routes queries using real embeddings"""

    def __init__(self, embedding_model: EmbeddingModel, threshold: float = 0.5):
        self.embedding_model = embedding_model
        self.routes: List[Route] = []
        self.threshold = threshold

    def add_route(self, name: str, examples: List[str]) -> Route:
        """Add a new route to the router"""
        route = Route(name, examples, self.embedding_model)
        self.routes.append(route)
        return route

    def route(self, query: str) -> Tuple[str, float]:
        """Route a query and return the route name and confidence score"""
        # Encode the query
        query_embedding = self.embedding_model.encode([query])[0]

        # Find best matching route
        best_route = None
        best_score = 0.0

        for route in self.routes:
            score = route.score(query_embedding)
            if score > best_score:
                best_score = score
                best_route = route

        if best_route and best_score >= self.threshold:
            return best_route.name, best_score
        else:
            return "default", best_score


class ThresholdAnalyzer:
    """Analyzes routing performance across different thresholds"""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        routes_config: List[Tuple[str, List[str]]],
    ):
        self.embedding_model = embedding_model
        self.routes_config = routes_config

    def evaluate(self, test_cases: List[TestCase], threshold: float) -> Dict:
        """Evaluate routing performance at a given threshold"""
        # Create router with specified threshold
        router = SemanticRouter(self.embedding_model, threshold)

        print(f"\nEvaluating threshold {threshold:.2f}...")
        for route_name, examples in self.routes_config:
            router.add_route(route_name, examples)

        # Evaluate
        correct = 0
        routed_to_default = 0
        confusion_matrix = {}
        scores = []

        for test_case in test_cases:
            predicted, score = router.route(test_case.query)
            expected = test_case.expected_route
            scores.append(score)

            if predicted == expected:
                correct += 1

            if predicted == "default":
                routed_to_default += 1

            key = (expected, predicted)
            confusion_matrix[key] = confusion_matrix.get(key, 0) + 1

        total = len(test_cases)
        accuracy = correct / total if total > 0 else 0
        default_rate = routed_to_default / total if total > 0 else 0
        avg_score = sum(scores) / len(scores) if scores else 0

        return {
            "threshold": threshold,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "default_rate": default_rate,
            "avg_score": avg_score,
            "confusion_matrix": confusion_matrix,
        }

    def analyze_threshold_range(
        self,
        test_cases: List[TestCase],
        start: float = 0.3,
        end: float = 0.9,
        step: float = 0.05,
    ) -> List[Dict]:
        """Analyze performance across a range of thresholds"""
        results = []
        threshold = start

        while threshold <= end:
            result = self.evaluate(test_cases, round(threshold, 2))
            results.append(result)
            threshold += step

        return results

    def find_optimal_threshold(
        self,
        test_cases: List[TestCase],
        metric: str = "accuracy",
        start: float = 0.3,
        end: float = 0.9,
        step: float = 0.05,
    ) -> Tuple[float, Dict]:
        """Find the threshold that maximizes the given metric"""
        results = self.analyze_threshold_range(test_cases, start, end, step)

        best_threshold = 0.0
        best_score = 0.0
        best_result = None

        for result in results:
            score = result[metric]
            if score > best_score:
                best_score = score
                best_threshold = result["threshold"]
                best_result = result

        return best_threshold, best_result


def create_example_system():
    """Create example routes configuration and test cases"""

    # Define routes (name, examples)
    routes_config = [
        (
            "get_mkt_name",
            [
                "What is the MKT name of this laptop?",
                "Can you tell me the model name of this laptop?",
                "What is this laptop model called?",
                "I'd like to know the MKT name of this laptop.",
                "Could you please provide the laptop's MKT name?",
            ],
        ),
        (
            "get_hw_info",
            [
                "What are the hardware specifications of this laptop?",
                "Check the hardware specifications of this laptop.",
                "I'd like to know the specs of this laptop, especially the CPU, RAM, and storage.",
                "Can you tell me the HW details of this laptop?",
                "Could you provide information about the processor and memory of this laptop?",
            ],
        ),
        (
            "get_speaker_device",
            [
                "Select audio output.",
                "Change sound device.",
                "Which audio device to use?",
                "Choose playback device.",
                "Change sound output.",
            ],
        ),
        (
            "open_live_caption",
            [
                "Turn on live captions.",
                "Enable captions.",
                "Activate live text.",
                "Display what is being said.",
                "Start displaying subtitles.",
            ],
        ),
    ]

    # Define test cases with expected routes
    test_cases = [
        TestCase("What's the brand/model designation of this laptop?", "get_mkt_name"),
        TestCase("Could you identify the MKT name for me?", "get_mkt_name"),
        TestCase("What model of laptop is this?", "get_mkt_name"),
        TestCase("Could you tell me what this laptop is called?", "get_mkt_name"),
        TestCase("What's the brand/model designation of this laptop?", "get_mkt_name"),
        TestCase(
            "What are the technical specifications of this laptop?", "get_hw_info"
        ),
        TestCase("Can you show me the hardware specs for this laptop?", "get_hw_info"),
        TestCase("Please check this laptop's hardware specifications", "get_hw_info"),
        TestCase(
            "I need to know this laptop's specs, particularly the CPU, RAM, and storage.?",
            "get_hw_info",
        ),
        TestCase("Choose audio output.", "get_speaker_device"),
        TestCase("Select the audio device.", "get_speaker_device"),
        TestCase("Switch audio output device.", "get_speaker_device"),
        TestCase("Change audio output device.", "get_speaker_device"),
        TestCase("Turn on captions.", "open_live_caption"),
        TestCase("Enable live captions.", "open_live_caption"),
        TestCase("Switch on live captions", "open_live_caption"),
        TestCase("Activate captions.", "open_live_caption"),
    ]

    return routes_config, test_cases


def print_recommendations(optimal_threshold: float, result: Dict):
    """Print recommendations for threshold selection"""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print(f"\nOptimal Threshold: {optimal_threshold:.2f}")
    print(f"Expected Accuracy: {result['accuracy']:.2%}")
    print(f"Default Route Rate: {result['default_rate']:.2%}")
    print(f"Average Similarity Score: {result['avg_score']:.3f}")


if __name__ == "__main__":

    # Initialize embedding model
    embedding_model = EmbeddingModel()

    # Create system
    routes_config, test_cases = create_example_system()

    # ***********[START] this section is optional, only if you want to find optimial threshold for your use case
    # Create analyzer
    analyzer = ThresholdAnalyzer(embedding_model, routes_config)

    # Analyze threshold range
    min_threshold = 0.3
    max_threshold = 0.9
    step = 0.05

    # Find optimal threshold
    optimal_threshold, best_result = analyzer.find_optimal_threshold(
        test_cases, start=min_threshold, end=max_threshold, step=step
    )

    # Print recommendations
    print_recommendations(optimal_threshold, best_result)
    # ***********this section is optional, only if you want to find optimial threshold for your use case [END]

    # route with recommended threshold
    # Create router
    router = SemanticRouter(embedding_model, optimal_threshold)
    for route_name, examples in routes_config:
        router.add_route(route_name, examples)
    for test_case in test_cases:
        query = test_case.query
        route_name, score = router.route(query)
        print(f"\nQuery: {query}")
        print(f"Route: {route_name}")
        print(f"Score: {score}")
        print("-" * 60)

    guidelines = """Guidelines for threshold selection with embeddings:
    • Low threshold (e.g., 0.3-0.5): More routing, captures semantic similarity
    • Medium threshold (e.g., 0.5-0.7): Balanced precision and recall
    • High threshold (e.g., 0.7-0.85): Conservative, high confidence matches only
    • Very high threshold (e.g., 0.85+): Extremely strict
    
Usually embedding works well with thresholds of 0.5-0.7
But the optimal threshold depends on your data!"""

    print("\n" + "=" * 80)
    print(guidelines)
    print("=" * 80 + "\n")

    print("\n" + "=" * 80)
    print("TIP: Adjust test_cases to match your real use case for accurate tuning!")
    print("=" * 80)
