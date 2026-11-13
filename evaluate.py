from pathlib import Path
from sentence_transformers import SentenceTransformer

from semantic_search import load_passages, retrieve_passages


def main():
    project_folder = Path(__file__).resolve().parent
    document_path = project_folder / "data" / "travel_policy.txt"

    passages = load_passages(document_path)

    print("Loading model...")
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    passage_embeddings = model.encode(passages)

    # Each question has one expected supporting passage.
    test_cases = [
        {
            "question": "What is the hotel reimbursement limit?",
            "expected_passage": (
                "Hotel expenses are reimbursed up to 100 euros per night."
            )
        },
        {
            "question": "When must I submit receipts?",
            "expected_passage": (
                "Employees must submit receipts within 30 days "
                "of completing their trip."
            )
        },
        {
            "question": "Are personal holidays reimbursed?",
            "expected_passage": (
                "Personal holidays and entertainment expenses "
                "are not reimbursed."
            )
        }
    ]

    passed = 0

    for case in test_cases:
        scores, indices = retrieve_passages(
            case["question"], model, passage_embeddings
        )

        retrieved_passages = [
            passages[index] for index in indices
        ]

        found = case["expected_passage"] in retrieved_passages
        first_is_correct = (
            bool(retrieved_passages)
            and retrieved_passages[0] == case["expected_passage"]
        )
        if found:
            passed += 1

        status = "PASS" if found else "FAIL"

        print(f"\nQuestion: {case['question']}")
        print(f"Expected evidence retrieved: {status}")
        print(f"Expected evidence ranked first: {first_is_correct}")
    print(
        f"\nEvidence found in top 3: "
        f"{passed}/{len(test_cases)} questions"
    )


if __name__ == "__main__":
    main()