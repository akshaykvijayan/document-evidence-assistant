from pathlib import Path

project_folder = Path(__file__).resolve().parent
document_path = project_folder / "data" / "travel_policy.txt"

document_text = document_path.read_text(encoding="utf-8")

print("Loaded document:", document_path.name)
print()
print(document_text)

#SEARCH A KEYWORD IN DOCUMENT 

# # Split the document into individual lines.
# lines = document_text.splitlines()

# # Ask the user what to search for.
# keyword = input("\nEnter a keyword: ").strip().lower()

# if not keyword:
#     print("Please enter a keyword.")
# else:
#     matches = []

#     for line in lines:
#         if keyword in line.lower():
#             matches.append(line)

#     if matches:
#         print("\nMatching passages:")

#         for match in matches:
#             print("-", match)
#     else:
#         print("No matching passages found.")

###########################################################################
import re

# Keep only non-empty lines.
passages = [
    line.strip()
    for line in document_text.splitlines()
    if line.strip()
]

# Ignore common words that add little value to this search.
stop_words = {
    "a", "an", "the", "is", "are", "for", "of", "to",
    "how", "much", "can", "i", "what", "my", "do"
}

question = input("\nAsk a question: ").strip()

# Extract lowercase words and remove duplicates.
question_words = set(re.findall(r"\b\w+\b", question.lower()))
question_words = question_words - stop_words

results = []

for passage in passages:
    passage_words = set(re.findall(r"\b\w+\b", passage.lower()))

    # The & operator finds words present in both sets.
    shared_words = question_words & passage_words
    score = len(shared_words)

    if score > 0:
        results.append((score, passage))

# Put passages with the highest scores first.
results.sort(key=lambda result: result[0], reverse=True)

if results:
    print("\nTop matching passages:")

    for score, passage in results[:3]:
        print(f"[Score: {score}] {passage}")
else:
    print("No matching passages found.")