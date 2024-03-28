from transformers import pipeline
import sys

classifier = pipeline('sentiment-analysis')

# text = "I love coding in python!"

# text = input("Tell me whatever you're thinking right now: ")

text = sys.argv[1]

result = classifier(text)[0]

print(f"The text \"{text}\" was classified as {result['label']} with a score of {round(result['score'], 4) * 100}%")