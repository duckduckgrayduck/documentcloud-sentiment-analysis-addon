"""
    Sentiment Analysis Add-on
"""

import csv

import nltk
from documentcloud.addon import AddOn
from happytransformer import HappyTextClassification

# Download the sentence parser from NLTK.
# Gives the ability to break a bunch of text down into sentences.
nltk.download("punkt")

# Initialize the text classification model
tc = HappyTextClassification(
    model_type="DISTILBERT",
    model_name="distilbert-base-uncased-finetuned-sst-2-english",
    num_labels=2,
)


class Sentiment(AddOn):
    """ Add-On that uses NLTK to analyze sentiment in documents """
    def main(self):
        """ Breaks up documents into sentences for analysis, saves scores to CSV """
        with open("sentiment.csv", "w+", encoding="utf-8") as file_:
            writer = csv.writer(file_)
            writer.writerow(
                ["document_title", "sentence", "sentiment_label", "sentiment_valence"]
            )

            for document in self.get_documents():
                # Break document text into sentences
                sentences = nltk.tokenize.sent_tokenize(document.full_text)

                # For each sentence, write the document's title, which sentence in the document
                # we've analyzed, and what the sentiment breakdown is.
                for sentence in sentences:
                    # Check if sentence length exceeds the model's maximum sequence length
                    if len(sentence.split()) > 512:
                        # Split the sentence into chunks if > 512
                        midpoint = len(sentence) // 2
                        chunks = [sentence[:midpoint],sentence[midpoint:]]
                        for chunk in chunks:
                            sentiment_object = tc.classify_text(chunk)
                            writer.writerow(
                                [
                                    document.title,
                                    chunk,
                                    sentiment_object.label,
                                    sentiment_object.score,
                                ]
                            )
                    else:
                        sentiment_object = tc.classify_text(sentence)
                        writer.writerow(
                            [
                                document.title,
                                sentence,
                                sentiment_object.label,
                                sentiment_object.score,
                            ]
                        )

            self.upload_file(file_)


if __name__ == "__main__":
    Sentiment().main()
