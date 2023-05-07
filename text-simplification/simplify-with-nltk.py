import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

if __name__ == '__main__':

    text = "While pondering over the intricacies of linguistics and the convolutions of grammar, the linguist, having been deep in thought for hours, inadvertently overlooked the time and missed his appointment with the head of the department, much to his chagrin and embarrassment."

    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w.lower() in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    simplified_text = ' '.join(words)

    print(simplified_text)