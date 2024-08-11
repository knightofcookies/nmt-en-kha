from gensim.models import Word2Vec

# Load and preprocess your Khasi corpus
with open("kha_monolingual.txt", "r", encoding="utf-8") as file:
    khasi_sentences = [line.strip().split() for line in file]

# Train Word2Vec model for Khasi
khasi_model = Word2Vec(
    khasi_sentences, vector_size=256, window=5, min_count=1, workers=4
)
khasi_model.save("khasi_word2vec.model")

# Load and preprocess your English corpus
with open("en_monolingual.txt", "r", encoding="utf-8") as file:
    english_sentences = [line.strip().split() for line in file]

# Train Word2Vec model for English
english_model = Word2Vec(
    english_sentences, vector_size=256, window=5, min_count=1, workers=4
)
english_model.save("english_word2vec.model")
