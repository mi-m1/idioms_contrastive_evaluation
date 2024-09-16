import pandas as pd
import statistics as s

df_lit = pd.read_csv("literal_1032.csv")

# print(len(df.Idiom))
# print(len(set(df.Idiom)))

# print(len(df.Sentence))
# df.sentence

def average_words_per_sentence(sentences):
    total_sentences = len(sentences)
    total_words = 0
    
    # Calculate total number of words
    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        total_words += words_in_sentence
    
    # Calculate average number of words per sentence
    if total_sentences > 0:
        average_words = total_words / total_sentences
        return average_words
    else:
        return 0
val = average_words_per_sentence(df_lit.Sentence)
print(val)

def average_words_per_sentence_v2(sentences):
    total_sentences = len(sentences)
    words_count_per_sentence = []
    
    for sent in sentences:

        words_in_sentence = len(sent.split())

        words_count_per_sentence.append(words_in_sentence)

    return s.mean(words_count_per_sentence)


df_fig = pd.read_csv("figurative_1032.csv")

val2 = average_words_per_sentence_v2(df_fig.Sentence)
print(val2)





# # Example usage:
# if __name__ == "__main__":
#     # Example list of sentences
#     sentence_list = [
#         "This is a sample sentence.",
#         "It contains a few words.",
#         "Each sentence is separated by a period."
#     ]
    
#     average = average_words_per_sentence(sentence_list)
#     print(f"Average number of words per sentence: {average:.2f}")
