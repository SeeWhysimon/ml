from nltk.tokenize import word_tokenize

pos = {"good", "better", "best", 
       "nice", "fantastic", "amazing", "pretty"}
neg = {"bad", "worse", "worst",
       "horrible", "disgusting", "shitty", "fucking"}

def find_sentiment(input_str: str, pos: set, neg: set) -> str:
    sentence = set(word_tokenize(input_str))
    
    num_pos = len(sentence.intersection(pos))
    num_neg = len(sentence.intersection(neg))

    if num_neg > num_pos:
        return "negative"
    elif num_pos > num_neg:
        return "positive"
    else:
        return "neutral"
    
if __name__ == "__main__":
    input_str = input("Enter a comment: ")
    result = find_sentiment(input_str, pos, neg)
    print(result)