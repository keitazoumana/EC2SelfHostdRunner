
import pandas as pd 
import re

def remove_special_characters(text):

    return re.sub('[^A-Za-z0-9]+', ' ', text)


def prepare_data(path_data):

    data = pd.read_csv(path_data)
    data["text"] = data["text"].apply(lambda x: remove_special_characters(x))
    return data

def encode_sentiments_values(df):
    
    possible_sentiments = df.airline_sentiment.unique()
    sentiment_dict = {}
    
    for index, possible_sentiment in enumerate(possible_sentiments):
        sentiment_dict[possible_sentiment] = index
    
    # Encode all the sentiment values
    df['label'] = df.airline_sentiment.replace(sentiment_dict)

    return df 
 

if __name__ == '__main__':

    path_to_data = "./data/raw/airline_sentiment_data.csv"
    processed_data = prepare_data(path_to_data)


    # Encode the labels
    processed_data = encode_sentiments_values(processed_data)

    # Save the preproccesed data
    processed_data.to_csv('./data/preprocessed/airline_sentiment_preprocessed_data.csv')