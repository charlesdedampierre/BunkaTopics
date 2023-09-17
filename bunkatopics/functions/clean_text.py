import re


def clean_tweet(tweet):
    temp = tweet.lower()
    temp = temp.replace("@ ", "@").replace("# ", "#")
    temp = re.sub("pic.twitter", "", temp)
    temp = re.sub("@[A-Za-z0-9_]+", "", temp)
    temp = re.sub("@[A-Za-z0-9_]+", "", temp)
    temp = re.sub("#[A-Za-z0-9_]+", "", temp)

    temp = re.sub(r"http\S+", "", temp)
    temp = re.sub(r"www.\S+", "", temp)

    temp = re.sub("[()!?]", " ", temp)
    temp = re.sub("\[.*?\]", " ", temp)

    temp = re.sub("[^a-z0-9]", " ", temp)

    return temp
