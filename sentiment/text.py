import re


def clean_tweet(text) -> str:
    """
    Basic tweet cleaning, applied the same way at training and prediction time:
    - strip URLs (no sentiment signal, just noise)
    - strip @mentions (usernames are not sentiment-bearing)
    - collapse repeated whitespace
    Hashtags, punctuation and emoticons are kept on purpose: ":)" or
    "#disappointed" carry real signal the model can learn from.
    """
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()
