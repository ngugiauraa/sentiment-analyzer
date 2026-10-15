from sentiment.text import clean_tweet


def test_removes_urls_and_mentions_but_keeps_hashtags_and_emoticons():
    assert clean_tweet("@jane loving it :) #happy http://t.co/x") == "loving it :) #happy"


def test_collapses_whitespace_and_handles_non_strings():
    assert clean_tweet("  so    tired \n today ") == "so tired today"
    assert clean_tweet(None) == "None"
    assert clean_tweet("www.example.com") == ""
