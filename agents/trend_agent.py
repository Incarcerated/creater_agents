from pytrends.request import TrendReq


def get_trending_topics(limit=5):
    pytrends = TrendReq(hl='en-IN', tz=330)

    trends = pytrends.trending_searches(pn='india')

    topics = trends[0].tolist()

    # basic cleanup
    return [t.strip() for t in topics if t][:limit]