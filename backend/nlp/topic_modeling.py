from typing import List
from bertopic import BERTopic

def get_topics(texts: List[str]) -> List[str]:
    model = BERTopic()
    topics, _ = model.fit_transform(texts)
    return model.get_topic_info()


