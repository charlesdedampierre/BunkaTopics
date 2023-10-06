promp_template_topics_terms = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant in Topic Modeling. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
<</SYS>>

I have data about {context}

Within the data I have a topic that is described by the following keywords: 
{terms}

Here are some examples of documents in the topic:
{documents}:

Based on the keywords documents in the topic, create a label in 4 or 5 words that summarizes best the topic.

Only give the name of the topic and nothing else in {language}:[/INST]

Topic Name:"""

promp_template_topics_terms_no_docs = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant in Topic Modeling. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
<</SYS>>


I have data about {context}

I have a topic that is described the following keywords: 
{terms}

Based on the keywrods about the topic, create a short label that summarizes best the topic.

Only give the name of the topic and nothing else in {language}:[/INST]

Topic Name:"""
