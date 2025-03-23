import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from keybert import KeyBERT
from urllib.parse import urlparse
from difflib import get_close_matches
import pandas as pd
import tldextract
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
sentiment_analyzer = pipeline('sentiment-analysis')
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
kw_model = KeyBERT()
nlp = spacy.load("en_core_web_sm")

# Load bias data
bias_df = pd.read_csv("allsides.csv")
bias_df.columns = bias_df.columns.str.strip().str.lower()
bias_df['name_clean'] = bias_df['name'].str.lower().str.replace(r"[^\w\s]", "", regex=True)

# Set API keys
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY"
SERPER_API_KEY = "YOUR_SERPER_API_KEY"

st.title("📰 News in Context")
article_url = st.text_input("Paste a news article URL here")

if article_url:
    st.info("Fetching and analyzing the article...")

    def fetch_article_text(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            return ' '.join(p.get_text() for p in paragraphs).strip()
        except Exception as e:
            return f"Error: {e}"

    def extract_domain(url):
        return urlparse(url).netloc.replace("www.", "").lower()

    def get_bias_from_url(article_url):
        domain = extract_domain(article_url)
        domain_root = domain.split('.')[0]
        for _, row in bias_df.iterrows():
            if domain_root in row['name_clean'].replace(" ", ""):
                return row['name'], row['bias']
        cleaned_names = bias_df['name_clean'].str.replace(" ", "").tolist()
        close_match = get_close_matches(domain_root, cleaned_names, n=1, cutoff=0.7)
        if close_match:
            matched_row = bias_df[bias_df['name_clean'].str.replace(" ", "") == close_match[0]].iloc[0]
            return matched_row['name'], matched_row['bias']
        return "Unknown Source", "Unknown"

    def generate_adaptive_query(text, max_keywords=5, fallback_terms=None):
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=max_keywords)
        extracted = [kw[0] for kw in keywords]
        doc = nlp(text)
        named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "NORP"]]
        noun_chunks = [chunk.root.text for chunk in doc.noun_chunks]
        candidates = list(dict.fromkeys(named_entities + extracted + noun_chunks))
        query = ' '.join(candidates[:max_keywords]).strip()
        return query if query else ' '.join(fallback_terms or [])

    def search_related_articles_newsapi(query):
        url = "https://newsapi.org/v2/everything"
        params = {"q": query, "language": "en", "pageSize": 10, "apiKey": NEWSAPI_KEY}
        response = requests.get(url, params=params)
        return response.json().get("articles", []) if response.status_code == 200 else []

    def is_similar_to_article(fact_text, article_summary, threshold=0.4):
        texts = [fact_text.lower(), article_summary.lower()]
        vec = TfidfVectorizer().fit_transform(texts)
        return cosine_similarity(vec[0:1], vec[1:2])[0][0] >= threshold

    def search_fact_check_serper(query, article_summary):
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        payload = {"q": f"{query} site:politifact.com OR site:snopes.com OR site:factcheck.org", "num": 3}
        response = requests.post(url, headers=headers, json=payload)
        filtered = []
        if response.status_code == 200:
            for result in response.json().get("organic", []):
                title, snippet, link = result.get("title", ""), result.get("snippet", ""), result.get("link", "")
                if is_similar_to_article(f"{title}. {snippet}", article_summary):
                    filtered.append({"title": title, "url": link, "snippet": snippet, "source": "Fact-Check"})
        if not filtered:
            filtered.append({"title": "No relevant fact-checks found.", "url": "#", "snippet": "", "source": "Fact-Check"})
        return filtered

    text = fetch_article_text(article_url)
    summary = summarizer(text[:2000], max_length=100, min_length=30)[0]['summary_text']
    sentiment = sentiment_analyzer(text[:1000])[0]
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=5, stop_words='english')
    extracted_keywords = [kw[0] for kw in keywords]

    topics = classifier(' '.join(extracted_keywords), ["Politics", "Economy", "Sports", "Healthcare", "Technology", "Entertainment", "Crime", "Religion", "Science", "Education", "Obituary"])
    top_category = topics['labels'][0].title()

    source_name, bias = get_bias_from_url(article_url)

    st.subheader("📄 Summary")
    st.write(summary)

    st.subheader("😎 Sentiment")
    st.markdown(f"**Label:** {sentiment['label']}  ")
    st.markdown(f"**Confidence:** {sentiment['score']:.2%}")

    st.subheader("🔹 Topics")
    st.markdown(f"**Category:** {top_category}")
    st.write(extracted_keywords[:3])

    st.subheader("📰 Source & Bias")
    st.write(f"Source: {source_name}")
    st.write(f"Bias: {bias}")

    keyword_query = generate_adaptive_query(text, max_keywords=5, fallback_terms=["news"])
    related = search_related_articles_newsapi(keyword_query)
    fact_checks = search_fact_check_serper(keyword_query, summary)

    st.subheader("🌐 Related Articles")
    for r in related[:3]:
        st.markdown(f"**{r['title']}**  ")
        st.markdown(f"_{r['source']['name']}_")
        st.markdown(f"[{r['url']}]({r['url']})")

    st.subheader("✅ Fact Checks")
    for f in fact_checks:
        st.markdown(f"**{f['title']}**")
        st.markdown(f"[{f['url']}]({f['url']})")
