# Databricks notebook source
# Run once, then comment out
# %pip install transformers==4.36.2 sentence-transformers keybert beautifulsoup4 requests tldextract

# COMMAND ----------

#%pip install -U spacy
#!python -m spacy download en_core_web_sm

# COMMAND ----------

# ✅ Create widget safely — only if it doesn't exist already
try:
    dbutils.widgets.get("Paste URL here")
except:
    dbutils.widgets.text("Paste URL here", "")

# ✅ Always capture the current value at runtime
article_url = dbutils.widgets.get("Paste URL here")

# COMMAND ----------

displayHTML("""
<div style="font-family:Segoe UI, sans-serif; margin-bottom:2px;">
  <h3 style="margin-bottom:2px;">Start by Pasting a News Article URL</h3>
  <p style="font-size:16px; color:#666;">
    Your article will be analyzed for topic, tone, and viewpoint context.
    This process takes just a few seconds. Paste the URL above and hit <b>Run All</b>.
  </p>
</div>
""")


# COMMAND ----------

import requests
from bs4 import BeautifulSoup

def fetch_article_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text (simplified extraction)
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])

        return text.strip()
    except Exception as e:
        return f"Error: {e}"

# Fetch text
article_text = fetch_article_text(article_url)
print(article_text[:500])  # Preview first 500 chars

# COMMAND ----------

from transformers import pipeline

# Quick summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Summarize your article text (limit input size for speed)
summary_result = summarizer(article_text[:2000], max_length=100, min_length=30, do_sample=False)

# Extract and print the summary
summary_text = summary_result[0]['summary_text']

# Example: summary_text = summary_result[0]['summary_text']

from urllib.parse import urlparse

# Example values (replace with dynamic vars)
article_title = "AI-Generated Summary"
summary_snippet = summary_text  # From summarizer
article_url = article_url  # Already pulled from widget
parsed_url = urlparse(article_url)
source_site = parsed_url.netloc.replace("www.", "").title()

# Unique ID for HTML toggle block
import uuid
block_id = f"summary_{uuid.uuid4().hex[:6]}"

# HTML block
displayHTML(f"""
<div style="font-size:17px; font-family:Segoe UI, sans-serif; padding: 10px 15px; border: 1px solid #ccc; border-radius: 8px; background-color: #f9f9f9;">
  <strong style="font-size: 18px;">📰 Summary</strong><br><br>
  {summary_text}
</div>
""")


# COMMAND ----------

from transformers import pipeline
import pandas as pd

# Sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

# Run analysis (truncate long texts)
sentiment_result = sentiment_analyzer(article_text[:1000])[0]

# Normalize score into position on a 1–10 scale
score = sentiment_result['score']
label = sentiment_result['label'].upper()

# Determine slider position (1–10, centered at 5)
pos = int(score * 10)
if label == "NEGATIVE":
    slider_position = 5 - pos
elif label == "NEUTRAL":
    slider_position = 5
else:
    slider_position = 5 + pos

slider_position = max(1, min(slider_position, 10))

# Build the slider string using emoji
bar = ''.join(['⬜'] * 10)
bar = bar[:slider_position - 1] + '🔵' + bar[slider_position:]
spectrum = f"Negative ⬅️ {bar} ➡️ Positive"

# Set sentiment color
if label == "POSITIVE":
    sentiment_color = "green"
elif label == "NEGATIVE":
    sentiment_color = "red"
else:
    sentiment_color = "blue"

# Build display content without emojis
sentiment_display = [
    f"<b>Sentiment:</b> <span style='color:{sentiment_color}'>{label.title()}</span>",
    f"<b>Meter:</b> {spectrum}",
    f"<b>Confidence:</b> {score:.1%}"
]

# Build and style DataFrame
df_sentiment_pretty = pd.DataFrame({' ': sentiment_display})
df_sentiment_pretty.index = [' ' * i for i in range(len(df_sentiment_pretty))]

styled_sentiment = df_sentiment_pretty.style.set_table_styles([
    {'selector': 'td', 'props': [
        ('font-size', '18px'),
        ('font-family', 'Segoe UI, sans-serif'),
        ('padding-top', '6px'),
        ('padding-bottom', '6px')
    ]}
])

displayHTML(styled_sentiment.to_html())




# COMMAND ----------

sentiment_result

# COMMAND ----------

from keybert import KeyBERT
from transformers import pipeline
import pandas as pd

# Step 1: Keyword extraction (KeyBERT)
kw_model = KeyBERT()
keywords = kw_model.extract_keywords(
    article_text, keyphrase_ngram_range=(1, 2), top_n=10, stop_words='english'
)
extracted_keywords = [kw[0] for kw in keywords]
print("Extracted Keywords:", extracted_keywords)

# Step 2: Zero-shot classification (Transformers)
high_level_topics = ["Politics", "Economy", "Sports", "Healthcare",
                     "Technology", "Entertainment", "Crime",
                     "Religion", "Science", "Education", "Obituary"]

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classification_result = classifier(' '.join(extracted_keywords), high_level_topics)

# Step 3: Display results in a clean dataframe
df_dynamic_topics = pd.DataFrame({
    'Topic': classification_result['labels'],
    'Score': classification_result['scores']
})

display(df_dynamic_topics)



# COMMAND ----------

import pandas as pd

# Assuming you've already extracted this info
top_category = classification_result['labels'][0].title()
top_keywords = extracted_keywords[:3]  # Top 3 topic keywords

# Define category colors (expand as needed)
category_colors = {
    'Politics': '#CD5C5C',
    'Economy': '#4682B4',
    'Obituary': '#4B4B4B',
    'Entertainment': '#DAA520',
    'Healthcare': '#2E8B57',
    'Sports': '#1E90FF',
    'Science': '#6A5ACD',
    'Technology': '#FF8C00',
    'Education': '#8A2BE2',
    'Crime': '#B22222',
    'Religion': '#8B0000'
}
category_color = category_colors.get(top_category, '#444')  # Default dark gray

# Build visual topic tags
topic_tags = ' '.join([
    f"<span style='background-color:#e0e0e0; padding:6px 10px; border-radius:6px; margin:6px 6px 6px 0; display:inline-block;'>{kw.title()}</span>"
    for kw in top_keywords
])

# Build final block
html_topic_card = f"""
<div style="font-size:18px; font-family:Segoe UI, sans-serif; padding:10px 5px;">
    <b>Category:</b> <span style="color:{category_color}; font-weight:600;">{top_category}</span><br><br>
    <b>Main Topics:</b><br>
    {topic_tags}
</div>
"""

# Display in dashboard
displayHTML(html_topic_card)



# COMMAND ----------

# MAGIC %pip install tldextract

# COMMAND ----------

from urllib.parse import urlparse
from difflib import get_close_matches
import pandas as pd

# Load and clean AllSides bias data
bias_df = pd.read_csv('/Volumes/workspace/default/hackathon2025/allsides.csv')
bias_df.columns = bias_df.columns.str.strip().str.lower()
bias_df['name_clean'] = bias_df['name'].str.lower().str.replace(r"[^\w\s]", "", regex=True)

# Extract domain from a URL
def extract_domain(url):
    return urlparse(url).netloc.replace("www.", "").lower()

# Improved bias lookup with substring and fuzzy fallback
def get_bias_from_url(article_url):
    domain = extract_domain(article_url)
    domain_root = domain.split('.')[0]  # "nbc" from "nbcnews.com"

    # Step 1: Try substring match in cleaned names
    for _, row in bias_df.iterrows():
        if domain_root in row['name_clean'].replace(" ", ""):
            return row['name'], row['bias']

    # Step 2: Fallback to fuzzy matching
    cleaned_names = bias_df['name_clean'].str.replace(" ", "").tolist()
    close_match = get_close_matches(domain_root, cleaned_names, n=1, cutoff=0.7)

    if close_match:
        matched_row = bias_df[bias_df['name_clean'].str.replace(" ", "") == close_match[0]].iloc[0]
        return matched_row['name'], matched_row['bias']

    return "Unknown Source", "Unknown"

# Example usage
source_name, bias = get_bias_from_url(article_url)



# COMMAND ----------



# Color code the bias
bias_color = {
    "left": "red",
    "left-center": "orangered",
    "center": "gray",
    "right-center": "dodgerblue",
    "right": "blue",
    "allsides": "darkgreen",
    "unknown": "black"
}.get(bias.lower(), "black")

# Display result
displayHTML(f"""
<div style="font-size:17px; font-family:Segoe UI, sans-serif; padding: 10px 0;">
    <strong>Source:</strong> {source_name}<br>
    <strong>Bias Rating:</strong> 
    <span style="color:{bias_color}; font-weight:600;">{bias.title()}</span>
</div>
""")


# COMMAND ----------

import re

def generate_search_query(summary_text, keywords, max_keywords=3):
    """
    Create a concise, relevant search query using top keywords and a cleaned-up version of the AI summary.
    
    Parameters:
    - summary_text: str - AI-generated article summary
    - keywords: list[str] - extracted keywords from KeyBERT
    - max_keywords: int - number of keywords to include

    Returns:
    - search_query: str
    """
    # Clean summary text
    cleaned_summary = re.sub(r'[^\w\s]', '', summary_text)
    summary_terms = ' '.join(cleaned_summary.lower().split()[:10])  # Take first 10 words

    # Filter and join keywords
    key_terms = [kw.lower() for kw in keywords[:max_keywords]]
    keyword_string = ' '.join(key_terms)

    # Combine both sources
    search_query = f"{summary_terms} {keyword_string}".strip()

    return search_query


# COMMAND ----------

search_query = generate_search_query(summary_text, extracted_keywords)
print("Generated Search Query:", search_query)


# COMMAND ----------

import requests
from keybert import KeyBERT
import spacy

# Load spaCy model (only once)
nlp = spacy.load("en_core_web_sm")

# Step 1: Adaptive Query Generator
def generate_adaptive_query(text, max_keywords=5, fallback_terms=None):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=max_keywords)
    extracted = [kw[0] for kw in keywords]

    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "NORP"]]
    noun_chunks = [chunk.root.text for chunk in doc.noun_chunks]

    # Prioritize: named entities > keybert > noun chunks
    candidates = list(dict.fromkeys(named_entities + extracted + noun_chunks))
    query = ' '.join(candidates[:max_keywords]).strip()

    if not query and fallback_terms:
        query = ' '.join(fallback_terms)

    return query

# Step 2: Generate query from article text
keyword_query = generate_adaptive_query(article_text, max_keywords=5, fallback_terms=["politics", "news"])
print("🧠 Adaptive Query:", keyword_query)

# Step 3: Search NewsAPI
NEWSAPI_KEY = "3e80d49d34a94c5fa008d6a32376ba2d"

def search_related_articles_newsapi(query, sources=None, language="en", page_size=10):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "apiKey": NEWSAPI_KEY
    }
    if sources:
        params["sources"] = sources

    response = requests.get(url, params=params)
    results = []

    if response.status_code == 200:
        articles = response.json().get("articles", [])
        for article in articles:
            results.append({
                "title": article.get("title"),
                "url": article.get("url"),
                "source": article.get("source", {}).get("name"),
                "publishedAt": article.get("publishedAt")
            })
    else:
        print(f"NewsAPI error: {response.status_code} - {response.text}")
    
    return results

# Step 4: Search with adaptive query
related_articles = search_related_articles_newsapi(keyword_query)


# COMMAND ----------

# Show results
if related_articles:
    for article in related_articles:
        print(f"- {article['title']}\n  ({article['source']})\n  {article['url']}\n")
else:
    print("No related articles found from NewsAPI using keywords.")

# COMMAND ----------

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set your Serper.dev API key
SERPER_API_KEY = "1bd9631fc5c88b5da0b5248448422f5ebbf9f0ca"

def is_similar_to_article(fact_text, article_summary, threshold=0.4):
    """Check if a fact-check snippet/title is similar to the article summary."""
    texts = [fact_text.lower(), article_summary.lower()]
    vec = TfidfVectorizer().fit_transform(texts)
    sim_score = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    return sim_score >= threshold

def search_fact_check_serper(query, article_summary, limit=3):
    """
    Uses Serper.dev (Google Search API) to look for relevant fact-checks.
    
    Filters based on semantic similarity to article summary.
    """
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {
        "q": f"{query} site:politifact.com OR site:snopes.com OR site:factcheck.org",
        "num": limit
    }

    response = requests.post(url, headers=headers, json=payload)
    filtered = []

    if response.status_code == 200:
        results = response.json().get("organic", [])
        for result in results:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            url = result.get("link", "")

            # Filter by semantic similarity
            combined_text = f"{title}. {snippet}"
            if is_similar_to_article(combined_text, article_summary):
                filtered.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": "Fact-Check"
                })
    else:
        print(f"Serper.dev error: {response.status_code} - {response.text}")
    # Add fallback if no fact-checks passed the similarity check
    if not filtered:
        filtered.append({
            "title": "No relevant fact-checks found for this topic.",
            "url": "#",
            "snippet": "We searched trusted fact-checking sources but did not find a close match for this article.",
            "source": "Fact-Check"
        })

    return filtered

# COMMAND ----------

fact_check_results = search_fact_check_serper(keyword_query, summary_text)

if fact_check_results:
    for fc in fact_check_results:
        print(f"✅ {fc['title']}\n{fc['url']}\n{fc['snippet']}\n")
else:
    print("No fact-checks found for this topic.")


# COMMAND ----------

def build_rag_display(related_articles, fact_checks, media_bias_lookup):
    bias_color_map = {
        "left": "red",
        "left-center": "orangered",
        "center": "gray",
        "right-center": "dodgerblue",
        "right": "blue",
        "allsides": "darkgreen",
        "fact-check": "green",
        "unknown": "black"
    }

    html = """
    <table style="font-size:16px; font-family:Segoe UI, sans-serif; border-collapse: collapse;">
        <tr style="text-align: left;">
            <th style="padding: 10px; font-size: 16px;">Source</th>
            <th style="padding: 10px; font-size: 16px;">Bias</th>
            <th style="padding: 10px; font-size: 16px;">Summary</th>
        </tr>
    """

    for article in related_articles:
        source = article.get("source", "Unknown")
        title = article.get("title", "")
        url = article.get("url", "")
        bias = get_bias_label_from_lookup(source, media_bias_lookup, url)
        bias_color = bias_color_map.get(bias.lower(), "black")

        html += f"""
        <tr>
            <td style="padding: 10px; font-size: 16px;"><a href="{url}" target="_blank">{source}</a></td>
            <td style="padding: 10px; font-size: 16px; color:{bias_color}; font-weight:600;">{bias.title()}</td>
            <td style="padding: 10px; font-size: 16px;">{title}</td>
        </tr>
        """

    for fc in fact_checks:
        fc_url = fc.get("url", "#")
        fc_summary = fc.get("title", "")
        fc_source = fc.get("source", "Fact-Check")

        html += f"""
        <tr>
            <td style="padding: 10px; font-size: 16px;"><a href="{fc_url}" target="_blank">{fc_source}</a></td>
            <td style="padding: 10px; font-size: 16px; color:green; font-weight:600;">Fact-Check</td>
            <td style="padding: 10px; font-size: 16px;">{fc_summary}</td>
        </tr>
        """

    html += "</table>"
    displayHTML(html)



# COMMAND ----------

# Step 1: Generate keywords
keywords = kw_model.extract_keywords(article_text, keyphrase_ngram_range=(1, 2), top_n=5, stop_words='english')
keyword_query = ' '.join([kw[0] for kw in keywords[:3]])

# Step 2: Search for related articles
related_articles = search_related_articles_newsapi(keyword_query)

# Step 3: Get relevant fact-checks using AI summary filtering
fact_check_results = search_fact_check_serper(keyword_query, summary_text)  # <-- Include summary for filtering

# Step 4: Balance across bias categories
fact_checker_domains = ["snopes.com", "politifact.com", "factcheck.org"]

def get_bias_label_from_lookup(source_name, bias_lookup, article_url=None):
    # Step 0: Force override if known fact-checker domain
    if article_url:
        domain = extract_domain(article_url)
        if domain in fact_checker_domains:
            return "Fact-Check"

    # Step 1: Try exact match
    if source_name in bias_lookup:
        return bias_lookup[source_name]

    # Step 2: Fuzzy fallback
    close = get_close_matches(source_name, bias_lookup.keys(), n=1, cutoff=0.6)
    if close:
        return bias_lookup[close[0]]

    return "Unknown"

def group_articles_by_bias(articles, bias_lookup):
    grouped = {"Left": None, "Center": None, "Right": None}
    for article in articles:
        bias_raw = get_bias_label_from_lookup(article['source'], bias_lookup).lower()
        if "left" in bias_raw:
            normalized_bias = "Left"
        elif "right" in bias_raw:
            normalized_bias = "Right"
        elif "center" in bias_raw:
            normalized_bias = "Center"
        else:
            normalized_bias = "Unknown"
        if normalized_bias in grouped and grouped[normalized_bias] is None:
            grouped[normalized_bias] = article
    return [a for a in grouped.values() if a]

# Step 5: Apply grouping
balanced_articles = group_articles_by_bias(related_articles, media_bias_lookup)

# Define fact-checker domains
fact_checker_domains = ["snopes.com", "politifact.com", "factcheck.org"]

# Filter out fact-check sources from related_articles
filtered_articles = [
    article for article in related_articles
    if extract_domain(article.get("url", "")) not in fact_checker_domains
]

# Step 6: Display
balanced_articles = group_articles_by_bias(filtered_articles, media_bias_lookup)
build_rag_display(balanced_articles, fact_check_results, media_bias_lookup)





# COMMAND ----------

bias_df

# COMMAND ----------

