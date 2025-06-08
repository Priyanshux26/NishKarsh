
# Define lexicons
left_lexicon = [
    'dalit', 'ambedkarite', 'protest', 'caste', 'reservation',
    "equality", "justice", "progressive", "welfare", "reform", "inclusive", "diversity", "socialism",
    "redistribution", "solidarity", "equity", "activism", "feminism", "environment", "sustainability",
    "affordable care", "public healthcare", "public education", "minimum wage", "universal basic income",
    "labor rights", "social safety net", "social democracy", "equal opportunity", "healthcare for all",
    "tax fairness", "worker rights", "human rights", "anti-discrimination", "social justice", "antiracism",
    "anti-poverty", "collective bargaining", "civil rights", "affordable housing", "free college education",
    "anti-austerity", "public transportation", "income inequality", "LGBTQ rights", "refugee rights",
    "universal healthcare", "climate change action", "carbon tax", "green economy", "clean energy",
    "renewable energy", "environmental protection", "progressive taxation", "worker empowerment",
    "anti-capitalism", "economic justice", "anti-globalization", "anti-imperialism", "anti-war", "anti-fascism",
    "redistribution of wealth", "abolition rights", "disability rights", "abolitionism", "nationalization",
    "anti-privatization", "gun control", "universal pension", "police reform", "universal voting rights",
    "free speech", "criminal justice reform", "wealth gap", "green policies", "fair wages", "immigrant rights",
    "political correctness", "racial justice", "abolition", "healthcare equity", "workers’ rights", "collective action",
    "anti-bigotry", "healthcare access", "unionization", "multilateralism", "Medicare for All", "gender equality",
    "workers’ compensation", "gun violence prevention", "economic democracy", "racial equity", "anti-corporatism",
    "economic empowerment", "human dignity", "anti-monopoly", "public funding", "disability accessibility",
    "fair access to education", "anti-deforestation", "public investment", "affordable public services",
    "economic mobility", "anti-banking system", "green taxes", "paid family leave", "anti-fracking", "national healthcare",
    "tax progressivity", "political reform", "free college", "public health system", "anti-corporate greed",
    "abolition of private prisons", "gender inclusion", "migrant rights", "non-discrimination", "anti-racism education",
    "tax justice", "housing justice", "voting rights", "affordable energy", "investment in public services",
    "gender diversity", "equal access", "equal treatment", "progressive foreign policy", "carbon emissions",
    "anti-capitalist feminism", "indigenous rights", "police abolition", "income redistribution", "financial reform",
    "land rights", "international justice", "free internet", "prison abolition", "nationalized healthcare",
    "social equity", "opposing gentrification", "pro-immigrant policies", "public utilities", "international solidarity",
    "anti-colonialism", "political transparency", "free university", "mental health services", "family benefits",
    "strong unions", "free public transport", "anti-exploitation", "end poverty", "green new deal", "affordable food",
    "ethical consumption", "radical inclusion", "fair housing", "socialized medicine", "community organizing",
    "state intervention", "anti-corporate lobbying", "anti-trade deals", "affordable electricity", "end hunger",
    "rights of workers", "gun reform", "childcare accessibility", "disability justice", "anti-extractionism",
    "gender-neutral policies", "anti-imperialist education", "women’s empowerment", "human rights education",
    "public transportation investment", "public infrastructure", "equal justice under law", "anti-union busting",
    "human rights violations", "humanitarian aid", "climate justice", "economic redistribution", "paid parental leave",
    "gender-based violence prevention", "workers’ rights to organize", "immigration sanctuary", "cultural diversity",
    "economic diversification", "equal access to resources", "anti-fascist resistance", "affordable transportation",
    "anti-war movements", "transgender rights", "anti-privatization movements", "democratic socialism",
    "public-sector unions", "anti-corporate corruption", "right to health", "community-led development",
    "abolition of class divisions", "renewable resources", "protection of public spaces", "workers' collective",
    "pro-immigrant", "pro-environment", "anti-militarism", "global solidarity", "free public healthcare",
    "pro-union", "reproductive justice", "democracy promotion", "anti-unfair trade", "end wage disparity",
    "dismantle corporate monopolies", "environmental justice", "global wealth tax", "public goods", "anti-eviction",
    "educational equity", "community organizing", "public interest", "welfare state", "voting reform", "community engagement",
    "class struggle", "pro-labor", "anti-fascist activism"
]

right_lexicon = [
    'hindutva', 'temple', 'nationalism', 'cow', 'ram mandir', 'patriotism', 'security', 'sovereignty',
    'tradition', 'culture', 'heritage', 'values', 'order', 'capitalism', 'market economy','Bharat',
    'privatization', 'economic growth', 'self-reliance', 'individualism', 'freedom', 'law and order',
    'family values', 'fiscal conservatism', 'economic liberalization', 'entrepreneurship', 'tax cuts',
    'national pride', 'strong defense', 'self-sufficiency', 'immigration control', 'free market',
    'social order', 'national security', 'capitalist', 'nationalist', 'privatization of services',
    'free trade', 'economic liberalism', 'entrepreneurial spirit', 'free enterprise',
    'business-friendly policies', 'traditional family', 'patriotic', 'cultural pride',
    'religious freedom', 'protecting traditions', 'economic competitiveness', 'law and justice',
    'strong military', 'conservative values', 'national integrity', 'domestic security',
    'traditional values', 'entrepreneurial freedom', 'privatize', 'deregulation',
    'law and justice reform', 'self-defense', 'policing', 'free market economy',
    'democratic conservatism', 'national unity', 'corporate tax reduction', 'defense spending',
    'political conservatism', 'family-first policies', 'market-driven economy', 'self-governance',
    'small government', 'law enforcement', 'patriotic values', 'limited government', 'security-first',
    'strong borders', 'Hindu nationalism', 'economic development', 'business empowerment',
    'opposing communism', 'political stability', 'cultural nationalism', 'anti-terrorism',
    'military preparedness', 'capitalist reforms', 'secure borders', 'anti-communism',
    'family-focused', 'religion in politics', 'traditional institutions', 'anti-secularism',
    'nation first', 'cultural identity', 'economic empowerment', 'pro-business', 'military strength',
    'pro-Hindu policies', 'unilateral decisions', 'state sovereignty', 'anti-leftist', 'defense reform',
    'capitalist economy', 'tax incentives', 'trade protectionism', 'economic conservatism',
    'empowering the market', 'immigration reform', 'defensive nationalism', 'right-wing',
    'militant nationalism', 'conservative policies', 'limited government intervention', 'market reforms',
    'strategic autonomy', 'promoting domestic industries', 'patriotic duty', 'strong central government',
    'pro-privatization', 'law and order policies', 'defending cultural heritage', 'nationalistic policies',
    'capitalist growth', 'anti-communist', 'secularism in decline', 'national integration',
    'opposing welfare state', 'defense of national interests', 'opposing foreign influence',
    'anti-globalization', 'empowering private sector', 'conservative economics', 'anti-terrorism policies',
    'national cohesion', 'civilizational nationalism', 'pro-business reforms', 'free enterprise system',
    'economic freedom', 'religious identity', 'political sovereignty', 'anti-leftist agenda',
    'right-wing populism', 'patriotic spirit', 'welfare reform', 'pro-national security',
    'traditional leadership', 'nationalistic identity', 'anti-multiculturalism', 'economic nationalism',
    'cultural conservatism', 'strong political leadership', 'self-determination', 'sovereign control',
    'defending national values', 'anti-foreign interference', 'self-rule', 'national government',
    'anti-socialism', 'economic protectionism', 'cultural preservation', 'family-centered policies',
    'national autonomy', 'limited welfare', 'strong political institutions', 'economic individualism',
    'traditional society'
]

centrist_lexicon = [
    "growth", "development", "policy", "neutral", "reform",
    "bipartisan", "compromise", "pragmatic", "middle ground", "consensus", "balanced policy", "moderation",
    "collaboration", "harmony", "independent", "unity", "objective", "open-minded", "cooperation",
    "nonpartisan", "common ground", "practical solutions", "realism", "reasonable", "flexibility", "rational",
    "open dialogue", "common sense", "civic engagement", "moderate", "pragmatism", "peace", "equitable",
    "stable government", "long-term solutions", "pluralism", "functionalism", "gradual change", "policy balance",
    "centrist reforms", "social harmony", "united governance", "inclusive policy", "calm approach", "practical politics",
    "policy implementation", "unifying agenda", "cross-party cooperation", "gradual progress", "public consensus",
    "governance reforms", "citizen-driven", "cooperative governance", "public-private partnerships", "peaceful resolution",
    "national unity", "national interest", "democratic participation", "policy collaboration", "center-ground solutions",
    "realpolitik", "legislative compromise", "public unity", "inclusive governance", "equitable reforms",
    "national development", "independent voices", "policy negotiations", "open debate", "social responsibility",
    "policy continuity", "grassroots participation", "incremental change", "fair governance", "strategic planning",
    "socio-economic growth", "equal participation", "moderate politics", "inclusive progress", "balanced growth",
    "mutual understanding", "economic stability", "cooperative politics", "unity in diversity", "positive governance",
    "international relations", "policy transparency", "public trust", "economic cooperation", "moderate social reforms",
    "peaceful coexistence", "regional balance", "policy inclusiveness", "right to dissent", "equitable development",
    "sustainable growth", "centrist policy agenda", "international diplomacy", "non-confrontational", "constructive debate",
    "social equilibrium", "collaborative governance", "open policy discussions", "civic responsibility", "sustainable policies",
    "grassroots collaboration", "balanced democracy", "deliberative democracy", "mutual benefit", "reliable leadership",
    "legislative process", "governance based on consensus", "fair electoral system", "pragmatic approach", "stable society",
    "moderate taxation", "eclectic approach", "cohesive society", "comprehensive governance", "community participation",
    "rational policies", "equal rights for all", "balanced economic policies", "inclusive development", "political moderation",
    "constitutional values", "moderate reforms", "equitable solutions", "strategic compromises",
    "equality of opportunity", "fairness in governance", "economic pragmatism", "democratic reforms", "peaceful transitions",
    "non-violent change", "comprehensive solutions", "moderate ideologies", "collaborative political agenda", "responsible governance",
    "institutional integrity", "just reforms", "economic compromise", "stable policy framework"
]
#Code3
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from readability import Document
from bs4 import BeautifulSoup
import tldextract
import re
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import AutoConfig
from transformers import pipeline


mpath='./deberta-political-leaning-model'




# ------------------- Source Lists ------------------- #
left_sources = ['the wire', 'scroll.in', 'newsclick', 'the quint','the hindu']
right_sources = ['opindia', 'swarajya', 'republic bharat', 'zee news','ndtv','abp','republic world']
centrist_sources = [ 'indian express', 'THE TIMES OF INDIA','economic times','tribune']



seen = set()
def remove_duplicates(lexicon):
    return [word for word in lexicon if not (word in seen or seen.add(word))]

def clean_lexicon(lexicon):
    return [w for w in lexicon if isinstance(w, str)]

left_lexicon = clean_lexicon(remove_duplicates(left_lexicon))
right_lexicon = clean_lexicon(remove_duplicates(right_lexicon))
centrist_lexicon = clean_lexicon(remove_duplicates(centrist_lexicon))

# ------------------- RoBERTa Model Setup + Classifier Prediction ------------------- #
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading main model")
# Load tokenizer and model from fine-tuned path
# Set device
tokenizer = AutoTokenizer.from_pretrained("priyanshux26/political-leaning-detection")
model = AutoModelForSequenceClassification.from_pretrained(
    "priyanshux26/political-leaning-detection",
    output_hidden_states=True
).to(device)

# Map label indices to human-readable labels (based on training)
id2label = {0: "Left", 1: "Center", 2: "Right"}

# Direct classification prediction using fine-tuned model
def predict_leaning(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()
    return {
        'label': id2label[predicted_class],
        'confidence': confidence,
        'raw_probs': probs.squeeze().tolist()
    }










# 1. Load config and force output_hidden_states = True
config = AutoConfig.from_pretrained(mpath)
config.output_hidden_states = True  # crucial!

# 2. Load model with config
tokenizer = AutoTokenizer.from_pretrained(mpath)
model = AutoModelForSequenceClassification.from_pretrained(mpath, config=config).to(device)
model.eval()

print("#1")
tokens = tokenizer("This is a test", return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**tokens)
print('#2')

print("Hidden states returned?", outputs.hidden_states is not None)

# Map class indices to labels (edit based on your training labels)
id2label = {0: "Left", 1: "Center", 2: "Right"}



sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)



def predict_leaning(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence, prediction = torch.max(probs, dim=1)
    
    return prediction.item(), confidence.item(), probs.squeeze().tolist()







def get_lexicon_embedding(lexicon):
    tokens = tokenizer(lexicon, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**tokens)
    last_hidden = outputs.hidden_states[-1]
    return last_hidden.mean(dim=1)



left_embedding = get_lexicon_embedding(left_lexicon)
right_embedding = get_lexicon_embedding(right_lexicon)
centrist_embedding = get_lexicon_embedding(centrist_lexicon)

# ------------------- TF-IDF Setup ------------------- #
full_vocab = left_lexicon + right_lexicon + centrist_lexicon
tfidf_vectorizer = TfidfVectorizer(vocabulary=full_vocab)
tfidf_vectorizer.fit([" ".join(full_vocab)])

def calculate_tfidf_scores(text):
    tfidf_vector = tfidf_vectorizer.transform([text]).toarray()[0]
    features = tfidf_vectorizer.get_feature_names_out()
    left = sum(tfidf_vector[i] for i, w in enumerate(features) if w in left_lexicon)
    right = sum(tfidf_vector[i] for i, w in enumerate(features) if w in right_lexicon)
    centrist = sum(tfidf_vector[i] for i, w in enumerate(features) if w in centrist_lexicon)
    return left, right, centrist

# ------------------- Source Bias Weights ------------------- #
def get_source_bias_weight(source):
    source = source.strip().lower()
    if source in left_sources:
        return (1.0, 0.0, 0.0)
    elif source in right_sources:
        return (0.0, 1.0, 0.0)
    elif source in centrist_sources:
        return (0.0, 0.0, 1.0)
    return (0.33, 0.33, 0.33)

# ------------------- Main Analysis Function ------------------- #
def analyze_article(article, source, weights=(0.4, 0.4, 0.2), w_sentiment=0.1):
    # Step 1: RoBERTa direct prediction
    direct_prediction = predict_leaning(article)
    predicted_label = direct_prediction[0]
    print(f"RoBERTa Model Prediction: {predicted_label} ({direct_prediction[1]*100:.2f}%)")

    # Step 2: TF-IDF Scores
    left_tfidf, right_tfidf, centrist_tfidf = calculate_tfidf_scores(article)
    print("TFIDF:", left_tfidf, right_tfidf, centrist_tfidf)

    # Step 3: Embedding similarity with lexicons
    tokens = tokenizer(article, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**tokens)
        article_embedding = outputs.hidden_states[-1].mean(dim=1)

    left_sim = cosine_similarity(article_embedding.cpu(), left_embedding.cpu())[0][0]
    right_sim = cosine_similarity(article_embedding.cpu(), right_embedding.cpu())[0][0]
    centrist_sim = cosine_similarity(article_embedding.cpu(), centrist_embedding.cpu())[0][0]
    print("LEXICON SIM:", left_sim, right_sim, centrist_sim)

    # Step 4: Source bias weights
    left_bias, right_bias, centrist_bias = get_source_bias_weight(source)
    print("Source Weights:", left_bias, right_bias, centrist_bias)

    # Step 5: Base scores (TF-IDF + embedding + source)
    left_final = weights[0]*left_tfidf + weights[1]*left_sim + weights[2]*left_bias
    right_final = weights[0]*right_tfidf + weights[1]*right_sim + weights[2]*right_bias
    centrist_final = weights[0]*centrist_tfidf + weights[1]*centrist_sim + weights[2]*centrist_bias

    # Step 6: Sentiment analysis (optional nudge)
    sentiment_score = get_sentiment_score(article)  # Range [-1, +1]

    if predicted_label == "Left":
        left_final += w_sentiment * sentiment_score
    elif predicted_label == "Right":
        right_final += w_sentiment * sentiment_score
    elif predicted_label == "Center":
        centrist_final += w_sentiment * sentiment_score

    # Step 7: Final result
    scores = {
        'Leftist': left_final,
        'Rightist': right_final,
        'Centristist': centrist_final
    }
    leaning = max(scores, key=scores.get)

    return {
        'Model_Prediction': direct_prediction,
        'TFIDF_Scores': (float(left_tfidf), float(right_tfidf), float(centrist_tfidf)),
        'RoBERTa_Similarities': (float(left_sim), float(right_sim), float(centrist_sim)),
        'Source_Weights': (float(left_bias), float(right_bias), float(centrist_bias)),
        'Final_Scores': scores,
        'Sentiment_Score': sentiment_score,
        'Final Conclusion': leaning
    }

def get_sentiment_score(text):
    result = sentiment_analyzer(text[:512])[0]  # Truncate long text for safe input
    label = result['label']
    score = result['score']
    return score if label == "POSITIVE" else -score


def extract_article_info_fallback(url):
    response = requests.get(url)
    doc = Document(response.text)
    html = doc.summary()
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator='\n')
    source = tldextract.extract(url).domain
    return {
        'title': doc.title(),
        'text': text,
        'source': source
    }


# Initialize summarizer globally (so it loads once)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_neutral_summary(text):
    # You can adjust max_length and min_length as needed
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

def print_structured_bias_report(result):
    print("\n===== Bias Analysis Report =====\n")
    print(f"0. RoBERTa Model Prediction:")
    print(f"   Predicted Label : {result['Model_Prediction'][0]}")
    print(f"   Confidence       : {result['Model_Prediction'][1]*100:.2f}%\n")

    print(f"1. TF-IDF Scores:")
    print(f"   Leftist   : {result['TFIDF_Scores'][0]:.4f}")
    print(f"   Rightist  : {result['TFIDF_Scores'][1]:.4f}")
    print(f"   Centrist  : {result['TFIDF_Scores'][2]:.4f}\n")

    print(f"2. RoBERTa Embedding Similarities:")
    print(f"   Leftist   : {result['RoBERTa_Similarities'][0]:.4f}")
    print(f"   Rightist  : {result['RoBERTa_Similarities'][1]:.4f}")
    print(f"   Centrist  : {result['RoBERTa_Similarities'][2]:.4f}\n")

    print(f"3. Source Bias Weights:")
    print(f"   Leftist   : {result['Source_Weights'][0]:.2f}")
    print(f"   Rightist  : {result['Source_Weights'][1]:.2f}")
    print(f"   Centrist  : {result['Source_Weights'][2]:.2f}\n")

    print(f"4. Final Aggregated Scores:")
    print(f"   Leftist   : {result['Final_Scores']['Leftist']:.4f}")
    print(f"   Rightist  : {result['Final_Scores']['Rightist']:.4f}")
    print(f"   Centristist: {result['Final_Scores']['Centristist']:.4f}\n")

    print(f"5. Sentiment Influence:")
    print(f"   Score toward predicted class: {result['Sentiment_Score']:.4f}\n")


    print(f">>> Final Conclusion:  {result['Final Conclusion']}\n")
    print("===============================\n")

def analyze_and_summarize(article, source):
    # Run bias analysis
    bias_result = analyze_article(article, source)

    # Generate neutral summary
    summary = generate_neutral_summary(article)

    # Print bias report
    print_structured_bias_report(bias_result)

    # Print neutral summary
    print("===== Neutral Summary =====\n")
    print(summary)
    print("\n===========================")

    return bias_result, summary