"""
13_nlp_notes.py — NLP-анализ текстового поля notes.

Методы: TF-IDF, CountVectorizer, WordCloud, тематическое моделирование (LDA).

Читает:  data/robot_wash_features.csv
Пишет:   reports/nlp_report.txt
         reports/notes_top_words.csv
         images/55-58*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter

try:
    from wordcloud import WordCloud
    HAS_WC = True
except ImportError:
    HAS_WC = False

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")

FEATURES_PATH = 'data/robot_wash_features.csv'
IMAGES_DIR = 'images'
REPORTS_DIR = 'reports'

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# ШАГ 1. ЗАГРУЗКА
# ============================================================
print("=" * 60)
print("ШАГ 1. Загрузка данных")
print("=" * 60)

df = pd.read_csv(FEATURES_PATH, dtype={'tnved_code': str}, parse_dates=['date'])
print(f"Строк: {len(df)}")

notes = df['notes'].dropna().astype(str).tolist()
notes = [n for n in notes if n.strip()]
print(f"Непустых notes: {len(notes)}")


# ============================================================
# ШАГ 2. ОЧИСТКА ТЕКСТА
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 2. Очистка текста")
print("=" * 60)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^а-яёa-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

notes_clean = [clean_text(n) for n in notes]
notes_clean = [n for n in notes_clean if n]
print(f"После очистки: {len(notes_clean)}")
print(f"Пример: {notes_clean[:5]}")


# ============================================================
# ШАГ 3. ЧАСТОТНЫЙ АНАЛИЗ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 3. Частотный анализ слов")
print("=" * 60)

all_words = ' '.join(notes_clean).split()
word_counts = Counter(all_words)

top_words = pd.DataFrame(word_counts.most_common(30),
                         columns=['word', 'count'])
print(f"Уникальных слов: {len(word_counts)}")
print("\nТОП-15 слов:")
print(top_words.head(15).to_string(index=False))

top_words.to_csv(f'data/notes_top_words.csv', index=False)


# ============================================================
# ШАГ 4. TF-IDF
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 4. TF-IDF")
print("=" * 60)

tfidf = TfidfVectorizer(max_features=100, min_df=2)
X_tfidf = tfidf.fit_transform(notes_clean)
print(f"TF-IDF матрица: {X_tfidf.shape}")

tfidf_words = pd.DataFrame({
    'word': tfidf.get_feature_names_out(),
    'tfidf_sum': np.asarray(X_tfidf.sum(axis=0)).flatten()
}).sort_values('tfidf_sum', ascending=False)

print("\nТОП-10 TF-IDF слов:")
print(tfidf_words.head(10).to_string(index=False))


# ============================================================
# ШАГ 5. ТЕМАТИЧЕСКОЕ МОДЕЛИРОВАНИЕ (LDA)
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 5. Тематическое моделирование (LDA)")
print("=" * 60)

cv = CountVectorizer(max_features=200, min_df=2)
X_counts = cv.fit_transform(notes_clean)

n_topics = 3
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
lda.fit(X_counts)

feature_names = cv.get_feature_names_out()
topics_words = {}
for topic_idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[-10:][::-1]
    topics_words[topic_idx] = [feature_names[i] for i in top_indices]
    print(f"\nТема {topic_idx + 1}: {', '.join(topics_words[topic_idx])}")


# ============================================================
# ГРАФИКИ
# ============================================================
print("\n" + "=" * 60)
print("Построение графиков")
print("=" * 60)

# 55. ТОП-20 слов (bar)
plt.figure(figsize=(12, 7))
top20 = top_words.head(20)
plt.barh(top20['word'], top20['count'], color='steelblue')
plt.title('ТОП-20 слов в поле notes')
plt.xlabel('Частота')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/55_nlp_top_words.png', dpi=150)
plt.close()
print("Сохранён: 55_nlp_top_words.png")

# 56. WordCloud
if HAS_WC:
    wc = WordCloud(width=1200, height=600, background_color='white',
                   colormap='viridis').generate_from_frequencies(word_counts)
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('WordCloud поля notes')
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/56_nlp_wordcloud.png', dpi=150)
    plt.close()
    print("Сохранён: 56_nlp_wordcloud.png")

# 57. TF-IDF ТОП-20
plt.figure(figsize=(12, 7))
top_tfidf = tfidf_words.head(20)
plt.barh(top_tfidf['word'], top_tfidf['tfidf_sum'], color='coral')
plt.title('ТОП-20 слов по TF-IDF')
plt.xlabel('Сумма TF-IDF')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/57_nlp_tfidf.png', dpi=150)
plt.close()
print("Сохранён: 57_nlp_tfidf.png")

# 58. Темы LDA
fig, axes = plt.subplots(1, n_topics, figsize=(5 * n_topics, 6))
for i, ax in enumerate(axes):
    words = topics_words[i]
    weights = [lda.components_[i][list(feature_names).index(w)] for w in words]
    ax.barh(words, weights, color=f'C{i}')
    ax.set_title(f'Тема {i + 1}')
    ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/58_nlp_topics.png', dpi=150)
plt.close()
print("Сохранён: 58_nlp_topics.png")


# ============================================================
# ОТЧЁТ
# ============================================================
with open(f'{REPORTS_DIR}/nlp_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЁТ: NLP-АНАЛИЗ ПОЛЯ notes\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Всего записей: {len(notes)}\n")
    f.write(f"Уникальных слов: {len(word_counts)}\n\n")

    f.write("ТОП-20 слов:\n")
    for _, r in top_words.head(20).iterrows():
        f.write(f"  {r['word']}: {r['count']}\n")
    f.write("\n")

    f.write("ТОП-10 слов по TF-IDF:\n")
    for _, r in tfidf_words.head(10).iterrows():
        f.write(f"  {r['word']}: {r['tfidf_sum']:.3f}\n")
    f.write("\n")

    f.write("Темы LDA:\n")
    for i, words in topics_words.items():
        f.write(f"  Тема {i+1}: {', '.join(words)}\n")

print(f"Отчёт сохранён: {REPORTS_DIR}/nlp_report.txt")


print("\n" + "=" * 60)
print("ГОТОВО! NLP завершён.")
print("=" * 60)