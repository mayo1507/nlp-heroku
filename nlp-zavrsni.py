#Import all the required livraries..

import pandas as pd
import feedparser
# Import packages
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import gensim
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
import pyLDAvis.gensim_models
from collections import Counter
import feedparser
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import base64
import seaborn as sns
from PIL import Image
import cufflinks
from sklearn.feature_extraction.text import CountVectorizer
from plotly.offline import iplot
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import plotly.express as px
import spacy
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
#from gensim.summarization import summarize
from sumy.utils import get_stop_words
from sumy.nlp.stemmers import Stemmer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as sumytoken
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.utils import get_stop_words
from sumy.nlp.stemmers import Stemmer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as sumytoken
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.utils import get_stop_words
from sumy.nlp.stemmers import Stemmer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as sumytoken
from sumy.summarizers.luhn import LuhnSummarizer

nltk.download('punkt')
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

# In[ ]:
class RSSFeed():
    feedurl = ""

    global ndf
    def __init__(self, paramrssurl):
        print(paramrssurl)
        self.feedurl = paramrssurl
        self.parse()


    def parse(self):
        thefeed = feedparser.parse(self.feedurl)
        global ndf
        ndf = pd.DataFrame(columns=['title', 'link', 'decription', 'content'])
        for thefeedentry in thefeed.entries:
            title = thefeedentry.get("title", "")
            link = thefeedentry.get("link", "")
            decr = thefeedentry.get("description", "")
            ndf = ndf.append({'title': title, 'link': link, 'decription': decr},
                           ignore_index=True)
        return ndf

#Beautiful Soup Code
@st.cache_data
def full_text(my_url):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    url = my_url
    article = requests.get(url)
    articles = BeautifulSoup(article.content, 'html.parser')
    articles_body = articles.findAll('body')
    p_blocks = articles_body[0].findAll('p')
    p_blocks_df = pd.DataFrame(columns=['element_name', 'parent_hierarchy', 'element_text', 'element_text_Count'])
    for i in range(0, len(p_blocks)):
        parents_list = []
        for parent in p_blocks[i].parents:
            Parent_id = ''
            try:
                Parent_id = parent['id']
            except:
                pass
            parents_list.append(parent.name + 'id: ' + Parent_id)
        parent_element_list = ['' if (x == 'None' or x is None) else x for x in parents_list]
        parent_element_list.reverse()
        parent_hierarchy = ' -> '.join(parent_element_list)
        p_blocks_df = p_blocks_df.append({"element_name": p_blocks[i].name
                                             , "parent_hierarchy": parent_hierarchy
                                             , "element_text": p_blocks[i].text
                                             , "element_text_Count": len(str(p_blocks[i].text))}
                                         , ignore_index=True
                                         , sort=False)
    if len(p_blocks_df) > 0:
        p_blocks_df_groupby_parent_hierarchy = p_blocks_df.groupby(by=['parent_hierarchy'])
        p_blocks_df_groupby_parent_hierarchy_sum = p_blocks_df_groupby_parent_hierarchy[['element_text_Count']].sum()
        p_blocks_df_groupby_parent_hierarchy_sum.reset_index(inplace=True)
    maxid = p_blocks_df_groupby_parent_hierarchy_sum.loc[
        p_blocks_df_groupby_parent_hierarchy_sum['element_text_Count'].idxmax()
        , 'parent_hierarchy']
    merge_text = '\n'.join(p_blocks_df.loc[p_blocks_df['parent_hierarchy'] == maxid, 'element_text'].to_list())
    return merge_text

    
#Matplt lib
@st.cache_data
def preprocess(ReviewText):
    ReviewText = ReviewText.str.replace("(<br/>)", "")
    ReviewText = ReviewText.str.replace('(<a).*(>).*(</a>)', '')
    ReviewText = ReviewText.str.replace('(&amp)', '')
    ReviewText = ReviewText.str.replace('(&gt)', '')
    ReviewText = ReviewText.str.replace('(&lt)', '')
    ReviewText = ReviewText.str.replace('(\xa0)', ' ')
    return ReviewText


#Get top words
@st.cache_data
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

@st.cache_data
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

@st.cache_data
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

@st.cache_data
def sentiment_textblob(text):
    x = TextBlob(text).sentiment.polarity

    if x < 0:
        return 'neg'
    elif x == 0:
        return 'neu'
    else:
        return 'pos'

#@st.cache(suppress_st_warning=True)
def plot_sentiment_barchart(text, method='TextBlob'):
    if method == 'TextBlob':
        sentiment = text.map(lambda x: sentiment_textblob(x))

    plt.bar(sentiment.value_counts().index,
            sentiment.value_counts(),color=['cyan', 'red', 'green', 'black'],edgecolor='yellow')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

#Parts of Speech Tagging
def plot_parts_of_speach_barchart(text):
    nltk.download('averaged_perceptron_tagger')

    def _get_pos(text):
        pos = nltk.pos_tag(word_tokenize(text))
        pos = list(map(list, zip(*pos)))[1]
        return pos

    tags = text.apply(lambda x: _get_pos(x))
    tags = [x for l in tags for x in l]
    counter = Counter(tags)
    x, y = list(map(list, zip(*counter.most_common(7))))

    sns.barplot(x=y, y=x)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

#st.set_page_config(layout="wide")
st.title('News Articles Analysis -NLP App')
st.header("""
This app displays the news articles appeared in the top News Publications!
""")

st.sidebar.header('Please select the news org from the dropdown list')
lnews = [ "NY Times","BuzzFeed","Huffington Post","The Wall Street Journal"]
s_news = st.sidebar.selectbox('News', lnews)
st.sidebar.header('Please select the Function')
lnlp = ["Intro","Snapshot","Unigrams","Bigrams","Trigrams","WordCloud","Sentiment Analysis TextBlob","Parts of Speech"]
s_nlp = st.sidebar.selectbox('Functions', lnlp)

def load_data(news,nlp):
   if news =="NY Times":
       #st.write(news)
       #st.write(nlp)
       if nlp =="Intro":
           #st.write("this is intro")
           image1 = Image.open(r'C:\Users\PC\Desktop\git\nlp-zavrsni\images\New-York-Times-logo-500x281.jpg')
           st.write( " ")
           st.image(image1, width=300)
           st.write(" ")
           st.write(" ")
           st.write(" ")
           image = Image.open(r'C:\Users\PC\Desktop\git\nlp-zavrsni\images\nytimes-building-ap-img.jpg')
           st.image(image, width=700)
           st.write(" ")
           st.write(" ")
           st.write(" ")
           st.markdown("""
           The New York Times (NYT or NY Times) is an American daily newspaper based in New York City with a worldwide readership.Founded in 1851, the Times has since won 130 Pulitzer Prizes (the most of any newspaper),and has long been regarded within the industry as a national "newspaper of record". It is ranked 18th in the world by circulation and 3rd in the U.S.
           The paper is owned by The New York Times Company, which is publicly traded. It has been governed by the Sulzberger family since 1896, through a dual-class share structure after its shares became publicly traded.A. G. Sulzberger and his father, Arthur Ochs Sulzberger Jr.—the paper's publisher and the company's chairman, respectively—are the fourth and fifth generation of the family to head the paper.
           Since the mid-1970s, The New York Times has expanded its layout and organization, adding special weekly sections on various topics supplementing the regular news, editorials, sports, and features. Since 2008,the Times has been organized into the following sections: News, Editorials/Opinions-Columns/Op-Ed, New York (metropolitan), Business, Sports, Arts, Science, Styles, Home, Travel, and other features.[15] On Sundays, the Times is supplemented by the Sunday Review (formerly the Week in Review),The New York Times Book Review, The New York Times Magazine,and T: The New York Times Style Magazine.
           The Times stayed with the broadsheet full-page set-up and an eight-column format for several years after most papers switched to six,and was one of the last newspapers to adopt color photography, especially on the front page.The paper's motto, "All the News That's Fit to Print", appears in the upper left-hand corner of the front page.
           """)

       df = pd.DataFrame(columns=['title', 'link', 'decription', 'content'])
       url_link = "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"
       RSSFeed(url_link)
       df = ndf
       #st.header('Display the dataframe')
       #st.dataframe(df)
       pd.set_option('display.max_rows', df.shape[0] + 1)
       df.reset_index(inplace=True, drop=True)
       for ind in df.index:
           # print(df['title'][ind], df['link'][ind], df['content'][ind])
           url = df['link'][ind]
           #print(url)
           text = full_text(url)
           df['content'][ind] = text
       #st.write(df['title'])
       # Build the corpus.
       corpus = []
       for ind in df.index:
           # corpus = df['content'][ind]
           corpus.append(df['title'][ind])
       #print(corpus)
       df = df.dropna()
       X_train1 = df['title']
       if nlp == "Snapshot":
           st.write(" ")
           st.write(" ")
           st.write(" ")
           st.subheader('Display the dataframe')
           st.write(" ")
           st.write(" ")
           st.write(" ")
           st.dataframe(df)
           st.write(" ")
           st.write(" ")
           st.write(" ")
           st.markdown("""
                                                                                        <style>
                                                                                        .big2-font {
                                                                                            font-size:30px !important;
                                                                                        }
                                                                                        </style>
                                                                                        """, unsafe_allow_html=True)
           st.markdown('<p class="big2-font">The no of articles :</p>', unsafe_allow_html=True)
           st.write(df.shape[0])
           st.write(" ")
           st.write("The Url Link ")
           for index, row in df.iterrows():
               st.write(row['link'])

       if nlp == "WordCloud":
           st.markdown("""
                                                                             <style>
                                                                             .big1-font {
                                                                                 font-size:20px !important;
                                                                             }
                                                                             </style>
                                                                             """, unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">WordCloud</p>', unsafe_allow_html=True)
           st.write(' ')
           st.markdown(
               '<p class="big1-font">Word clouds or tag clouds are graphical representations of word frequency that give greater prominence to words that appear more frequently in a source text. The larger the word in the visual the more common the word was in the document(s).</p>',
               unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')
           long_string = ','.join(list(X_train1.values))
           # Create a WordCloud object
           wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
           # Generate a word cloud
           wordcloud.generate(long_string)
           # Visualize the word cloud
           plt.figure(figsize=(20, 10))
           plt.imshow(wordcloud)
           st.image(wordcloud.to_array(), width=700)
           st.write("Word Cloud")
           # Generate word cloud
           long_string = ','.join(list(X_train1.values))
           wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='salmon', colormap='Pastel1',
                                 collocations=False, stopwords=STOPWORDS).generate(long_string)
           # Visualize the word cloud
           plt.figure(figsize=(20, 10))
           plt.imshow(wordcloud)
           st.image(wordcloud.to_array(), width=700)
           st.write("Word Cloud")
           wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='black', colormap='Set2',
                                 collocations=False, stopwords=STOPWORDS).generate(long_string)
           # Visualize the word cloud
           plt.figure(figsize=(20, 10))
           plt.imshow(wordcloud)
           st.image(wordcloud.to_array(), width=700)

       df_n = df
       df_n['title'] = preprocess(df['title'])

       if nlp == "Unigrams":
           st.markdown("""
                                                       <style>
                                                       .big1-font {
                                                           font-size:20px !important;
                                                       }
                                                       </style>
                                                       """, unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">N Grams</p>', unsafe_allow_html=True)
           st.write(' ')
           st.markdown(
               '<p class="big1-font">N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). When N=1, this is referred to as unigrams and this is essentially the individual words in a sentence. When N=2, this is called bigrams and when N=3 this is called trigrams. When N>3 this is usually referred to as four grams or five grams and so on.</p>',
               unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')

           common_words = get_top_n_words(df_n['title'], 10)
           #for word, freq in common_words:
           #    (word, freq)
           df2 = pd.DataFrame(common_words, columns=['Words', 'Count'])
           st.table(df2)
           fig = px.scatter(
               x=df2["Words"],
               y=df2["Count"],
               color=df2["Count"],
           )
           fig.update_layout(
               xaxis_title="Words",
               yaxis_title="Count",
           )

           # st.write(fig)
           st.plotly_chart(fig)

       if nlp == "Bigrams":
           st.markdown("""
                                                                  <style>
                                                                  .big1-font {
                                                                      font-size:20px !important;
                                                                  }
                                                                  </style>
                                                                  """, unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">N Grams</p>', unsafe_allow_html=True)
           st.write(' ')
           st.markdown(
               '<p class="big1-font">N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). When N=1, this is referred to as unigrams and this is essentially the individual words in a sentence. When N=2, this is called bigrams and when N=3 this is called trigrams. When N>3 this is usually referred to as four grams or five grams and so on.</p>',
               unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')
           common_words = get_top_n_bigram(df_n['title'], 10)
           df4 = pd.DataFrame(common_words, columns=['Bigrams', 'Count'])
           st.table(df4)
           fig = px.scatter(
               x=df4["Bigrams"],
               y=df4["Count"],
               color=df4["Count"],
           )
           fig.update_layout(
               xaxis_title="Bigrams",
               yaxis_title="Count",
           )

           # st.write(fig)
           st.plotly_chart(fig)
       # wordcloud.to_image()

       if nlp == 'Trigrams':
           st.markdown("""
                                                                  <style>
                                                                  .big1-font {
                                                                      font-size:20px !important;
                                                                  }
                                                                  </style>
                                                                  """, unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">N Grams</p>', unsafe_allow_html=True)
           st.write(' ')
           st.markdown(
               '<p class="big1-font">N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). When N=1, this is referred to as unigrams and this is essentially the individual words in a sentence. When N=2, this is called bigrams and when N=3 this is called trigrams. When N>3 this is usually referred to as four grams or five grams and so on.</p>',
               unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')
           common_words = get_top_n_trigram(df_n['title'], 10)
           df6 = pd.DataFrame(common_words, columns=['Trigrams', 'Count'])
           st.table(df6)
           fig = px.scatter(
               x=df6["Trigrams"],
               y=df6["Count"],
               color=df6["Count"],
           )
           fig.update_layout(
               xaxis_title="Trigrams",
               yaxis_title="Count",
           )

           # st.write(fig)
           st.plotly_chart(fig)

       if nlp =="Sentiment Analysis TextBlob":
           st.markdown("""
                                 <style>
                                 .big1-font {
                                     font-size:20px !important;
                                 }
                                 </style>
                                 """, unsafe_allow_html=True)
           t_word = "The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective."
           st.write(' ')
           st.markdown('<p class="big1-font">TextBlob Sentiment Analyzer</p>',unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a float within the range [0.0, 1.0]</p>',unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')
           plot_sentiment_barchart(df['title'], method='TextBlob')

       if nlp == "Parts of Speech":
           st.markdown("""
                      <style>
                      .big1-font {
                          font-size:20px !important;
                      }
                      </style>
                      """, unsafe_allow_html=True)
           st.write(" ")
           st.markdown('<p class="big1-font">Noun (NN)- Joseph, London, table, cat, teacher, pen, city</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Verb (VB)- read, speak, run, eat, play, live, walk, have, like, are, is</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Adjective(JJ)- beautiful, happy, sad, young, fun, three</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Adverb(RB)- slowly, quietly, very, always, never, too, well, tomorrow</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Preposition (IN)- at, on, in, from, with, near, between, about, under</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Conjunction (CC)- and, or, but, because, so, yet, unless, since, if</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Pronoun(PRP)- I, you, we, they, he, she, it, me, us, them, him, her,this</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Interjection (INT)- Ouch! Wow! Great! Help! Oh! Hey! Hi!</p>',unsafe_allow_html=True)
           plot_parts_of_speach_barchart(df['content'])

   if news =="BuzzFeed":
       #st.write(news)
       #st.write(nlp)
       if nlp =="Intro":
           #st.write("this is intro")
           image1 = Image.open(r'C:\Users\PC\Desktop\git\nlp-zavrsni\images\5113082_BuzzFeed_Logo.jpg')
           st.write( " ")
           st.image(image1, width=300)
           st.write(" ")
           st.write(" ")
           st.write(" ")
           image = Image.open(r'C:\Users\PC\Desktop\git\nlp-zavrsni\images\buzzfeed-officejpg.jpg')
           st.image(image, width=700)
           st.write(" ")
           st.write(" ")
           st.write(" ")
           st.markdown("""
           BuzzFeed, Inc. is an American Internet media, news and entertainment company with a focus on digital media. Based in New York City, BuzzFeed was founded in 2006 by Jonah Peretti and John S. Johnson III to focus on tracking
           viral content. Kenneth Lerer, co-founder and chairman of The Huffington Post, started as a co-founder and investor in BuzzFeed and is now the executive chairman. Originally known for online quizzes, "listicles", and pop
           culture articles, the company has grown into a global media and technology company, providing coverage on a variety of topics including politics, DIY, animals, and business. In late 2011, BuzzFeed hired Ben Smith of Politico
           as editor-in-chief, to expand the site into long-form journalism and reportage. After years of investment in investigative journalism, by 2021 BuzzFeed News had won the National Magazine Award, the George Polk Award,
           and the Pulitzer Prize, and was nominated for the Michael Kelly Award. BuzzFeed generates revenue by native advertising, a strategy that helps with increasing the likelihood of viewers read through the content of
           advertisement. Despite BuzzFeed's entrance into serious journalism, a 2014 Pew Research Center survey found that in the United States, BuzzFeed was viewed as an unreliable source by the majority of respondents,
           regardless of age or political affiliation. The company's audience has been described as "left-leaning". BuzzFeed News has since moved to its own domain rather than existing as a section of the main BuzzFeed website.
           """)

       df = pd.DataFrame(columns=['title', 'link', 'decription', 'content'])
       url_link = "https://www.buzzfeed.com/politics.xml"
       RSSFeed(url_link)
       df = ndf
       #st.header('Display the dataframe')
       #st.dataframe(df)
       pd.set_option('display.max_rows', df.shape[0] + 1)
       df.reset_index(inplace=True, drop=True)
       for ind in df.index:
           # print(df['title'][ind], df['link'][ind], df['content'][ind])
           url = df['link'][ind]
           #print(url)
           text = full_text(url)
           df['content'][ind] = text
       #st.write(df['title'])
       # Build the corpus.
       corpus = []
       for ind in df.index:
           # corpus = df['content'][ind]
           corpus.append(df['title'][ind])
       #print(corpus)
       df = df.dropna()
       X_train1 = df['title']
       if nlp == "Snapshot":
           st.write(" ")
           st.write(" ")
           st.write(" ")
           st.subheader('Display the dataframe')
           st.write(" ")
           st.write(" ")
           st.write(" ")
           st.dataframe(df)
           st.write(" ")
           st.write(" ")
           st.write(" ")
           st.markdown("""
                                                                                        <style>
                                                                                        .big2-font {
                                                                                            font-size:30px !important;
                                                                                        }
                                                                                        </style>
                                                                                        """, unsafe_allow_html=True)
           st.markdown('<p class="big2-font">The no of articles :</p>', unsafe_allow_html=True)
           st.write(df.shape[0])
           st.write(" ")
           st.write("The Url Link ")
           for index, row in df.iterrows():
               st.write(row['link'])

       if nlp == "WordCloud":
           st.markdown("""
                                                                             <style>
                                                                             .big1-font {
                                                                                 font-size:20px !important;
                                                                             }
                                                                             </style>
                                                                             """, unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">WordCloud</p>', unsafe_allow_html=True)
           st.write(' ')
           st.markdown(
               '<p class="big1-font">Word clouds or tag clouds are graphical representations of word frequency that give greater prominence to words that appear more frequently in a source text. The larger the word in the visual the more common the word was in the document(s).</p>',
               unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')
           long_string = ','.join(list(X_train1.values))
           # Create a WordCloud object
           wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
           # Generate a word cloud
           wordcloud.generate(long_string)
           # Visualize the word cloud
           plt.figure(figsize=(20, 10))
           plt.imshow(wordcloud)
           st.image(wordcloud.to_array(), width=700)
           st.write("Word Cloud")
           # Generate word cloud
           long_string = ','.join(list(X_train1.values))
           wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='salmon', colormap='Pastel1',
                                 collocations=False, stopwords=STOPWORDS).generate(long_string)
           # Visualize the word cloud
           plt.figure(figsize=(20, 10))
           plt.imshow(wordcloud)
           st.image(wordcloud.to_array(), width=700)
           st.write("Word Cloud")
           wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='black', colormap='Set2',
                                 collocations=False, stopwords=STOPWORDS).generate(long_string)
           # Visualize the word cloud
           plt.figure(figsize=(20, 10))
           plt.imshow(wordcloud)
           st.image(wordcloud.to_array(), width=700)

       df_n = df
       df_n['title'] = preprocess(df['title'])

       if nlp == "Unigrams":
           st.markdown("""
                                                       <style>
                                                       .big1-font {
                                                           font-size:20px !important;
                                                       }
                                                       </style>
                                                       """, unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">N Grams</p>', unsafe_allow_html=True)
           st.write(' ')
           st.markdown(
               '<p class="big1-font">N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). When N=1, this is referred to as unigrams and this is essentially the individual words in a sentence. When N=2, this is called bigrams and when N=3 this is called trigrams. When N>3 this is usually referred to as four grams or five grams and so on.</p>',
               unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')

           common_words = get_top_n_words(df_n['title'], 10)
           #for word, freq in common_words:
           #    (word, freq)
           df2 = pd.DataFrame(common_words, columns=['Words', 'Count'])
           st.table(df2)
           fig = px.scatter(
               x=df2["Words"],
               y=df2["Count"],
               color=df2["Count"],
           )
           fig.update_layout(
               xaxis_title="Words",
               yaxis_title="Count",
           )

           # st.write(fig)
           st.plotly_chart(fig)

       if nlp == "Bigrams":
           st.markdown("""
                                                                  <style>
                                                                  .big1-font {
                                                                      font-size:20px !important;
                                                                  }
                                                                  </style>
                                                                  """, unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">N Grams</p>', unsafe_allow_html=True)
           st.write(' ')
           st.markdown(
               '<p class="big1-font">N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). When N=1, this is referred to as unigrams and this is essentially the individual words in a sentence. When N=2, this is called bigrams and when N=3 this is called trigrams. When N>3 this is usually referred to as four grams or five grams and so on.</p>',
               unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')
           common_words = get_top_n_bigram(df_n['title'], 10)
           df4 = pd.DataFrame(common_words, columns=['Bigrams', 'Count'])
           st.table(df4)
           fig = px.scatter(
               x=df4["Bigrams"],
               y=df4["Count"],
               color=df4["Count"],
           )
           fig.update_layout(
               xaxis_title="Bigrams",
               yaxis_title="Count",
           )

           # st.write(fig)
           st.plotly_chart(fig)
       # wordcloud.to_image()

       if nlp == 'Trigrams':
           st.markdown("""
                                                                  <style>
                                                                  .big1-font {
                                                                      font-size:20px !important;
                                                                  }
                                                                  </style>
                                                                  """, unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">N Grams</p>', unsafe_allow_html=True)
           st.write(' ')
           st.markdown(
               '<p class="big1-font">N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). When N=1, this is referred to as unigrams and this is essentially the individual words in a sentence. When N=2, this is called bigrams and when N=3 this is called trigrams. When N>3 this is usually referred to as four grams or five grams and so on.</p>',
               unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')
           common_words = get_top_n_trigram(df_n['title'], 10)
           df6 = pd.DataFrame(common_words, columns=['Trigrams', 'Count'])
           st.table(df6)
           fig = px.scatter(
               x=df6["Trigrams"],
               y=df6["Count"],
               color=df6["Count"],
           )
           fig.update_layout(
               xaxis_title="Trigrams",
               yaxis_title="Count",
           )

           # st.write(fig)
           st.plotly_chart(fig)

       if nlp =="Sentiment Analysis TextBlob":
           st.markdown("""
                                 <style>
                                 .big1-font {
                                     font-size:20px !important;
                                 }
                                 </style>
                                 """, unsafe_allow_html=True)
           t_word = "The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective."
           st.write(' ')
           st.markdown('<p class="big1-font">TextBlob Sentiment Analyzer</p>',unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a float within the range [0.0, 1.0]</p>',unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')
           plot_sentiment_barchart(df['title'], method='TextBlob')

       if nlp == "Parts of Speech":
           st.markdown("""
                      <style>
                      .big1-font {
                          font-size:20px !important;
                      }
                      </style>
                      """, unsafe_allow_html=True)
           st.write(" ")
           st.markdown('<p class="big1-font">Noun (NN)- Joseph, London, table, cat, teacher, pen, city</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Verb (VB)- read, speak, run, eat, play, live, walk, have, like, are, is</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Adjective(JJ)- beautiful, happy, sad, young, fun, three</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Adverb(RB)- slowly, quietly, very, always, never, too, well, tomorrow</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Preposition (IN)- at, on, in, from, with, near, between, about, under</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Conjunction (CC)- and, or, but, because, so, yet, unless, since, if</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Pronoun(PRP)- I, you, we, they, he, she, it, me, us, them, him, her,this</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Interjection (INT)- Ouch! Wow! Great! Help! Oh! Hey! Hi!</p>',unsafe_allow_html=True)
           plot_parts_of_speach_barchart(df['content'])

   if news =="Huffington Post":
       #st.write(news)
       #st.write(nlp)
       if nlp =="Intro":
           #st.write("this is intro")
           image1 = Image.open(r'C:\Users\PC\Desktop\git\nlp-zavrsni\images\HuffPost.svg.jpg')
           st.write( " ")
           st.image(image1, width=300)
           st.write(" ")
           st.write(" ")
           st.write(" ")
           image = Image.open(r'C:\Users\PC\Desktop\git\nlp-zavrsni\images\huffington-post-office.jpg')
           st.image(image, width=700)
           st.write(" ")
           st.write(" ")
           st.write(" ")
           st.markdown("""
           HuffPost (formerly The Huffington Post until 2017 and sometimes abbreviated HuffPo) is an American news aggregator and blog, with localized and international editions. The site offers news, satire, blogs, and original content
           , and covers politics, business, entertainment, environment, technology, popular media, lifestyle, culture, comedy, healthy living, women's interests, and local news featuring columnists. It was created to provide a liberal
           alternative to the conservative news websites such as the Drudge Report. The site offers content posted directly on the site as well as user-generated content via video blogging, audio, and photo. In 2012, the website became
           the first commercially run United States digital media enterprise to win a Pulitzer Prize. Founded by Andrew Breitbart, Arianna Huffington, Kenneth Lerer, and Jonah Peretti, the site was launched on May 9, 2005, as
           counterpart to the Drudge Report. In March 2011, it was acquired by AOL, making Arianna Huffington editor-in-chief. In June 2015, Verizon Communications acquired AOL and the site became
           a part of Verizon Media. In November 2020, BuzzFeed acquired the company. Weeks after the acquisition, BuzzFeed laid off 47 HuffPost staff in the U.S. (mostly journalists) and closed down HuffPost Canada, laying off 23 staff
           working for the Canadian and Quebec divisions of the company.
           """)

       df = pd.DataFrame(columns=['title', 'link', 'decription', 'content'])
       url_link = "http://www.huffingtonpost.com/feeds/verticals/world/index.xml"
       RSSFeed(url_link)
       df = ndf
       #st.header('Display the dataframe')
       #st.dataframe(df)
       pd.set_option('display.max_rows', df.shape[0] + 1)
       df.reset_index(inplace=True, drop=True)
       for ind in df.index:
           # print(df['title'][ind], df['link'][ind], df['content'][ind])
           url = df['link'][ind]
           #print(url)
           text = full_text(url)
           df['content'][ind] = text
       #st.write(df['title'])
       # Build the corpus.
       corpus = []
       for ind in df.index:
           # corpus = df['content'][ind]
           corpus.append(df['title'][ind])
       #print(corpus)
       df = df.dropna()
       X_train1 = df['title']
       if nlp == "Snapshot":
           st.write(" ")
           st.write(" ")
           st.write(" ")
           st.subheader('Display the dataframe')
           st.write(" ")
           st.write(" ")
           st.write(" ")
           st.dataframe(df)
           st.write(" ")
           st.write(" ")
           st.write(" ")
           st.markdown("""
                                                                                        <style>
                                                                                        .big2-font {
                                                                                            font-size:30px !important;
                                                                                        }
                                                                                        </style>
                                                                                        """, unsafe_allow_html=True)
           st.markdown('<p class="big2-font">The no of articles :</p>', unsafe_allow_html=True)
           st.write(df.shape[0])
           st.write(" ")
           st.write("The Url Link ")
           for index, row in df.iterrows():
               st.write(row['link'])

       if nlp == "WordCloud":
           st.markdown("""
                                                                             <style>
                                                                             .big1-font {
                                                                                 font-size:20px !important;
                                                                             }
                                                                             </style>
                                                                             """, unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">WordCloud</p>', unsafe_allow_html=True)
           st.write(' ')
           st.markdown(
               '<p class="big1-font">Word clouds or tag clouds are graphical representations of word frequency that give greater prominence to words that appear more frequently in a source text. The larger the word in the visual the more common the word was in the document(s).</p>',
               unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')
           long_string = ','.join(list(X_train1.values))
           # Create a WordCloud object
           wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
           # Generate a word cloud
           wordcloud.generate(long_string)
           # Visualize the word cloud
           plt.figure(figsize=(20, 10))
           plt.imshow(wordcloud)
           st.image(wordcloud.to_array(), width=700)
           st.write("Word Cloud")
           # Generate word cloud
           long_string = ','.join(list(X_train1.values))
           wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='salmon', colormap='Pastel1',
                                 collocations=False, stopwords=STOPWORDS).generate(long_string)
           # Visualize the word cloud
           plt.figure(figsize=(20, 10))
           plt.imshow(wordcloud)
           st.image(wordcloud.to_array(), width=700)
           st.write("Word Cloud")
           wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='black', colormap='Set2',
                                 collocations=False, stopwords=STOPWORDS).generate(long_string)
           # Visualize the word cloud
           plt.figure(figsize=(20, 10))
           plt.imshow(wordcloud)
           st.image(wordcloud.to_array(), width=700)

       df_n = df
       df_n['title'] = preprocess(df['title'])

       if nlp == "Unigrams":
           st.markdown("""
                                                       <style>
                                                       .big1-font {
                                                           font-size:20px !important;
                                                       }
                                                       </style>
                                                       """, unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">N Grams</p>', unsafe_allow_html=True)
           st.write(' ')
           st.markdown(
               '<p class="big1-font">N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). When N=1, this is referred to as unigrams and this is essentially the individual words in a sentence. When N=2, this is called bigrams and when N=3 this is called trigrams. When N>3 this is usually referred to as four grams or five grams and so on.</p>',
               unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')

           common_words = get_top_n_words(df_n['title'], 10)
           #for word, freq in common_words:
           #    (word, freq)
           df2 = pd.DataFrame(common_words, columns=['Words', 'Count'])
           st.table(df2)
           fig = px.scatter(
               x=df2["Words"],
               y=df2["Count"],
               color=df2["Count"],
           )
           fig.update_layout(
               xaxis_title="Words",
               yaxis_title="Count",
           )

           # st.write(fig)
           st.plotly_chart(fig)

       if nlp == "Bigrams":
           st.markdown("""
                                                                  <style>
                                                                  .big1-font {
                                                                      font-size:20px !important;
                                                                  }
                                                                  </style>
                                                                  """, unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">N Grams</p>', unsafe_allow_html=True)
           st.write(' ')
           st.markdown(
               '<p class="big1-font">N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). When N=1, this is referred to as unigrams and this is essentially the individual words in a sentence. When N=2, this is called bigrams and when N=3 this is called trigrams. When N>3 this is usually referred to as four grams or five grams and so on.</p>',
               unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')
           common_words = get_top_n_bigram(df_n['title'], 10)
           df4 = pd.DataFrame(common_words, columns=['Bigrams', 'Count'])
           st.table(df4)
           fig = px.scatter(
               x=df4["Bigrams"],
               y=df4["Count"],
               color=df4["Count"],
           )
           fig.update_layout(
               xaxis_title="Bigrams",
               yaxis_title="Count",
           )

           # st.write(fig)
           st.plotly_chart(fig)
       # wordcloud.to_image()

       if nlp == 'Trigrams':
           st.markdown("""
                                                                  <style>
                                                                  .big1-font {
                                                                      font-size:20px !important;
                                                                  }
                                                                  </style>
                                                                  """, unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">N Grams</p>', unsafe_allow_html=True)
           st.write(' ')
           st.markdown(
               '<p class="big1-font">N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). When N=1, this is referred to as unigrams and this is essentially the individual words in a sentence. When N=2, this is called bigrams and when N=3 this is called trigrams. When N>3 this is usually referred to as four grams or five grams and so on.</p>',
               unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')
           common_words = get_top_n_trigram(df_n['title'], 10)
           df6 = pd.DataFrame(common_words, columns=['Trigrams', 'Count'])
           st.table(df6)
           fig = px.scatter(
               x=df6["Trigrams"],
               y=df6["Count"],
               color=df6["Count"],
           )
           fig.update_layout(
               xaxis_title="Trigrams",
               yaxis_title="Count",
           )

           # st.write(fig)
           st.plotly_chart(fig)

       if nlp =="Sentiment Analysis TextBlob":
           st.markdown("""
                                 <style>
                                 .big1-font {
                                     font-size:20px !important;
                                 }
                                 </style>
                                 """, unsafe_allow_html=True)
           t_word = "The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective."
           st.write(' ')
           st.markdown('<p class="big1-font">TextBlob Sentiment Analyzer</p>',unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a float within the range [0.0, 1.0]</p>',unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')
           plot_sentiment_barchart(df['title'], method='TextBlob')

       if nlp == "Parts of Speech":
           st.markdown("""
                      <style>
                      .big1-font {
                          font-size:20px !important;
                      }
                      </style>
                      """, unsafe_allow_html=True)
           st.write(" ")
           st.markdown('<p class="big1-font">Noun (NN)- Joseph, London, table, cat, teacher, pen, city</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Verb (VB)- read, speak, run, eat, play, live, walk, have, like, are, is</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Adjective(JJ)- beautiful, happy, sad, young, fun, three</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Adverb(RB)- slowly, quietly, very, always, never, too, well, tomorrow</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Preposition (IN)- at, on, in, from, with, near, between, about, under</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Conjunction (CC)- and, or, but, because, so, yet, unless, since, if</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Pronoun(PRP)- I, you, we, they, he, she, it, me, us, them, him, her,this</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Interjection (INT)- Ouch! Wow! Great! Help! Oh! Hey! Hi!</p>',unsafe_allow_html=True)
           plot_parts_of_speach_barchart(df['content'])

   if news =="The Wall Street Journal":
       #st.write(news)
       #st.write(nlp)
       if nlp =="Intro":
           #st.write("this is intro")
           image1 = Image.open(r'C:\Users\PC\Desktop\git\nlp-zavrsni\images\Wall-Street-Journal-logo.jpg')
           st.write( " ")
           st.image(image1, width=300)
           st.write(" ")
           st.write(" ")
           st.write(" ")
           image = Image.open(r'C:\Users\PC\Desktop\git\nlp-zavrsni\images\wall-street-journal-headquarters-john-wisniewski-700-700x525.jpg')
           st.image(image, width=700)
           st.write(" ")
           st.write(" ")
           st.write(" ")
           st.markdown("""
           The Wall Street Journal is an American business-focused, international daily newspaper based in New York City, with international editions also available in Chinese and Japanese. The Journal, along with its Asian editions,
           is published six days a week by Dow Jones & Company, a division of News Corp. The newspaper is published in the broadsheet format and online. The Journal has been printed continuously since its inception on July 8, 1889,
           by Charles Dow, Edward Jones, and Charles Bergstresser. The Journal is regarded as a newspaper of record, particularly in terms of business and financial news. The newspaper has won 38 Pulitzer Prizes, the most recent in
           2019. The Wall Street Journal is one of the largest newspapers in the United States by circulation, with a circulation of about 2.834 million copies (including nearly 1,829,000 digital sales) as of August 2019, compared
           with USA Today's 1.7 million. The Journal publishes the luxury news and lifestyle magazine WSJ, which was originally launched as a quarterly but expanded to 12 issues in 2014. An online version was launched in 1995, which
           has been accessible only to subscribers since it began.
           """)

       df = pd.DataFrame(columns=['title', 'link', 'decription', 'content'])
       url_link = "https://feeds.a.dj.com/rss/RSSWorldNews.xml"
       RSSFeed(url_link)
       df = ndf
       #st.header('Display the dataframe')
       #st.dataframe(df)
       pd.set_option('display.max_rows', df.shape[0] + 1)
       df.reset_index(inplace=True, drop=True)
       for ind in df.index:
           # print(df['title'][ind], df['link'][ind], df['content'][ind])
           url = df['link'][ind]
           #print(url)
           text = full_text(url)
           df['content'][ind] = text
       #st.write(df['title'])
       # Build the corpus.
       corpus = []
       for ind in df.index:
           # corpus = df['content'][ind]
           corpus.append(df['title'][ind])
       #print(corpus)
       df = df.dropna()
       X_train1 = df['title']
       if nlp == "Snapshot":
           st.write(" ")
           st.write(" ")
           st.write(" ")
           st.subheader('Display the dataframe')
           st.write(" ")
           st.write(" ")
           st.write(" ")
           st.dataframe(df)
           st.write(" ")
           st.write(" ")
           st.write(" ")
           st.markdown("""
                                                                                        <style>
                                                                                        .big2-font {
                                                                                            font-size:30px !important;
                                                                                        }
                                                                                        </style>
                                                                                        """, unsafe_allow_html=True)
           st.markdown('<p class="big2-font">The no of articles :</p>', unsafe_allow_html=True)
           st.write(df.shape[0])
           st.write(" ")
           st.write("The Url Link ")
           for index, row in df.iterrows():
               st.write(row['link'])

       if nlp == "WordCloud":
           st.markdown("""
                                                                             <style>
                                                                             .big1-font {
                                                                                 font-size:20px !important;
                                                                             }
                                                                             </style>
                                                                             """, unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">WordCloud</p>', unsafe_allow_html=True)
           st.write(' ')
           st.markdown(
               '<p class="big1-font">Word clouds or tag clouds are graphical representations of word frequency that give greater prominence to words that appear more frequently in a source text. The larger the word in the visual the more common the word was in the document(s).</p>',
               unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')
           long_string = ','.join(list(X_train1.values))
           # Create a WordCloud object
           wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
           # Generate a word cloud
           wordcloud.generate(long_string)
           # Visualize the word cloud
           plt.figure(figsize=(20, 10))
           plt.imshow(wordcloud)
           st.image(wordcloud.to_array(), width=700)
           st.write("Word Cloud")
           # Generate word cloud
           long_string = ','.join(list(X_train1.values))
           wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='salmon', colormap='Pastel1',
                                 collocations=False, stopwords=STOPWORDS).generate(long_string)
           # Visualize the word cloud
           plt.figure(figsize=(20, 10))
           plt.imshow(wordcloud)
           st.image(wordcloud.to_array(), width=700)
           st.write("Word Cloud")
           wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='black', colormap='Set2',
                                 collocations=False, stopwords=STOPWORDS).generate(long_string)
           # Visualize the word cloud
           plt.figure(figsize=(20, 10))
           plt.imshow(wordcloud)
           st.image(wordcloud.to_array(), width=700)

       df_n = df
       df_n['title'] = preprocess(df['title'])

       if nlp == "Unigrams":
           st.markdown("""
                                                       <style>
                                                       .big1-font {
                                                           font-size:20px !important;
                                                       }
                                                       </style>
                                                       """, unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">N Grams</p>', unsafe_allow_html=True)
           st.write(' ')
           st.markdown(
               '<p class="big1-font">N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). When N=1, this is referred to as unigrams and this is essentially the individual words in a sentence. When N=2, this is called bigrams and when N=3 this is called trigrams. When N>3 this is usually referred to as four grams or five grams and so on.</p>',
               unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')

           common_words = get_top_n_words(df_n['title'], 10)
           #for word, freq in common_words:
           #    (word, freq)
           df2 = pd.DataFrame(common_words, columns=['Words', 'Count'])
           st.table(df2)
           fig = px.scatter(
               x=df2["Words"],
               y=df2["Count"],
               color=df2["Count"],
           )
           fig.update_layout(
               xaxis_title="Words",
               yaxis_title="Count",
           )

           # st.write(fig)
           st.plotly_chart(fig)

       if nlp == "Bigrams":
           st.markdown("""
                                                                  <style>
                                                                  .big1-font {
                                                                      font-size:20px !important;
                                                                  }
                                                                  </style>
                                                                  """, unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">N Grams</p>', unsafe_allow_html=True)
           st.write(' ')
           st.markdown(
               '<p class="big1-font">N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). When N=1, this is referred to as unigrams and this is essentially the individual words in a sentence. When N=2, this is called bigrams and when N=3 this is called trigrams. When N>3 this is usually referred to as four grams or five grams and so on.</p>',
               unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')
           common_words = get_top_n_bigram(df_n['title'], 10)
           df4 = pd.DataFrame(common_words, columns=['Bigrams', 'Count'])
           st.table(df4)
           fig = px.scatter(
               x=df4["Bigrams"],
               y=df4["Count"],
               color=df4["Count"],
           )
           fig.update_layout(
               xaxis_title="Bigrams",
               yaxis_title="Count",
           )

           # st.write(fig)
           st.plotly_chart(fig)
       # wordcloud.to_image()

       if nlp == 'Trigrams':
           st.markdown("""
                                                                  <style>
                                                                  .big1-font {
                                                                      font-size:20px !important;
                                                                  }
                                                                  </style>
                                                                  """, unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">N Grams</p>', unsafe_allow_html=True)
           st.write(' ')
           st.markdown(
               '<p class="big1-font">N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). When N=1, this is referred to as unigrams and this is essentially the individual words in a sentence. When N=2, this is called bigrams and when N=3 this is called trigrams. When N>3 this is usually referred to as four grams or five grams and so on.</p>',
               unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')
           common_words = get_top_n_trigram(df_n['title'], 10)
           df6 = pd.DataFrame(common_words, columns=['Trigrams', 'Count'])
           st.table(df6)
           fig = px.scatter(
               x=df6["Trigrams"],
               y=df6["Count"],
               color=df6["Count"],
           )
           fig.update_layout(
               xaxis_title="Trigrams",
               yaxis_title="Count",
           )

           # st.write(fig)
           st.plotly_chart(fig)

       if nlp =="Sentiment Analysis TextBlob":
           st.markdown("""
                                 <style>
                                 .big1-font {
                                     font-size:20px !important;
                                 }
                                 </style>
                                 """, unsafe_allow_html=True)
           t_word = "The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective."
           st.write(' ')
           st.markdown('<p class="big1-font">TextBlob Sentiment Analyzer</p>',unsafe_allow_html=True)
           st.write(' ')
           st.markdown('<p class="big1-font">The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a float within the range [0.0, 1.0]</p>',unsafe_allow_html=True)
           st.write(' ')
           st.write(' ')
           plot_sentiment_barchart(df['title'], method='TextBlob')           

       if nlp == "Parts of Speech":
           st.markdown("""
                      <style>
                      .big1-font {
                          font-size:20px !important;
                      }
                      </style>
                      """, unsafe_allow_html=True)
           st.write(" ")
           st.markdown('<p class="big1-font">Noun (NN)- Joseph, London, table, cat, teacher, pen, city</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Verb (VB)- read, speak, run, eat, play, live, walk, have, like, are, is</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Adjective(JJ)- beautiful, happy, sad, young, fun, three</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Adverb(RB)- slowly, quietly, very, always, never, too, well, tomorrow</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Preposition (IN)- at, on, in, from, with, near, between, about, under</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Conjunction (CC)- and, or, but, because, so, yet, unless, since, if</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Pronoun(PRP)- I, you, we, they, he, she, it, me, us, them, him, her,this</p>',unsafe_allow_html=True)
           st.markdown('<p class="big1-font">Interjection (INT)- Ouch! Wow! Great! Help! Oh! Hey! Hi!</p>',unsafe_allow_html=True)
           plot_parts_of_speach_barchart(df['content'])

load_data(s_news,s_nlp)

