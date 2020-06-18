# NLP Analysis: Song Lyrics

How well do songs separate into different topics? Humans are pretty good at spotting love songs, sad songs, dance songs, and many more types. But how easy is it for computers to pick up on these themes and emotions? This project aims to use natural language processing to analyze song lyrics.  

## The Data

A list of songs was collected from the Billboard Hot 100, a prominent song chart with a new chart coming out every week. The scraper collected songs going back 25 years, jumping 13 weeks at a time in order to minimize time scraping and to maximize variety amongst each chart collected. It is not uncommon for songs to spend several weeks on the Hot 100.  

Once this list of songs was collected, the API lyrics.ovh was used to collect song lyrics. Lyrics.ovh was selected because of its ease of use and simple output. All song information and lyrics were stored on MongoDB.  

Duplicates and songs with missing lyrics were removed and 6252 songs remained in the dataset.  

## Exploratory Data Analysis

We start our EDA looking at the word count and character count of each song. We see that a lot of these songs from the Billboard Hot 100 contain between 200 and 400 words, but we see a few that have more than 800.  

![Word_char_count](<https://github.com/sn-ekstrand/lyric-clustering/blob/master/images/word_char_jointplot.png?raw=true>)  

["Rap God"](<https://youtu.be/XbGs_qK2PQA>) by Eminem is in the upper right with 1,539 words. In fact, we find that most in the upper right are rap songs.  

But wait, who's that extreme outlier in the bottom left? Are they using really long words or what?  

Nope, it's just ["Barbra Streisand"](<https://youtu.be/wWhtcU4-xAM>) by Duck Sauce and vocalizations showing up as long words. If you need to stop reading this and go listen, I'd understand. Nay, I'd respect it.  

Next, let's see how many songs are talking about love and hate. The following bar plot shows how many songs only mention love, only mention hate, or mention both.  

![love_hate](<https://github.com/sn-ekstrand/lyric-clustering/blob/master/images/love_hate.png?raw=true>)  

That's not surprising. How many of us want to listen to songs about hate when we can listen to love songs. Also, it's no secret that [love songs](<https://youtu.be/qi7Yh16dA0w>) are pushed by label companies.  

## Natural Language Processing

To be able to do a more complex analysis of our lyrics with a computer we need to translate the information into something a computer can understand. Numbers are easy to work with while text is not.  

We split up the text into separate words, lower case the words, convert the words to their lemmas, and then pass the many lists of words through a tf-idf vectorizer. This gives us a number that represents how common a word is in a song and how rare it is throughout all songs. We also use a standard list of English stopwords to remove some words from our list.  

## Classification

Running an initial NMF model to see if any topics pop up doesn't turn out to be very informative. We usually get one topic that is rap (lots of swear words) and a lot of topics that look like generic pop songs.  

While the goal is to explore topics amongst songs, we should probably do some classification first. There are a lot of words that don't mean much to us like "oh", "yeah", and "baby". Let's see if we can find some stop words to add by running some classification models.  

To do this we try to predict an artist based on the tf-idf vectorized lyrics. After running a Random Forrest Classifier we can then look at the Gini index of each word to see which words are the most important for making predictions. We do this with several artists and combine the results to get a new list of words to add to our stop words.  

## Topic Modeling

We now go back to our NMF model and try again. An NMF model was chosen for its interpretability. We see a little more clarity in our topics but not much. There are still a lot of words that are not helpful. At this point, a few stop words are added by hand and we see some improvement.  

## Future Work

Using spaCy we could look at just the nouns or adjectives in a song. This would be a lot more simple that using all types of words and could give us a good starting point.  

Lyrics can be complex. With the mind of a human, you might argue that a lot of pop songs are not complex but the computer disagrees. While n-grams were experimented with, n-grams were set 1 for this study. We can push that up to 2, 3, or more and should see some improvements.  

Stop words were implemented before any models were run. It would be interesting to see what our topics look like without any stop words. This would give us a chance to be more systematic while choosing stop words.  

Word vectors could be used to distill ideas down to a single vector. The computer could go about interpreting things differently.  

It could be worthwhile to see how effective a neural network is at understanding meaning or the relationship between words.  

And finally, more data is always nice. More songs should be collected and not just those that make it to the Hot 100. One of the problems faced while doing classification was making predictions with imbalanced data, having at most only a handful of songs for one artist. The artist with the most songs in the dataset was Taylor Swift with 58 songs.
