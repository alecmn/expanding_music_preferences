# Feedback-Driven Gradual Discovery for Expanding Musical Preferences

This repository contains supplementary content to aid in the understanding of our methods and provide implementation details necessary to reproduce our experiments.

## CKG Embeddings

Details on the size and contents of the CKG can be found in `ckg_schema.png`. Embeddings can be found in `embeddings.pkl` and were generated from the CKG using Node2Vec that incorporates edge weights in its random walks with these parameters:

- dimensions = 128
- walk-length = 40-
- context-window = 5
- p = 0.5
- q = 0.5

We include an additional figure `embedding_visualization.png` displaying a t-SNE visualization of the node2vec embeddings showing the separation of genres. For added context the number of songs per genre are below.

| Genre               | # Songs |
| ------------------- | ------- |
| Pop                 | 17593   |
| Rap/Hip-Hop         | 16824   |
| Alternative/Indie   | 16636   |
| Latin               | 12250   |
| Electronic/Dance    | 6676    |
| R&B/Soul            | 6574    |
| Rock                | 5146    |
| Country             | 4808    |
| Hard Rock/Metal     | 4656    |
| Holiday             | 1180    |
| World/Roots         | 1056    |
| Soundtracks         | 1028    |
| Religious/Spiritual | 990     |
| Classical           | 922     |
| Reggae              | 866     |
| Folk/Americana      | 835     |
| Jazz                | 792     |
| Blues               | 371     |
| Children's          | 316     |

## Playlist Completion Protocol

Playlist completion done using a holdout set from the 760 playlists in the graph and usign a subset of Spotify playlists from a dataset with 7,405 playlists containing 20,007 unique songs.

1. Hold-out split\
   Each curated playlist $P$ is removed entirely from the graph before training.

2. Seeding\
   Randomly sample $n$ = 50 % of P’s tracks as the seed set S.

3. Retrieval\
   Compute the centroid and return the $|P| - n$ nearest neighbours in cosine space.

4. Metrics\
   Report R-precision and nDCG@$|P|$

## Code

All code for generating paths, sampling paths and updating their distributions can be found in `utils.py`.

## User Experiments

The jupyter notebook `results_analysis.ipynb` contains the code used for analyzing the results of the user experiments from the `responses.csv` and `user_likes.csv` files. For more detail on the user experiment we show several screenshots of the experiment platform in the `Figures` folder and we provide an overview of the subjective factors and their survey questions.

| **EXP/SSA**                | **Question item**                                                                          |
| -------------------------- | ------------------------------------------------------------------------------------------ |
| **Perceived helpfulness**  | This approach supports me in getting to know the new genre.                                |
|                            | This approach motivates me to more delve into the new genre.                               |
|                            | This approach is useful in exploring a new genre.                                          |
| **Affinity Toward Target** | I enjoy the music from [target taste].                                                     |
|                            | My enjoyment of music from [target taste] has increased since the start of the experiment. |
|                            | My enjoyment of music from [target taste] has increased since the previous step.           |
| **Quality of Direction**   | I can notice the recommendations going in the direction of the target.                     |
|                            | The recommended songs seem to be in between my preferences and the target.                 |
| **Personalization**        | I feel like the recommended songs take my preferences into account.                        |
|                            | I find the songs from the playlist appealing.                                              |
|                            | I would listen to the playlist again.                                                      |
| **Control**                | I found it easy to modify the recommendations in the recommender.                          |
|                            | The recommender allows only limited control to modify the recommendations.                 |
|                            | I feel in control of modifying the recommendations.                                        |
| **Understandability**      | I understand how the recommended songs relate to my musical taste.                         |
|                            | It is easy to grasp why I received these recommended songs.                                |
|                            | The recommendation process is clear to me.                                                 |
