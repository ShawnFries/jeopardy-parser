# Simple deep learning model with 2 hidden layers to try to predict whether or not the clue text is indicative of a very difficult clue
# Accuracy is only ~91.5% at most (baseline ~90% if entirely random)
# Credit: Much of the basic structure here is from an official keras tutorial:
# https://tensorflow.rstudio.com/keras/articles/tutorial_basic_text_classification.html

library(dplyr)
library(dbplyr)
library(keras)
clues <- DBI::dbConnect(RSQLite::SQLite(), "/jeopardy-parser-master/clues.db")
src_dbi(clues)

clues_and_dollar_values <- as.data.frame(tbl(clues, sql("SELECT clue,
                                                                value,
                                                                airdate
                                                           FROM documents AS d
                                                                INNER JOIN clues AS c
                                                                ON d.id = c.id
                                                                
                                                                INNER JOIN airdates AS a
                                                                ON a.game = c.game
                                                          WHERE airdate > '2001-11-25'
                                                            AND round IN (1, 2)
                                                            AND value BETWEEN 200 AND 2000
                                                            AND value % 200 = 0")))
tokenizer <- text_tokenizer()
fit_text_tokenizer(tokenizer, x=clues_and_dollar_values$clue)
int_sequences <- texts_to_sequences(tokenizer, clues_and_dollar_values$clue)

padded_sequences <- pad_sequences(int_sequences, padding="post")

is_most_difficult_clue <- ifelse(clues_and_dollar_values$value==2000,1,0)

vocab_size <- max(padded_sequences) + 1
vocab_size

model <- keras_model_sequential()
model %>% 
  layer_embedding(input_dim=vocab_size, output_dim=16) %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units=16, activation='relu') %>%
  layer_dense(units=1, activation='relu')

model %>% compile(optimizer='adam', loss='binary_crossentropy', metrics=list('accuracy'))

history <- model %>% fit(
  head(padded_sequences, -10000),
  head(is_most_difficult_clue, -10000),
  epochs=3,
  batch_size=16,
  verbose=1
)

results <- model %>% evaluate(tail(padded_sequences, 10000), tail(is_most_difficult_clue, 10000))
results

plot(history)