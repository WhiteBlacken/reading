# class ReadingArticle:
#     all_values = {
#         'syllable': [],
#         'length': [],
#         'fam': [],
#         'ent_flag': [],
#         'stop_flag': [],
#         'topic_score': [],
#     }
#     mean_values = {}
#     std_values = {}
#
#     word_fam_map = {}
#     with open('mrc2.dct', 'r') as fp:
#         i = 0
#         for line in fp:
#             line = line.strip()
#
#             word, phon, dphon, stress = line[51:].split('|')
#
#             w = {
#                 'wid': i,
#                 'nlet': int(line[0:2]),
#                 'nphon': int(line[2:4]),
#                 'nsyl': int(line[4]),
#                 'kf_freq': int(line[5:10]),
#                 'kf_ncats': int(line[10:12]),
#                 'kf_nsamp': int(line[12:15]),
#                 'tl_freq': int(line[15:21]),
#                 'brown_freq': int(line[21:25]),
#                 'fam': int(line[25:28]),
#                 'conc': int(line[28:31]),
#                 'imag': int(line[31:34]),
#                 'meanc': int(line[34:37]),
#                 'meanp': int(line[37:40]),
#                 'aoa': int(line[40:43]),
#                 'tq2': line[43],
#                 'wtype': line[44],
#                 'pdwtype': line[45],
#                 'alphasyl': line[46],
#                 'status': line[47],
#                 'var': line[48],
#                 'cap': line[49],
#                 'irreg': line[50],
#                 'word': word,
#                 'phon': phon,
#                 'dphon': dphon,
#                 'stress': stress
#             }
#             if word not in word_fam_map:
#                 word_fam_map[word] = w['fam']
#             word_fam_map[word] = max(word_fam_map[word], w['fam'])
#             i += 1
#
#     def __init__(self, article_text):
#         self.text = article_text
#         self._spacy_doc = nlp(article_text)
#         self._word_list = [token.text for token in self._spacy_doc]
#         self._transformer_features = self._generate_word_embedding(xlnet_model)
#         self._difficulty_features = self._generate_word_difficulty()
#         self._topic_features = self._generate_topic_feature(top_n=1000)
#         self._sentence_word_mapping = self._generate_sentence_mapping()
#
#     def _generate_word_embedding(self, language_model):
#         inputs = xlnet_tokenizer(self.text, return_tensors='pt')
#         word_token_mapping = self.generate_token_mapping(self._word_list, inputs.tokens())
#         outputs = language_model(**inputs)
#         token_embeddings = outputs[0].squeeze()
#         word_embeddings = []
#         for start_id, end_id in word_token_mapping:
#             if start_id <= end_id:
#                 word_embeddings.append(
#                     torch.mean(token_embeddings[start_id: end_id + 1, :], 0, dtype=torch.float32))
#             else:
#                 word_embeddings.append(torch.zeros((token_embeddings.shape[1],), dtype=torch.float32))
#         return word_embeddings
#
#     def _generate_sentence_mapping(self):
#         self._sentence_list = sent_tokenize(self.text)
#         sentence_word_mapping = self.generate_token_mapping(self._sentence_list, self._word_list)
#         return sentence_word_mapping
#
#     @staticmethod
#     def standardization(value, column):
#         if ReadingArticle.std_values[column] != 0.:
#             return (value - ReadingArticle.mean_values[column]) / ReadingArticle.std_values[column]
#         else:
#             return 0.
#
#     @staticmethod
#     def generate_token_mapping(string_list, token_list):
#         string_pos = 0
#         string_idx = 0
#         token_string_idx_list = []
#         max_cross_count = 3
#         for token_idx, token in enumerate(token_list):
#             original_token = token.replace('▁', '')
#             flag = False
#             while string_idx < len(string_list) and string_list[string_idx][
#                                                     string_pos: string_pos + len(original_token)] != original_token:
#                 string_pos += 1
#                 if string_pos >= len(string_list[string_idx]):
#                     cross_count = 1
#                     prefix = string_list[string_idx]
#                     pre_length = len(string_list[string_idx])
#                     while cross_count <= max_cross_count and string_idx + cross_count < len(string_list):
#                         prefix += string_list[string_idx + cross_count]
#                         new_string_pos = 0
#                         while new_string_pos + len(original_token) <= len(prefix) and new_string_pos < len(
#                                 string_list[string_idx]):
#                             if prefix[new_string_pos: new_string_pos + len(
#                                     original_token)] == original_token and new_string_pos + len(
#                                 original_token) > len(
#                                 string_list[string_idx]):
#                                 string_pos = new_string_pos + len(original_token) - pre_length
#                                 flag = True
#                                 break
#                             new_string_pos += 1
#                         if flag:
#                             break
#                         pre_length += len(string_list[string_idx + cross_count])
#                         cross_count += 1
#                     if flag:
#                         for delta_idx in range(cross_count + 1):
#                             token_string_idx_list.append((token_idx, string_idx + delta_idx))
#                         string_idx += cross_count
#                         if string_idx < len(string_list) and string_pos == len(string_list[string_idx]):
#                             string_pos = 0
#                             string_idx += 1
#                         break
#                     else:
#                         string_pos = 0
#                         string_idx += 1
#             if flag:
#                 continue
#             if string_idx < len(string_list) and string_pos == len(string_list[string_idx]):
#                 string_pos = 0
#                 string_idx += 1
#             if string_idx >= len(string_list):
#                 continue
#             token_string_idx_list.append((token_idx, string_idx))
#             string_pos += len(original_token)
#
#         # for token_idx, string_idx in token_string_idx_list:
#         #     print(inputs.tokens()[token_idx], string_list[string_idx])
#
#         string_token_mapping = [(float('inf'), 0)] * len(string_list)
#         for token_idx, string_idx in token_string_idx_list:
#             string_token_mapping[string_idx] = (
#                 min(string_token_mapping[string_idx][0], token_idx),
#                 max(string_token_mapping[string_idx][1], token_idx))
#
#         return string_token_mapping
#
#     @staticmethod
#     def get_word_familiar_rate(word_text):
#         capital_word = word_text.upper()
#         return ReadingArticle.word_fam_map.get(capital_word, 0)
#
#     def _generate_word_difficulty(self):
#         word_difficulties = []
#         for token in self._spacy_doc:
#             if token.is_alpha:  # and not token.is_stop:
#                 fam = self.get_word_familiar_rate(token.text)
#                 if fam == 0:
#                     fam = self.get_word_familiar_rate(token.lemma_)
#                 syllable = textstat.syllable_count(token.text)
#                 length = len(token.text)
#                 ent_flag = token.ent_iob != 2
#                 stop_flag = token.is_stop
#                 difficulty_feat = {
#                     # float(textstat.syllable_count(token.text) > 2),
#                     # float(len(token.text) > 7),
#                     # float(fam < 482),
#                     'syllable': float(syllable),
#                     'length': float(length),
#                     'fam': float(fam),
#                     'ent_flag': float(ent_flag),
#                 }
#                 for column in difficulty_feat:
#                     ReadingArticle.all_values[column].append(difficulty_feat[column])
#             else:
#                 difficulty_feat = {
#                     'syllable': 0.,
#                     'length': 0.,
#                     'fam': 645.,
#                     'ent_flag': 0.,
#                 }
#             word_difficulties.append(difficulty_feat)
#         return word_difficulties
#
#     def _generate_topic_feature(self, top_n):
#         kw_extractor.load_document(input=self.text, language='en')
#         kw_extractor.candidate_selection(1)
#         kw_extractor.candidate_weighting()
#         keywords = kw_extractor.get_n_best(n=top_n)
#         keywords = {k: min(max(v, 0), 1) for k, v in keywords}
#
#         topic_feature = []
#         for token in self._spacy_doc:
#             doc_level_score = keywords.get(token.text, 1.)
#             topic_feature.append({
#                 'topic_score': doc_level_score
#             })
#             ReadingArticle.all_values['topic_score'].append(doc_level_score)
#         return topic_feature
#
#     def get_word_filter_id_set(self, only_alpha=True, filter_digit=True, filter_punctuation=True,
#                                filter_stop_words=False):
#         word_filter_id_set = set()
#         for word in self._spacy_doc:
#             if only_alpha and not word.is_alpha:
#                 word_filter_id_set.add(word.i)
#             if filter_digit and word.is_digit:
#                 word_filter_id_set.add(word.i)
#             if filter_punctuation and word.is_punct:
#                 word_filter_id_set.add(word.i)
#             if filter_stop_words and word.is_stop:
#                 word_filter_id_set.add(word.i)
#         return word_filter_id_set
#
#     def get_word_list(self):
#         return self._word_list
#
#     def get_sentence_list(self):
#         return self._sentence_list
#
#     def get_transformer_features(self):
#         return self._transformer_features
#
#     def get_difficulty_features(self):
#         rst = []
#         if not ReadingArticle.mean_values or not ReadingArticle.std_values:
#             for column, values in ReadingArticle.all_values.items():
#                 ReadingArticle.mean_values[column] = np.mean(values)
#                 ReadingArticle.std_values[column] = np.std(values)
#         for difficulty_feature in self._difficulty_features:
#             rst.append(torch.tensor(
#                 [self.standardization(value, column) for column, value in difficulty_feature.items()],
#                 dtype=torch.float32)
#             )
#         return rst
#
#     def get_topic_features(self):
#         rst = []
#         if not ReadingArticle.mean_values or not ReadingArticle.std_values:
#             for column, values in ReadingArticle.all_values.items():
#                 ReadingArticle.mean_values[column] = np.mean(values)
#                 ReadingArticle.std_values[column] = np.std(values)
#         for topic_feature in self._topic_features:
#             rst.append(
#                 torch.tensor([self.standardization(value, column) for column, value in topic_feature.items()],
#                              dtype=torch.float32))
#         return rst
#
#     def get_original_features(self):
#         rst = []
#         for difficulty_feat, topic_feat in zip(self._difficulty_features, self._topic_features):
#             line = {}
#             for column, value in difficulty_feat.items():
#                 line[column] = value
#             for column, value in topic_feat.items():
#                 line[column] = value
#             rst.append(line)
#         return rst
#
#     def get_sentence_word_mapping(self):
#         return self._sentence_word_mapping


# class SentFeature:
#
#     def __init__(self, num, sent_list, word_list):
#         super().__init__()
#         self.num = num
#         self.sent_list = sent_list
#         self.word_list = word_list
#
#         self.backward_times_of_sentence = [0 for _ in range(num)]
#         self.forward_times_of_sentence = [0 for _ in range(num)]
#         # self.horizontal_saccade_proportion = [0 for _ in range(num)]
#         self.saccade_duration = [0 for _ in range(num)]
#         self.saccade_times_of_sentence = [0 for _ in range(num)]
#         self.saccade_velocity = [0 for _ in range(num)]
#         self.total_dwell_time_of_sentence = [0 for _ in range(num)]
#
#         self.saccade_distance = [0 for _ in range(num)]
#
#         # self.horizontal_saccade = [0 for _ in range(num)]
#
#         self.backward_times_of_sentence_div_syllable = [0 for _ in range(num)]
#         self.forward_times_of_sentence_div_syllable = [0 for _ in range(num)]
#         self.horizontal_saccade_proportion_div_syllable = [0 for _ in range(num)]
#         self.saccade_duartion_div_syllable = [0 for _ in range(num)]
#         self.saccade_times_of_sentence_div_syllable = [0 for _ in range(num)]
#         self.saccade_velocity_div_syllable = [0 for _ in range(num)]
#         self.total_dwell_time_of_sentence_div_syllable = [0 for _ in range(num)]
#
#     def update(self):
#         self.backward_times_of_sentence_div_syllable = self.div_syllable(self.backward_times_of_sentence)
#         self.forward_times_of_sentence_div_syllable = self.div_syllable(self.forward_times_of_sentence)
#
#         self.horizontal_saccade_proportion = self.get_list_div(self.horizontal_saccade, self.saccade_times_of_sentence)
#         self.horizontal_saccade_proportion_div_syllable = self.div_syllable(self.horizontal_saccade_proportion)
#
#         self.saccade_duration_div_syllable = self.div_syllable(self.saccade_duration)
#         self.saccade_times_of_sentence_div_syllable = self.div_syllable(self.saccade_times_of_sentence)
#
#         self.saccade_velocity = self.get_list_div(self.saccade_distance, self.saccade_duration)
#         self.saccade_velocity_div_syllable = self.div_syllable(self.saccade_velocity)
#
#         self.total_dwell_time_of_sentence_div_syllable = self.div_syllable(self.total_dwell_time_of_sentence)
#
#     def norm(self, list1):
#         results = []
#         mean = np.mean(list1)
#         var = np.var(list1)
#         for item in list1:
#             if var != 0:
#                 a = (item - mean) / var
#             else:
#                 a = 0
#             results.append(a)
#         return results
#
#     def get_list_div(self, list_a, list_b):
#         div_list = [0 for _ in range(self.num)]
#         for i in range(len(list_b)):
#             if list_b[i] != 0:
#                 div_list[i] = list_a[i] / list_b[i]
#
#         return div_list
#
#     def div_syllable(self, feature):
#         assert len(feature) == len(self.sent_list)
#         results = []
#         for i in range(len(feature)):
#             sent = self.sent_list[i]
#             syllable_len = self.get_syllable(self.word_list[sent[1]:sent[2]])
#             if syllable_len > 0:
#                 results.append(feature[i] / syllable_len)
#             else:
#                 results.append(0)
#         return results
#
#     def get_syllable(self, word_list):
#         syllable_len = 0
#         for word in word_list:
#             syllable_len += textstat.syllable_count(word)
#         return syllable_len
#
#     def to_dataframe(self):
#         data = pd.DataFrame({
#             'backward_times_of_sentence_div_syllable': self.norm(self.backward_times_of_sentence_div_syllable),
#             'forward_times_of_sentence_div_syllable': self.norm(self.forward_times_of_sentence_div_syllable),
#             'horizontal_saccade_proportion_div_syllable': self.norm(self.horizontal_saccade_proportion_div_syllable),
#             'saccade_duration_div_syllable': self.norm(self.saccade_duration_div_syllable),
#             'saccade_times_of_sentence_div_syllable': self.norm(self.saccade_times_of_sentence_div_syllable),
#             'saccade_velocity_div_syllable': self.norm(self.saccade_velocity_div_syllable),
#             'total_dwell_time_of_sentence_div_syllable': self.norm(self.total_dwell_time_of_sentence_div_syllable)
#         })
#         print(data)
#         return data