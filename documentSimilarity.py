
'''
implements doc2vec over the existing sklearn method
'''sentence = LabeledSentence(words=[u'some', u'words', u'here'], labels=[u'SENT_1'])

class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for uid, line in enumerate(open(filename)):
            yield LabeledSentence(words=line.split(), labels=['SENT_%s' % uid])

model = Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
model.build_vocab(sentences)
for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay

def word2vec_similarity(documents, document_ids):
    sentence_vectors = list(map(word2vec.tovector, documents))

    output_list = []
    
    for idx_root, root_vector in enumerate(sentence_vectors):
        print("Calculating similarities of:", str(idx_root))
        for idx_tmp, tmp_vector in enumerate(sentence_vectors):
            if idx_root <= idx_tmp:
                try:
                    similarity = 1 - scipy.spatial.distance.cosine(root_vector, tmp_vector)
                except:
                    similarity = 0.0
                output_list.append([document_ids[idx_root], document_ids[idx_tmp], similarity])
            
    return output_list

model = Doc2Vec(sentences)
...
# store the model to mmap-able files
model.save('/tmp/my_model.doc2vec')
# load the model back
model_loaded = Doc2Vec.load('/tmp/my_model.doc2vec')

print model.most_similar(&quot;SENT_0&quot;)
[('SENT_48859', 0.2516525387763977),
 (u'paradox', 0.24025458097457886),
 (u'methodically', 0.2379375547170639),
 (u'tongued', 0.22196565568447113),
 (u'cosmetics', 0.21332012116909027),
 (u'Loos', 0.2114654779434204),
 (u'backstory', 0.2113303393125534),
 ('SENT_60862', 0.21070502698421478),
 (u'gobble', 0.20925869047641754),
 ('SENT_73365', 0.20847654342651367)]