# Download 2003 CoNLL NER task dataset
wget -nc -P ./data/ http://fprodrigues.com//deep_ner-mturk.tar.gz
tar -zxvf ./data/deep_ner-mturk.tar.gz -C ./data/

# Download glove.6B
# https://nlp.stanford.edu/projects/glove/
wget -nc -P ./data/ https://nlp.stanford.edu/data/glove.6B.zip

# unzip only glove.6B.300d.txt
unzip -u ./data/glove.6B.zip -d ./data/ -x glove.6B.50d.txt glove.6B.100d.txt glove.6B.200d.txt
