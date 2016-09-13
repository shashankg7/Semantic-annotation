#D1 : Evaluate with papers dataset
import sys
from gensim.models import Doc2Vec
from gensim.corpora.dictionary import Dictionary
from gensim import utils
from read_wiki import stream
import pdb
import json
#read in papers dataset
file=sys.argv[1];

wiki_path = './enwiki-latest-pages-articles.xml.bz2'

#Read the keyword and abstract
uniqueFields = {}
keyAbs=[]
with open(file) as infile:
	field = None;
	abstract = None;
	for line in infile:
		#if len(line)==1 :
		if line in ['\n', '\r\n']:
			if abstract != None:
				keyAbs.append([field,abstract])
				#print abstract
				#print field
			field = None			
			abstract = None
		else:
			#if line[1]=='*' : # field is title
				#field = line.replace("#*","")
			if line[1]=='f' : #field is keyword
				field = line.replace("#f","")
				field = field.replace("\n","")
				field = field.replace("\r","")
				field = field.replace("_"," ")
				field = field.capitalize()
				#print field
				uniqueFields[field]=0
			if line[1]=='!' :
				abstract = line.replace("#!","")
print 'Citations loaded = '+str(len(keyAbs))
keys = map(lambda x:x[0], keyAbs)
keys = set(keys)
n_cats = len(keys)
# avg length of each article's abstract
avg_len = map(lambda x:len(x[1].split()), keyAbs)
avg_len1 = sum(avg_len)/float(len(avg_len))

#print 'Unique fields = '+str(uniqueFields.keys())

#Load model
#model = Doc2Vec.load('./data/wiki_model2.doc2vec')#model_loaded = Doc2Vec.load_word2vec_format('/tmp/my_model.doc2vec')
#print 'Words in loaded model = '+str(len(model.vocab))
#print 'Entities in loaded model = '+str(len(model.docvecs))

#wikiKeyTitleMap = {}
#with utils.smart_open("./wikiTitleKeyMapFull.txt") as fin:
#    for line in fin:
#        if line.endswith('\n'):
#        	line = line.replace("\n","")
#    	#line = line.replace("\n","")
#        words = utils.to_unicode(line).split(' ',1);
#        wikiKeyTitleMap[words[0]] = words[1]

#wiki_file = open('wikiIdTitleMap.json','r')
#wikiKeyTitleMap = json.load(wiki_file)

#map Citation field name to Wiki article name for string inconsistencies
mapFields={} 
mapFields['Programming languages'] = ['Programming language']; mapFields['Real time and embedded systems'] = ['Modeling and Analysis of Real Time and Embedded systems'];
mapFields['Scientific computing'] = ['Computational science']; mapFields['Natural language and speech'] = ['Natural language processing','Speech recognition'];
mapFields['Machine learning and pattern recognition'] = ['Machine learning','Pattern recognition']; mapFields['Operating systems'] = ['Operating system']; 
mapFields['World wide web'] = ['World Wide Web'];  mapFields['Bioinformatics and computational biology'] = ['Bioinformatics','Computational biology'];
mapFields['Security and privacy']=['Information security', 'Internet privacy']; mapFields['Distributed and parallel computing'] = ['Distributed computing','Parallel computing'];
mapFields['Databases'] = ['Database'];  mapFields['Simulation'] = ['Computer simulation']; 
mapFields['Algorithms and theory'] = ['Algorithm', 'Theoretical computer science']; mapFields['Computer education']=['Computer literacy'];
mapFields['Human-computer interaction']= []; #REVISIT ['Human-computer interaction' is missing in wiki_model2.doc2vec
mapFields['Hardware and architecture']=['Hardware architecture'];
mapFields['Networks and communications'] = ['Computer network', 'Telecommunications engineering']
mapFields['Artificial intelligence'] = ['Artificial intelligence']; mapFields['Data mining'] = ['Data mining'];
mapFields['Computer vision'] = ['Computer vision']; mapFields['Simulation'] = ['Simulation'] ; mapFields['Software engineering'] = ['Software engineering']; 
mapFields['Information retrieval']=['Information retrieval']; mapFields['Multimedia'] = ['Multimedia']; mapFields['Graphics'] = ['Graphics']
#print 'Mappings exist for '+str(len(mapFields.keys()))

print("Scaning wikipedia pages for categories")

titles = mapFields.keys()
titles1 = mapFields.values()
titles1 = sum(titles1, [])
titles.extend(titles1)
Titles = set(titles)
n_cats = len(Titles)
pdb.set_trace()
i = 0
# Get wiki page corresponding to categories
CategoriesWikiPage = {}
for title, tokens in stream:
	if i == n_cats:
		print("All category's related articles found, breaking the loop (reduce unncesaary time searching whole wiki dump")
		break
	for field in uniqueFields.keys():
		keyFields = mapFields[field]
		for keyField in keyFields:
			keyField = keyField.encode('utf-8')
			if keyField == title:
				i += 1
				print("Title found matching paper's category, : %s %s"%(title, keyField))
				CategoriesWikiPage[keyField] = tokens	

#or title, text in stream:
#if title in keys:
#	print("%d th category found in wikipedia dump"%(i))
#	i += 1
#	CategoriesWikiPage[title] = text
			
print("Wikipedia articles correspoding to each category found")
print("Dumping it to disk")
f = open('wikiCatArticlesFull.json','w')
json.dump(CategoriesWikiPage, f)
print("Dumped to disk")
pdb.set_trace()
modelledKeys = {}
#dic = Dictionary.load('wiki_dict.dict')

for field in uniqueFields.keys():
	keyFields = mapFields[field]
	for keyField in keyFields:
		keyField = keyField.encode('utf-8');
		for k,v in wikiKeyTitleMap.iteritems():
			if v==keyField:
				print(k,v)
				modelledKeys[v] = k;
				break;			


pdb.set_trace()

'''	       	
modelledEntities = []
for key in modelledKeys.keys():
    try:
        if(model.docvecs[key] != None):
            modelledEntities.append(modelledKeys.get(key));
    except KeyError, e:
        #TODO  Dictionary and model should have bigrams .
        print (' KeyError - reason "%s"' % str(e))
    except:
        raise
'''

print 'Unique Keywords = '+str(len(uniqueFields))+', Mapped Keywords = '+str(len(modelledKeys));#, modelledKeys)

citationCount = 0
for citation in keyAbs:
	print str(citationCount)+'. Keyword = '+citation[0]
	
	testVec = model.infer_vector(citation[1].split(), alpha=0.1, min_alpha=0.0001, steps=5)
	#print 'Abstract = '+citation[1]
	sims = model.docvecs.most_similar([testVec])
	#print(sims)

	goldKeys = []; silverKeys = [];
	field = citation[0].replace("\n","")
	field = field.replace("\r","")
	field = field.replace("_"," ")
	field = field.capitalize()
	mappedKeywords = mapFields.get(field)
	#print (len(mappedKeywords) , mappedKeywords)
	for mapKey in mappedKeywords: #	print( modelledKeys[mapKey].encode('utf-8') )
		goldKeys.append(modelledKeys[mapKey].encode('utf-8'))
		goldVec = model.docvecs[modelledKeys[mapKey].encode('utf-8')]
		#print(' goldvec size =',len(goldVec))
		silverVecs = model.docvecs.most_similar([goldVec])
		for silverVec in silverVecs:
			silverKeys.append(silverVec[0].encode('utf-8'))
	print(len(goldKeys), goldKeys)
	print(len(silverKeys), silverKeys)
	
	for simRec in sims:
		print(wikiKeyTitleMap.get(simRec[0]),simRec[1])#print(simRec)
		if simRec[0] in goldKeys:
			print 'GOLDEN HURRAY'
		if simRec[0] in silverKeys:
			print 'SILVER HURRAY'
	else:
		print str(citationCount)+' == No sim docVec. '
	citationCount = citationCount + 1
#d_result = model.docvecs.most_similar('bmw')
#print(d_result)
