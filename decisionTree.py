import pandas as pd 
import numpy as np 
import math

# داده های آموزش را مدیریت کنید تا بتوان یک بار بارگیری کرد و از آنجا به آنها مراجعه کرد
# تغییراتی را در تعداد ویژگی های مورد استفاده در داده های آموزشی ایجاد کنید
#  داده های اصلی را بخوانید و جدول زیر مجموعه را با ستون دریافت کنید: اسامی ستون ها
## Is_Home_or_Away
## Is_Opponent_in_AP25_Preseason
## Label
from pip._vendor.distlib.compat import raw_input

DF_TRAIN = pd.read_csv('Dataset-football-train.txt',sep='\t')
DF_TRAIN = DF_TRAIN[['Is_Home_or_Away','Is_Opponent_in_AP25_Preseason','Media','Label']]


class Tree:
	def __init__(self,observationIDs,features,currLvl=0,subTree={},bestFeature=None,majorityLabel=None,parentMajorityLabel=None):
		self.observationIDs = observationIDs
		self.features = features
		self.currLvl = currLvl
		self.subTree = subTree
		self.bestFeature = bestFeature
		self.majorityLabel = majorityLabel
		self.parentMajorityLabel = parentMajorityLabel
		self.setBestFeatureID(bestFeature)

	# predicts using a tree and 
	# observation: [Is_Home_or_Away, Is_Opponent_in_AP25_Preseason, Media]

	def setBestFeatureID(self, feature):
		idx = None
		if feature == 'Is_Home_or_Away':
			idx = 0
		elif feature == 'Is_Opponent_in_AP25_Preseason':
			idx = 1
		else:
			idx = 2
		self.bestFeatureID = int(idx)

def predict(tree, obs):
	if tree.bestFeature == None:
		return tree.majorityLabel
	featVal = obs[tree.bestFeatureID]
	if not featVal in tree.subTree: # val with no subtree
		return tree.majorityLabel
	else: # recurse on subtree
		return predict(tree.subTree[featVal],obs)

def displayDecisionTree(tree):
	print('\t'*tree.currLvl + '(lvl {}) {}'.format(tree.currLvl,tree.majorityLabel))
	if tree.bestFeature == None:
		return

	print('\t'*tree.currLvl + '{}'.format(tree.bestFeature) + ': ')
	for [val,subTree] in sorted(tree.subTree.items()):
		print('\t'*(tree.currLvl+1) + 'choice: {}'.format(val))
		displayDecisionTree(subTree)

def Entropy(ns):
	entropy = 0.0
	total = sum(ns)
	for x in ns:
		entropy += -1.0*x/total*math.log(1.0*x/total,2)
	return entropy

# Information Gain
def IG(observationIDs, feature):

	df = DF_TRAIN.loc[observationIDs]
	# تعداد برنده ها / باخت ها را برای هر دسته از ویژگی ها جمع کنید
	labelCountDict = {}
	valueLabelCountDict = {}
	for index, row in df.iterrows():
		label = row['Label']
		if not label in labelCountDict:
			labelCountDict[label] = 0
		labelCountDict[label] += 1
		featureValue = row[feature]
		if not featureValue in valueLabelCountDict:
			valueLabelCountDict[featureValue] = {}
		if not label in valueLabelCountDict[featureValue]:
			valueLabelCountDict[featureValue][label] = 0
		valueLabelCountDict[featureValue][label] += 1

	ns = []
	for [label,count] in labelCountDict.items():
		ns.append(count)

	H_Y = Entropy(ns)

	H_Y_X = 0.0
	for [featureValue, labelCountDict] in valueLabelCountDict.items():
		nsHYX = []
		for [label,count] in labelCountDict.items():
			nsHYX.append(count)
		H_Y_X += 1.0*sum(nsHYX)/len(df)*Entropy(nsHYX)
	return H_Y - H_Y_X

def GR(observationIDs, feature):
	ig = IG(observationIDs,feature)
	if ig == 0:
		return 0
	df = DF_TRAIN.loc[observationIDs]
	valueLabelDict = {}
	for index, row in df.iterrows():
		label = row['Label']
		featureValue = row[feature]
		if featureValue not in valueLabelDict:
			valueLabelDict[featureValue] = 0
		valueLabelDict[featureValue] += 1
	ns = []
	for [val,count] in valueLabelDict.items():
		ns.append(count)
	ent = Entropy(ns)
	return float(ig)/ent

def fillDecisionTree(tree,decisionTreeAlgo):
	# بیشترین برچسب را پیدا کنید
	df = DF_TRAIN.loc[tree.observationIDs] # smaller df
	counts = df['Label'].value_counts()
	majorityLabel = df['Label'].value_counts().idxmax()
	if len(counts) > 1:
		if counts['Win'] == counts['Lose']:
			majorityLabel = tree.parentMajorityLabel
	tree.majorityLabel = majorityLabel

	# پایان اگر فقط یک LABELوجود داتشه باشد
	if len(counts) == 1:
		return
	# پایان اگر ویژگی صفر باشد
	if len(tree.features) == 0: 
		return

	# بهترین ویژگی را پیدا کنید
	featureValueDict = {}
	for feature in tree.features: 
		if decisionTreeAlgo == 'ID3':
			metricScore = IG(tree.observationIDs,feature)
		if decisionTreeAlgo == 'C45':
			metricScore = GR(tree.observationIDs,feature)
		featureValueDict[feature] = metricScore
	bestFeature, bestFeatureValue = sorted(featureValueDict.items(),reverse=True)[0]
	# پایان اگر هر کدام از درخت ها بهترین ویژگیشان صفر باشد
	if bestFeatureValue == 0.0:
		return
	tree.bestFeature = bestFeature

	# زیر مجموعه ویژگی ها را پیدا کنید
	subFeatures = set()
	for feature in tree.features:
		if feature == bestFeature: # skip the current best feature
			continue
		subFeatures.add(feature)
	
	# بهترین شناسه ویژگی را پیدا کنید
	bestFeatureIdx = 0
	if bestFeature == 'Is_Home_or_Away':
		bestFeatureIdx = 0
	elif bestFeature == 'Is_Opponent_in_AP25_Preseason':
		bestFeatureIdx = 1
	else:
		bestFeatureIdx = 2
	
	# زیر مجموعه مشاهدات را پیدا کنید
	subObservationsDict = {}
	for obs in tree.observationIDs:
		val = DF_TRAIN.values[obs][bestFeatureIdx]
		if not val in subObservationsDict:
			subObservationsDict[val] = set()
		subObservationsDict[val].add(obs)

	for [val,obs] in subObservationsDict.items():

		tree.subTree[val] = Tree(obs, subFeatures, tree.currLvl + 1,{},None,None,majorityLabel)
		
		fillDecisionTree(tree.subTree[val],decisionTreeAlgo)

def predictAndAnalyze(tree, data):
	TP = 0
	FN = 0
	FP = 0
	TN = 0
	for obs in data:
		prediction = predict(tree,obs)
		ground = obs[3]
		if prediction == 'Win' and ground == 'Win':
			TP += 1
		if prediction == 'Win' and ground == 'Lose':
			FP += 1
		if prediction == 'Lose' and ground == 'Win':
			FN += 1
		if prediction == 'Lose' and ground == 'Lose':
			TN += 1

	accuracy = float(TP+TN)/len(data)
	precision = float(TP)/(TP + FP)
	recall = float(TP)/(TP + FN)
	F1 = 2*(recall*precision)/(recall+precision)
	print('\nانالیز:')
	print('دقت = {}'.format(accuracy))
	print('دقت، درستی = {}'.format(precision))
	print('recall = {}'.format(recall))
	print('F1 score = {}'.format(F1))


#  در داده های اصلی بخوانید و جدول زیر مجموعه ها را با ستون دریافت کنید:اسامی صفت ها
## Is_Home_or_Away
## Is_Opponent_in_AP25_Preseason
## Label
dfTest = pd.read_csv('Dataset-football-test.txt',sep='\t')
dfTest = dfTest[['Is_Home_or_Away','Is_Opponent_in_AP25_Preseason','Media','Label']]


initialObservationIDs = set(range(len(DF_TRAIN)))
initialFeatures = set(dfTest.columns.values[:-1])

print("این پروژه احتمال برد و باخت یک تیم را بر اساس ویژگی های رسانه و خارج از خانه بودن یا نبودن تیم با کمک INFORMATION GAIN حساب میکند")
print("شما می خواهید از کدام الگوریتم درخت تصمیم استفاده کنید ('ID3' or 'C45)?")
algoChoice = str(raw_input())
if algoChoice not in {'ID3','C45'}:
	print("انتخاب الگوریتم نامعتبر است. شما باید انتخاب کنید 'ID3' or 'C45'")
	exit()

print("choice: {}".format(algoChoice))

MyTree = Tree(initialObservationIDs,initialFeatures)
fillDecisionTree(MyTree,algoChoice)

print('درخت تصمیم من:')
displayDecisionTree(MyTree)


print('برچسب های پیش بینی شده از داده های آزمون:')
predictAndAnalyze(MyTree,dfTest.values)

