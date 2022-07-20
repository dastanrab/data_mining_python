import numpy as np
from sklearn.metrics import precision_recall_fscore_support
#این پروژه شامل محاسبه 3 روش برای انتخاب معیار بود که من فقط معیار gini index توضیح دادم
def column(matrix, i):
    return [row[i] for row in matrix]


def IG(D, index, value):


    colXofData = column(D[0], index)
    columnofNums = D[1]
    classesZero = []
    classesOne = []
    dataZero = []
    dataOne = []

    OriginaltrueCount = 0
    OriginalfalseCount = 0
    for x in range(0, len(D[1])):
        if (D[1][x] == 0):
            OriginaltrueCount += 1
        elif (D[1][x] == 1):
            OriginalfalseCount += 1

    op1 = OriginaltrueCount / (OriginaltrueCount + OriginalfalseCount)
    op2 = OriginalfalseCount / (OriginaltrueCount + OriginalfalseCount)
    XY = - (op1 * np.log2(op1) + ((op2) * np.log2((op2))))
    print("این انتروپی کل مجموعه داده است: {}".format(XY))


    for x in range(0, len(columnofNums)):
        if(colXofData[x] >= value):
            dataZero.append(colXofData[x])
            classesZero.append(0)
        elif(colXofData[x] < value):
            dataOne.append(colXofData[x])
            classesOne.append(1)


    trueCount = len(dataZero)
    falseCount = len(dataOne)
    datasetNP = np.array(colXofData)
    classesNP = np.array(columnofNums)
    Yes = 0
    No = 0
    Yes = len(classesZero)
    No = len(classesOne)


    p1 = trueCount / (trueCount + falseCount)
    p2 = falseCount / (trueCount + falseCount)


    DYDN = - (((Yes)/(Yes+No))*p1*np.log2(p1) + ((No)/(Yes+No)) * p2*np.log2(p2))



    informationGain = XY - DYDN
    print("این مقدار information gain است: {}".format(informationGain))
    return informationGain


def G(D, index, value):
    """شاخص gini یک تقسیم بر شاخص ویژگی را با مقدار محاسبه کنید
     برای مجموعه داده D.

     آرگس:
         D: یک مجموعه داده ، tuple (X ، y) که X داده است ، y کلاس ها
         index: شاخص ویژگی (ستون X) برای تقسیم
         value: مقدار ویژگی در شاخص برای تقسیم در

     بازده:
         مقدار شاخص جینی برای تقسیم داده شده
    """

    trueCount = 0
    falseCount = 0
    for x in range(0, len(D[1])):
        if (D[1][x] == 0):
            trueCount += 1
        elif (D[1][x] == 1):
            falseCount += 1

    p1 = trueCount / (trueCount + falseCount)
    p2 = falseCount / (trueCount + falseCount)

    colXofData = column(D[0], index)
    columnofNums = D[1]
    classesZero = []
    classesOne = []
    dataZero = []
    dataOne = []

    for x in range(0, len(columnofNums)):
        if(colXofData[x] >= value):
            dataZero.append(colXofData[x])
            classesZero.append(0)
        elif(colXofData[x] < value):
            dataOne.append(colXofData[x])
            classesOne.append(1)

    trueCount = len(dataZero)
    falseCount = len(dataOne)

    Yes = 0
    No = 0
    Yes = len(classesZero)
    No = len(classesOne)
    yesPercent = Yes/(Yes+No)
    noPercent = No/(Yes+No)

    GiniDY = (yesPercent)*(1 - (yesPercent*yesPercent))
    GiniDN = (noPercent)*(1 - (noPercent*noPercent))
    GiniIndex = GiniDY + GiniDN
    print("این مقدار gini index است: {}".format(GiniIndex))
    return GiniIndex

def CART(D, index, value):

    trueCount = 0
    falseCount = 0
    for x in range(0, len(D[1])):
        if (D[1][x] == 0):
            trueCount += 1
        elif (D[1][x] == 1):
            falseCount += 1

    p1 = trueCount / (trueCount + falseCount)
    p2 = falseCount / (trueCount + falseCount)

    colXofData = column(D[0], index)
    columnofNums = D[1]
    classesZero = []
    classesOne = []
    dataZero = []
    dataOne = []

    for x in range(0, len(columnofNums)):
        if (colXofData[x] >= value):
            dataZero.append(colXofData[x])
            classesZero.append(0)
        elif (colXofData[x] < value):
            dataOne.append(colXofData[x])
            classesOne.append(1)

    trueCount = len(dataZero)
    falseCount = len(dataOne)
    Yes = 0
    No = 0

    Yes = len(classesZero)
    No = len(classesOne)
    yesPercent = Yes / (Yes + No)
    noPercent = No / (Yes + No)

    CARTMeasure = 2*(yesPercent*noPercent)*((abs(p1 - p2) + (abs(p1-p2))))
    print("This is the CART: {}".format(CARTMeasure))
    return CART
def bestSplit(D, criterion):
    """با استفاده از معیار مشخص شده بهترین تقسیم را برای مجموعه داده D محاسبه می کند

     آرگس:
         D: یک مجموعه داده ، tuple (X ، y) که X داده است ، y کلاس ها
         معیار: یکی از "IG" ، "GINI" ، "CART"

     بازده:
         یک تاپل (i ، مقدار) که i شاخص ویژگی برای تقسیم در مقدار است
    """


    if(criterion == "IG"):
        infoGainList = []
        indexlist = []
        valuelist = []
        columnofNums = D[1]

        for index in range(0, 10):
            colXofData = column(D[0], index)
            classesZero = []
            classesOne = []
            dataZero = []
            dataOne = []
            OriginaltrueCount = 0
            OriginalfalseCount = 0
            for x in range(0, len(D[1])):
                if (D[1][x] == 0):
                    OriginaltrueCount += 1
                elif (D[1][x] == 1):
                    OriginalfalseCount += 1
            for value in range(int(min(colXofData)), int(max(colXofData))):
                op1 = OriginaltrueCount / (OriginaltrueCount + OriginalfalseCount)
                op2 = OriginalfalseCount / (OriginaltrueCount + OriginalfalseCount)
                XY = - (op1 * np.log2(op1) + ((op2) * np.log2((op2))))

                for x in range(0, len(colXofData)):
                    if (colXofData[x] >= value):
                        dataZero.append(colXofData[x])
                        classesZero.append(0)
                    elif (colXofData[x] < value):
                        dataOne.append(colXofData[x])
                        classesOne.append(1)

                trueCount = len(dataZero)
                falseCount = len(dataOne)
                Yes = 0
                No = 0
                Yes = len(classesZero)
                No = len(classesOne)
                if (Yes > 0 and No > 0):

                    p1 = trueCount / (trueCount + falseCount)
                    p2 = falseCount / (trueCount + falseCount)

                    # Split-Entropy را برای مجموعه داده محاسبه کنید
                    DYDN = - (((Yes) / (Yes + No)) * p1 * np.log2(p1) + ((No) / (Yes + No)) * p2 * np.log2(p2))


                    # قسمت محاسبه information gain
                    informationGain = XY - DYDN
                    infoGainList.append(informationGain)
                    indexlist.append(index)
                    valuelist.append(value)
                    #print("This is the Information Gain: {}".format(informationGain))
                    #print("This is the index: {}".format(index))
                    #print("This is the value: {}".format(value))
        infoGain = infoGainList.index(max(infoGainList))
        #print(infoGainList[infoGain])
        i = indexlist[infoGain]
        value = valuelist[infoGain]
        return (i, value)


    if(criterion == "GINI"):
        trueCount = 0
        falseCount = 0
        giniList = []
        indexlist = []
        valuelist = []
        for x in range(0, len(D[1])):
            if (D[1][x] == 0):
                trueCount += 1
            elif (D[1][x] == 1):
                falseCount += 1
            p1 = trueCount / (trueCount + falseCount)
            p2 = falseCount / (trueCount + falseCount)

        for index in range(0, 10):
            colXofData = column(D[0], index)
            columnofNums = D[1]
            classesZero = []
            classesOne = []
            dataZero = []
            dataOne = []

            for value in range(int(min(colXofData)), int(max(colXofData))):
                if (colXofData[x] >= value):
                    dataZero.append(colXofData[x])
                    classesZero.append(0)
                elif (colXofData[x] < value):
                    dataOne.append(colXofData[x])
                    classesOne.append(1)

                trueCount = len(dataZero)
                falseCount = len(dataOne)
                Yes = 0
                No = 0
                Yes = len(classesZero)
                No = len(classesOne)
                if Yes > 0 and No > 0:
                    yesPercent = Yes / (Yes + No)
                    noPercent = No / (Yes + No)

                    GiniDY = (yesPercent) * (1 - (yesPercent * yesPercent))
                    GiniDN = (noPercent) * (1 - (noPercent * noPercent))
                    GiniIndex = GiniDY + GiniDN
                    giniList.append(GiniIndex)
                    indexlist.append(index)
                    valuelist.append(value)
                    #print("This is the Gini Index: {}".format(GiniIndex))
                    #print("This is the index: {}".format(index))
                    #print("This is the value: {}".format(value))
        Gini = giniList.index(min(giniList))
        i = indexlist[Gini]
        value = valuelist[Gini]
        return (i, value)

    if(criterion == "CART"):
        trueCount = 0
        falseCount = 0
        CARTList = []
        indexlist = []
        valuelist = []
        for x in range(0, len(D[1])):
            if (D[1][x] == 0):
                trueCount += 1
            elif (D[1][x] == 1):
                falseCount += 1

        p1 = trueCount / (trueCount + falseCount)
        p2 = falseCount / (trueCount + falseCount)
        for index in range(0, 10):
            colXofData = column(D[0], index)
            columnofNums = D[1]
            classesZero = []
            classesOne = []
            dataZero = []
            dataOne = []

            for value in range(int(min(colXofData)), int(max(colXofData))):
                if (colXofData[x] >= value):
                    dataZero.append(colXofData[x])
                    classesZero.append(0)
                elif (colXofData[x] < value):
                    dataOne.append(colXofData[x])
                    classesOne.append(1)
                trueCount = len(dataZero)
                falseCount = len(dataOne)
                truePercent = trueCount/(trueCount+falseCount)
                falsePercent = falseCount/(trueCount+falseCount)
                Yes = 0
                No = 0
                Yes = len(classesZero)
                No = len(classesOne)
                if Yes > 0 and No > 0:
                    yesPercent = Yes / (Yes + No)
                    noPercent = No / (Yes + No)
                    CARTMeasure = 2 * (truePercent * falsePercent) * ((abs(yesPercent - noPercent) + (abs(yesPercent - noPercent))))
                    #print("This is the CART: {}".format(CARTMeasure))
                    CARTList.append(CARTMeasure)
                    indexlist.append(index)
                    valuelist.append(value)
        Cartnum = CARTList.index(max(CARTList))
        #print(CARTList[Cartnum])
        i = indexlist[Cartnum]
        value = valuelist[Cartnum]
        return (i, value)

def load(filename):
    """نام فایل را به عنوان یک مجموعه داده بارگیری می کند. فرض کنید آخرین ستون کلاسها باشد ، و
     مشاهدات به صورت ردیف سازماندهی می شوند.

     آرگس:
         filename: پرونده ای برای خواندن

     بازده:
         یک تاپل D = (X ، y) ، که در آن X یک لیست یا یک تقسیم عددی از صفات مشاهده است
         که در آن X [i] از ردیف i-th در نام پرونده آمده است. y یک لیست یا تقسیم بندی از است
         کلاسهای مشاهدات ، به همان ترتیب
    """
    with open(filename) as f:
        dataset = f.readlines()
    dataset = [x.strip().split() for x in dataset]
    dataset = [[float((float(j))) for j in i] for i in dataset]
    for x in range(0, len(dataset)):
        dataset[x][0] = int(dataset[x][0])
        dataset[x][1] = int(dataset[x][1])
        dataset[x][2] = int(dataset[x][2])
        dataset[x][3] = int(dataset[x][3])
        dataset[x][4] = int(dataset[x][4])
        dataset[x][5] = int(dataset[x][5])
        #dataset[x][6] = int(dataset[x][6])
        dataset[x][7] = int(dataset[x][7])
        dataset[x][8] = int(dataset[x][8])
        dataset[x][9] = int(dataset[x][9])
        dataset[x][10] = int(dataset[x][10])

    classes = []
    for x in range(0, len(dataset)):
        classes.append(dataset[x][10])
        dataset[x] = dataset[x][:-1]
    X = (dataset, classes)
    return X

def classifyIG(train, test):

    X = bestSplit(train, "IG")
    bestSplitColTrain = X[0]
    bestSplitValueTrain = X[1]
    allValuesTrain = train[0] # list of lists
    allClassesTrain = train[1]

    allClassesTest = test[1]

    data = []
    classes = []

    colXofData = column(test[0], bestSplitColTrain)
    for x in range(0, len(allClassesTest)):
        if(colXofData[x] >= bestSplitValueTrain):
            data.append(colXofData[x])
            classes.append(0)
        elif(colXofData[x] < bestSplitValueTrain):
            data.append(colXofData[x])
            classes.append(1)
    return(classes)


def classifyG(train, test):
    """با استفاده از معیار GINI یک درخت تصمیم منفرد ایجاد می کند
     و مجموعه داده ها ، و لیستی از کلاس های پیش بینی شده برای آزمون مجموعه داده را برمی گرداند

     آرگس:
         train: یک تاپل (X ، y) ، جایی که X داده است ، y کلاس ها
         test: مجموعه آزمون ، همان قالب قطار

     بازده:
         لیستی از کلاسهای پیش بینی شده برای مشاهدات در آزمون (به ترتیب)
    """
    X = bestSplit(train, "GINI")
    bestSplitColTrain = X[0]
    bestSplitValueTrain = X[1]
    allValuesTrain = train[0]  # list of lists
    allClassesTrain = train[1]

    allClassesTest = test[1]

    data = []
    classes = []

    colXofData = column(test[0], bestSplitColTrain)
    for x in range(0, len(allClassesTest)):
        if (colXofData[x] >= bestSplitValueTrain):
            data.append(colXofData[x])
            classes.append(0)
        elif (colXofData[x] < bestSplitValueTrain):
            data.append(colXofData[x])
            classes.append(1)
    return(classes)


def classifyCART(train, test):
    """


         train: یک تاپل (X ، y) ، جایی که X داده است ، y کلاس ها
         test: مجموعه آزمون ، همان قالب قطار


         لیستی از کلاسهای پیش بینی شده برای مشاهدات در آزمون (به ترتیب)
    """
    X = bestSplit(train, "CART")
    bestSplitColTrain = X[0]
    bestSplitValueTrain = X[1]
    allValuesTrain = train[0]  # list of lists
    allClassesTrain = train[1]

    allClassesTest = test[1]

    data = []
    classes = []

    colXofData = column(test[0], bestSplitColTrain)
    for x in range(0, len(allClassesTest)):
        if (colXofData[x] >= bestSplitValueTrain):
            data.append(colXofData[x])
            classes.append(0)
        elif (colXofData[x] < bestSplitValueTrain):
            data.append(colXofData[x])
            classes.append(1)
    return (classes)

def main():

    file = load('train.txt')
    fileTest = load('test.txt')
    IG(file, 1, 21)
    G(file, 8, 6)

    print("بهترین تقسیم information gain: {}".format(bestSplit(file, "IG")))
    print("بهترین تقسیم gini indix: {}".format(bestSplit(file, "GINI")))



    exit()

if __name__=="__main__":

    main()
