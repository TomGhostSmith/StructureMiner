from StructureMiner import StructureMiner
from config import config
import torch
import random
import IOUtils
import pickle

def getTrainset(device, trainCount, validateCount):
    posset = list()
    negset = list()
    posValidSet = list()
    negValidSet = list()
    positiveCount = trainCount / 2
    negativeCount = trainCount / 2
    posValidateCount = validateCount / 2
    negValidateCount = validateCount / 2
    # pendingCount = 0
    with open(config.trainsetPath) as fp:
        line = fp.readline()   # skip the title line
        line = fp.readline().strip()
        while (line):
            terms = line.split(',')
            name = terms[1]
            refSeq = terms[2]
            altSeq = terms[3]
            startPos = int(terms[4]) - 1
            if (len(refSeq) == len(altSeq)):  # only focus on missense
                significance = terms[-1].lower()
                significance = 1 if ('pathogenic' in significance) else 0
                if (significance == 1):
                    if (len(posset) < positiveCount):
                        significance = torch.tensor([significance], dtype=torch.float, device=device)
                        posset.append(((name, refSeq, altSeq, startPos), significance))
                    elif (len(posValidSet) < posValidateCount):
                        significance = torch.tensor([significance], dtype=torch.float, device=device)
                        posValidSet.append(((name, refSeq, altSeq, startPos), significance))
                    # else:
                    #     pendingCount += 1
                else:
                    if (len(negset) < negativeCount):
                        significance = torch.tensor([significance], dtype=torch.float, device=device)
                        negset.append(((name, refSeq, altSeq, startPos), significance))
                    elif (len(negValidSet) < negValidateCount):
                        significance = torch.tensor([significance], dtype=torch.float, device=device)
                        negValidSet.append(((name, refSeq, altSeq, startPos), significance))
                    # print(len(negset))
                


                # if (pendingCount > 10000)

                # if (positiveCount == totalCount / 2 and negativeCount < totalCount / 2):
                #     times = totalCount / 2
                    

            line = fp.readline().strip()
            # if (len(posset) == positiveCount and len(negset) == negativeCount and len(posValidSet) == posValidateCount and len(negValidSet) == negValidateCount):
            if (len(posset) == positiveCount and len(negset) == negativeCount and len(posValidSet) == posValidateCount and len(negValidSet) == negValidateCount):
                break

    trainset = posset + negset
    validateSet = posValidSet + negValidSet
    random.shuffle(trainset)
    random.shuffle(validateSet)
    return trainset, validateSet

# def splitDataset(dataset, ratio):
#     # random.shuffle(dataset)
#     threshold = int(len(dataset) * ratio)
#     return dataset[:threshold], dataset[threshold:]

def test(modelPath):
    device = torch.device('cuda:0')
    trainset, validationset = getTrainset(device, 5000, 250)
    with open(modelPath, 'rb') as fp:
        structureMiner:StructureMiner = pickle.load(fp)
    
    structureMiner.test(validationset)

def train(modelPath=None):
    device = torch.device('cuda:1')
    trainset, validationset = getTrainset(device, 5000, 250)
    # pathoCount = sum([t[1] for t in validationset]).item()
    # IOUtils.showInfo(f"patho = {pathoCount}, benign = {len(validationset) - pathoCount}")
    args = {
        "lr": 1e-5,
        "iter": 15000,
        "validationStep": 300,
        "checkpointStep": 1500,
        "device": device
    }
    if (modelPath):
        with open(modelPath, 'rb') as fp:
            structureMiner:StructureMiner = pickle.load(fp)
        args['originIter'] = int(modelPath.split('_')[-1][:-4])
        structureMiner.setArgs(args)
    else:
        structureMiner = StructureMiner(args)
    structureMiner.train(trainset, validationset)

def main():
    # train()
    train("./log/05130154/checkpoint_7500.pkl")
    # test("./log/202405130109/checkpoint_30.pkl")



    

if (__name__ == '__main__'):
    main()