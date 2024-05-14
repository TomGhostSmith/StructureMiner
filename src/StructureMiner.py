import torch

from Embedder import Embedder
from SiameseNetwork import SiameseNetwork
from Classifier import Classifier
from config import config
import random
import os
import datetime
from tqdm import tqdm
import pickle

import pynvml  # monitor for NVIDIA GPU. Modify batch size if GRAM is sufficient
pynvml.nvmlInit()

import IOUtils


class StructureMiner(torch.nn.Module):
    def __init__(self, args) -> None:
        super(StructureMiner, self).__init__()
        # parse args
        self.learningRate = args['lr'] if ('lr' in args) else 1e-3
        self.batchSize = args['batch'] if ('batch' in args) else 1
        # self.epochCount = args['epoch'] if ('epoch' in args) else 100
        self.iterCount = args['iter'] if ('iter' in args) else 10000
        self.validationStep = args['validationStep'] if ('validationStep' in args) else 100
        self.checkpointStep = args['checkpointStep'] if ('checkpointStep' in args) else 1000
        self.device = args['device'] if ('device' in args) else torch.device('cuda:0')
        self.attentionDevices = args['attentionDevices'] if ('attentionDevices' in args) else [self.device]
        self.autoBatch = args['autoBatch'] if ('autoBatch' in args) else False
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)

        if (self.autoBatch):
            self.batchSize = 1
            self.attentionDevices = [self.device]

        self.embedder = Embedder(self.device)
        self.siameseNetwork = SiameseNetwork(self.device)
        self.classifier = Classifier('LR', self.device)
        self.optimizer = torch.optim.SGD(
            list(self.embedder.parameters()) + 
            list(self.siameseNetwork.parameters()) + 
            list(self.classifier.parameters()),
            lr=self.learningRate,
            momentum=0.9)
        self.iter = 0
        
    def setArgs(self, args):
        self.learningRate = args['lr'] if ('lr' in args) else 1e-3
        self.batchSize = args['batch'] if ('batch' in args) else 1
        # self.epochCount = args['epoch'] if ('epoch' in args) else 100
        self.iterCount = args['iter'] if ('iter' in args) else 10000
        self.validationStep = args['validationStep'] if ('validationStep' in args) else 100
        self.checkpointStep = args['checkpointStep'] if ('checkpointStep' in args) else 1000
        self.device = args['device'] if ('device' in args) else torch.device('cuda:0')
        self.iter = args['originIter'] if ('originIter' in args) else 0
        self.attentionDevices = args['attentionDevices'] if ('attentionDevices' in args) else [self.device]
        self.autoBatch = args['autoBatch'] if ('autoBatch' in args) else False
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)

        if (self.autoBatch):
            self.batchSize = 1
            self.attentionDevices = [self.device]

        self.embedder.to(self.device)
        self.siameseNetwork.to(self.device)
        self.classifier.to(self.device)
        self.embedder.setDevice(self.device)

        self.optimizer = torch.optim.SGD(
            list(self.embedder.parameters()) + 
            list(self.siameseNetwork.parameters()) + 
            list(self.classifier.parameters()),
            lr=self.learningRate,
            momentum=0.9)
        # torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()


    def train(self, trainset, validationset):
        # for i in range(1, self.epochCount + 1):
        logPath = f"{config.logPath}/{datetime.datetime.now().strftime('%m%d%H%M')}" 
        os.mkdir(logPath)
        while (self.iter < self.iterCount):
            # progressTrain = tqdm(total=len(trainset), desc="Training", unit=" iter")
            progressTrain = tqdm(total=self.iterCount, desc="Training", unit="iter", initial=self.iter)
            length = min(self.iterCount - self.iter, self.validationStep)
            random.shuffle(trainset)
            totalLoss = 0

            batchLoss = 0

            currentBatchSize = 0

            for x, y in trainset[:length]:
                geneName, refSeq, altSeq, startPos = x
                # IOUtils.showInfo('-- Get Embedding --')
                attentionDevice = self.attentionDevices[self.iter % self.batchSize]
                refEmbedding = self.embedder.forward(refSeq, geneName, startPos, attentionDevice)
                altEmbedding = self.embedder.forward(altSeq, geneName, startPos, attentionDevice)
                # IOUtils.showInfo('-- Siamese Network --')
                embedding = self.siameseNetwork.forward(refEmbedding, altEmbedding)
                # IOUtils.showInfo('-- Classifier --')
                result = self.classifier.forward(embedding)
                diff = torch.abs(result - y)
                # resultClz = 0 if result < 0.5 else 1
                # IOUtils.showInfo('-- Get Loss --')
                loss = self.classifier.getLoss(y)
                # IOUtils.showInfo('-- Backward --')

                batchLoss += loss
                
                # loss.backward()
                # self.optimizer.step()
                # IOUtils.showInfo('-- End iter --')
                self.iter += 1
                currentBatchSize += 1

                # totalLoss += loss.item()
            
                # backward after a batch
                normalBatch = (not self.autoBatch) and self.iter % self.batchSize == 0
                autoBatch = self.autoBatch
                if (autoBatch):
                    pass
                    # IOUtils.showInfo(f"{memory} MB is free")
                    memory = int(pynvml.nvmlDeviceGetMemoryInfo(self.handle).free/1048576)
                    autoBatch = (memory < 10000)
                    logs = {"loss": loss.detach().item(), "diff": diff.detach().item(), 'mem': memory, 'bs': currentBatchSize}
                else:
                    logs = {"loss": loss.detach().item(), "diff": diff.detach().item()}

                progressTrain.update(1)
                progressTrain.set_postfix(**logs)

                if (normalBatch or autoBatch):
                    self.optimizer.zero_grad()
                    batchLoss.backward()
                    self.optimizer.step()
                    totalLoss += batchLoss.detach().item()
                    batchLoss = 0
                    currentBatchSize = 0

                    if (autoBatch):
                        torch.cuda.empty_cache()


            
            progressTrain.close()

            if (batchLoss != 0):  # do batch backward for non-complete batch
                self.optimizer.zero_grad()
                batchLoss.backward()
                self.optimizer.step()
                totalLoss += batchLoss.detach().item()
                batchLoss = 0
                currentBatchSize = 0

            if (self.iter % self.validationStep == 0):
                accuracy = 0
                progressValidation = tqdm(total=len(validationset), desc="Validating", unit="iter")
                # res = list()
                totalDiff = 0
                for x, y in validationset:
                    predictY = self.predict(x)
                    progressValidation.update(1)
                    resultClz = 0 if predictY < 0.5 else 1
                    # res.append((x[0], predictY, y))
                    totalDiff += abs(predictY - y).item()
                    if (resultClz == y):
                        accuracy += 1
                progressValidation.close()
                IOUtils.showInfo(f'Iter {self.iter} finished. Loss = {totalLoss / length}. Validation accuracy: {(accuracy/len(validationset)*100):.2f}%, diff = {totalDiff / len(validationset)}')
                # for name, preY, y in res:
                #     IOUtils.showInfo(f'{name}: {preY.item()}, {y.item()}, abs={abs(preY - y).item()}')
            
            if (self.iter % self.checkpointStep == 0):
                h = self.handle
                self.handle = None
                with open(f"{logPath}/checkpoint_{self.iter}.pkl", 'wb') as fp:
                    pickle.dump(self, fp)
                self.handle = h

    def predict(self, x):
        with torch.no_grad():
            geneName, refSeq, altSeq, startPos = x
            refEmbedding = self.embedder.forward(refSeq, geneName, startPos, self.device)
            altEmbedding = self.embedder.forward(altSeq, geneName, startPos, self.device)
            embedding = self.siameseNetwork.forward(refEmbedding, altEmbedding)
            result = self.classifier.forward(embedding)
            # resultClz = 0 if result < 0.5 else 1
            return result
        
    def test(self, testset):
        progress = tqdm(total=len(testset), desc="Testing", unit="iter")
        accuracy = 0
        # res = list()
        for x, y in testset:
            predictY = self.predict(x)
            resultClz = 0 if predictY < 0.5 else 1
            # res.append((x[0], predictY, y))
            progress.update(1)
            if (resultClz == y):
                accuracy += 1

        progress.close()
            
        IOUtils.showInfo(f'accuracy: {(accuracy/len(testset)*100):.2f}%')
        # for name, preY, y in res:
        #     IOUtils.showInfo(f'{name}: {preY.item()}, {y.item()}, abs={abs(preY - y).item()}')