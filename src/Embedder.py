import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import numpy
from tqdm.auto import tqdm
from config import config
import json
import IOUtils

torch.cuda.set_device(0)

class Embedder(torch.nn.Module):
    def __init__(self, device) -> None:
        super(Embedder, self).__init__()
        self.device = device
        # modelName = 'Rostlab/prot_bert'
        modelName = '/Software/PythonCache/huggingface/hub/models--Rostlab--prot_bert/snapshots/7a894481acdc12202f0a415dd567f6cfdb698908'
        self.tokenizer = AutoTokenizer.from_pretrained(modelName, do_lower_case=False)
        self.model = AutoModel.from_pretrained(modelName)
        self.fe = pipeline('feature-extraction', model=self.model, tokenizer=self.tokenizer,device=device)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.headDimension = config.embeddingSize // config.headCount

        self.query = torch.nn.Linear(config.embeddingSize, config.embeddingSize).to(self.device)
        self.key = torch.nn.Linear(config.embeddingSize, config.embeddingSize).to(self.device)
        self.value = torch.nn.Linear(config.embeddingSize, config.embeddingSize).to(self.device)

        self.outLinear = torch.nn.Linear(config.embeddingSize, config.embeddingSize).to(self.device)

        # get context from json
        with open(config.secondaryStructureJsonPath) as fp:
            self.geneStructure = json.load(fp)
    
    def setDevice(self, device):
        self.fe = pipeline('feature-extraction', model=self.model, tokenizer=self.tokenizer,device=device)
        self.device = device

    def forward(self, sequence, geneName, pos, attentionScoreDevice):
        # get origin embedding
        # IOUtils.showInfo('start BERT')
        tokenizedSequence = ' '.join(list(sequence))
        originEmbedding = torch.tensor(self.fe(tokenizedSequence)[0][1:-1], device=self.device)

        # if (isinstance(sequence, list)):  # this will no longer support batch
        #     raise Exception()

        seqLength = len(sequence)
        # seqLength = torch.tensor([len(seq) for seq in sequence])
        # IOUtils.showInfo('prepare Attention')

        queries = self.query(originEmbedding).to(attentionScoreDevice)
        keys = self.key(originEmbedding).to(attentionScoreDevice)
        values = self.value(originEmbedding).to(attentionScoreDevice)

        queries = queries.view(seqLength, config.headCount, self.headDimension).transpose(0, 1)
        keys = keys.view(seqLength, config.headCount, self.headDimension).transpose(0, 1)
        values = values.view(seqLength, config.headCount, self.headDimension).transpose(0, 1)


        # IOUtils.showInfo('get divider')

        # get divider
        # originally, dividers begins with 0
        # here, we define context: geq than left, less than right. That is to say, [a, b)
        structure = self.geneStructure.get(geneName)
        if (structure != None):
            dividers:list = structure['divider']
            dividers.insert(0, 0)
            if (dividers[-1] != seqLength):
                dividers.append(seqLength)
        else:
            dividers = [0, seqLength]
        
        ranges = list()
        for begin, end in zip(dividers[:-1], dividers[1:]):
            if (begin < seqLength):
                if (end <= seqLength):
                    ranges += [range(begin, end)] * (end - begin)
                else:
                    ranges += [range(begin, seqLength)] * (seqLength - begin)

            
        # IOUtils.showInfo('start Attention')

        # init attention score
        # attentionScore = torch.zeros((config.headCount, seqLength, seqLength), device = originEmbedding.device)
        attentionScore = torch.zeros((config.headCount, seqLength, seqLength), device = attentionScoreDevice)
        for head in range(config.headCount):
            for idx in range(seqLength):
                contextRange = ranges[idx]
                # if (head >= attentionScore.shape[0] or idx >= attentionScore.shape[1] or max(contextRange) >= attentionScore.shape[2]):
                try:
                    attentionScore[head, idx, contextRange] = torch.matmul(
                        queries[head, idx],
                        keys[head, contextRange].transpose(0, 1)
                    )
                except:
                    IOUtils.showInfo(f'gene={geneName}, head={head}, idx={idx}, contextRange={contextRange}, scoreShape={attentionScore.shape}', 'ERROR')

        # IOUtils.showInfo('sum Attention')


        attentionWeights = torch.softmax(attentionScore, dim=-1)
        weightedValues = torch.matmul(attentionWeights, values)
        weightedValues = weightedValues.transpose(0, 1).contiguous().view(seqLength, config.embeddingSize).to(self.device)

        result = self.outLinear(weightedValues)

        mainContext = ranges[pos]


        # TODO: use protein embedding, or secondary structure embedding?
        result = torch.mean(result[mainContext], dim=0)
        # IOUtils.showInfo('end Attention')
        return result





        # a = self.fe('AET')
        # b = self.fe('A E T')
        # print(a)
        # print(b)
        

# e = Embedder()
# e.forward('MALRGVSVRLLSRGPGLHVLRTWVSSAAQTEKGGRTQSQLAKSSRPEFDWQDPLVLEEQLTTDEILIRDTFRTYCQERLMPRILLANRNEVFHREIISEM' + 
# 'GELGVLGPTIKGYGCAGVSSVAYGLLARELERVDSGYRSAMSVQSSLVMHPIYAYGSEEQRQKYLPQLAKGELLGCFGLTEPNSGSDPSSMETRAHYNSS' + 
# 'NKSYTLNGTKTWITNSPMADLFVVWARCEDGCIRGFLLEKGMRGLSAPRIQGKFSLRASATGMIIMDGVEVPEENVLPGASSLGGPFGCLNNARYGIAWG' + 
# 'VLGASEFCLHTARQYALDRMQFGVPLARNQLIQKKLADMLTEITLGLHACLQLGRLKDQDKAAPEMVSLLKRNNCGKALDIARQARDMLGGNGISDEYHV' + 
# 'IRHAMNLEAVNTYEGTHDIHALILGRAITGIQAFTASK', 'GCDH')