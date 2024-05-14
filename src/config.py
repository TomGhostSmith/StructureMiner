class Config():
    def __init__(self) -> None:
        self.headCount = 8
        self.embeddingSize = 1024
        self.secondaryStructureJsonPath = './data/preprocess/geneStructure.json'
        self.comparativeEmbedding = 32
        # self.trainsetPath = '/Data/1000GenomesData/variant_summary-missense.csv'
        self.trainsetPath = '/Data/1000GenomesData/variant_summary.csv'
        # self.trainsetPath = '/Data/1000GenomesData/GRAMtest.csv'
        self.logPath = './log'

config = Config()