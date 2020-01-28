# Add parent directory to sys path to be able to import modules from the root dir
import sys
sys.path.append('.')

# Custom imports
from model.utils import *
from model.transformer import Transformer
from model.bracketing import IdentityChunker, NNSimilarityChunker
from model.generators import IdentityGenerator
from model.classifiers import AttentionClassifier, SeqPairAttentionClassifier
from model.model import MultiTaskNet, End2EndModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device being used: {device}")

transformer_net = Transformer()
bracketing_net = NNSimilarityChunker()
generator_net = IdentityGenerator()

task1_classifier = AttentionClassifier(embedding_dim=768,
                                       sentset_size=2,
                                       batch_size=32,
                                       dropout=0.3,
                                       n_sentiments=4,
                                       pool_mode='concat',
                                       device=device)

task2_classifier = SeqPairAttentionClassifier(embedding_dim=768,
                                              num_classes=4,
                                              batch_size=32,
                                              dropout=0.3,
                                              n_attention_vecs=4,
                                              pool_mode='concat',
                                              device=device)

multitask_net = MultiTaskNet(task1_classifier, task1_classifier, device=device)

end2end_model = End2EndModel(transformer=transformer_net,
                             bracketer=bracketing_net,
                             generator=generator_net,
                             multitasknet=multitask_net,
                             device=device)

batch_sequence = ['this is one sentence', 'this is a second sentence', 'this is a third sentnen.']

optimizer = torch.optim.Adam(end2end_model.parameters())
output = end2end_model.forward(batch_sequence)
print(output.size())
raise ValueError("Whoha it workeddd!!!")
