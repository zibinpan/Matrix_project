from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse


class TuckerExperiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200, 
                 num_iterations=500, batch_size=128, decay_rate=0., 
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0.,
                 d=None, dataset_name=None):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.d = d
        self.dataset_name = dataset_name
        
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = torch.as_tensor(er_vocab_pairs[idx:idx+self.batch_size], dtype=torch.long).to(self.device)
        targets = np.zeros((len(batch), len(self.d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.as_tensor(targets, dtype=torch.float32).to(self.device)
        return batch, targets

    
    def evaluate(self, model, data, epoch):
        hits = []
        ranks = []
        for i in range(10): hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(self.d.data))

        print("Number of data points: %d" % len(test_data_idxs))
        
        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.as_tensor(data_batch[:,0], dtype=torch.long).to(self.device)  # heads
            r_idx = torch.as_tensor(data_batch[:,1], dtype=torch.long).to(self.device)  # relations
            e2_idx = torch.as_tensor(data_batch[:,2], dtype=torch.long).to(self.device)  # tails
            predictions = model.forward(e1_idx, r_idx)
            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            _, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
                ranks.append(rank+1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))
        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        # print('Mean rank: {0}'.format(np.mean(ranks)))
        metrics = np.array([np.mean(1./np.array(ranks)), np.mean(hits[9]), np.mean(hits[2]), np.mean(hits[0])])
        np.savetxt('MRR_Hits_'+ self.dataset_name + '_' + str(epoch) + '.csv', metrics, delimiter=',')



    def train_and_eval(self):
        print("Training the TuckER model...")
        self.entity_idxs = {self.d.entities[i]:i for i in range(len(self.d.entities))}
        self.relation_idxs = {self.d.relations[i]:i for i in range(len(self.d.relations))}

        train_data_idxs = self.get_data_idxs(self.d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))
        model = TuckER(self.d, self.ent_vec_dim, self.rel_vec_dim, self.device, **self.kwargs).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        epoch_loss = []

        print("Starting training...")
        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()  # 切换成训练模式
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                
                opt.zero_grad()
                e1_idx = data_batch[:,0]  # 获得heads
                r_idx = data_batch[:,1]  # 获得relations
                
                predictions = model.forward(e1_idx, r_idx)  # 预测tails
                
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print(it)
            # print(time.time()-start_train)
            # print(np.mean(losses))
            epoch_loss.append(np.mean(losses))

            if it == 50 or it == 200 or it == 350 or it == 500:
                model.eval()  # 切换成测试模式
                with torch.no_grad():
                    # print("Validation:")
                    # start = time.time()
                    # self.evaluate(model, self.d.valid_data)
                    # print('耗时：', time.time() - start)
                    print("Test:")
                    start_test = time.time()
                    self.evaluate(model, self.d.test_data, it)
                    print(time.time()-start_test)
        print('Finish training.')
        
        
        np.savetxt('loss_'+ self.dataset_name +'.csv', epoch_loss, delimiter=',')

           

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="WN18RR", nargs="?",
                    help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                    help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    help="Amount of label smoothing.")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    data_dir = "data/%s/" % dataset_name
    torch.backends.cudnn.deterministic = True 
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 
    d = Data(data_dir=data_dir, reverse=True)  # 获取data
    experiment = TuckerExperiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr, 
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing,
                            d=d, dataset_name=dataset_name)
    # print(len(d.data))
    experiment.train_and_eval()

