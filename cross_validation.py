from preprocess import *
from model import *
from graph import *
import statistics

def cross_validation_sets(D, k):
    result = list()
    
    # Split edge set for training and testing
    u, v = D.edges()

    eids = np.arange(D.number_of_edges())
    eids = DEFAULT_RNG.permutation(eids)
    test_size = int(len(eids) / k)
    train_size = D.number_of_edges() - test_size
    
    edge_dict = {(int(u[i]), int(v[i])): i for i in eids}
    
    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(D.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_eids = DEFAULT_RNG.choice(len(neg_u), D.number_of_edges())
    
    for i in range(k):
        test_pos_u, test_pos_v = u[eids[test_size*i:test_size*(i+1)]], v[eids[test_size*i:test_size*(i+1)]]
        train_pos_u, train_pos_v = np.delete(u, eids[test_size*i:test_size*(i+1)]), \
                                        np.delete(v, eids[test_size*i:test_size*(i+1)])

        test_neg_u, test_neg_v = neg_u[neg_eids[test_size*i:test_size*(i+1)]], neg_v[neg_eids[test_size*i:test_size*(i+1)]]
        train_neg_u, train_neg_v = neg_u[neg_eids[test_size*(k-i):test_size*(k-i+1)]], neg_v[neg_eids[test_size*(k-i):test_size*(k-i+1)]]
        train_neg_u, train_neg_v = torch.tensor(train_neg_u), torch.tensor(train_neg_v)

        test_indexes = []
        for i in range(len(test_pos_u)):
            test_indexes.append(edge_dict[(int(test_pos_v[i]), int(test_pos_u[i]))])
            test_indexes.append(edge_dict[(int(test_pos_u[i]), int(test_pos_v[i]))])

        train_g = dgl.remove_edges(D, np.array(test_indexes))

        train_pos = torch.cat((train_pos_u, train_pos_v)), torch.cat((train_pos_v, train_pos_u))
        train_neg = torch.cat((train_neg_u, train_neg_v)), torch.cat((train_neg_v, train_neg_u))

        train_pos_g = dgl.graph(train_pos, num_nodes=D.number_of_nodes())
        train_neg_g = dgl.graph(train_neg, num_nodes=D.number_of_nodes())

        test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=D.number_of_nodes())
        test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=D.number_of_nodes())

        result.append((train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g))
    return result



def logreg_cross_validation(G, folds, weight_decays, learning_rates, epochs):
    aucs = dict()
    
    for weight_decay in weight_decays:
        for learning_rate in learning_rates:
            for epoch in epochs:
                    local_aucs = list()
                    print(f"Learning rate: {learning_rate}, epochs: {epoch}, weight decay: {weight_decay}")

                    #train-test split with sampling negative edges
                    cross_validation_folds = cross_validation_sets(G, k=folds)

                    #defining identity model and predictor model to handle final node embeddings            
                    model = Identity()
                    predictor = LogisticRegression(600, 1)

                    #training
                    for i in range(len(cross_validation_folds)):
                        optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate, weight_decay=weight_decay)
                        train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = cross_validation_folds[i]

                        auc = training_loop(model, predictor, optimizer, compute_loss_bce_logits, epoch, train_g, train_pos_g, train_neg_g, test_pos_g, train_neg_g)

                        local_aucs.append(auc)
                    key = (weight_decay, learning_rate, epoch)
                    aucs[key] = statistics.mean(local_aucs)

    sorted_aucs = sorted(aucs.items(), key=lambda x:x[1], reverse=True)
    weight_decay, learning_rate, epoch = sorted_aucs[0][0]
    return weight_decay, learning_rate, epoch



def gcn_cross_validation(G, folds, weight_decays, learning_rates, epochs, channels):
    aucs = dict()
    
    for weight_decay in weight_decays:
        for learning_rate in learning_rates:
            for epoch in epochs:
                for channel in channels:
                    local_aucs = list()
                    print(f"Learning rate: {learning_rate}, epochs: {epoch}, weight decay: {weight_decay}")

                    #train-test split with sampling negative edges
                    cross_validation_folds = cross_validation_sets(G, k=folds)

                    #defining GCN model and predictor model to handle final node embeddings
                    model = GCN(G.ndata['feat1'].shape[1], channel)
                    predictor = NNPredictor(channel)

                    #training
                    for i in range(len(cross_validation_folds)):
                        optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate, weight_decay=weight_decay)
                        train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = cross_validation_folds[i]

                        auc = training_loop(model, predictor, optimizer, compute_loss_bce_logits, epoch, train_g, train_pos_g, train_neg_g, test_pos_g, train_neg_g)

                        local_aucs.append(auc)
                    key = (weight_decay, learning_rate, epoch, channel)
                    aucs[key] = statistics.mean(local_aucs)

    sorted_aucs = sorted(aucs.items(), key=lambda x:x[1], reverse=True)
    weight_decay, learning_rate, epoch, channel = sorted_aucs[0][0]
    return weight_decay, learning_rate, epoch, channel


