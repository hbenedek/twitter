from preprocess import *
from model import *
from graph import *
from cross_validation import *
from clustering import *

if __name__ == "__main__":
    question = input("Do you want to run the preprocessing step? [y/n] ")
    if question == ("y"):
        lang = input("Provide language code (e.g., 'en'): ")
        # returns two dataframe containing all preprocessed, embedded tweets with usernames
        df_mep, df_lobby = process_wrapper(lang)
        
        # returns two dataframes containing mbedding vectors by users
        df_mep_users, df_lobby_users = get_embedding_by_user(df_mep, df_lobby)
        df_mep_users, df_lobby_users = average_embedding_by_user(df_mep, df_lobby)
        
        # build graph based on data
        G  = build_graph(main_dir = '../lobbymeptweets/', lobby_dir = 'lobbytweetsep8', mep_dir = 'meptweetsep8')
        D = build_dgl_graph(df_mep_users, df_lobby_users, G)
        dump_pickle(D, "DGL_graph.pickle")
        
        # baseline model (maximum inner product)
        B = build_bipartite_edges(df_mep_users, df_lobby_users)
        baseline_df = graph_to_edge_dataframe(B, G)
        print_baseline_auc(baseline_df)
    
    question = input("Do you want to run an unsupervised model or one of the supervised ones? [u/s]")
    if(question.lower() == 's'.lower()):
        question = input("Do you want to run GCN or logistic regression? [GCN/log] ")
        if question.lower() == "GCN".lower():
            #loading DGL object from pickle
            D = load_pickle("DGL_graph.pickle")
            
            train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = train_test_split(D)
            #train-test split with sampling negative edges    
            
            #hyperparameters for cross validation
            weight_decays = [0]
            learning_rates = [0.01]
            epochs = [100]
            channels = [32]
            folds = 5
            
            #run cross validation
            #weight_decay, learning_rate, epoch, channel = gcn_cross_validation(train_g, folds, weight_decays, learning_rates, epochs, channels)
            weight_decay, learning_rate, epoch, channel = 0, 0.01, 100, 32
            # defining GCN model
            model = GCN(train_g.ndata['feat1'].shape[1], channel)
            #defining predictor model to handle final node embeddings
            predictor = NNPredictor(channel)
            optimizer = torch.optim.Adam(itertools.chain(model.parameters(), predictor.parameters()), lr=learning_rate)
            
            # training   
            training_loop(model, predictor, optimizer, compute_loss_bce_logits, epoch, train_g, train_pos_g, train_neg_g, test_pos_g, train_neg_g)
            
        elif question.lower() == "log".lower():
            #loading DGL object from pickle
            D = load_pickle("DGL_graph.pickle")
            
            #train-test split with sampling negative edges
            train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = train_test_split(D)
            
            #hyperparameters for cross validation
            weight_decays = [0, 0.01, 0.001, 0.0001]
            learning_rates = [0.02, 0.01, 0.001, 0.0001]
            epochs = [100, 200, 500, 1000]
            folds = 5
            
            #run cross validation
            #weight_decay, learning_rate, epoch = logreg_cross_validation(train_g, folds, weight_decays, learning_rates, epochs)
            weight_decay, learning_rate, epoch  = 0.01, 0.001, 1000
            print(f"Using learning rate: {learning_rate}, epochs: {epoch}, weight decay: {weight_decay}")

            #defining identity model and predictor model to handle final node embeddings
            model = Identity()
            predictor = LogisticRegression(600, 1)
            optimizer = torch.optim.Adam(itertools.chain(model.parameters(), predictor.parameters()), lr=learning_rate, weight_decay=weight_decay)
            
            #training
            training_loop(model, predictor, optimizer, compute_loss_bce, epoch, train_g, train_pos_g, train_neg_g, test_pos_g, train_neg_g)
    elif question.lower() == 'u'.lower():
        #Import the bipartite graph
        G = nx.read_edgelist("./graph-29-11.bz2")
        node_to_community = community_louvain.best_partition(G)
        
        # returns two dataframe containing all preprocessed, embedded tweets with usernames
        lang='en'
        df_mep, df_lobby = process_wrapper(lang)
        df=pd.concat([df_mep,df_lobby])[['id','text']]
        df=df.set_index('id')
        #Run the louvain algorithm to cluster the graph
        
        s=set(v for (k,v) in node_to_community.items())
        for cluster_num in list(s):
            print("topic in cluster  ", cluster_num,end='\n')
            print(topic_per_cluster(cluster_num,df))
            print('\n')
