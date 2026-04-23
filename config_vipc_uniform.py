class params:
    def __init__(self): 
        #General Parameters
        self.cat = "car"
        # plane cabinet car chair lamp couch table watercraft all
        
        #Training parameters
        self.batch_size = 64
        self.nThreads = 8
        self.lr = 0.001
        self.milestones = [20, 40, 60, 80, 100, 120, 140, 160, 180]
        self.data_root = "/scratch/tbalamur/vipc_data/ShapeNetViPC-Dataset"
        self.n_epochs = 100
        self.eval_epoch = 1
        self.resume = True
        self.ckpt = "./log/AMG_PC_uniform_64_car_train_vipc_Mon Apr 20 19:39:49 2026/ckpt_82_1.9505.pt"