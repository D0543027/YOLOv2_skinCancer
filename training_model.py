from darkflow.net.build import TFNet
if __name__ == "__main__":
    
    #If first training, change load to bin/yolov2-tiny.weights
    options = {"model": "cfg/tiny-yolov2-custom.cfg", "load": -1,
               "gpu":0.7,"dataset": "data/training/images","annotation": "data/training/annotations", "train":True}

    tfnet = TFNet(options)
    tfnet.train()
    tfnet.savepb()