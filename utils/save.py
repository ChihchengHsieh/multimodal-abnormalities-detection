def get_train_data(loger):
    train_data = {}
    for k in loger.meters.keys(): 
        train_data [k] = loger.meters[k].avg
        
    return train_data