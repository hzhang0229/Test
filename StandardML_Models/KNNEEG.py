
class KNNEEG:

    def __init__(self, model_name, **model_params):
        self.model_name = model_name
        self.model = None

        if self.model_name == "KNNEEG":
            from KNN import returnmodel
            self.model = returnmodel()


    def fit(self, trainX, trainY, validX, validY):
        trainX = trainX.reshape((-1, 258))  # TODO: A hack for now
        print("Using KNNEEG")
        self.model.train()

    def predict(self, testX):
        testX = testX.reshape((-1, 258))  # TODO: A hack for now
        print("Evaluating")
        return self.model.eval()

    def save(self, path):
        # save the model to disk
        import pickle
        filename = path + self.model_name + '.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def load(self, path):
        # Test
        # load the model from disk
        import pickle
        filename = path + self.model_name + '.sav'
        self.model = pickle.load(open(filename, 'rb'))