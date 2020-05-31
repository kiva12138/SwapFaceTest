import DataOperation
import Setting
from Model import autoencoder_A, autoencoder_B


def save_model(modelA, modelB):
    modelA.save("./modelA.h5")
    modelB.save("./modelB.h5")


if __name__ == '__main__':
    for epoch in range(Setting.epochs):
        print('Epoch {} ......'.format(epoch))
        warped_A, target_A = DataOperation.get_training_data_A()
        warped_B, target_B = DataOperation.get_training_data_B()
        loss_A = autoencoder_A.train_on_batch(warped_A, target_A)
        loss_B = autoencoder_B.train_on_batch(warped_B, target_B)
        print("lossA:{},lossB:{}".format(loss_A, loss_B))
        if epoch+1 % 10 == 0:
            save_model(autoencoder_A, autoencoder_B)
