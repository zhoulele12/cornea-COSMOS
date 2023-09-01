
import time
import torch
from torchvision.utils import save_image
from COSMOS import cosmosModel
from data import create_dataset

if __name__ == '__main__':
    batch_size = 1
    # opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(batch_size)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    # model = create_model(opt)      # create a model given opt.model and other options
    device = torch.device("cuda")
    model = cosmosModel()
    model.to(device)
    model.eval()
    model.load_networks(50,'Sep1st')
    # summary(model, input_size=(batch_size, 1, 28, 28),device='cpu')
    # model.setup(opt)               # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
    # model.update_learning_rate()    # update learning rates in the beginning of every epoch.
    for i, data in enumerate(dataset):  # inner loop within one epoch
        # print(data['A']['image'].size())
        # print(data['A']['mask'].size())
        # print(data['B']['image'].size())
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.forward()
        save_image(model.real_A_image,'test_out/test%drealA.png'%i)
        save_image(model.fake_B,'test_out/test%dfakeB.png'%i)
        save_image(model.real_B_image, 'test_out/test%drealB.png'%i)
        save_image(model.fake_A, 'test_out/test%dfakeA.png'%i)


    # print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
