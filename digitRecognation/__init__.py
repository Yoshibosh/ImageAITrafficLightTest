from __future__ import print_function
import argparse

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision
from PIL import Image
from torch.nn.functional import normalize
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.io import ImageReadMode

from Net import Net, test, train


# Training settings
def main():
    model_path = 'mnist_cnn.pt'
    img_path = 'Images/RFpasport_cifra_8.png'
    img = Image.open(img_path).convert('L')
    convert_tensor = transforms.ToTensor()
    print(img)

    args = modelArgs()
    device, train_kwargs, test_kwargs = initDeviceAndKwargs(args)

    model = Net().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    train_loader, test_loader, mnist_img = initMnistDataSet(train_kwargs, test_kwargs)
    test(model, device, test_loader)

    tensor = torchvision.io.read_image(img_path, ImageReadMode.GRAY).type(torch.float)
    tensor = normalize(tensor)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform(img).float()

    mnist_img = mnist_img[None, :]
    tensor = tensor[None, :]

    # plt.imshow(mnist_img[0, 0])
    plt.imshow(tensor[0, 0])
    plt.show()
    # print("tensor : \n", tensor)
    with torch.no_grad():
        print("shape : \t", mnist_img.shape)
        print("result : \n", model(mnist_img.to(device)))
        print("shape : \t", tensor.shape)
        print("result : \n", model(tensor.to(device)))


def modelArgs():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    return parser.parse_args()


def initDeviceAndKwargs(args):
    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    return device, train_kwargs, test_kwargs


def initMnistDataSet(train_kwargs, test_kwargs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    print(dataset1[0][0].shape)
    print(dataset2[0][0].shape)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader, dataset2[0][0]


def startModelTraining():
    args = modelArgs()
    device, train_kwargs, test_kwargs = initDeviceAndKwargs(args)
    train_loader, test_loader = initMnistDataSet(train_kwargs, test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
