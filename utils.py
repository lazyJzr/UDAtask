import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Fault diagnosis')
    parser.add_argument('--dataset', default='./engine/',
                        help='Dataset names.')
    parser.add_argument('--num_steps', type=int, default=300,
                        help='Number of diffusion steps')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='input batch size for training')
    parser.add_argument('--num_epoch', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--class_epochs', type=int, default=50,
                        help='Number of classifier training rounds')
    parser.add_argument('--lr_diff', type=float, default=1e-3,
                        help='number of epochs to train')
    parser.add_argument('--lr_class', type=float, default=1e-2,
                        help='number of epochs to train')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Training classifier steps')
    parser.add_argument('--schedule_name', default='exponential',
                        help='Learning rate scheduling strategy')
    parser.add_argument('--domain_strategy', default='residual', choices=['residual',"noresidual"],
                        help='Domain alignment strategy')
    parser.add_argument('--scale_factor', type=int, default=4.0,
                        help='Scale factor')
    parser.add_argument('--scale_factor', type=int, default=4.0,
                        help='Scale factor')
    parser.add_argument('--width', type=int, default=256,
                        help='Time frequency graph width')
    parser.add_argument('--height', type=int, default=256,
                        help='Time frequency graph height')
    
    args = parser.parse_args()

    return args


def compute_accuracy(model, feature, labels):
    correct_pred, num_examples = 0, 0
    l = 0
    N = feature.size(0)
    total_batch = int(np.ceil(N / batch_size))
    indices = np.arange(N)
    np.random.shuffle(indices)
    for i in range(total_batch):
        rand_index = indices[batch_size * i:batch_size * (i + 1)]
        features = feature[rand_index, :]

        targets = labels[rand_index]

        features = features.cuda()
        targets = targets.cuda()

        logits = model(features)
        probas = F.softmax(logits, dim=1)
        print(logits.shape, targets.shape)
        loss = criterion(logits, targets.long())
        _, predicted_labels = torch.max(probas, 1)

        num_examples += targets.size(0)
        l += loss.item()
        correct_pred += (predicted_labels == targets).sum()

    return l / num_examples, correct_pred.float() / num_examples * 100


