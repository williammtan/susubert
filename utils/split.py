import argparse
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data')
    parser.add_argument('-s', '--split', type=float, nargs='+', default=[0.8, 0.1, 0.1])
    parser.add_argument('-f', '--out-format', default='save/{set}.txt')
    args = parser.parse_args()

    data = open(args.data).read().split('\n')
    split = args.split
    train, val = train_test_split(data, train_size=split[0], test_size=split[1] + split[2])
    val, test = train_test_split(val, test_size=split[2] / (split[1] + split[2]))

    for (set_name, set_value) in {'train': train, 'val': val, 'test': test}.items():
        output = open(args.out_format.format(set=set_name), 'w')
        for line in set_value:
            output.write(line + '\n')
