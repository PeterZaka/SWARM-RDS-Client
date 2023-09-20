import argparse
import pickle
import typing

import matplotlib.pyplot as plt


def load_pickle_file(file_name: str) -> typing.Any:
    """
    Load in the pickle file that was selected
    """
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data


def get_arguements() -> argparse.Namespace:
    """
    Get the arguements from the command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="The name of the pickle file")
    return parser.parse_args()


def visualize_occupancy_map(data: typing.Any) -> None:
    """
    Visualize the occupancy map
    """
    plt.imshow(data)
    plt.show()


def main() -> None:
    """
    Main function
    """
    args = get_arguements()
    data = load_pickle_file(args.file_name)
    if args.file_name == "occupancy_map.pickle":
        visualize_occupancy_map(data)
    else:
        print(data)


if __name__ == "__main__":
    main()