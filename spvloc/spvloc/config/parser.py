import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,help='path to image file',default="/home/godling/Downloads/deneme-main/layout_pred/example.jpg")
    parser.add_argument("-c", "--config_file", default="", type=str)
    parser.add_argument("-l", "--checkpoint_file", default="", type=str)
    parser.add_argument("-t", "--test_ckpt", default="", type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args
