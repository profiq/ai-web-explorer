import argparse


parser = argparse.ArgumentParser(description="Explore a website")
parser.add_argument("domain", help="The domain to start the exploration from")


def main():
    args = parser.parse_args()
    print(args)

