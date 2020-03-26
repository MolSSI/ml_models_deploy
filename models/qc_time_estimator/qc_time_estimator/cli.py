import click
from qc_time_estimator.train_pipeline import run_training

@click.group()
def main():
    pass

@main.command()
@click.option('--overwrite', is_flag=True, default=False, help='Overwrite existing model file')
@click.option('--with_accuracy', is_flag=True, default=False, help='Calculate prediction and test accuracy')
@click.option('--train_test_split', is_flag=True, default=False, help='Split data for test and tain. Not recommended for production')
def train_model(overwrite, with_accuracy, train_test_split):

    click.echo("Running model trainning...")
    run_training(overwrite=overwrite, with_accuracy=with_accuracy,
                 use_all_data=not train_test_split)

if __name__ == "__main__":
    main()