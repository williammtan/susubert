import kfp
from kfp import dsl
from kfp.components import func_to_container_op, load_component_from_url, OutputPath, InputPath

def read_df(path : InputPath(str), output_path: OutputPath(str)) -> None:
    import pandas as pd
    df = pd.read_csv(path)
    print(df)
    df.to_csv(output_path)

@dsl.pipeline(name='train pipeline')
def train_pipeline():
    read_df_op = func_to_container_op(func=read_df, packages_to_install=['pandas'])
    download = load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/0795597562e076437a21745e524b5c960b1edb68/components/google-cloud/storage/download/component.yaml')

    download_task = download('gs://data_external_backup/EXTERNAL_PRODUCTS.csv')

    read_df_op(download_task.output)

if __name__ == '__main__':
    # kfp.compiler.Compiler().compile(train_pipeline, 'train_pipline.yaml')
    client = kfp.Client(host='https://118861cf2b92c13d-dot-us-central1.pipelines.googleusercontent.com')
    client.create_run_from_pipeline_func(train_pipeline, arguments={})
